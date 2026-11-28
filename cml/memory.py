"""Repo memory: an exact delta chain over one file's git history.

The first version is stored raw; every later version is coded against the
previous one with the shipped delta arm.  Decode replays the chain and
checks every SHA-256, so random access is a prefix replay, not a search.

Container `CMLH`: magic, u32 header length, JSON header, then a sequence of
u32-length-prefixed payloads.  Payload 0 is raw; payload i is a delta.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import struct
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import codec

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
MAGIC = b"CMLH"


@dataclass(frozen=True, slots=True)
class ChainConfig:
    window: int = 1 << 20
    min_match: int = 8
    cap: int = 256
    depth: int = 16


def _git(repo: Path, args: list[str], data: bytes | None = None) -> bytes:
    p = subprocess.run(
        ["git", "-C", str(repo), *args],
        input=data,
        capture_output=True,
        check=False,
    )
    if p.returncode:
        raise RuntimeError(p.stderr.decode("utf-8", "replace"))
    return p.stdout


def git_versions(repo: Path, path: str) -> tuple[list[str], list[str], list[bytes]]:
    """Return commits, blob SHAs, and blobs oldest to newest, no repeats."""
    raw = _git(repo, ["log", "--follow", "--format=%H", "--", path])
    commits = [c for c in raw.decode().split() if c]
    commits.reverse()
    if not commits:
        raise SystemExit(f"chain: no history for {path}")
    kept: list[str] = []
    shas: list[str] = []
    seen: set[str] = set()
    for commit in commits:
        p = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", f"{commit}:{path}"],
            capture_output=True,
            check=False,
        )
        if p.returncode:
            continue
        sha = p.stdout.decode().strip()
        if sha and sha not in seen:
            seen.add(sha)
            kept.append(commit)
            shas.append(sha)
    blobs = [_git(repo, ["cat-file", "blob", sha]) for sha in shas]
    return kept, shas, blobs


def _arm(cfg: ChainConfig, old: int, new: int) -> codec.DeltaArm:
    limit = max(old + new, cfg.window, 2)
    return codec.DeltaArm(
        cfg.window, cfg.min_match, cfg.cap, cfg.depth, max_distance=limit
    )


def _header(
    path: str, commits: list[str], versions: list[bytes], cfg: ChainConfig
) -> bytes:
    header = {
        "path": path,
        "commits": commits,
        "sizes": [len(v) for v in versions],
        "shas": [hashlib.sha256(v).hexdigest() for v in versions],
        **asdict(cfg),
    }
    return json.dumps(header, separators=(",", ":")).encode()


def _container(header: bytes, payloads: bytes) -> bytes:
    return MAGIC + struct.pack("<I", len(header)) + header + payloads


def _payloads(
    cfg: ChainConfig,
    versions: list[bytes],
    start: int = 0,
    prev: bytes | None = None,
) -> bytes:
    """Encode payloads from `start`; earlier payloads are reused by the caller.

    With `prev=None` the stream is a fresh chain; with `prev` set the first
    payload deltas against `prev` (a global chain appends after another file).
    """
    out = bytearray()
    i = start
    if prev is None and i == 0:
        out += struct.pack("<I", len(versions[0]))
        out += versions[0]
        i = 1
    for j in range(i, len(versions)):
        old = prev if j == start and prev is not None else versions[j - 1]
        new = versions[j]
        tokens = codec.delta_parse(
            new, old, cfg.window, cfg.min_match, cfg.cap, cfg.depth
        )
        payload = _arm(cfg, len(old), len(new)).encode(new, old, tokens)
        out += struct.pack("<I", len(payload))
        out += payload
    return bytes(out)


def encode_chain(
    versions: list[bytes], path: str, commits: list[str], cfg: ChainConfig
) -> bytes:
    if not versions:
        raise ValueError("chain needs at least one version")
    return _container(_header(path, commits, versions, cfg), _payloads(cfg, versions))


def read_container(path: Path) -> tuple[dict, bytes]:
    """Return the JSON header and the raw length-prefixed payload section."""
    blob = path.read_bytes()
    if blob[:2] == b"\x1f\x8b":
        blob = gzip.decompress(blob)
    if blob[:4] != MAGIC:
        raise ValueError(f"{path}: bad magic")
    n_head = struct.unpack_from("<I", blob, 4)[0]
    header = json.loads(blob[8 : 8 + n_head])
    return header, blob[8 + n_head :]


def _row_versions(versions: list[bytes], row: dict) -> list[bytes]:
    return [versions[i] for start, end in row["ranges"] for i in range(start, end)]


def _checkpoint_name(index: int) -> str:
    return f"ckpt/{index:06d}.gz"


def sync_checkpoints(
    out: Path,
    versions: list[bytes],
    header: dict,
    interval: int,
    existing: list[dict] = (),
) -> list[dict]:
    """Write one gzip snapshot every `interval` versions; reuse valid ones."""
    ckpt_dir = out / "ckpt"
    if not interval:
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        return []
    by_index = {c["index"]: c for c in existing}
    rows = []
    for i in range(interval, len(versions), interval):
        old = by_index.get(i)
        if old is not None and old["sha"] == header["shas"][i]:
            rows.append(old)
            continue
        data = gzip.compress(versions[i], 9)
        name = _checkpoint_name(i)
        write_chain(out / name, data)
        rows.append(
            {
                "index": i,
                "file": name,
                "bytes": len(data),
                "sha": header["shas"][i],
                "raw_bytes": len(versions[i]),
            }
        )
    live = {row["file"] for row in rows}
    ckpt_dir = out / "ckpt"
    if ckpt_dir.exists():
        for path in ckpt_dir.glob("*.gz"):
            if str(path.relative_to(out)) not in live:
                path.unlink()
    return rows


def decode_chain(blob: bytes) -> tuple[dict, list[bytes]]:
    if blob[:4] != MAGIC:
        raise ValueError("chain: bad magic")
    n_head = struct.unpack_from("<I", blob, 4)[0]
    header = json.loads(blob[8 : 8 + n_head])
    at = 8 + n_head
    versions: list[bytes] = []
    for i, size in enumerate(header["sizes"]):
        (n,) = struct.unpack_from("<I", blob, at)
        at += 4
        payload = blob[at : at + n]
        at += n
        if i == 0:
            new = payload
        else:
            new = _arm(
                ChainConfig(**{k: header[k] for k in ChainConfig.__annotations__}),
                len(versions[-1]),
                size,
            ).decode(payload, size, versions[-1])
        if len(new) != size:
            raise ValueError("chain: size mismatch")
        if hashlib.sha256(new).hexdigest() != header["shas"][i]:
            raise ValueError("chain: digest mismatch")
        versions.append(new)
    if at != len(blob):
        raise ValueError("chain: trailing bytes")
    return header, versions


def payload_offset(blob: bytes, index: int) -> int:
    """Byte offset of payload `index`; scans only u32 lengths, no deltas."""
    n_head = struct.unpack_from("<I", blob, 4)[0]
    at = 8 + n_head
    for _ in range(index):
        (n,) = struct.unpack_from("<I", blob, at)
        at += 4 + n
    return at


def decode_slice(
    blob: bytes, start: int, stop: int, seed: bytes | None = None
) -> list[bytes]:
    """Decode versions `start..stop` inclusive, from a seed when `start > 0`."""
    if blob[:4] != MAGIC:
        raise ValueError("chain: bad magic")
    n_head = struct.unpack_from("<I", blob, 4)[0]
    header = json.loads(blob[8 : 8 + n_head])
    n_versions = len(header["sizes"])
    if not 0 <= start <= stop < n_versions:
        raise ValueError(f"chain: range {start}..{stop} outside {n_versions}")
    cfg = ChainConfig(**{k: header[k] for k in ChainConfig.__annotations__})
    at = payload_offset(blob, start)
    (n,) = struct.unpack_from("<I", blob, at)
    at += 4
    if seed is None:
        if start != 0:
            raise ValueError("chain: a nonzero slice needs a seed")
        seed = blob[at : at + n]
    at += n
    if (
        len(seed) != header["sizes"][start]
        or hashlib.sha256(seed).hexdigest() != header["shas"][start]
    ):
        raise ValueError(f"chain: seed digest mismatch at {start}")
    out = [seed]
    prev = seed
    for i in range(start + 1, stop + 1):
        (n,) = struct.unpack_from("<I", blob, at)
        at += 4
        payload = blob[at : at + n]
        at += n
        new = _arm(cfg, len(prev), header["sizes"][i]).decode(
            payload, header["sizes"][i], prev
        )
        if (
            len(new) != header["sizes"][i]
            or hashlib.sha256(new).hexdigest() != header["shas"][i]
        ):
            raise ValueError(f"chain: digest mismatch at {i}")
        out.append(new)
        prev = new
    return out


def load_chain(path: Path) -> tuple[dict, list[bytes]]:
    """Read a chain, accepting a gzip-wrapped `CMLH` container."""
    blob = path.read_bytes()
    if blob[:2] == b"\x1f\x8b":
        blob = gzip.decompress(blob)
    return decode_chain(blob)


def write_chain(path: Path, blob: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(blob)
    os.replace(tmp, path)


def summarize(header: dict, blob: bytes, versions: list[bytes], seconds: float) -> dict:
    raw = sum(len(v) for v in versions)
    payload = len(blob)
    gz = len(gzip.compress(b"".join(versions), 9))
    return {
        "path": header["path"],
        "versions": len(versions),
        "raw_bytes": raw,
        "chain_bytes": payload,
        "gzip_bytes": gz,
        "ratio": raw / payload if payload else 0.0,
        "gzip_ratio": raw / gz if gz else 0.0,
        "bpb": 8 * payload / raw if raw else 0.0,
        "seconds": round(seconds, 3),
        "first_bytes": header["sizes"][0],
        "last_bytes": header["sizes"][-1],
        "config": {k: header[k] for k in ChainConfig.__annotations__},
    }


def render_report(stats: dict) -> str:
    lines = [
        "# Repo memory: exact history chain",
        "",
        f"- path `{stats['path']}`; {stats['versions']} versions; "
        f"{stats['raw_bytes']:,} B raw",
        f"- chain {stats['chain_bytes']:,} B; **{stats['ratio']:.1f}x**; "
        f"{stats['bpb']:.3f} bpb; decode+verify {stats['seconds']:.1f} s",
        f"- gzip of the concatenation: {stats['gzip_bytes']:,} B "
        f"({stats['gzip_ratio']:.1f}x); the chain must beat it to count",
        "",
        "## What it is",
        "",
        "The first version is stored raw. Every later version is a delta",
        "against the previous one; decode replays the chain and checks each",
        "SHA-256. This is exact version storage, not a fuzzy patch.",
        "",
        "## Numbers",
        "",
        "| metric | bytes | ratio |",
        "|---|---:|---:|",
        f"| raw versions | {stats['raw_bytes']:,} | 1.0x |",
        f"| gzip concat | {stats['gzip_bytes']:,} | {stats['gzip_ratio']:.1f}x |",
        f"| chain | {stats['chain_bytes']:,} | **{stats['ratio']:.1f}x** |",
        "",
        "## Use",
        "",
        "```bash",
        "uv run cml chain --repo /path/to/repo.git --path fastapi/routing.py \\",
        "  --out routing.cmlh",
        "uv run cml unchain routing.cmlh --out-dir out/",
        "```",
        "",
        "`unchain` writes the last version by default; `--index i` writes a",
        "prefix version. Every write is digest-checked.",
    ]
    return "\n".join(lines) + "\n"


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    os.replace(tmp, path)


def run_chain(
    repo: Path,
    path: str,
    out: Path,
    report: Path,
    data_out: Path,
    window: int,
    min_match: int,
    cap: int,
    depth: int,
) -> int:
    cfg = ChainConfig(window, min_match, cap, depth)
    t0 = time.perf_counter()
    commits, shas, versions = git_versions(repo, path)
    blob = encode_chain(versions, path, commits, cfg)
    t1 = time.perf_counter()
    header, back = decode_chain(blob)
    if back != versions:
        raise SystemExit("chain: round-trip failed")
    stats = summarize(header, blob, versions, t1 - t0)
    stats["repo"] = str(repo)
    stats["shas"] = shas[-3:]
    write_chain(out, blob)
    _write_json(data_out, {"stats": stats, "header": header})
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(stats))
    print(
        f"chain: {stats['versions']} versions {stats['raw_bytes']:,} B -> "
        f"{stats['chain_bytes']:,} B ({stats['ratio']:.1f}x, "
        f"gzip {stats['gzip_ratio']:.1f}x) exact, {stats['seconds']:.1f} s"
    )
    print(f"chain: {out}")
    print(f"chain: {report}")
    return 0


def run_unchain(blob_path: Path, out_dir: Path, index: int) -> int:
    header, versions = load_chain(blob_path)
    if index < 0:
        index = len(versions) - 1
    if not 0 <= index < len(versions):
        raise SystemExit("unchain: index out of range")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"v{index:04d}.bin"
    target.write_bytes(versions[index])
    print(
        f"{blob_path} -> {target}: version {index} of {len(versions)}, "
        f"{len(versions[index]):,} B, digest ok"
    )
    return 0


TEXT_SUFFIXES = (
    ".py",
    ".md",
    ".txt",
    ".json",
    ".toml",
    ".cfg",
    ".ini",
    ".yml",
    ".yaml",
    ".rst",
    ".sh",
    ".c",
    ".h",
    ".cc",
    ".cpp",
    ".js",
    ".ts",
    ".tsx",
    ".css",
    ".html",
)


def head_paths(repo: Path, max_file: int) -> list[tuple[str, int]]:
    """All text files at HEAD, with sizes, path-sorted."""
    raw = _git(repo, ["ls-tree", "-r", "-l", "HEAD"])
    rows = []
    for line in raw.decode("utf-8", "replace").splitlines():
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) < 4:
            continue
        size = int(parts[3])
        if size > max_file or not path.endswith(TEXT_SUFFIXES):
            continue
        rows.append((path, size))
    return rows


def _safe_name(path: str) -> str:
    return path.replace("..", "_").replace("/", "__") + ".cmlh"


def _finalize(
    out: Path,
    report: Path,
    data_out: Path,
    repo: Path,
    rows: list[dict],
    global_mode: bool,
    cfg: ChainConfig,
    secs: float,
    head: str,
    checkpoint: int = 0,
    checkpoints: list[dict] | None = None,
) -> int:
    raw = sum(r["raw_bytes"] for r in rows)
    if global_mode:
        chain = rows[0]["chain_bytes"] if rows else 0
    else:
        chain = sum(r["chain_bytes"] for r in rows)
    gz = sum(r["gzip_bytes"] for r in rows)
    ratio = raw / max(chain, 1)
    gzip_ratio = raw / max(gz, 1)
    vs_gzip = gz / max(chain, 1)
    ckpt_bytes = sum(c["bytes"] for c in checkpoints or [])
    stats = {
        "repo": str(repo),
        "head": head,
        "files": len(rows),
        "global": global_mode,
        "raw_bytes": raw,
        "chain_bytes": chain,
        "gzip_bytes": gz,
        "ratio": ratio,
        "gzip_ratio": gzip_ratio,
        "vs_gzip": vs_gzip,
        "exact": True,
        "seconds": round(secs, 1),
        "checkpoint": checkpoint,
        "checkpoint_bytes": ckpt_bytes,
        "config": asdict(cfg),
    }
    payload = {
        "stats": stats,
        "files": rows,
        "checkpoints": checkpoints or [],
    }
    _write_json(out / "manifest.json", payload)
    _write_json(data_out, payload)
    top = sorted(rows, key=lambda r: -r["raw_bytes"])[:20]
    ckpt_line = (
        [
            f"- checkpoints every {checkpoint} versions: {len(checkpoints or [])} "
            f"snapshots, {ckpt_bytes:,} B; access replays ≤{checkpoint - 1} deltas",
        ]
        if checkpoint
        else []
    )
    lines = [
        "# Repo memory: exact history index",
        "",
        f"- repo `{repo}`; {len(rows)} files, {raw:,} B raw; "
        f"{'one global chain' if global_mode else 'one chain per file'}",
        f"- chain **{chain:,} B = {ratio:.1f}x**; gzip {gz:,} B "
        f"({gzip_ratio:.1f}x); the chain is **{vs_gzip:.1f}x better than gzip**",
        *ckpt_line,
        f"- every file decoded and every SHA-256 checked; {secs:.1f} s on CPU",
        "",
        "## Largest files",
        "",
    ]
    if global_mode:
        lines += ["| path | versions | raw B |", "|---|---:|---:|"]
        for r in top:
            lines.append(f"| `{r['path']}` | {r['versions']} | {r['raw_bytes']:,} |")
    else:
        lines += [
            "| path | versions | raw B | chain B | ratio |",
            "|---|---:|---:|---:|---:|",
        ]
        for r in top:
            lines.append(
                f"| `{r['path']}` | {r['versions']} | {r['raw_bytes']:,} | "
                f"{r['chain_bytes']:,} | {r['ratio']:.1f}x |"
            )
    lines += [
        "",
        "## Use",
        "",
        "```bash",
        "uv run cml repo --repo /path/to/repo.git --out repo-index --limit 50",
        "uv run cml repo --repo /path/to/repo.git --out repo-index --update",
        "uv run cml repo --index repo-index --search 'def routing'",
        "```",
        "",
        "The index is exact: each `.cmlh` replays to the same bytes and the",
        "manifest carries the HEAD commit and blob SHA. Search decodes the last",
        "version and reports path, line, and the exact matching text. `--update`",
        "appends new versions in place; unchanged chains are not re-encoded.",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    print(
        f"repo: {len(rows)} files, {raw:,} B -> {chain:,} B "
        f"({ratio:.1f}x, {vs_gzip:.1f}x better than gzip), "
        f"{secs:.1f} s, exact"
    )
    print(f"repo: {out}")
    print(f"repo: {report}")
    return 0


def _candidates(
    repo: Path, paths: str, limit: int, max_file: int
) -> list[tuple[str, int]]:
    wanted = [p for p in paths.split(",") if p]
    cand = [(p, 0) for p in wanted] if wanted else head_paths(repo, max_file)
    return cand[:limit] if limit else cand


def _file_row(
    out: Path,
    path: str,
    commits: list[str],
    shas: list[str],
    versions: list[bytes],
    cfg: ChainConfig,
    old: dict | None,
) -> dict:
    name = _safe_name(path)
    if old is not None:
        old_header, old_payloads = read_container(out / name)
        n_old = len(old_header["sizes"])
        if (
            len(versions) < n_old
            or commits[:n_old] != old_header["commits"]
            or [hashlib.sha256(v).hexdigest() for v in versions[:n_old]]
            != old_header["shas"]
        ):
            raise SystemExit(f"repo update: history rewritten for {path}; rebuild")
        if len(versions) == n_old:
            return old
        blob = _container(
            _header(path, commits, versions, cfg),
            old_payloads + _payloads(cfg, versions, n_old),
        )
    else:
        blob = encode_chain(versions, path, commits, cfg)
    header, back = decode_chain(blob)
    if back != versions:
        raise SystemExit(f"repo update: round-trip failed for {path}")
    raw = sum(len(v) for v in versions)
    gz = sum(len(gzip.compress(v, 9)) for v in versions)
    stored = gzip.compress(blob, 9)
    compressed = len(stored) < len(blob)
    data = stored if compressed else blob
    write_chain(out / name, data)
    row = {
        "path": path,
        "file": name,
        "versions": len(versions),
        "raw_bytes": raw,
        "chain_bytes": len(data),
        "compressed": compressed,
        "gzip_bytes": gz,
        "ratio": raw / max(len(data), 1),
        "gzip_ratio": raw / max(gz, 1),
        "head": shas[-1],
    }
    print(
        f"  {path}: {len(versions)} v, {raw:,} B -> {len(data):,} B "
        f"({row['ratio']:.1f}x)",
        flush=True,
    )
    return row


def _update_files(
    repo: Path,
    out: Path,
    cand: list[tuple[str, int]],
    old_rows: dict[str, dict],
    cfg: ChainConfig,
) -> list[dict]:
    rows = []
    for path, _ in cand:
        try:
            commits, shas, versions = git_versions(repo, path)
        except RuntimeError, SystemExit:
            continue
        rows.append(
            _file_row(out, path, commits, shas, versions, cfg, old_rows.get(path))
        )
    live = {row["path"] for row in rows}
    for path, row in old_rows.items():
        if path not in live:
            (out / row["file"]).unlink(missing_ok=True)
    return rows


def _update_global(
    repo: Path,
    out: Path,
    cand: list[tuple[str, int]],
    old_rows: dict[str, dict],
    cfg: ChainConfig,
    interval: int,
    existing: list[dict],
) -> tuple[list[dict], list[dict]]:
    old_blob = (out / "repo.cmlh").read_bytes()
    raw = gzip.decompress(old_blob) if old_blob[:2] == b"\x1f\x8b" else old_blob
    _, old_payloads = read_container(out / "repo.cmlh")
    _, flat = decode_chain(raw)
    payloads = bytearray(old_payloads)
    rows = []
    for path, _ in cand:
        try:
            commits, shas, versions = git_versions(repo, path)
        except RuntimeError, SystemExit:
            continue
        old = old_rows.get(path)
        if old is None:
            start = len(flat)
            payloads += _payloads(cfg, versions, prev=flat[-1] if flat else None)
            flat.extend(versions)
            ranges = [[start, len(flat)]]
        else:
            old_versions = _row_versions(flat, old)
            n_old = len(old_versions)
            if len(versions) < n_old or versions[:n_old] != old_versions:
                raise SystemExit(f"repo update: history rewritten for {path}; rebuild")
            ranges = list(old["ranges"])
            if len(versions) > n_old:
                start = len(flat)
                payloads += _payloads(cfg, versions, n_old, prev=flat[-1])
                flat.extend(versions[n_old:])
                ranges.append([start, len(flat)])
            else:
                rows.append(old)
                continue
        raw_bytes = sum(len(v) for v in versions)
        gz = sum(len(gzip.compress(v, 9)) for v in versions)
        rows.append(
            {
                "path": path,
                "file": "repo.cmlh",
                "ranges": ranges,
                "versions": len(versions),
                "raw_bytes": raw_bytes,
                "gzip_bytes": gz,
                "head": shas[-1],
            }
        )
    blob = _container(_header("repo", [], flat, cfg), bytes(payloads))
    header, back = decode_chain(blob)
    if back != flat:
        raise SystemExit("repo update: global round-trip failed")
    stored = gzip.compress(blob, 9)
    data = stored if len(stored) < len(blob) else blob
    write_chain(out / "repo.cmlh", data)
    checkpoints = sync_checkpoints(out, flat, header, interval, existing)
    total = len(data) + sum(c["bytes"] for c in checkpoints)
    for row in rows:
        row["chain_bytes"] = total
        row["ratio"] = row["raw_bytes"] / max(total, 1)
        row["gzip_ratio"] = row["raw_bytes"] / max(row["gzip_bytes"], 1)
    return rows, checkpoints


def _run_update(
    repo: Path,
    root: Path,
    report: Path,
    data_out: Path,
    paths: str,
    limit: int,
    max_file: int,
    global_mode: bool,
    checkpoint: int,
) -> int:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"repo update: missing {manifest_path}; build it first")
    manifest = json.loads(manifest_path.read_text())
    stats = manifest["stats"]
    if bool(stats.get("global")) != global_mode:
        raise SystemExit("repo update: --global does not match the index")
    if checkpoint and not global_mode:
        raise SystemExit("repo update: --checkpoint needs --global")
    interval = stats.get("checkpoint", 0)
    if checkpoint and interval and checkpoint != interval:
        raise SystemExit(
            f"repo update: index uses --checkpoint {interval}, not {checkpoint}"
        )
    if checkpoint and not interval:
        interval = checkpoint
    cfg = ChainConfig(**{k: stats["config"][k] for k in ChainConfig.__annotations__})
    head = _git(repo, ["rev-parse", "HEAD"]).decode().strip()
    if stats.get("head") == head and interval == stats.get("checkpoint", 0):
        print(f"repo: up to date at {head[:12]}; nothing to append")
        return 0
    cand = _candidates(repo, paths, limit, max_file)
    old_rows = {row["path"]: row for row in manifest["files"]}
    existing = manifest.get("checkpoints", [])
    t0 = time.perf_counter()
    if global_mode:
        rows, checkpoints = _update_global(
            repo, root, cand, old_rows, cfg, interval, existing
        )
    else:
        rows, checkpoints = _update_files(repo, root, cand, old_rows, cfg), []
    secs = time.perf_counter() - t0
    print(f"repo: appended history up to {head[:12]} in {secs:.1f} s", flush=True)
    return _finalize(
        root,
        report,
        data_out,
        repo,
        rows,
        global_mode,
        cfg,
        secs,
        head,
        interval,
        checkpoints,
    )


def run_repo(
    repo: Path | None,
    out: Path,
    report: Path,
    data_out: Path,
    search: str,
    index: Path | None,
    paths: str,
    limit: int,
    max_file: int,
    window: int,
    min_match: int,
    cap: int,
    depth: int,
    global_mode: bool = False,
    update: bool = False,
    checkpoint: int = 0,
    context: int = 0,
    hits_json: Path | None = None,
    max_hits: int = 50,
) -> int:
    """Repo-wide exact memory: chain every text file's history, or search it."""
    cfg = ChainConfig(window, min_match, cap, depth)
    root = index or out
    if checkpoint and not global_mode:
        raise SystemExit("repo: --checkpoint needs --global")
    if checkpoint == 1:
        raise SystemExit("repo: --checkpoint must be 0 or >= 2")
    if context < 0:
        raise SystemExit("repo: --context must be >= 0")
    if max_hits < 1:
        raise SystemExit("repo: --max-hits must be >= 1")
    if search:
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"repo search: missing {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        needle = search.encode()
        hits: list[dict] = []
        shared = None
        if manifest["stats"].get("global"):
            shared = load_chain(root / "repo.cmlh")[1]
        for row in sorted(manifest["files"], key=lambda r: r["path"]):
            if len(hits) >= max_hits:
                break
            if shared is not None:
                last = _row_versions(shared, row)[-1]
            else:
                last = load_chain(root / row["file"])[1][-1]
            lines = last.split(b"\n")
            seen: set[int] = set()
            at = last.find(needle)
            while at >= 0 and len(hits) < max_hits:
                line = last.count(b"\n", 0, at) + 1
                if line not in seen:
                    seen.add(line)
                    start = last.rfind(b"\n", 0, at) + 1
                    end = last.find(b"\n", at)
                    text = last[start : end if end >= 0 else len(last)].decode(
                        "utf-8", "replace"
                    )
                    hit = {"path": row["path"], "line": line, "text": text}
                    if context:
                        index = line - 1
                        first = max(0, index - context)
                        last_line = min(len(lines), index + context + 1)
                        span = lines[first:last_line]
                        hit["context"] = [
                            {
                                "line": first + i + 1,
                                "text": value.decode("utf-8", "replace"),
                                "hit": first + i == index,
                            }
                            for i, value in enumerate(span)
                        ]
                        for item in hit["context"]:
                            sep = ":" if item["hit"] else "-"
                            print(
                                f"{row['path']}{sep}{item['line']}{sep} "
                                f"{item['text'][:120]}"
                            )
                    else:
                        print(f"{row['path']}:{line}: {text[:120]}")
                    hits.append(hit)
                at = last.find(needle, at + 1)
        if hits_json is not None:
            _write_json(
                hits_json,
                {"query": search, "context": context, "hits": hits},
            )
        print(f"repo search: {len(hits)} exact hit(s) for {search!r}")
        return 0
    if repo is None:
        raise SystemExit("repo: pass --repo to build or --search to query")
    if update:
        return _run_update(
            repo,
            root,
            report,
            data_out,
            paths,
            limit,
            max_file,
            global_mode,
            checkpoint,
        )
    wanted = [p for p in paths.split(",") if p]
    cand = [(p, 0) for p in wanted] if wanted else head_paths(repo, max_file)
    if limit:
        cand = cand[:limit]
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.perf_counter()
    collected = []
    for path, _ in cand:
        try:
            commits, shas, versions = git_versions(repo, path)
        except RuntimeError, SystemExit:
            continue
        collected.append((path, commits, shas, versions))
        print(
            f"  {path}: {len(versions)} v, {sum(len(v) for v in versions):,} B",
            flush=True,
        )
    if global_mode:
        flat: list[bytes] = []
        for path, commits, shas, versions in collected:
            start = len(flat)
            flat.extend(versions)
            rows.append(
                {
                    "path": path,
                    "file": "repo.cmlh",
                    "ranges": [[start, len(flat)]],
                    "versions": len(versions),
                    "raw_bytes": sum(len(v) for v in versions),
                    "gzip_bytes": sum(len(gzip.compress(v, 9)) for v in versions),
                    "head": shas[-1],
                }
            )
        blob = encode_chain(flat, "repo", [], cfg)
        header, back = decode_chain(blob)
        if back != flat:
            raise SystemExit("repo: global round-trip failed")
        stored = gzip.compress(blob, 9)
        data = stored if len(stored) < len(blob) else blob
        write_chain(out / "repo.cmlh", data)
        checkpoints = sync_checkpoints(out, flat, header, checkpoint)
        total = len(data) + sum(c["bytes"] for c in checkpoints)
        for row in rows:
            row["chain_bytes"] = total
            row["ratio"] = row["raw_bytes"] / max(total, 1)
            row["gzip_ratio"] = row["raw_bytes"] / max(row["gzip_bytes"], 1)
    else:
        checkpoints = []
        for path, commits, shas, versions in collected:
            blob = encode_chain(versions, path, commits, cfg)
            header, back = decode_chain(blob)
            if back != versions:
                raise SystemExit(f"repo: round-trip failed for {path}")
            raw = sum(len(v) for v in versions)
            gz = sum(len(gzip.compress(v, 9)) for v in versions)
            name = _safe_name(path)
            stored = gzip.compress(blob, 9)
            compressed = len(stored) < len(blob)
            data = stored if compressed else blob
            write_chain(out / name, data)
            row = {
                "path": path,
                "file": name,
                "versions": len(versions),
                "raw_bytes": raw,
                "chain_bytes": len(data),
                "compressed": compressed,
                "gzip_bytes": gz,
                "ratio": raw / max(len(data), 1),
                "gzip_ratio": raw / max(gz, 1),
                "head": shas[-1],
            }
            rows.append(row)
            print(
                f"  {path}: {len(versions)} v, {raw:,} B -> "
                f"{len(data):,} B ({row['ratio']:.1f}x)",
                flush=True,
            )
    secs = time.perf_counter() - t0
    head = _git(repo, ["rev-parse", "HEAD"]).decode().strip()
    return _finalize(
        out,
        report,
        data_out,
        repo,
        rows,
        global_mode,
        cfg,
        secs,
        head,
        checkpoint,
        checkpoints,
    )
