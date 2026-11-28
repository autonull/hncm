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


def encode_chain(
    versions: list[bytes], path: str, commits: list[str], cfg: ChainConfig
) -> bytes:
    if not versions:
        raise ValueError("chain needs at least one version")
    payloads = [versions[0]]
    for i in range(1, len(versions)):
        old, new = versions[i - 1], versions[i]
        tokens = codec.delta_parse(
            new, old, cfg.window, cfg.min_match, cfg.cap, cfg.depth
        )
        payloads.append(_arm(cfg, len(old), len(new)).encode(new, old, tokens))
    header = {
        "path": path,
        "commits": commits,
        "sizes": [len(v) for v in versions],
        "shas": [hashlib.sha256(v).hexdigest() for v in versions],
        **asdict(cfg),
    }
    head = json.dumps(header, separators=(",", ":")).encode()
    out = bytearray(MAGIC)
    out += struct.pack("<I", len(head))
    out += head
    for payload in payloads:
        out += struct.pack("<I", len(payload))
        out += payload
    return bytes(out)


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
    header, versions = decode_chain(blob_path.read_bytes())
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
