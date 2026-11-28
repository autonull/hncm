"""Small exact products: patch guard, log deltas, and commit deltas."""

from __future__ import annotations

import gzip
import json
import subprocess
import tempfile
import time
from pathlib import Path

import codec
import memory


def run_patch(base: Path, diff: Path, expect: Path | None, strip: int = 0) -> int:
    """Apply a unified diff to `base`; refuse unless it equals `expect`."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out"
        p = subprocess.run(
            ["patch", "-s", f"-p{strip}", "-o", str(out), str(base), str(diff)],
            capture_output=True,
        )
        if p.returncode:
            raise SystemExit(f"patch: apply failed: {p.stderr.decode()[:300]}")
        got = out.read_bytes()
    if expect is not None and got != expect.read_bytes():
        raise SystemExit("patch: result is not byte-exact with --expect")
    print(f"patch: exact ({len(got):,} B)")
    return 0


def run_logdelta(directory: Path, out: Path, report: Path, data_out: Path) -> int:
    """Chain a directory of CI logs oldest-to-newest and verify every byte."""
    logs = sorted(directory.glob("*.log"))
    if not logs:
        raise SystemExit(f"logdelta: no *.log in {directory}")
    versions = [p.read_bytes() for p in logs]
    cfg = memory.ChainConfig()
    t0 = time.perf_counter()
    blob = memory.encode_chain(versions, "logs", [p.name for p in logs], cfg)
    header, back = memory.decode_chain(blob)
    if back != versions:
        raise SystemExit("logdelta: round-trip failed")
    raw = sum(len(v) for v in versions)
    gz = len(gzip.compress(b"".join(versions), 9))
    stored = gzip.compress(blob, 9)
    data = stored if len(stored) < len(blob) else blob
    ratio = raw / max(len(data), 1)
    gz_ratio = raw / max(gz, 1)
    memory.write_chain(out, data)
    stats = {
        "logs": [p.name for p in logs],
        "versions": len(versions),
        "raw_bytes": raw,
        "chain_bytes": len(data),
        "compressed": len(stored) < len(blob),
        "gzip_bytes": gz,
        "ratio": ratio,
        "gzip_ratio": gz_ratio,
        "vs_gzip": gz / max(len(data), 1),
        "exact": True,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    data_out.parent.mkdir(parents=True, exist_ok=True)
    data_out.write_text(json.dumps({"stats": stats, "header": header}, indent=1) + "\n")
    lines = [
        "# Log delta: exact CI log history",
        "",
        f"- {len(logs)} logs, {raw:,} B raw",
        f"- chain **{len(data):,} B = {ratio:.1f}x**; gzip {gz:,} B "
        f"({gz_ratio:.1f}x); the chain is **{gz / max(len(data), 1):.1f}x better**",
        f"- all versions decoded and every SHA-256 checked; "
        f"{stats['seconds']:.1f} s CPU",
        "",
        "| log | versions | raw B |",
        "|---|---:|---:|",
    ]
    for p, v in zip(logs, versions):
        lines.append(f"| `{p.name}` | 1 | {len(v):,} |")
    lines += ["", f"Chain written to `{out}`.", ""]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    print(
        f"logdelta: {len(logs)} logs, {raw:,} B -> {len(data):,} B "
        f"({ratio:.1f}x, {gz / max(len(data), 1):.1f}x better than gzip), exact"
    )
    print(f"logdelta: {out}")
    print(f"logdelta: {report}")
    return 0


def _cat(repo: Path, sha: str, path: str) -> bytes:
    p = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "blob", f"{sha}:{path}"],
        capture_output=True,
    )
    if p.returncode:
        raise SystemExit(f"repo-delta: {sha}:{path}: {p.stderr.decode()[:200]}")
    return p.stdout


def run_repo_delta(repo: Path, path: str, before: str, after: str, out: Path) -> int:
    """Exact delta between two commits of one path, verified on decode."""
    old = _cat(repo, before, path)
    new = _cat(repo, after, path)
    cfg = memory.ChainConfig()
    arm = memory._arm(cfg, len(old), len(new))
    tokens = codec.delta_parse(new, old, cfg.window, cfg.min_match, cfg.cap, cfg.depth)
    payload = arm.encode(new, old, tokens)
    if memory._arm(cfg, len(old), len(new)).decode(payload, len(new), old) != new:
        raise SystemExit("repo-delta: round-trip failed")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    ratio = len(new) / max(len(payload), 1)
    print(
        f"repo-delta: {path} {before[:8]}..{after[:8]}: "
        f"{len(new):,} B -> {len(payload):,} B ({ratio:.1f}x), exact"
    )
    print(f"repo-delta: {out}")
    return 0
