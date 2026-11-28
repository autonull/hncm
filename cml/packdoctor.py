"""Pack doctor: rank spans by net bits at the measured base rate.

The shipped pack gain assumed 8 bits/byte.  A 1 bpb trunk changes the trade:
a copy of L bytes saves about L*bpb bits and pays an id.  This tool re-ranks
an existing counted pack and keeps only spans whose net is positive.
"""

from __future__ import annotations

import time
from pathlib import Path


def run_fit(
    pack_path: Path,
    out: Path,
    report: Path,
    bpb: float,
    id_bits: float,
    units: int,
    min_len: int,
) -> int:
    import cml

    t0 = time.perf_counter()
    entries = cml.read_pack(pack_path)
    scored = []
    for e in entries:
        length = len(e.data)
        if length < min_len:
            continue
        net = e.count * (length * bpb - id_bits) - 8 * (length + 6)
        if net > 0:
            scored.append((net, e))
    scored.sort(key=lambda x: (-x[0], x[1].data))
    keep = [e for _, e in scored[:units]]
    if not keep:
        raise SystemExit("pack fit: no span has positive net at this bpb/id")
    pack = cml.Pack(tuple(keep), cml.Config())
    blob = cml.pack_bytes(pack)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    raw = sum(len(e.data) for e in keep)
    mean_len = raw / len(keep)
    lines = [
        "# Pack doctor: net-bit span ranking",
        "",
        f"- source `{pack_path.name}`: {len(entries):,} spans",
        f"- base {bpb:.3f} bpb, id {id_bits:.1f} bits, min length {min_len}",
        f"- kept **{len(keep):,}** spans, mean {mean_len:.1f} B, {len(blob):,} B pack",
        "- each kept span has positive net: `count*(L*bpb - id) - 8*(L+6)`",
        "",
        "| path | rank | count | len | net bits |",
        "|---:|---:|---:|---:|---:|",
    ]
    for rank, (net, e) in enumerate(scored[:20], 1):
        lines.append(
            f"| `{e.provenance[0] if e.provenance else ''}` | {rank} | "
            f"{e.count} | {len(e.data)} | {net:.0f} |"
        )
    lines += ["", f"Took {time.perf_counter() - t0:.1f} s CPU.", ""]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    print(
        f"pack fit: {len(entries):,} -> {len(keep):,} spans "
        f"({len(blob):,} B, mean {mean_len:.1f} B), net > 0"
    )
    print(f"pack fit: {out}")
    print(f"pack fit: {report}")
    return 0
