"""Stage 0: the CPU analyses that rank the levers before any GPU run.

`coverage` prices the parse: how much of the held-out stream each pack covers,
how many literals survive, the static cost of a fixed unit table, and the
measured unit arm.  `copyprobe` charges an LZ77 preprocessor with its own token
codec and compares it to the byte arm.  `perfile` amortizes the pack charge
over 1/4/10/100 KB files, the product gate G5.  All three are exact and only
use the CPU.
"""

from __future__ import annotations

import math
import time
from collections import Counter

import codec


def coverage_rows(data: bytes, packs: list[tuple[int, codec.Lexicon]]) -> list[dict]:
    """One longest-match parse and one unit-arm encode per pack."""
    rows = []
    for units, lex in packs:
        t0 = time.perf_counter()
        parse, covered = codec.greedy_parse(data, lex)
        stats = codec.parse_stats(parse, lex, len(data), covered)
        classes = Counter(-1 if sym < 0 else sym for sym in parse)
        total = max(len(parse), 1)
        h0 = -sum((c / total) * math.log2(c / total) for c in classes.values())
        static = (h0 * total + 8 * stats["literals"]) / max(len(data), 1)
        pack_bytes = len(lex.blob())
        payload = len(codec.UnitArm(lex).encode(data, parse))
        rows.append(
            {
                "units": units,
                "pack_bytes": pack_bytes,
                "coverage": stats["coverage"],
                "literals": stats["literals"],
                "literal_fraction": stats["literals"] / total,
                "mean_unit": stats["mean_unit"],
                "decisions": total,
                "static_bpb": static,
                "payload_bytes": payload,
                "payload_bpb": 8 * payload / max(len(data), 1),
                "shared_bpb": 8 * (payload + pack_bytes) / max(len(data), 1),
                "seconds": round(time.perf_counter() - t0, 3),
            }
        )
    return rows


def render_coverage(report: dict) -> str:
    rows = report["rows"]
    rise = rows[-1]["coverage"] - rows[0]["coverage"]
    gain = rows[0]["payload_bpb"] - rows[-1]["payload_bpb"]
    lines = [
        "# Coverage: the pack lever, priced before training",
        "",
        f"- corpus `{report['corpus']}` sha `{report['sha'][:12]}`, held-out "
        f"{report['n']:,} B",
        f"- packs induce from the frozen train split, min count "
        f"{report['min_count']}, unit cap {report['cap']}",
        "- `static` is a fixed unit table: zero-order symbol entropy plus 8",
        "  bits per literal. It is the cost without context, not a floor.",
        "- `payload` and `shared` are the measured unit arm, the second one",
        "  charging the pack once. `delta` is against the byte arm.",
        "",
        "| units | pack B | coverage | literals | lit frac | mean unit "
        "| decisions | static | payload | shared | delta |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['units']} | {r['pack_bytes']:,} | {r['coverage']:.3f} | "
            f"{r['literals']:,} | {r['literal_fraction']:.3f} | "
            f"{r['mean_unit']:.2f} | {r['decisions']:,} | "
            f"{r['static_bpb']:.3f} | {r['payload_bpb']:.3f} | "
            f"{r['shared_bpb']:.3f} | "
            f"{r['payload_bpb'] - report['byte_bpb']:+.3f} |"
        )
    lines += [
        "",
        "## Read",
        "",
        f"- byte arm: {report['byte_bpb']:.3f} bpb on the same bytes.",
        f"- coverage moves {rise:+.3f} from the smallest to the largest pack; "
        "the Stage 0 gate is +0.10.",
        f"- the measured payload gains {gain:.3f} bpb across the ladder; the",
        "  pack charge must stay below that gain.",
        f"- the unit arm is {rows[-1]['payload_bpb'] - report['byte_bpb']:+.3f} bpb "
        "against the byte arm at its best: a static",
        "  pack is not a win. The pack pays only through the pointer, which",
        "  predicts a span from context (`reports/SCALE.md`: 1.518 shared).",
        "",
    ]
    return "\n".join(lines)


def copy_probe(data: bytes, window: int, min_match: int, cap: int, depth: int) -> dict:
    """LZ77 parse, its own token codec, and the byte arm on the same bytes."""
    t0 = time.perf_counter()
    tokens = codec.lz_parse(data, window, min_match, cap, depth)
    parse_secs = time.perf_counter() - t0
    payload = codec.CopyArm(window, min_match, cap).encode(data, tokens)
    if codec.CopyArm(window, min_match, cap).decode(payload, len(data)) != data:
        raise ValueError("copy arm: round-trip failed")
    base = codec.ByteArm().encode(data)
    n = max(len(data), 1)
    copies = sum(ell for ell, _ in tokens if ell)
    matches = sum(1 for ell, _ in tokens if ell)
    return {
        "window": window,
        "min_match": min_match,
        "cap": cap,
        "depth": depth,
        "n": len(data),
        "tokens": len(tokens),
        "matches": matches,
        "copy_bytes": copies,
        "copy_fraction": copies / n,
        "literal_fraction": 1 - copies / n,
        "mean_match": copies / max(matches, 1),
        "payload_bytes": len(payload),
        "payload_bpb": 8 * len(payload) / n,
        "base_bytes": len(base),
        "base_bpb": 8 * len(base) / n,
        "saving": 1 - len(payload) / max(len(base), 1),
        "parse_seconds": round(parse_secs, 3),
        "seconds": round(time.perf_counter() - t0, 3),
    }


def render_copy(report: dict) -> str:
    lines = [
        "# Copy: the LZ77 preprocessor, charged",
        "",
        f"- corpus `{report['corpus']}` sha `{report['sha'][:12]}`, held-out "
        f"{report['n']:,} B",
        f"- window {report['window']:,} B, min match {report['min_match']}, "
        f"match cap {report['cap']}, depth {report['depth']}",
        "- charge: token bit, distance slot, raw slot bits, length symbol;",
        "  literals escape to the order-2 byte arm.",
        "",
        "| token | value |",
        "|---|---:|",
        f"| tokens | {report['tokens']:,} |",
        f"| matches | {report['matches']:,} |",
        f"| copy fraction | {report['copy_fraction']:.3f} |",
        f"| mean match B | {report['mean_match']:.2f} |",
        f"| copy payload | {report['payload_bytes']:,} B "
        f"({report['payload_bpb']:.3f} bpb) |",
        f"| byte payload | {report['base_bytes']:,} B ({report['base_bpb']:.3f} bpb) |",
        f"| saving | {report['saving']:.1%} |",
        "",
        "## Read",
        "",
        f"- copy saves {report['saving']:.1%} at this window; the Stage 0 gate "
        "is 5%." + (" The lever holds." if report["saving"] > 0.05 else " Park copy."),
        "- The preprocessor is exact and costs one parse pass; a Stage 3 run",
        "  can reuse it in front of the trained trunk without new training.",
        "",
        f"Parse took {report['parse_seconds']:.1f} s; codec took "
        f"{report['seconds'] - report['parse_seconds']:.1f} s.",
        "",
    ]
    return "\n".join(lines)


def perfile_rows(
    test: bytes,
    packs: list[tuple[int, codec.Lexicon]],
    sizes: list[int],
    baseline: bool = True,
) -> dict:
    """Split `test` into files of each size; charge the pack once per file."""
    rows = []
    for size in sizes:
        chunks = [test[a : a + size] for a in range(0, len(test) - size + 1, size)]
        if not chunks:
            continue
        base = None
        if baseline:
            base = sorted(
                8 * len(codec.ByteArm().encode(chunk)) / len(chunk) for chunk in chunks
            )
        for units, lex in packs:
            pack_bytes = len(lex.blob())
            shared = sorted(
                8 * (len(codec.UnitArm(lex).encode(chunk)) + pack_bytes) / len(chunk)
                for chunk in chunks
            )
            rows.append(
                {
                    "size": size,
                    "units": units,
                    "pack_bytes": pack_bytes,
                    "files": len(chunks),
                    "shared_mean": sum(shared) / len(shared),
                    "shared_min": shared[0],
                    "shared_max": shared[-1],
                    "base_mean": sum(base) / len(base) if base else None,
                    "base_min": base[0] if base else None,
                    "base_max": base[-1] if base else None,
                }
            )
    return {"rows": rows, "sizes": sizes}


def render_perfile(report: dict) -> str:
    lines = [
        "# Per-file: the pack charge, amortized",
        "",
        f"- corpus `{report['corpus']}` sha `{report['sha'][:12]}`, held-out "
        f"{report['n']:,} B",
        "- each file is coded alone; the pack is charged once per file:",
        "  `shared = payload + pack_bytes`.",
        "- the byte column is the order-2 arm on the same files.",
        "",
        "| file B | files | units | pack B | shared mean | shared min | "
        "shared max | byte mean | delta |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in report["rows"]:
        base = f"{r['base_mean']:.3f}" if r["base_mean"] is not None else "—"
        delta = (
            f"{r['shared_mean'] - r['base_mean']:+.3f}"
            if r["base_mean"] is not None
            else "—"
        )
        lines.append(
            f"| {r['size']:,} | {r['files']} | {r['units']} | "
            f"{r['pack_bytes']:,} | {r['shared_mean']:.3f} | "
            f"{r['shared_min']:.3f} | {r['shared_max']:.3f} | {base} | {delta} |"
        )
    wins = [r for r in report["rows"] if r["base_mean"] is not None]
    lines += ["", "## Read", ""]
    if not wins:
        lines += [
            "- no byte baseline requested; read the pack charge against a",
            "  parallel run of the same sizes.",
            "",
        ]
        return "\n".join(lines)
    best = min(wins, key=lambda r: r["shared_mean"] - r["base_mean"])
    worst = max(wins, key=lambda r: r["shared_mean"] - r["base_mean"])
    if best["shared_mean"] < best["base_mean"]:
        lines.append(
            f"- best case {best['size']:,} B with {best['units']} units wins "
            f"{best['base_mean'] - best['shared_mean']:.3f} bpb; G5 holds there."
        )
    else:
        lines.append(
            f"- best case {best['size']:,} B with {best['units']} units still "
            f"loses {best['shared_mean'] - best['base_mean']:.3f} bpb; G5 fails."
        )
    lines += [
        f"- worst case {worst['size']:,} B with {worst['units']} units loses "
        f"{worst['shared_mean'] - worst['base_mean']:.3f} bpb.",
        "- The charge is per file: a static pack cannot pay below its own",
        "  size, so the product needs an adaptive pack, a delta codec, or a",
        "  shared installation that is charged once (`reports/COPY.md`).",
        "",
    ]
    return "\n".join(lines)
