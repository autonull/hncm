"""Code bench: exact structural recall and hybrid bits for a byte LM.

Split a Python repo by file.  Build a template/fill pack on the train files,
then parse held-out files.  Report structural coverage, fully exact lines,
fill copy sources, decisions per byte, and a hybrid bits/byte estimate that
uses the frozen byte trunk for literal bytes.
"""

from __future__ import annotations

import math
import zlib
from pathlib import Path

import cml
import fillcopy
import templates

TRUNK = "codec-byte-model-code-756752-d448-l8-q2048"


def split(root: Path, max_bytes: int = 2_000_000) -> tuple[list[Path], list[Path]]:
    files = [p for p in sorted(root.rglob("*.py")) if ".venv" not in p.parts]
    train, test = [], []
    for path in files:
        rel = str(path.relative_to(root))
        (test if zlib.crc32(rel.encode()) % 10 == 0 else train).append(path)
    if not test:
        test = train[-1:]
        train = train[:-1]
    total = 0
    keep = []
    for path in train:
        total += path.stat().st_size
        if total <= max_bytes:
            keep.append(path)
    return keep, test


def measure(
    train: Path,
    test: Path,
    root: Path,
    min_fill: int = 8,
    min_count: int = 3,
    max_n: int = 6,
    use_trunk: bool = True,
) -> dict:
    train_data = b"\n".join(p.read_bytes() for p in train)
    test_data = b"\n".join(p.read_bytes() for p in test)
    lib = fillcopy.induce(train_data, min_fill, min_count, max_n)
    events = fillcopy.parse(test_data, lib, min_fill, max_n)
    n = len(test_data)
    covered = bytearray(n)
    literal = bytearray(n)
    struct = decisions = 0
    fill_local = fill_pack = fill_literal = 0
    templates_n = fills_n = local_n = pack_n = 0
    pos = 0
    for event in events:
        if isinstance(event, templates.Lit):
            literal[pos : pos + len(event.value)] = b"\x01" * len(event.value)
            pos += len(event.value)
            decisions += len(event.value)
            continue
        decisions += 1
        templates_n += 1
        j = 0
        for element in event.key:
            if element.startswith(b"F"):
                length = len(element) - 1
                covered[pos : pos + length] = b"\x01" * length
                struct += length
                pos += length
                continue
            value = event.fills[j]
            j += 1
            decisions += 1
            fills_n += 1
            local = test_data.rfind(value, max(0, pos - (1 << 16)), pos)
            if local >= 0 and local + len(value) <= pos:
                fill_local += len(value)
                local_n += 1
                covered[pos : pos + len(value)] = b"\x01" * len(value)
                struct += len(value)
            elif train_data.find(value) >= 0:
                fill_pack += len(value)
                pack_n += 1
                covered[pos : pos + len(value)] = b"\x01" * len(value)
                struct += len(value)
            else:
                fill_literal += len(value)
                literal[pos : pos + len(value)] = b"\x01" * len(value)
            pos += len(value)
    lines = test_data.splitlines(keepends=True)
    exact = total_lines = 0
    at = 0
    for line in lines:
        end = at + len(line)
        total_lines += 1
        if all(covered[at:end]):
            exact += 1
        at = end
    costs = None
    if use_trunk:
        import bytecodec

        model = bytecodec.load_model(cml.RUNS / TRUNK / "model.pt")
        costs = bytecodec.literal_bits(bytecodec.FastByteCodec(model), test_data)
        trunk_bpb = float(costs.sum()) / n
    else:
        trunk_bpb = 0.0
    literal_bytes = sum(literal)
    if costs is None:
        literal_bits = 0.0
    else:
        idx = [i for i, value in enumerate(literal) if value]
        literal_bits = float(costs[idx].sum()) if idx else 0.0
    bits = (
        templates_n * math.log2(max(len(lib), 1))
        + local_n * 11
        + pack_n * 21
        + literal_bits
    )
    pack_bpb = bits / n
    return {
        "train_files": len(train),
        "test_files": len(test),
        "train_bytes": len(train_data),
        "test_bytes": n,
        "templates": len(lib),
        "table_bytes": sum(
            sum(len(e) - 1 for e in key if e.startswith(b"F")) + 3 * len(key)
            for key in lib
        ),
        "coverage": struct / n,
        "exact_lines": exact / max(total_lines, 1),
        "literal_share": sum(literal) / n,
        "literal_bytes": literal_bytes,
        "fill_local": fill_local / n,
        "fill_pack": fill_pack / n,
        "fill_literal": fill_literal / n,
        "decisions_per_byte": decisions / n,
        "bytes_per_decision": n / max(decisions, 1),
        "trunk_bpb": trunk_bpb,
        "pack_bpb": pack_bpb,
    }


def run(
    repo: Path,
    max_bytes: int = 2_000_000,
    min_fill: int = 8,
    min_count: int = 3,
    max_n: int = 6,
) -> int:
    train, test = split(repo, max_bytes)
    stats = measure(train, test, repo, min_fill, min_count, max_n)
    text = "\n".join(
        [
            "# Code bench: exact structural memory",
            "",
            f"- root `{repo}`; train {stats['train_files']} files/"
            f"{stats['train_bytes']:,} B; held-out {stats['test_files']} files/"
            f"{stats['test_bytes']:,} B",
            f"- pack {stats['templates']:,} templates, "
            f"{stats['table_bytes']:,} B table",
            "",
            "| item | value |",
            "|---|---:|",
            f"| structural coverage | {stats['coverage']:.3f} |",
            f"| fully exact held-out lines | {stats['exact_lines']:.3f} |",
            f"| literal share | {stats['literal_share']:.3f} |",
            f"| literal bytes | {stats['literal_bytes']:,} |",
            f"| fill copy local / pack / literal | {stats['fill_local']:.3f} / "
            f"{stats['fill_pack']:.3f} / {stats['fill_literal']:.3f} |",
            f"| decisions/B | {stats['decisions_per_byte']:.3f} |",
            f"| bytes/decision | {stats['bytes_per_decision']:.2f} |",
            f"| byte trunk bpb | {stats['trunk_bpb']:.3f} |",
            f"| hybrid pack bpb | {stats['pack_bpb']:.3f} |",
            "",
            "## Read",
            "",
            "Structural coverage is the fraction of held-out bytes that are",
            "fixed template bytes or exact fills. Fully exact lines are the",
            "strong LLM capability: no literal byte on the line. The hybrid",
            "bpb charges the frozen trunk for literals and a flat id cost for",
            "events, so it is an estimate, not a trained model result.",
            f"The table is {stats['table_bytes'] / max(stats['train_bytes'], 1):.1%} "
            "of train bytes and is **not** in the hybrid bpb; prune or share "
            "it before a product claim.",
            "",
        ]
    )
    cml.write_text(cml.REPORTS / "CODEBENCH.md", text)
    cml.write_json(cml.REPORTS / "codebench.json", stats)
    print(text)
    return 0
