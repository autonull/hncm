"""Fill-copy codec: long template fills, exact pointers, byte fallback.

A template event has exact fixed bytes and typed holes.  A long fill is
copied from earlier output or from an installed pack by a pointer; a fill
that is not present is sent literally.  Fixed bytes are free after the
template table is charged.  The decoder expands the same events and the
round-trip is checked byte for byte.
"""

from __future__ import annotations

import gzip
import keyword
import math
import re
import time
from collections import Counter
from dataclasses import dataclass

import cml
import templates

WORD = re.compile(rb"[A-Za-z_]\w*")
DOTTED = re.compile(rb"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
NUMBER = re.compile(rb"\d+(?:\.\d+)?")
KEYWORDS = {k.encode() for k in keyword.kwlist}


def tokenize(
    line: bytes, min_fill: int = 8, cap: int = 1024
) -> list[tuple[bytes, bytes]]:
    """Fixed syntax plus long typed holes for strings, names, groups, atoms."""
    out: list[tuple[bytes, bytes]] = []
    fixed = bytearray()
    i = 0
    n = len(line)

    def flush() -> None:
        if fixed:
            out.append((b"F", bytes(fixed)))
            fixed.clear()

    def hole(kind: bytes, value: bytes) -> None:
        if len(value) >= min_fill:
            flush()
            out.append((kind, value))
        else:
            fixed.extend(value)

    while i < n:
        b = line[i : i + 1]
        if b in (b"'", b'"'):
            quote = b
            j = i + 1
            while j < n and j - i < cap:
                if line[j : j + 1] == b"\\":
                    j += 2
                    continue
                if line[j : j + 1] == quote:
                    j += 1
                    break
                j += 1
            hole(b"S", line[i:j])
            i = j
            continue
        match = DOTTED.match(line, i)
        if match:
            hole(b"D", match.group(0))
            i = match.end()
            continue
        if b in b"([{":
            close = {b"(": b")", b"[": b"]", b"{": b"}"}[b]
            depth = 0
            j = i
            while j < n and j - i < cap:
                c = line[j : j + 1]
                if c in (b"'", b'"'):
                    quote = c
                    j += 1
                    while j < n and j - i < cap:
                        if line[j : j + 1] == b"\\":
                            j += 2
                            continue
                        if line[j : j + 1] == quote:
                            j += 1
                            break
                        j += 1
                    continue
                if c == b:
                    depth += 1
                elif c == close:
                    depth -= 1
                    j += 1
                    if depth == 0:
                        break
                    continue
                j += 1
            if depth == 0:
                hole(b"E", line[i:j])
                i = j
                continue
        match = NUMBER.match(line, i)
        if match:
            hole(b"N", match.group(0))
            i = match.end()
            continue
        match = WORD.match(line, i)
        if match:
            value = match.group(0)
            if value in KEYWORDS:
                fixed.extend(value)
            else:
                hole(b"I", value)
            i = match.end()
            continue
        fixed += b
        i += 1
    flush()
    return out


def keyof(tokens: list[tuple[bytes, bytes]]) -> tuple[bytes, ...]:
    return tuple(
        b"F" + value if kind == b"F" else b"H" + kind for kind, value in tokens
    )


def induce(
    train: bytes, min_fill: int = 8, min_count: int = 3, max_n: int = 6
) -> dict[tuple[bytes, ...], int]:
    counts: Counter[tuple[bytes, ...]] = Counter()
    for line in train.splitlines(keepends=True):
        ts = tokenize(line, min_fill)
        for n in range(2, max_n + 1):
            for i in range(len(ts) - n + 1):
                win = ts[i : i + n]
                holes = sum(1 for kind, _ in win if kind != b"F")
                fixed = sum(len(value) for kind, value in win if kind == b"F")
                if holes and fixed >= 6 and holes < n - 1:
                    counts[keyof(win)] += 1
    return {key: count for key, count in counts.items() if count >= min_count}


def parse(
    data: bytes,
    templates_: dict[tuple[bytes, ...], int],
    min_fill: int = 8,
    max_n: int = 6,
) -> list[templates.Hit | templates.Lit]:
    events: list[templates.Hit | templates.Lit] = []
    for line in data.splitlines(keepends=True):
        ts = tokenize(line, min_fill)
        i = 0
        while i < len(ts):
            found = None
            for n in range(max_n, 1, -1):
                if i + n > len(ts):
                    continue
                win = ts[i : i + n]
                key = keyof(win)
                if key in templates_:
                    found = (key, tuple(value for kind, value in win if kind != b"F"))
                    i += n
                    break
            if found is None:
                events.append(templates.Lit(*ts[i]))
                i += 1
            else:
                events.append(templates.Hit(*found))
    return events


def put_varint(out: bytearray, value: int) -> None:
    while value >= 128:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)


def get_varint(blob: bytes, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = blob[pos]
        pos += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, pos
        shift += 7


@dataclass(slots=True)
class Stats:
    templates: int = 0
    fills: int = 0
    fixed_bytes: int = 0
    fill_bytes: int = 0
    literal_bytes: int = 0
    local_bytes: int = 0
    pack_bytes: int = 0
    literal_fill_bytes: int = 0
    template_bits: float = 0.0
    decisions: int = 0
    table_bytes: int = 0
    roundtrip: bool = False
    seconds: float = 0.0
    blob_bytes: int = 0

    def as_dict(self) -> dict:
        return {key: getattr(self, key) for key in self.__slots__}


def encode(
    data: bytes,
    templates_: dict[tuple[bytes, ...], int],
    pack: bytes = b"",
    min_fill: int = 8,
    max_n: int = 6,
    window: int = 1 << 16,
) -> tuple[bytes, Stats]:
    """Encode with template ids, local/pack fill pointers, and literal runs."""
    t0 = time.perf_counter()
    stats = Stats(templates=len(templates_))
    order = sorted(templates_, key=lambda key: (-templates_[key], key))
    tid = {key: i for i, key in enumerate(order)}
    total_events = sum(templates_.values())
    blob = bytearray()
    decoded = bytearray()
    pending = bytearray()

    def flush_literal() -> None:
        if not pending:
            return
        put_varint(blob, 1)
        put_varint(blob, len(pending))
        blob.extend(pending)
        stats.literal_bytes += len(pending)
        stats.decisions += 1
        pending.clear()

    for event in parse(data, templates_, min_fill, max_n):
        if isinstance(event, templates.Lit):
            pending.extend(event.value)
            decoded.extend(event.value)
            continue
        flush_literal()
        put_varint(blob, 0)
        put_varint(blob, tid[event.key])
        stats.template_bits += -math.log2(templates_[event.key] / total_events)
        stats.decisions += 1
        pos = len(decoded)
        fill_index = 0
        for element in event.key:
            if element.startswith(b"F"):
                value = element[1:]
                decoded.extend(value)
                stats.fixed_bytes += len(value)
                continue
            value = event.fills[fill_index]
            fill_index += 1
            stats.fills += 1
            stats.fill_bytes += len(value)
            stats.decisions += 1
            local = data.rfind(value, max(0, pos - window), pos)
            use_local = local >= 0 and local + len(value) <= pos
            if use_local:
                put_varint(blob, 1)
                put_varint(blob, len(value))
                put_varint(blob, local)
                stats.local_bytes += len(value)
            elif pack and (pack_at := pack.find(value)) >= 0:
                put_varint(blob, 2)
                put_varint(blob, len(value))
                put_varint(blob, pack_at)
                stats.pack_bytes += len(value)
            else:
                put_varint(blob, 0)
                put_varint(blob, len(value))
                blob.extend(value)
                stats.literal_fill_bytes += len(value)
            decoded.extend(value)
            pos = len(decoded)
    flush_literal()
    stats.blob_bytes = len(blob)
    stats.table_bytes = sum(
        sum(len(e) - 1 for e in key if e.startswith(b"F")) + len(key) * 3
        for key in templates_
    )
    stats.roundtrip = decode(blob, order, pack) == data
    stats.seconds = round(time.perf_counter() - t0, 1)
    return blob, stats


def decode(blob: bytes, order: list[tuple[bytes, ...]], pack: bytes = b"") -> bytes:
    out = bytearray()
    pos = 0
    while pos < len(blob):
        tag, pos = get_varint(blob, pos)
        if tag == 1:
            length, pos = get_varint(blob, pos)
            out.extend(blob[pos : pos + length])
            pos += length
            continue
        tid, pos = get_varint(blob, pos)
        key = order[tid]
        for element in key:
            if element.startswith(b"F"):
                out.extend(element[1:])
                continue
            code, pos = get_varint(blob, pos)
            length, pos = get_varint(blob, pos)
            if code == 0:
                value = blob[pos : pos + length]
                pos += length
            elif code == 1:
                start, pos = get_varint(blob, pos)
                value = out[start : start + length]
            else:
                offset, pos = get_varint(blob, pos)
                value = pack[offset : offset + length]
            out.extend(value)
    return bytes(out)


def run(
    train_bytes: int = 2_000_000,
    test_bytes: int = 100_000,
    min_fill: int = 8,
    min_count: int = 3,
    max_n: int = 6,
) -> int:
    corpus = cml.code_scale_corpus(48_000_000)
    train = corpus.train[:train_bytes]
    test = corpus.test[:test_bytes]
    templates_ = induce(train, min_fill, min_count, max_n)
    blob, stats = encode(test, templates_, train, min_fill, max_n)
    n = len(test)
    gz = len(gzip.compress(test, 9))
    stream = 0.97 / 8 * n
    literal_fill = stats.literal_fill_bytes + stats.literal_bytes
    ideal_bits = (len(blob) - literal_fill) * 8 + literal_fill * 1
    ideal = ideal_bits / 8 / n
    total = len(blob) + stats.table_bytes
    rows = [
        ("blob", len(blob), len(blob) / n),
        ("blob 1 bpb literals", ideal_bits / 8, ideal),
        ("blob + table", total, total / n),
        ("gzip -9", gz, gz / n),
        ("byte trunk 0.97", stream, 0.97 / 8),
    ]
    text = "\n".join(
        [
            "# Fill-copy codec: long fills, exact pointers",
            "",
            f"- code corpus `{corpus.sha[:12]}`, train {len(train):,} B, "
            f"held-out {n:,} B",
            f"- tokenizer: min fill {min_fill} B, cap 1024 B; templates "
            f"{min_count}+ occurrences",
            f"- library {stats.templates:,} templates, {stats.table_bytes:,} B table",
            "",
            "| item | bytes | bytes/byte |",
            "|---|---:|---:|",
            *[f"| {name} | {size:,.0f} | {ratio:.3f} |" for name, size, ratio in rows],
            "",
            "## Stream",
            "",
            f"- fixed {stats.fixed_bytes / n:.3f}, fill {stats.fill_bytes / n:.3f}, "
            f"literal {stats.literal_bytes / n:.3f}",
            f"- fill copy: local {stats.local_bytes / n:.3f}, pack "
            f"{stats.pack_bytes / n:.3f}, literal {stats.literal_fill_bytes / n:.3f}",
            f"- templates {stats.decisions / n:.3f} decisions/B; "
            f"{n / max(stats.decisions, 1):.2f} B/decision",
            f"- round-trip {'OK' if stats.roundtrip else 'FAIL'}; "
            f"table charge {stats.table_bytes:,} B",
            "",
            "## Read",
            "",
            "Fixed syntax is free after the table charge. Fills are exact "
            "pointers into earlier output or the train pack; a fill that is "
            "not present is sent literally. The one-bpb row replaces literal "
            "bytes with the byte trunk's measured rate, so it is an estimate.",
            "",
            f"Took {stats.seconds} s CPU.",
            "",
        ]
    )
    cml.write_text(cml.REPORTS / "FILL-COPY.md", text)
    cml.write_json(
        cml.REPORTS / "fill-copy.json",
        {"stats": stats.as_dict(), "git": cml.git_sha()},
    )
    print(text)
    return 0
