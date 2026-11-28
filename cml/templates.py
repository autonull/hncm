"""Template-fill probe: exact fixed syntax, typed holes, content fills.

The event unit is a token run whose fixed tokens are exact bytes and whose
variable atoms are typed holes.  A hole is filled from an installable
lexicon; a missed template falls back to literal tokens.  The probe
measures the headroom of that unit before any training: how many bytes are
fixed for free, how many are holes, how many need the byte fallback, and
how many decisions a sequential model would take.
"""

from __future__ import annotations

import keyword
import re
import time
from collections import Counter
from dataclasses import dataclass

import cml

KW = {k.encode() for k in keyword.kwlist}
ATOM = re.compile(
    rb"""(?P<s>'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*")|"""
    rb"""(?P<n>\d+(?:\.\d+)?)|(?P<i>[A-Za-z_]\w*)""",
    re.S,
)


@dataclass(frozen=True, slots=True)
class Hit:
    key: tuple[bytes, ...]
    fills: tuple[bytes, ...]


@dataclass(frozen=True, slots=True)
class Lit:
    kind: bytes
    value: bytes


def tokenize(line: bytes) -> list[tuple[bytes, bytes]]:
    """Split a line into fixed syntax and typed atoms."""
    out: list[tuple[bytes, bytes]] = []
    last = 0
    for m in ATOM.finditer(line):
        if m.start() > last:
            out.append((b"F", line[last : m.start()]))
        value = m.group(0)
        if m.group("s") is not None:
            out.append((b"S", value))
        elif m.group("n") is not None:
            out.append((b"N", value))
        elif value in KW:
            out.append((b"F", value))
        else:
            out.append((b"I", value))
        last = m.end()
    if last < len(line):
        out.append((b"F", line[last:]))
    return out


def keyof(tokens: list[tuple[bytes, bytes]]) -> tuple[bytes, ...]:
    return tuple(
        b"F" + value if kind == b"F" else b"H" + kind for kind, value in tokens
    )


def induce(
    lines: list[bytes], min_count: int = 3, max_n: int = 6
) -> tuple[dict[tuple[bytes, ...], int], Counter[tuple[bytes, bytes]]]:
    """Count token-run templates and typed fill values."""
    counts: Counter[tuple[bytes, ...]] = Counter()
    fills: Counter[tuple[bytes, bytes]] = Counter()
    for line in lines:
        ts = tokenize(line)
        for kind, value in ts:
            if kind != b"F":
                fills[(kind, value)] += 1
        for n in range(2, max_n + 1):
            for i in range(len(ts) - n + 1):
                win = ts[i : i + n]
                holes = sum(1 for kind, _ in win if kind != b"F")
                fixed = sum(len(value) for kind, value in win if kind == b"F")
                if holes and fixed >= 6 and holes < n - 1:
                    counts[keyof(win)] += 1
    keep = {key: count for key, count in counts.items() if count >= min_count}
    return keep, fills


def parse(
    data: bytes,
    templates: dict[tuple[bytes, ...], int],
    max_n: int = 6,
) -> list[Hit | Lit]:
    """Greedy longest-template parse over token boundaries."""
    events: list[Hit | Lit] = []
    for line in data.splitlines(keepends=True):
        ts = tokenize(line)
        i = 0
        while i < len(ts):
            found = None
            for n in range(max_n, 1, -1):
                if i + n > len(ts):
                    continue
                win = ts[i : i + n]
                key = keyof(win)
                if key in templates:
                    found = (key, tuple(value for kind, value in win if kind != b"F"))
                    i += n
                    break
            if found is None:
                events.append(Lit(*ts[i]))
                i += 1
            else:
                events.append(Hit(*found))
    return events


def render(events: list[Hit | Lit]) -> bytes:
    """Expand events back to bytes, consuming fixed and fill elements."""
    out = bytearray()
    for event in events:
        if isinstance(event, Lit):
            out += event.value
            continue
        fills = iter(event.fills)
        for element in event.key:
            out += element[1:] if element.startswith(b"F") else next(fills)
    return bytes(out)


def probe(train: bytes, test: bytes, min_count: int = 3, max_n: int = 6) -> dict:
    """Measure template and fill headroom on a held-out byte slice."""
    t0 = time.perf_counter()
    templates, fills = induce(train.splitlines(keepends=True), min_count, max_n)
    slots: dict[tuple[tuple[bytes, ...], int], Counter[bytes]] = {}
    for line in train.splitlines(keepends=True):
        for event in parse(line, templates, max_n):
            if isinstance(event, Hit):
                for j, value in enumerate(event.fills):
                    slots.setdefault((event.key, j), Counter())[value] += 1
    events = parse(test, templates, max_n)
    rebuilt = render(events)
    fixed = hole = lit_fixed = lit_atom = decisions = hits = 0
    atom_bytes = atom_vocab = seen_hole = combo_hole = novel_hole = 0
    for event in events:
        if isinstance(event, Lit):
            decisions += 1
            if event.kind == b"F":
                lit_fixed += len(event.value)
            else:
                lit_atom += len(event.value)
                atom_bytes += len(event.value)
                atom_vocab += (
                    len(event.value) if (event.kind, event.value) in fills else 0
                )
            continue
        hits += 1
        decisions += 1 + len(event.fills)
        for j, (element, value) in enumerate(
            zip((e for e in event.key if not e.startswith(b"F")), event.fills)
        ):
            kind = element[1:]
            hole += len(value)
            atom_bytes += len(value)
            if (kind, value) in fills:
                atom_vocab += len(value)
            if value in slots.get((event.key, j), {}):
                seen_hole += len(value)
            elif (kind, value) in fills:
                combo_hole += len(value)
            else:
                novel_hole += len(value)
        fixed += sum(len(e) - 1 for e in event.key if e.startswith(b"F"))
    n = max(len(test), 1)
    table = sum(len(e) for key, _ in templates.items() for e in key) + 3 * sum(
        len(key) for key in templates
    )
    table += sum(len(value) + 3 for _, value in fills)
    return {
        "train_bytes": len(train),
        "test_bytes": len(test),
        "templates": len(templates),
        "fills": len(fills),
        "table_bytes": table,
        "hit_events": hits,
        "events": len(events),
        "coverage": (fixed + hole + lit_fixed + lit_atom) / n,
        "fixed": fixed / n,
        "hole": hole / n,
        "lit_fixed": lit_fixed / n,
        "lit_atom": lit_atom / n,
        "atom_vocab": atom_vocab / max(atom_bytes, 1),
        "seen_hole": seen_hole / max(hole, 1),
        "combo_hole": combo_hole / max(hole, 1),
        "novel_hole": novel_hole / max(hole, 1),
        "slot_pairs": len(slots),
        "slot_mean": sum(len(c) for c in slots.values()) / max(len(slots), 1),
        "decisions": decisions / n,
        "bytes_per_decision": n / max(decisions, 1),
        "roundtrip": rebuilt == test,
        "seconds": round(time.perf_counter() - t0, 1),
    }


def run_probe(
    train_bytes: int = 2_000_000,
    test_bytes: int = 100_000,
    min_count: int = 3,
    max_n: int = 6,
) -> int:
    corpus = cml.code_scale_corpus(48_000_000)
    train = corpus.train[:train_bytes]
    test = corpus.test[:test_bytes]
    stats = probe(train, test, min_count, max_n)
    stats["corpus_sha"] = corpus.sha
    stats["git"] = cml.git_sha()
    lines = [
        "# Template-fill probe",
        "",
        f"- code corpus `{corpus.sha[:12]}`, train {len(train):,} B, "
        f"held-out {len(test):,} B",
        f"- templates: token runs 2-{max_n}, min count {min_count}, "
        f"at least 6 fixed bytes, fewer holes than tokens",
        "- fill lexicon: exact typed atoms from train",
        "",
        "| item | value |",
        "|---|---:|",
        f"| templates | {stats['templates']:,} |",
        f"| fill values | {stats['fills']:,} |",
        f"| table bytes | {stats['table_bytes']:,} |",
        f"| fixed bytes | {stats['fixed']:.3f} |",
        f"| hole bytes | {stats['hole']:.3f} |",
        f"| literal fixed | {stats['lit_fixed']:.3f} |",
        f"| literal atom | {stats['lit_atom']:.3f} |",
        f"| atom in lexicon | {stats['atom_vocab']:.3f} |",
        f"| hole: seen slot | {stats['seen_hole']:.3f} |",
        f"| hole: new combination | {stats['combo_hole']:.3f} |",
        f"| hole: novel value | {stats['novel_hole']:.3f} |",
        f"| slot pairs | {stats['slot_pairs']:,} |",
        f"| distinct fills/slot | {stats['slot_mean']:.1f} |",
        f"| decisions/B | {stats['decisions']:.3f} |",
        f"| bytes/decision | {stats['bytes_per_decision']:.2f} |",
        f"| round-trip | {'OK' if stats['roundtrip'] else 'FAIL'} |",
        "",
        "## Read",
        "",
        f"- fixed syntax is {stats['fixed']:.1%} of held-out bytes and needs "
        f"no model call; holes are another {stats['hole']:.1%}.",
        f"- a sequential event model would take {stats['decisions']:.3f} "
        f"decisions/B ({stats['bytes_per_decision']:.2f} B/decision), "
        "before any learned predictor.",
        f"- {stats['atom_vocab']:.1%} of atom bytes are in the 2 MB train "
        "lexicon; the rest need context copy or the byte fallback.",
        f"- of hole bytes, {stats['seen_hole']:.1%} repeat the same slot, "
        f"{stats['combo_hole']:.1%} are new combinations with a known fill, "
        f"and {stats['novel_hole']:.1%} are novel values. New combinations "
        "are the capability target: a flat memorizer must learn each pair; "
        "a template model can share the slot.",
        "- this is parser headroom, not a trained LM. The decisive test is "
        "held-out fill composition, not the raw decision count.",
        "",
        "## Architecture: TF-LM",
        "",
        "Bytes -> deterministic tokenizer -> greedy template parse -> events.",
        "An event is `T(key, fills)` or `L(token)`. The model is a",
        "decoder-only transformer over event embeddings:",
        "",
        "- template embedding = a small causal encoder of the fixed tokens",
        "  plus a hole-type vector;",
        "- fill embedding = a content encoder of the fill bytes, tied across",
        "  templates, so a new combination lands in a shared space;",
        "- heads: next template, per-slot fill under the hole type, and a",
        "  byte fallback for novel values;",
        "- exact output by table lookup; a digest verifies every event;",
        "- library = templates + fills (the table above), installable and",
        "  diffable like a repo pack.",
        "",
        "One forward per event, not per byte. Fixed bytes are memcpy.",
        "Novel fills or missed templates go to the byte fallback.",
        "",
        "## Decisive experiment (T1)",
        "",
        "Event model ~30M: d512, 12 layers, 8 heads, 512 events, batch 8,",
        "AdamW lr 3e-4, 20k steps on 8 MB code. Baselines: the byte trunk",
        "and a BPE token model at equal FLOPs. Split: hold out 20% of",
        "(template, slot, fill) pairs while each held-out fill still appears",
        "in other slots; the model must share content, not memorize the pair.",
        "",
        "Gate: 100% exact round-trip, <=0.03 bpb regression against the byte",
        "trunk, >=+10 points exact fill accuracy on held-out combinations,",
        "and >=2x bytes/decision at equal quality.",
        "",
        "T2: install a library from repo A on repo B with an empty-library",
        "control; gate >=0.15 bpb improvement from installation alone.",
        "",
        f"Took {stats['seconds']} s CPU.",
        "",
    ]
    text = "\n".join(lines)
    cml.write_text(cml.REPORTS / "TEMPLATES.md", text)
    cml.write_json(cml.REPORTS / "templates.json", stats)
    print(text)
    return 0
