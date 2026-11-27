"""Exact-edit grammar, assembler, metrics, and the retrieval baseline.

Two tracks share one benchmark and one verifier:

- Track C: a serialized copy/literal action program over `old` bytes.
- Track E: changed spans in `old`, each with replacement bytes.

The assembler is exact and deterministic: a program either rebuilds `new`
byte-for-byte or scores zero.  This module builds targets from real commits.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import codec
import editdata

TOKEN = re.compile(r"[A-Za-z0-9_]+")
G0_MAX_EXACT = 0.20


@dataclass(frozen=True, slots=True)
class Action:
    """Phase-1 copies address `old`; later copies may address emitted bytes."""

    kind: str
    offset: int = 0
    length: int = 0
    data: bytes = b""


@dataclass(frozen=True, slots=True)
class Span:
    """Bytes `old[old_start:old_start+old_len]` become `new`."""

    old_start: int
    old_len: int
    new: bytes


def _varint(out: bytearray, value: int) -> None:
    if value < 0:
        raise ValueError("varint must be non-negative")
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)


def _read_varint(data: bytes, i: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if i >= len(data):
            raise ValueError("truncated varint")
        byte = data[i]
        i += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, i
        shift += 7


def copy_actions(
    new: bytes,
    old: bytes,
    window: int = 1 << 20,
    min_match: int = 8,
    cap: int = 256,
    depth: int = 16,
) -> list[Action]:
    """Track C target: merge adjacent literals around the LZ copy parse."""
    actions: list[Action] = []
    literals = bytearray()
    for length, value in codec.delta_parse(new, old, window, min_match, cap, depth):
        if length == 0:
            literals.append(value)
            continue
        if literals:
            actions.append(Action("LIT", data=bytes(literals)))
            literals.clear()
        actions.append(Action("COPY", value, length))
    if literals:
        actions.append(Action("LIT", data=bytes(literals)))
    return actions


def assemble(actions: list[Action], old: bytes, expect: int | None = None) -> bytes:
    """Exact phase-1/phase-2 assembler; raises on any invalid action."""
    buf = bytearray(old)
    start = len(buf)
    for action in actions:
        if action.kind == "COPY":
            if action.offset <= 0 or action.offset > len(buf) or action.length <= 0:
                raise ValueError("invalid copy action")
            for _ in range(action.length):
                buf.append(buf[-action.offset])
            continue
        if action.kind != "LIT":
            raise ValueError(f"unknown action {action.kind!r}")
        buf.extend(action.data)
    out = bytes(buf[start:])
    if expect is not None and len(out) != expect:
        raise ValueError("assembled length does not match the target")
    return out


def dump_actions(actions: list[Action]) -> bytes:
    out = bytearray()
    for action in actions:
        if action.kind == "COPY":
            out.append(1)
            _varint(out, action.offset)
            _varint(out, action.length)
        else:
            out.append(2)
            _varint(out, len(action.data))
            out.extend(action.data)
    out.append(0)
    return bytes(out)


def load_actions(blob: bytes) -> list[Action]:
    actions: list[Action] = []
    i = 0
    while True:
        if i >= len(blob):
            raise ValueError("action stream has no end marker")
        tag = blob[i]
        i += 1
        if tag == 0:
            if i != len(blob):
                raise ValueError("trailing bytes after the end marker")
            return actions
        if tag == 1:
            offset, i = _read_varint(blob, i)
            length, i = _read_varint(blob, i)
            actions.append(Action("COPY", offset, length))
        elif tag == 2:
            size, i = _read_varint(blob, i)
            if i + size > len(blob):
                raise ValueError("truncated literal")
            actions.append(Action("LIT", data=blob[i : i + size]))
            i += size
        else:
            raise ValueError(f"unknown action tag {tag}")


def _offsets(lines: list[bytes]) -> list[int]:
    off = [0]
    for line in lines:
        off.append(off[-1] + len(line))
    return off


def _trim(old: bytes, new: bytes) -> tuple[int, int]:
    """Common prefix and suffix lengths; never overlap."""
    limit = min(len(old), len(new))
    prefix = 0
    while prefix < limit and old[prefix] == new[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < limit - prefix
        and old[len(old) - 1 - suffix] == new[len(new) - 1 - suffix]
    ):
        suffix += 1
    return prefix, suffix


def spans_from(old: bytes, new: bytes) -> list[Span]:
    """Track E target: changed byte spans, from a line diff trimmed per block."""
    a = old.splitlines(keepends=True)
    b = new.splitlines(keepends=True)
    a_off, b_off = _offsets(a), _offsets(b)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    spans: list[Span] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        ob = old[a_off[i1] : a_off[i2]]
        nb = new[b_off[j1] : b_off[j2]]
        prefix, suffix = _trim(ob, nb)
        start = a_off[i1] + prefix
        old_len = len(ob) - prefix - suffix
        repl = nb[prefix : len(nb) - suffix]
        if old_len or repl:
            spans.append(Span(start, old_len, repl))
    return spans


def assemble_spans(old: bytes, spans: list[Span]) -> bytes:
    """Exact span assembler; raises on overlap."""
    out = bytearray()
    pos = 0
    for span in sorted(spans, key=lambda s: (s.old_start, s.old_len)):
        if span.old_start < pos:
            raise ValueError("overlapping spans")
        if span.old_len < 0:
            raise ValueError("negative span length")
        out.extend(old[pos : span.old_start])
        out.extend(span.new)
        pos = span.old_start + span.old_len
    out.extend(old[pos:])
    return bytes(out)


def span_scores(pred: list[Span], target: list[Span]) -> dict[str, float | bool]:
    """Span F1 and location exact on the changed extent in `old`."""
    p = [(s.old_start, s.old_len) for s in pred]
    t = [(s.old_start, s.old_len) for s in target]
    pc: dict[tuple[int, int], int] = {}
    tc: dict[tuple[int, int], int] = {}
    for key in p:
        pc[key] = pc.get(key, 0) + 1
    for key in t:
        tc[key] = tc.get(key, 0) + 1
    tp = sum(min(n, tc.get(key, 0)) for key, n in pc.items())
    fp = sum(pc.values()) - tp
    fn = sum(tc.values()) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "location_exact": pc == tc,
    }


def line_distance(a: bytes, b: bytes) -> float:
    """Normalized line edit distance: 1 - Dice similarity."""
    if a == b:
        return 0.0
    al, bl = a.splitlines(), b.splitlines()
    if not al and not bl:
        return 0.0
    sm = difflib.SequenceMatcher(None, al, bl, autojunk=False)
    same = sum(j2 - j1 for tag, _, _, j1, j2 in sm.get_opcodes() if tag == "equal")
    return 1.0 - 2.0 * same / (len(al) + len(bl))


def message_tokens(text: str) -> frozenset[str]:
    return frozenset(m.group(0).lower() for m in TOKEN.finditer(text))


def path_bonus(query: Retrieval, candidate: Retrieval) -> float:
    if query.path == candidate.path:
        return 0.6
    if query.base == candidate.base:
        return 0.35
    return 0.1 if query.suffix == candidate.suffix else 0.0


def _splice(old: bytes, edits: list[tuple[int, int, bytes]]) -> bytes:
    out = bytearray()
    pos = 0
    for start, end, repl in sorted(edits):
        if start < pos or end < start:
            continue
        out.extend(old[pos:start])
        out.extend(repl)
        pos = end
    out.extend(old[pos:])
    return bytes(out)


def transplant(query_old: bytes, ret_old: bytes, ret_new: bytes) -> bytes | None:
    """Replay a retrieved edit on `query_old` via aligned lines.

    Returns None when the retrieved change has no matching context; a valid
    unchanged answer is `query_old` itself.
    """
    if ret_old == query_old:
        return ret_new
    rlines = ret_old.splitlines(keepends=True)
    qlines = query_old.splitlines(keepends=True)
    q_off = _offsets(qlines)
    align = difflib.SequenceMatcher(None, rlines, qlines, autojunk=False)
    qmap = [-1] * len(rlines)
    for tag, i1, i2, j1, j2 in align.get_opcodes():
        if tag != "equal" or (i2 - i1) != (j2 - j1):
            continue
        for k in range(i2 - i1):
            qmap[i1 + k] = j1 + k
    newlines = ret_new.splitlines(keepends=True)
    changes = difflib.SequenceMatcher(None, rlines, newlines, autojunk=False)
    n_off = _offsets(newlines)
    edits: list[tuple[int, int, bytes]] = []
    for tag, i1, i2, j1, j2 in changes.get_opcodes():
        if tag == "equal":
            continue
        repl = ret_new[n_off[j1] : n_off[j2]]
        if i1 == i2:
            if i1 < len(qmap) and qmap[i1] >= 0:
                a = b = q_off[qmap[i1]]
            elif i1 > 0 and qmap[i1 - 1] >= 0:
                a = b = q_off[qmap[i1 - 1] + 1]
            else:
                return None
        else:
            mapped = qmap[i1:i2]
            if any(v < 0 for v in mapped):
                return None
            if any(mapped[k + 1] != mapped[k] + 1 for k in range(len(mapped) - 1)):
                return None
            a = q_off[mapped[0]]
            b = q_off[mapped[-1] + 1]
        edits.append((a, b, repl))
    return _splice(query_old, edits)


@dataclass(frozen=True, slots=True)
class Retrieval:
    record: dict
    old: bytes
    new: bytes
    tokens: frozenset[str] = field(repr=False)
    path: str = ""
    base: str = ""
    suffix: str = ""


def _load(manifest: dict, cache: Path, split: str) -> list[Retrieval]:
    out: list[Retrieval] = []
    for rec, old, new in editdata.load_edits(manifest, cache, split):
        path = rec["path"]
        out.append(
            Retrieval(
                rec,
                old,
                new,
                message_tokens(rec["message"]),
                path,
                Path(path).name,
                Path(path).suffix,
            )
        )
    return out


def _score(query: Retrieval, cand: Retrieval) -> float:
    inter = len(query.tokens & cand.tokens)
    union = len(query.tokens) + len(cand.tokens) - inter
    similarity = inter / union if union else 0.0
    return similarity + path_bonus(query, cand)


def run_retrieval(manifest: dict, cache: Path, split: str, limit: int = 0) -> dict:
    """Arm R: nearest train (old, new) pair by message and path, replayed."""
    train = _load(manifest, cache, "train")
    queries = _load(manifest, cache, split)
    if limit:
        queries = queries[:limit]
    if not train:
        raise SystemExit("editlm: the train split is empty")
    if not queries:
        raise SystemExit(f"editlm: the {split} split is empty")

    items = []
    exact = valid = 0
    distance = 0.0
    output = 0.0
    for query in queries:
        best = max(train, key=lambda cand: _score(query, cand))
        pred = transplant(query.old, best.old, best.new)
        applied = pred is not None
        ok = applied and pred == query.new
        if not applied:
            pred = query.old
        exact += int(ok)
        valid += int(applied)
        distance += line_distance(pred, query.new)
        output += len(pred) / max(len(query.new), 1)
        items.append(
            {
                "id": query.record["id"],
                "repo": query.record["repo"],
                "path": query.record["path"],
                "commit": query.record["commit"],
                "exact": ok,
                "valid": applied,
                "edit_distance": round(line_distance(pred, query.new), 4),
                "output_ratio": round(len(pred) / max(len(query.new), 1), 4),
                "retrieved": best.record["id"],
            }
        )
    n = len(items)
    exact_rate = exact / n
    return {
        "arm": "retrieval",
        "split": split,
        "n": n,
        "exact": exact_rate,
        "valid": valid / n,
        "edit_distance": distance / n,
        "output_ratio": output / n,
        "g0": {
            "threshold": G0_MAX_EXACT,
            "pass": exact_rate < G0_MAX_EXACT,
            "meaning": "a pass means retrieval is not a fake win",
        },
        "items": items,
    }


def render_report(manifest: dict, result: dict, cache: Path) -> str:
    counts = manifest["counts"]
    config = manifest["config"]
    lines = [
        "# Exact Edit Bench: retrieval baseline (G0)",
        "",
        "- arm **R retrieval**; deterministic; one job; no GPU",
        f"- split `{result['split']}`; n = {result['n']}",
        f"- manifest repos: {', '.join(config['repos'])}",
        f"- cache `{cache}`",
        f"- manifest sha256 `{result['manifest']['sha256'][:16]}`",
        "",
        "## Task",
        "",
        "Given `old`, the commit message, and the file path, predict `new`",
        "exactly. Retrieval picks the nearest train `(old, new)` pair by",
        "message Jaccard and path match, then replays its line changes on",
        "`old` where the context aligns. A wrong result scores zero.",
        "",
        "## Data",
        "",
        "| split | repos | commits |",
        "|---|---|---:|",
    ]
    for name in ("train", "val", "test"):
        repos = ", ".join(manifest["splits"][name])
        lines.append(f"| {name} | {repos} | {counts[name]} |")
    lines += [
        "",
        f"- filter: one text file, ≤ {config['max_file']} B, "
        f"diff ≤ {config['max_diff']:.0%} ({config['diff_metric']})",
        "- split by repository, deduplicated by `(old, new)` within a split",
    ]
    for repo, head in sorted(manifest["heads"].items()):
        lines.append(f"- `{repo}` head `{head[:12]}`")
    lines += [
        "",
        "## G0",
        "",
        "| test | threshold | observed | result |",
        "|---|---:|---:|---|",
        f"| retrieval exact | < {G0_MAX_EXACT:.0%} | "
        f"{result['exact']:.1%} | {'pass' if result['g0']['pass'] else 'harden'} |",
        "",
        "## Results",
        "",
        "| arm | n | exact | valid | edit distance | output bytes |",
        "|---|---:|---:|---:|---:|---:|",
        f"| R retrieval | {result['n']} | {result['exact']:.3f} | "
        f"{result['valid']:.3f} | {result['edit_distance']:.3f} | "
        f"{result['output_ratio']:.3f} |",
        "",
        "## Reading",
        "",
    ]
    if result["g0"]["pass"]:
        lines += [
            "Retrieval only sees the train repositories, so it cannot replay",
            "the query repository's own history. Its exact rate is a lower",
            "bound, not an upper bound. G0 passes: the benchmark is hard",
            "enough to train against. Proceed to E1 (locator) and C1",
            "(copy-SFT) on the same split. G0 is recorded.",
        ]
    else:
        lines += [
            "Retrieval solves at least 20% of held-out edits. Harden the",
            "benchmark before training: longer edits, counterfactual messages,",
            "and no near-duplicate pairs across the split.",
        ]
    lines += [
        "",
        "## Next",
        "",
        "| gate | arm | needs |",
        "|---|---|---|",
        "| GE1 | E1 locator | span F1 +20 points over message-only |",
        "| GC1 | C1 copy-SFT | +10 exact or +20 test pass, ≤5% output bytes |",
        "| GE2 | E2 compiler | +10 exact or +20 test pass, ≤20% output bytes |",
        "",
        "Val is `pytest-dev/pytest`; test is the last repository by name.",
        "Do not tune on test. One job at a time.",
    ]
    return "\n".join(lines) + "\n"


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    os.replace(tmp, path)


def run(
    arm: str,
    manifest_path: Path,
    cache: Path,
    split: str,
    limit: int,
    report: Path,
    out: Path,
    seed: int,
    steps: int,
    base: str,
    pretrain: str,
    pairs: int,
    samples: int,
    mode: str,
) -> int:
    if arm != "retrieval":
        raise SystemExit(
            f"editlm: arm {arm} is not built; the retrieval arm records G0 "
            f"first (TODO9.md §8)"
        )
    if seed or steps or base or pretrain or pairs or samples or mode:
        raise SystemExit("editlm: the retrieval arm takes no training arguments")
    manifest = editdata.load_manifest(manifest_path)
    cache = editdata.cache_root(cache)
    result = run_retrieval(manifest, cache, split, limit)
    result["manifest"] = {
        "path": str(manifest_path),
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "config": manifest["config"],
        "heads": manifest["heads"],
        "counts": manifest["counts"],
        "splits": manifest["splits"],
    }
    _write_json(out, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(manifest, result, cache))
    print(
        f"editlm: R retrieval n={result['n']} exact={result['exact']:.3f} "
        f"valid={result['valid']:.3f} g0={'pass' if result['g0']['pass'] else 'harden'}"
    )
    print(f"editlm: {report}")
    return 0
