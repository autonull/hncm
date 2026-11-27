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
import math
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import codec
import editdata

TOKEN = re.compile(r"[A-Za-z0-9_]+")
G0_MAX_EXACT = 0.20
ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
LOCATOR_D = 256
LOCATOR_LAYERS = 6
LOCATOR_HEADS = 8
LOCATOR_MSG = 192
LOCATOR_WIN = 1024
COPY_ACT = 256
COPY_SEQ = LOCATOR_MSG + 1 + LOCATOR_WIN + COPY_ACT


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


class EncBlock(nn.Module):
    """A bidirectional pre-LN block for byte-span classification."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d)
        self.out = nn.Linear(4 * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        shape = (b, t, self.heads, d // self.heads)
        q, k, v = (z.view(shape).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).reshape(b, t, d)
        x = x + self.proj(y)
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class Locator(nn.Module):
    """Message + file window -> one changed-byte logit per file byte."""

    def __init__(
        self,
        d: int = LOCATOR_D,
        layers: int = LOCATOR_LAYERS,
        heads: int = LOCATOR_HEADS,
        msg: int = LOCATOR_MSG,
        win: int = LOCATOR_WIN,
    ):
        super().__init__()
        self.msg_tok = nn.Embedding(256, d)
        self.file_tok = nn.Embedding(256, d)
        self.pos = nn.Embedding(msg + win, d)
        self.seg = nn.Embedding(2, d)
        self.blocks = nn.ModuleList(EncBlock(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        self.msg = msg
        self.win = win
        nn.init.normal_(self.msg_tok.weight, std=0.02)
        nn.init.normal_(self.file_tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(
        self, msg_ids: torch.Tensor, file_ids: torch.Tensor, zero_file: bool = False
    ) -> torch.Tensor:
        b = msg_ids.size(0)
        m = msg_ids.size(1)
        w = file_ids.size(1)
        pos = self.pos.weight[: m + w]
        message = self.msg_tok(msg_ids) + pos[:m] + self.seg.weight[0]
        if zero_file:
            files = pos[m : m + w] + self.seg.weight[1]
            files = files.unsqueeze(0).expand(b, -1, -1)
        else:
            files = self.file_tok(file_ids) + pos[m : m + w] + self.seg.weight[1]
        h = torch.cat([message, files], dim=1)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln(h[:, m:])).squeeze(-1)


def _byte_tensor(data: bytes, size: int) -> torch.Tensor:
    out = torch.zeros(size, dtype=torch.long)
    if data:
        out[: len(data)] = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
    return out


def _mask_from_spans(old: bytes, spans: list[Span]) -> np.ndarray:
    mask = np.zeros(len(old), dtype=np.uint8)
    for span in spans:
        if span.old_len > 0:
            mask[span.old_start : span.old_start + span.old_len] = 1
    return mask


def _mask_to_spans(mask: np.ndarray, min_len: int = 2, gap: int = 0) -> list[Span]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, on in enumerate(mask.tolist() + [False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append((start, i))
            start = None
    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return [Span(s, e - s, b"") for s, e in merged if e - s >= min_len]


@dataclass(frozen=True, slots=True)
class Labeled:
    record: dict
    old: bytes
    spans: list[Span]
    mask: np.ndarray
    focus: list[tuple[int, int]]


def load_labeled(
    manifest: dict, cache: Path, split: str, limit: int = 0
) -> list[Labeled]:
    """Load files with at least one non-empty changed span in `old`."""
    out: list[Labeled] = []
    for rec, old, new in editdata.load_edits(manifest, cache, split):
        spans = [s for s in spans_from(old, new) if s.old_len > 0]
        if not spans:
            continue
        focus = [(s.old_start, s.old_start + s.old_len) for s in spans]
        out.append(Labeled(rec, old, spans, _mask_from_spans(old, spans), focus))
    if limit and len(out) > limit:
        step = max(1, len(out) // limit)
        out = out[::step][:limit]
    return out


def _synthetic_example(files: list[bytes], rng: random.Random) -> Labeled | None:
    old = files[rng.randrange(len(files))]
    lines = old.splitlines(keepends=True)
    if len(lines) < 2 or len(old) < 32:
        return None
    i = rng.randrange(len(lines))
    line = lines[i]
    tokens = TOKEN.findall(line.decode("utf-8", "ignore"))
    if len(tokens) < 2:
        return None
    token = min(tokens, key=lambda t: (old.count(t.encode()), -len(t)))
    if len(token) < 4 or old.count(token.encode()) > 3:
        return None
    at = line.find(token.encode())
    if at < 0:
        return None
    start = sum(len(x) for x in lines[:i]) + at
    span = Span(start, len(token), b"")
    mask = _mask_from_spans(old, [span])
    return Labeled(
        {"message": f"change {token}"},
        old,
        [span],
        mask,
        [(start, start + len(token))],
    )


def _sample(
    example: Labeled,
    rng: random.Random,
    win: int,
    msg_len: int,
    positive: bool = True,
) -> tuple[bytes, bytes, np.ndarray, int]:
    old = example.old
    if positive and example.focus:
        a, b = example.focus[rng.randrange(len(example.focus))]
        lo = max(0, b - win)
        hi = min(a, max(0, len(old) - win))
        start = rng.randint(lo, hi) if hi >= lo else max(0, min(a, len(old) - win))
    else:
        start = rng.randint(0, max(0, len(old) - win))
        for _ in range(8):
            if not example.mask[start : start + win].any():
                break
            start = rng.randint(0, max(0, len(old) - win))
    end = min(len(old), start + win)
    labels = np.zeros(win, dtype=np.uint8)
    labels[: end - start] = example.mask[start:end]
    msg = example.record["message"].encode("utf-8")[:msg_len]
    return msg, old[start:end], labels, end - start


def _batch(
    examples: list[Labeled],
    files: list[bytes],
    rng: random.Random,
    device: torch.device,
    synthetic: bool,
    batch: int,
    win: int,
    msg_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    msg = torch.zeros(batch, msg_len, dtype=torch.long)
    file = torch.zeros(batch, win, dtype=torch.long)
    labels = torch.zeros(batch, win)
    valid = torch.zeros(batch, win)
    for k in range(batch):
        example = (
            _synthetic_example(files, rng)
            if synthetic
            else examples[rng.randrange(len(examples))]
        )
        if example is None:
            continue
        m, chunk, label, length = _sample(
            example, rng, win, msg_len, rng.random() < 0.5
        )
        if m:
            msg[k, : len(m)] = torch.frombuffer(bytearray(m), dtype=torch.uint8).long()
        file[k, :length] = torch.frombuffer(bytearray(chunk), dtype=torch.uint8).long()
        labels[k, :length] = torch.from_numpy(label[:length].astype(np.float32))
        valid[k, :length] = 1.0
    return (
        msg.to(device, non_blocking=True),
        file.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
        valid.to(device, non_blocking=True),
    )


def _loss(model: Locator, batch: tuple, zero_file: bool) -> torch.Tensor:
    msg, file, labels, valid = batch
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=msg.is_cuda):
        logits = model(msg, file, zero_file)
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)
    prob = torch.sigmoid(logits) * valid
    target = labels * valid
    num = 2 * (prob * target).sum(dim=1) + 1
    den = prob.sum(dim=1) + target.sum(dim=1) + 1
    dice = (1 - num / den).mean()
    return bce + dice


def train_locator(
    examples: list[Labeled],
    steps: int,
    seed: int,
    pretrain: str,
    zero_file: bool,
    device: torch.device,
    batch: int = 16,
    model: Locator | None = None,
) -> tuple[Locator, list[dict]]:
    """Pretrain on synthetic line edits, then fine-tune on real commits."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    files = [ex.old for ex in examples]
    model = model or Locator()
    model.to(device)
    log: list[dict] = []

    def run_phase(phase: str, n: int, lr: float, synth: bool) -> None:
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        warm = min(100, max(1, n // 10))

        def factor(step: int) -> float:
            if step < warm:
                return (step + 1) / warm
            progress = (step - warm) / max(1, n - warm)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
        model.train()
        for step in range(n):
            batch_in = _batch(
                examples, files, rng, device, synth, batch, model.win, model.msg
            )
            loss = _loss(model, batch_in, zero_file)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % 100 == 0 or step == n - 1:
                log.append(
                    {
                        "phase": phase,
                        "step": step,
                        "loss": round(float(loss.detach()), 4),
                    }
                )

    if pretrain == "synthetic":
        run_phase("synthetic", max(200, steps // 2), 3e-4, True)
    run_phase("real", steps, 2e-5 if pretrain else 3e-4, False)
    return model, log


def _window_starts(n: int, win: int, stride: int) -> list[int]:
    if n <= win:
        return [0]
    starts = list(range(0, n - win + 1, stride))
    if starts[-1] != n - win:
        starts.append(n - win)
    return starts


@torch.no_grad()
def predict_spans(
    model: Locator,
    old: bytes,
    message: str,
    device: torch.device,
    threshold: float = 0.5,
    min_len: int = 2,
    zero_file: bool = False,
    smooth: int = 1,
    gap: int = 0,
) -> list[Span]:
    """Slide windows, average byte probabilities, and group changed runs."""
    if not old:
        return []
    msg = _byte_tensor(message.encode("utf-8")[: model.msg], model.msg)
    msg = msg.to(device)
    starts = _window_starts(len(old), model.win, model.win // 2)
    probs = np.zeros(len(old), dtype=np.float32)
    counts = np.zeros(len(old), dtype=np.int32)
    model.eval()
    for at in range(0, len(starts), 32):
        chunk = starts[at : at + 32]
        file = torch.zeros(len(chunk), model.win, dtype=torch.long)
        for k, start in enumerate(chunk):
            seg = old[start : start + model.win]
            file[k, : len(seg)] = torch.frombuffer(
                bytearray(seg), dtype=torch.uint8
            ).long()
        file = file.to(device)
        msg_batch = msg.unsqueeze(0).expand(len(chunk), -1)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=file.is_cuda
        ):
            logits = model(msg_batch, file, zero_file)
        probs_win = torch.sigmoid(logits.float()).cpu().numpy()
        for k, start in enumerate(chunk):
            length = min(model.win, len(old) - start)
            probs[start : start + length] += probs_win[k, :length]
            counts[start : start + length] += 1
    probs = probs / np.maximum(counts, 1)
    if smooth > 1:
        kernel = np.ones(smooth, dtype=np.float32) / smooth
        probs = np.convolve(probs, kernel, mode="same")
    return _mask_to_spans(probs >= threshold, min_len, gap)


def span_prf(pred: list[Span], target: list[Span], iou: float = 0.5) -> dict:
    used: set[int] = set()
    tp = 0
    for p in pred:
        best = -1
        best_iou = 0.0
        for j, t in enumerate(target):
            if j in used:
                continue
            inter = max(
                0,
                min(p.old_start + p.old_len, t.old_start + t.old_len)
                - max(p.old_start, t.old_start),
            )
            union = p.old_len + t.old_len - inter
            score = inter / union if union else 0.0
            if score >= iou and score > best_iou:
                best, best_iou = j, score
        if best >= 0:
            used.add(best)
            tp += 1
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(target) if target else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall}


def mask_prf(pred: np.ndarray, target: np.ndarray) -> dict:
    p = pred.astype(bool)
    t = target.astype(bool)
    tp = int((p & t).sum())
    precision = tp / int(p.sum()) if p.any() else 0.0
    recall = tp / int(t.sum()) if t.any() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall}


def evaluate_locator(
    model: Locator,
    examples: list[Labeled],
    device: torch.device,
    threshold: float = 0.5,
    zero_file: bool = False,
    smooth: int = 1,
    gap: int = 0,
) -> dict:
    rows = []
    for ex in examples:
        pred = predict_spans(
            model,
            ex.old,
            ex.record["message"],
            device,
            threshold=threshold,
            zero_file=zero_file,
            smooth=smooth,
            gap=gap,
        )
        span = span_prf(pred, ex.spans)
        pmask = _mask_from_spans(ex.old, pred)
        byte = mask_prf(pmask, ex.mask)
        exact = [(s.old_start, s.old_len) for s in pred] == [
            (s.old_start, s.old_len) for s in ex.spans
        ]
        rows.append(
            {
                "id": ex.record["id"],
                "repo": ex.record["repo"],
                "path": ex.record["path"],
                "span_f1": byte["f1"],
                "span_precision": byte["precision"],
                "span_recall": byte["recall"],
                "match_f1": span["f1"],
                "location_exact": exact,
                "pred_spans": len(pred),
                "target_spans": len(ex.spans),
            }
        )
    n = len(rows)

    def mean(key: str) -> float:
        return sum(r[key] for r in rows) / n if n else 0.0

    return {
        "n": n,
        "span_f1": mean("span_f1"),
        "span_precision": mean("span_precision"),
        "span_recall": mean("span_recall"),
        "match_f1": mean("match_f1"),
        "location_exact": mean("location_exact"),
        "items": rows,
    }


def render_locator_report(result: dict) -> str:
    cfg = result["config"]
    loc = result["locator"]
    base = result["baseline"]
    delta = loc["span_f1"] - base["span_f1"]
    lines = [
        "# Exact Edit Bench: E1 locator (GE1)",
        "",
        f"- arm **E1 locator**; split `{result['split']}`; n = {loc['n']}",
        f"- seed `{cfg['seed']}`; steps `{cfg['steps']}`; "
        f"pretrain `{cfg['pretrain'] or 'none'}`",
        f"- manifest sha256 `{result['manifest']['sha256'][:16]}`",
        f"- model `d{cfg['d']} l{cfg['layers']} h{cfg['heads']}`; "
        f"{cfg['params']:,} params; window {cfg['win']} B",
        "",
        "## Task",
        "",
        "Given `old` and the commit message, predict the changed byte spans",
        "in `old`. The gate metric is byte-overlap F1 over changed bytes,",
        "the natural segmentation score. Span-match F1 (IoU ≥ 0.5) is a",
        "stricter secondary label. Insertions (`old_len == 0`) are excluded",
        "because a byte classifier cannot localize an empty span; the writer",
        "and Track C handle them.",
        "",
        "## Gate",
        "",
        "| test | bar | locator | message-only | delta | result |",
        "|---|---:|---:|---:|---:|---|",
        f"| byte F1 | +20 points | {loc['span_f1']:.3f} | "
        f"{base['span_f1']:.3f} | {delta:+.3f} | "
        f"{'pass' if delta >= 0.20 else 'fail'} |",
        "",
        "## Results",
        "",
        "| arm | byte F1 | byte P | byte R | match F1 | location exact |",
        "|---|---:|---:|---:|---:|---:|",
        f"| E1 locator | {loc['span_f1']:.3f} | {loc['span_precision']:.3f} | "
        f"{loc['span_recall']:.3f} | {loc['match_f1']:.3f} | "
        f"{loc['location_exact']:.3f} |",
        f"| message-only | {base['span_f1']:.3f} | {base['span_precision']:.3f} | "
        f"{base['span_recall']:.3f} | {base['match_f1']:.3f} | "
        f"{base['location_exact']:.3f} |",
        "",
        "## Data",
        "",
        "| split | repos | files |",
        "|---|---|---:|",
    ]
    for name in ("train", "val", "test"):
        lines.append(
            f"| {name} | {', '.join(result['manifest']['splits'][name])} | "
            f"{result['manifest']['counts'][name]} |"
        )
    lines += [
        "",
        "The message-only baseline is the same model and schedule with the",
        "file embedding zeroed; it sees the message and positions but not",
        "the old bytes. The delta isolates the value of file content.",
        "",
        f"Usable files with a non-empty changed span: train {cfg['train_files']}, "
        f"val/test {cfg['eval_files']}.",
        "",
        "## Reading",
        "",
    ]
    if delta >= 0.20:
        lines += [
            "GE1 passes: file content adds more than 20 byte-F1 points over",
            "the message-only baseline. E2 (locator + writer) is next; do",
            "not touch test until the Track E arm is final.",
        ]
    else:
        lines += [
            "GE1 fails: file content does not add 20 byte-F1 points. The",
            "next lever is the locator input, not a longer schedule: try",
            "message-to-file cross-attention, line-level labels, or a",
            "retrieval-located prior. Record the failure; do not widen the",
            "benchmark.",
        ]
    lines += ["", "## Next", ""]
    if delta >= 0.20:
        lines += [
            "E2 compiler: reuse this locator, add the writer on the",
            "predicted spans, and verify exact assembly.",
        ]
    else:
        lines += [
            "E1b locator: change the input, not the budget. Try line-level",
            "labels or explicit message-to-file cross-attention; rerun GE1",
            "before E2. C1 copy-SFT is independent and can run in parallel.",
        ]
    lines += [
        "Val is `pytest-dev/pytest`; test is `tiangolo/fastapi`. Do not",
        "tune on test.",
    ]
    return "\n".join(lines) + "\n"


def _param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def run_locator(
    manifest: dict,
    manifest_path: Path,
    cache: Path,
    split: str,
    limit: int,
    seed: int,
    steps: int,
    pretrain: str,
    report: Path,
    out: Path,
) -> int:
    if pretrain not in ("", "synthetic"):
        raise SystemExit("editlm: locator pretrain must be synthetic or empty")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = load_labeled(manifest, cache, "train")
    queries = load_labeled(manifest, cache, split, limit)
    if not train or not queries:
        raise SystemExit("editlm: locator needs a non-empty train and eval split")
    steps = steps or 1000
    threshold, smooth, gap = 0.5, 8, 8
    model, log = train_locator(train, steps, seed, pretrain, False, device)
    baseline, log0 = train_locator(train, steps, seed, pretrain, True, device)
    loc = evaluate_locator(
        model, queries, device, threshold=threshold, smooth=smooth, gap=gap
    )
    base = evaluate_locator(
        baseline,
        queries,
        device,
        threshold=threshold,
        zero_file=True,
        smooth=smooth,
        gap=gap,
    )
    config = {
        "seed": seed,
        "steps": steps,
        "pretrain": pretrain,
        "split": split,
        "limit": limit,
        "win": LOCATOR_WIN,
        "msg": LOCATOR_MSG,
        "d": LOCATOR_D,
        "layers": LOCATOR_LAYERS,
        "heads": LOCATOR_HEADS,
        "params": _param_count(model),
        "threshold": threshold,
        "smooth": smooth,
        "gap": gap,
        "min_span": 2,
        "train_files": len(train),
        "eval_files": len(queries),
    }
    result = {
        "arm": "locator",
        "split": split,
        "manifest": {
            "path": str(manifest_path),
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "config": manifest["config"],
            "heads": manifest["heads"],
            "counts": manifest["counts"],
            "splits": manifest["splits"],
        },
        "config": config,
        "locator": loc,
        "baseline": base,
        "log": log,
        "baseline_log": log0,
    }
    _write_json(out, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_locator_report(result))
    ckpt = RUNS / "edit-locator"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config}, ckpt / "model.pt")
    torch.save(
        {"model": baseline.state_dict(), "config": config},
        ckpt / "message-only.pt",
    )
    delta = loc["span_f1"] - base["span_f1"]
    print(
        f"editlm: E1 locator n={loc['n']} spanF1={loc['span_f1']:.3f} "
        f"baseline={base['span_f1']:.3f} delta={delta:+.3f} "
        f"ge1={'pass' if delta >= 0.20 else 'fail'}"
    )
    print(f"editlm: {report}")
    return 0


class CausalBlock(nn.Module):
    """A pre-LN causal block for the copy-action LM."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d)
        self.out = nn.Linear(4 * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        shape = (b, t, self.heads, d // self.heads)
        q, k, v = (z.view(shape).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(b, t, d)
        x = x + self.proj(y)
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class CopyLM(nn.Module):
    """Causal byte LM over `message + old + action target`."""

    def __init__(
        self,
        d: int = LOCATOR_D,
        layers: int = LOCATOR_LAYERS,
        heads: int = LOCATOR_HEADS,
        seq: int = LOCATOR_MSG + 1 + LOCATOR_WIN + COPY_ACT,
    ):
        super().__init__()
        self.tok = nn.Embedding(256, d)
        self.pos = nn.Embedding(seq, d)
        self.blocks = nn.ModuleList(CausalBlock(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 256, bias=False)
        self.head.weight = self.tok.weight
        self.seq = seq
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tok(x) + self.pos.weight[: x.size(1)]
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln(h))


def copy_example(rec: dict, old: bytes, new: bytes) -> tuple[bytes, bytes] | None:
    """One window: message + old as the prompt, serialized actions as target."""
    if len(old) > LOCATOR_WIN:
        return None
    actions = copy_actions(new, old, min_match=8, cap=256)
    target = dump_actions(actions)
    if len(target) > COPY_ACT:
        return None
    prompt = rec["message"].encode("utf-8")[:LOCATOR_MSG] + b"\x00" + old
    return prompt, target


def load_copy(
    manifest: dict, cache: Path, split: str, limit: int = 0
) -> list[tuple[dict, bytes, bytes, bytes, bytes]]:
    out = []
    for rec, old, new in editdata.load_edits(manifest, cache, split):
        pair = copy_example(rec, old, new)
        if pair is None:
            continue
        out.append((rec, old, new, *pair))
    if limit and len(out) > limit:
        step = max(1, len(out) // limit)
        out = out[::step][:limit]
    return out


def _copy_batch(examples: list, rng: random.Random, device: torch.device, batch: int):
    seq = torch.zeros(batch, COPY_SEQ, dtype=torch.long)
    labels = torch.zeros(batch, COPY_SEQ, dtype=torch.long)
    mask = torch.zeros(batch, COPY_SEQ, dtype=torch.bool)
    for k in range(batch):
        _, _, _, prompt, target = examples[rng.randrange(len(examples))]
        if len(prompt) + len(target) >= COPY_SEQ:
            continue
        p = len(prompt)
        t = len(target)
        seq[k, : p + t - 1] = torch.cat(
            [
                torch.frombuffer(bytearray(prompt), dtype=torch.uint8),
                torch.frombuffer(bytearray(target[:-1]), dtype=torch.uint8),
            ]
        ).long()
        labels[k, p - 1 : p - 1 + t] = torch.frombuffer(
            bytearray(target), dtype=torch.uint8
        ).long()
        mask[k, p - 1 : p - 1 + t] = True
    return (
        seq.to(device),
        labels.to(device),
        mask.to(device),
    )


def train_copy(
    examples: list,
    steps: int,
    seed: int,
    device: torch.device,
    batch: int = 8,
) -> tuple[CopyLM, list[dict]]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = CopyLM().to(device)
    opt = torch.optim.AdamW(model.parameters(), 3e-4)
    warm = min(100, max(1, steps // 10))

    def factor(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        progress = (step - warm) / max(1, steps - warm)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
    log = []
    model.train()
    for step in range(steps):
        seq, labels, mask = _copy_batch(examples, rng, device, batch)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=seq.is_cuda
        ):
            logits = model(seq)
        logits = logits.float()
        loss = F.cross_entropy(logits[mask], labels[mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == steps - 1:
            log.append({"step": step, "loss": round(float(loss.detach()), 4)})
    return model, log


@torch.no_grad()
def generate_actions(
    model: CopyLM, prompt: bytes, device: torch.device, max_new: int = COPY_ACT
) -> bytes:
    """Constrained decode: only bytes the action grammar admits."""
    seq = list(prompt)
    if len(seq) >= model.seq:
        seq = seq[: model.seq - 1]
    out = bytearray()
    model.eval()
    state = "tag"
    value = 0
    shift = 0
    remaining = 0
    for _ in range(max_new):
        if len(seq) >= model.seq:
            break
        x = torch.frombuffer(bytearray(seq), dtype=torch.uint8).long()
        x = x.unsqueeze(0).to(device)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=x.is_cuda
        ):
            logits = model(x)
        keep = torch.zeros(256, dtype=torch.bool, device=logits.device)
        if state == "tag":
            keep[[0, 1, 2]] = True
        else:
            keep[:] = True
        scores = logits[0, -1].float().masked_fill(~keep, float("-inf"))
        nxt = int(scores.argmax())
        out.append(nxt)
        seq.append(nxt)
        if state == "tag":
            if nxt == 0:
                break
            state = "off" if nxt == 1 else "size"
            value = 0
            shift = 0
        elif state in ("off", "len", "size"):
            value |= (nxt & 0x7F) << shift
            if nxt < 0x80:
                if state == "size":
                    remaining = value
                    state = "lit" if remaining > 0 else "tag"
                else:
                    state = "len" if state == "off" else "tag"
                value = 0
                shift = 0
            else:
                shift += 7
        else:
            remaining -= 1
            if remaining <= 0:
                state = "tag"
    return bytes(out)


def eval_copy(model: CopyLM, examples: list, device: torch.device) -> dict:
    items = []
    exact = valid = 0
    distance = 0.0
    output = 0.0
    for rec, old, new, prompt, target in examples:
        blob = generate_actions(model, prompt, device)
        parsed = False
        try:
            actions = load_actions(blob)
            pred = assemble(actions, old)
            parsed = True
        except ValueError:
            pred = old
        ok = parsed and pred == new
        valid += int(parsed)
        exact += int(ok)
        distance += line_distance(pred, new)
        output += len(blob) / max(len(new), 1)
        items.append(
            {
                "id": rec["id"],
                "repo": rec["repo"],
                "path": rec["path"],
                "exact": ok,
                "actions": len(blob),
                "target_actions": len(target),
                "output_ratio": round(len(blob) / max(len(new), 1), 4),
                "edit_distance": round(line_distance(pred, new), 4),
            }
        )
    n = len(items)
    return {
        "n": n,
        "exact": exact / n,
        "valid": valid / n,
        "edit_distance": distance / n,
        "output_ratio": output / n,
        "items": items,
    }


def render_copy_report(result: dict) -> str:
    c1 = result["copy"]
    cfg = result["config"]
    lines = [
        "# Exact Edit Bench: C1 copy-SFT",
        "",
        f"- arm **C1 copy-SFT**; split `{result['split']}`; n = {c1['n']}",
        f"- seed `{cfg['seed']}`; steps `{cfg['steps']}`; "
        f"{cfg['params']:,} params; one window ≤{cfg['win']} B",
        f"- manifest sha256 `{result['manifest']['sha256'][:16]}`",
        "",
        "## Task",
        "",
        "Emit a serialized copy/literal action program that rebuilds `new`",
        "from `old`, given `old` and the commit message. Exact match is the",
        "label; an invalid program scores zero. Only files that fit one",
        "window are in this pilot.",
        "",
        "## Results",
        "",
        "| arm | n | exact | valid | line distance | output bytes |",
        "|---|---:|---:|---:|---:|---:|",
        f"| C1 copy-SFT | {c1['n']} | {c1['exact']:.3f} | "
        f"{c1['valid']:.3f} | {c1['edit_distance']:.3f} | "
        f"{c1['output_ratio']:.3f} |",
        "",
        "## Gate GC1",
        "",
    ]
    if c1["exact"] > 0:
        lines += [
            "The full-file byte LM arm is not built on this split, so the",
            "relative GC1 comparison is blocked. GC1 needs +10 exact or +20",
            "test pass over that baseline at ≤5% output bytes. The absolute",
            "exact rate above is the pilot result; do not claim a pass.",
        ]
    else:
        lines += [
            "GC1 fails: C1 is exact on `0` files. Every baseline has exact",
            "rate ≥ 0, so a +10-point lift is impossible from here. The",
            "full-file arm is not needed to reject this arm.",
        ]
    lines += [
        "",
        "## Data",
        "",
        "| split | files |",
        "|---|---:|",
        f"| train | {cfg['train_files']} |",
        f"| eval | {cfg['eval_files']} |",
        "",
        "## Reading",
        "",
    ]
    if c1["exact"] > 0:
        lines += [
            "C1 produces at least one exact reconstruction. The next step is",
            "the full-file baseline and the GC1 comparison on the same",
            "files; then DPO only if GC1 has a recorded baseline.",
        ]
    else:
        lines += [
            "C1 is not exact on any file. The action grammar and decoder",
            "work for valid programs only if `valid` is high; that is not",
            "an edit capability. Do not start DPO from this arm. The next",
            "lever is the target (shorter action grammar) or a copy-aware",
            "decoder, not more steps.",
        ]
    return "\n".join(lines) + "\n"


def run_copy(
    manifest: dict,
    manifest_path: Path,
    cache: Path,
    split: str,
    limit: int,
    seed: int,
    steps: int,
    report: Path,
    out: Path,
) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = load_copy(manifest, cache, "train")
    queries = load_copy(manifest, cache, split, limit)
    if not train or not queries:
        raise SystemExit("editlm: copy-SFT needs non-empty train and eval splits")
    steps = steps or 1000
    model, log = train_copy(train, steps, seed, device)
    metrics = eval_copy(model, queries, device)
    config = {
        "seed": seed,
        "steps": steps,
        "split": split,
        "limit": limit,
        "win": LOCATOR_WIN,
        "msg": LOCATOR_MSG,
        "params": _param_count(model),
        "train_files": len(train),
        "eval_files": len(queries),
        "actions_max": COPY_ACT,
    }
    result = {
        "arm": "copy-sft",
        "split": split,
        "manifest": {
            "path": str(manifest_path),
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "config": manifest["config"],
            "heads": manifest["heads"],
            "counts": manifest["counts"],
            "splits": manifest["splits"],
        },
        "config": config,
        "copy": metrics,
        "log": log,
    }
    _write_json(out, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_copy_report(result))
    ckpt = RUNS / "edit-copy-sft"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config}, ckpt / "model.pt")
    print(
        f"editlm: C1 copy-SFT n={metrics['n']} exact={metrics['exact']:.3f} "
        f"valid={metrics['valid']:.3f} output={metrics['output_ratio']:.3f}"
    )
    print(f"editlm: {report}")
    return 0


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
    report: Path | None,
    out: Path | None,
    seed: int,
    steps: int,
    base: str,
    pretrain: str,
    pairs: int,
    samples: int,
    mode: str,
) -> int:
    if arm == "locator":
        if base or pairs or samples or mode:
            raise SystemExit(
                "editlm: the locator arm takes --steps, --pretrain, and --seed only"
            )
        manifest = editdata.load_manifest(manifest_path)
        cache = editdata.cache_root(cache)
        return run_locator(
            manifest,
            manifest_path,
            cache,
            split,
            limit,
            seed,
            steps,
            pretrain,
            report or REPORTS / "EDIT-LOCATOR.md",
            out or REPORTS / "edit-locator.json",
        )
    if arm == "copy":
        if mode != "sft":
            raise SystemExit("editlm: copy supports --sft in this session")
        if base or pretrain or pairs or samples:
            raise SystemExit("editlm: copy-SFT takes --steps, --limit, and --seed only")
        manifest = editdata.load_manifest(manifest_path)
        cache = editdata.cache_root(cache)
        return run_copy(
            manifest,
            manifest_path,
            cache,
            split,
            limit,
            seed,
            steps,
            report or REPORTS / "EDIT-COPY.md",
            out or REPORTS / "edit-copy.json",
        )
    if arm != "retrieval":
        raise SystemExit(
            f"editlm: arm {arm} is not built; E1 locator and the retrieval "
            f"G0 baseline come first (TODO9.md §8)"
        )
    if seed or steps or base or pretrain or pairs or samples or mode:
        raise SystemExit("editlm: the retrieval arm takes no training arguments")
    report = report or REPORTS / "EDIT-LM.md"
    out = out or REPORTS / "editlm.json"
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
