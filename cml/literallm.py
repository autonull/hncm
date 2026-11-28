"""Literal LM: condition the byte fallback on the event stream.

The stream interleaves template events with literal byte runs.  The mixed
model sees template and fill ids before each run; the ablated control sees
one generic EVENT token in their place.  Both have the same sequence
length, parameters, data, and steps, so the difference on literal bytes is
the value of the event state.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import cml
import fillcopy
import templates

LIT = 0
EVENT = 1
UNK = 2


@dataclass(slots=True)
class Vocab:
    tids: dict[tuple[bytes, ...], int]
    fills: dict[bytes, int]
    size: int
    byte_offset: int


def build_vocab(
    train: bytes,
    min_fill: int = 16,
    min_count: int = 16,
    max_n: int = 10,
    max_holes: int = 2,
    min_fixed: int = 16,
) -> tuple[Vocab, dict[tuple[bytes, ...], int]]:
    lib = fillcopy.induce(train, min_fill, min_count, max_n, max_holes, min_fixed)
    tids = {key: i + 3 for i, key in enumerate(sorted(lib))}
    fills: Counter[bytes] = Counter()
    for event in fillcopy.parse(train, lib, min_fill, max_n):
        if isinstance(event, templates.Hit):
            fills.update(event.fills)
    order = [value for value, _ in fills.most_common()]
    fids = {value: len(tids) + i for i, value in enumerate(order)}
    byte_offset = len(tids) + len(order) + 3
    return Vocab(tids, fids, byte_offset + 256, byte_offset), lib


def stream(
    data: bytes,
    vocab: Vocab,
    lib: dict[tuple[bytes, ...], int],
    ablate: bool,
    min_fill: int = 16,
    max_n: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixed ids plus a byte mask.  Ablation keeps the length, not the ids."""
    ids: list[int] = []
    is_byte: list[bool] = []
    pending: list[int] = []

    def flush() -> None:
        if not pending:
            return
        ids.append(LIT)
        is_byte.append(False)
        for value in pending:
            ids.append(vocab.byte_offset + value)
            is_byte.append(True)
        pending.clear()

    for event in fillcopy.parse(data, lib, min_fill, max_n):
        if isinstance(event, templates.Lit):
            pending.extend(event.value)
            continue
        flush()
        tid = vocab.tids[event.key]
        ids.append(EVENT if ablate else tid)
        is_byte.append(False)
        for value in event.fills:
            if ablate:
                ids.append(EVENT)
            else:
                ids.append(vocab.fills.get(value, UNK))
            is_byte.append(False)
    flush()
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(is_byte, dtype=torch.bool),
    )


def literal_bits(
    model: cml.Trunk,
    ids: torch.Tensor,
    mask: torch.Tensor,
    seq: int,
    batch: int,
) -> tuple[float, int]:
    wins = ids.unfold(0, seq + 1, seq).contiguous()
    mwins = mask.unfold(0, seq + 1, seq).contiguous()
    model.eval()
    bits = 0.0
    nbytes = 0
    with torch.no_grad():
        for x, m in zip(wins.split(batch), mwins.split(batch)):
            x = x.cuda()
            logits = model(x[:, :-1])
            target = x[:, 1:]
            keep = m[:, 1:].reshape(-1)
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
                reduction="none",
            )
            bits += float(ce[keep].double().sum()) / math.log(2)
            nbytes += int(keep.sum())
    return bits, nbytes


def run(
    train_bytes: int = 2_000_000,
    test_bytes: int = 100_000,
    steps: int = 2_000,
    batch: int = 32,
    seq: int = 256,
    d: int = 192,
    layers: int = 4,
    heads: int = 4,
    seed: int = 0,
) -> int:
    corpus = cml.code_scale_corpus(48_000_000)
    train = corpus.train[:train_bytes]
    test = corpus.test[:test_bytes]
    vocab, lib = build_vocab(train)
    tr_ids, tr_mask = stream(train, vocab, lib, False)
    te_ids, te_mask = stream(test, vocab, lib, False)
    ab_tr_ids, ab_tr_mask = stream(train, vocab, lib, True)
    ab_te_ids, ab_te_mask = stream(test, vocab, lib, True)
    print(
        f"library {len(lib):,} templates, {len(vocab.fills):,} fills; "
        f"mixed {len(tr_ids):,} tokens; test bytes {int(te_mask.sum()):,}",
        flush=True,
    )
    rows = []
    for name, ids, mask in (
        ("mixed", tr_ids, tr_mask),
        ("ablated", ab_tr_ids, ab_tr_mask),
    ):
        model = cml.Trunk(vocab.size, d, layers, heads, seq).cuda()
        wins = cml.windows(ids, seq)
        secs = cml.train_loop(model, wins, steps, batch, 3e-4)
        te = te_ids if name == "mixed" else ab_te_ids
        tm = te_mask if name == "mixed" else ab_te_mask
        bits, nbytes = literal_bits(model, te, tm, seq, batch)
        rows.append(
            {
                "arm": name,
                "literal_bpb": bits / max(nbytes, 1),
                "params": sum(p.numel() for p in model.parameters()),
                "seconds": round(secs, 1),
            }
        )
        print(f"{name}: {rows[-1]}", flush=True)
    gain = rows[1]["literal_bpb"] - rows[0]["literal_bpb"]
    text = "\n".join(
        [
            "# Literal LM: event-conditioned byte fallback",
            "",
            f"- code corpus `{corpus.sha[:12]}`, train {len(train):,} B, "
            f"held-out {len(test):,} B",
            f"- library {len(lib):,} templates; vocab {vocab.size:,}; "
            f"{int(te_mask.sum()):,} held-out literal bytes",
            f"- trunk d{d} x {layers} layers x {heads} heads, seq {seq}, "
            f"batch {batch}, {steps} steps",
            "- mixed: template and fill ids precede each literal run; "
            "ablated: one EVENT token occupies the same positions",
            "",
            "| arm | literal bpb | params | seconds |",
            "|---|---:|---:|---:|",
            *[
                f"| {r['arm']} | {r['literal_bpb']:.4f} | "
                f"{r['params']:,} | {r['seconds']} |"
                for r in rows
            ],
            "",
            "## Read",
            "",
            f"- event context changes literal bits by **{gain:+.4f} bpb** "
            f"(positive means the mixed arm is better).",
            f"- gate: mixed literal bpb at least 0.03 below ablated. "
            f"{'PASS' if gain >= 0.03 else 'MISS'}.",
            "- this is a small control model, not the byte trunk; the "
            "ablation isolates the event identity at equal length and cost.",
            "",
            f"Took {sum(r['seconds'] for r in rows):.0f} s on one GPU.",
            "",
        ]
    )
    cml.write_text(cml.REPORTS / "LITERAL-LM.md", text)
    cml.write_json(
        cml.REPORTS / "literal-lm.json",
        {"rows": rows, "gain": gain, "git": cml.git_sha()},
    )
    print(text)
    return 0
