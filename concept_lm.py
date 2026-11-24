"""Probe: a first-order concept LM over dictionary indices, solved exactly.

v7 prices each concept by its *unigram* (no context). The concept alphabet is
predictable, and the previous concept is already decoded, so a bigram over the
indices is a decodable context model. The DP state becomes the previous concept,
which makes the tiling a first-order Viterbi: exact, and no longer myopic.

Two measured facts make this the solution path, not a guess:

- The iterated bigram is *worse* than the unigram (3.128 vs 2.905 on
  shakespeare). Its costs come from the previous token of the *unigram* tiling, so
  the DP picks tokens under a model built for a different path. The model's cost
  must condition on the actual state of the path.
- The first-order Viterbi is *better*: 2.490 vs 3.003 on shakespeare and 0.886 vs
  1.283 on synseg, with a plan-aligned first token. The count bigram is the
  ceiling; a neural bigram replaces its table (a training step) and the tail
  coder covers spans the dictionary lacks.

This probe charges the dictionary but not the bigram table. The next step is the
charged budget curve (pruned bigrams) and a neural bigram for the shared model.
"""

from __future__ import annotations

import collections
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import hncm7 as v  # noqa: E402

CAP = 16
GRID = (2048, 4096, 8192, 16384)


def bigrams(toks: list[bytes]) -> tuple[dict, collections.Counter]:
    big = collections.defaultdict(collections.Counter)
    for a, b in zip(toks, toks[1:]):
        big[a][b] += 1
    return big, collections.Counter(toks)


def viterbi(
    bs: bytes, d: v.Dict, big: dict, uni: collections.Counter, backoff: float = 0.0
) -> tuple[list[bytes], float]:
    """Exact first-order DP: the state is the concept that just ended."""
    L = len(bs)
    best: list[dict] = [dict() for _ in range(L + 1)]
    best[0][None] = (0.0, None)
    for t in range(L):
        if not best[t]:
            continue
        for p, (cst, _) in best[t].items():
            node = d.trie
            for e in range(t, min(L, t + CAP)):
                if bs[e] not in node:
                    break
                node = node[bs[e]]
                if "#" not in node:
                    continue
                s = bs[t : e + 1]
                if p is None:
                    c = d.cost_of(s)
                else:
                    row = big.get(p)
                    c = (
                        -math.log2(row[s] / uni[p])
                        if row and row.get(s, 0)
                        else d.cost_of(s) + backoff
                    )
                nc = cst + c
                cur = best[e + 1].get(s)
                if cur is None or nc < cur[0]:
                    best[e + 1][s] = (nc, p)
    if not best[L]:
        return [], float("inf")
    s, (cost, _) = min(best[L].items(), key=lambda kv: kv[1][0])
    toks, e = [], L
    while e > 0:
        toks.append(s)
        p = best[e][s][1]
        e -= len(s)
        s = p
    return toks[::-1], cost


def run(name: str, ks: tuple = GRID, backoff: float = 0.0) -> list[dict]:
    corpus = v.base.get_corpus(name)
    tr = corpus["tr"].cpu().numpy().astype(np.uint8).tobytes()
    va = corpus["va"].cpu().numpy().astype(np.uint8).tobytes()
    va = va[: len(va) // 16 * 16]
    bc = v.byte_costs(tr)
    counts = v.counts_onsets(tr, v.base.onsets(name, np.frombuffer(tr, np.uint8)))
    d, _, usage = v.em_dict(counts, tr, bc)
    top = sorted((s for s in usage if len(s) > 1), key=lambda s: -usage[s])
    rows = []
    for k in ks:
        keep = top[:k]
        n = sum(usage[s] for s in keep)
        dk = v.assemble(
            keep,
            [
                float(bc[s[0]]) if len(s) == 1 else -math.log2(usage[s] / n)
                for s in keep
            ],
            bc,
        )
        uni_toks = v.cover(tr, dk, 0.0)
        big, ucount = bigrams(uni_toks)
        uni_rate = v.bits_of(v.cover(va, dk, 0.0), dk) / len(va)
        toks, cost = viterbi(va, dk, big, ucount, backoff)
        rows.append(
            {
                "corpus": name,
                "vocab": k,
                "dict": len(dk.strings),
                "model_bpb": 8 * dk.model_bytes / (len(tr) + len(va)),
                "unigram_rate": uni_rate,
                "bigram_rate": cost / len(va),
                "tokens": len(toks),
            }
        )
        print(
            f"{name:11s} vocab={k:6d} dict={len(dk.strings):6d} "
            f"uni={uni_rate:.3f} bigram={cost / len(va):.3f}",
            flush=True,
        )
    return rows


if __name__ == "__main__":
    for name in ("shakespeare", "synseg", "syngrid"):
        run(name)
