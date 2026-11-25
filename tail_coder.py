"""M5: the tail coder. The measurement is the synergy: bits *and* calls.

The dictionary covers a span exactly; the spans it lacks pay the byte unigram,
one byte at a time (~1 call/byte). On shakespeare that tail is 56% of the val
and ~70% of its bits, so it is the bill, not a detail. The tail coder replaces
it with one call per span. The arms, all on the *same* uncovered spans:

- `uni`: the byte unigram. The current cost: the control.
- `pos`: P(byte | length, offset), with the span length at H(L).
- `ar`: P(byte | previous byte, length, offset), plus H(L).
- `prev`: P(byte | concept class of the previous dictionary token, length,
  offset), plus H(L). The slow-tier concept predicts the fast-tier bytes and
  costs no side information: the previous token is already decoded.
- `prev+ar`: both contexts, plus H(L).
- `concept`: a hard k-means code over the span's (position, byte-class) profile,
  *transmitted*: H(concept) + H(L), then P(byte | concept, length, offset).

Every arm is a decodable code: its bits are -log2 of the model's probability
along the span, and its tables are charged. Count versions are the honest
ceiling test of the *code*: if a hard concept with counts cannot pay, a neural
decoder over the same code can only smooth it, and the next step is the code,
not the decoder.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import concept_lm as c  # noqa: E402
import hncm7 as v  # noqa: E402

CAP = c.CAP
ALPHA = 0.2
CONCEPTS = 64
BINS = 4
CLASSES = 32
SEED = 0
ARMS = ("uni", "pos", "ar", "prev", "prev+ar", "concept")
DIMS = BINS * CLASSES


@dataclass(frozen=True)
class Piece:
    """One uncovered span: its bytes, the concept class before it, its code, and
    the first byte of the token that follows it (256 = end of stream)."""

    raw: bytes
    prev: int
    cid: int = 0
    post: int = 256
    ctx: bytes = b""


# %%
def pieces(toks: list[bytes], rank: dict[bytes, int]) -> list[Piece]:
    """The maximal runs of single-byte tokens (the spans outside the dictionary),
    each with the concept class of the token that precedes it, and the first byte
    of the token that follows it. A run ends at a dictionary token, which is also
    the predecessor of the next run; `post` lets the boundary be a space instead
    of a transmitted length."""
    out, cur, last, last_tok = [], bytearray(), len(rank), b""
    for s in toks:
        if len(s) == 1:
            cur += s
        else:
            if cur:
                out.append(Piece(bytes(cur), last, 0, s[0], last_tok[-3:]))
                cur = bytearray()
            last, last_tok = rank.get(s, len(rank)), s
    if cur:
        out.append(Piece(bytes(cur), last, 0, 256, last_tok[-3:]))
    return out


def tail(toks: list[bytes]) -> list[bytes]:
    """The tail spans only: the runs of single-byte tokens."""
    return [p.raw for p in pieces(toks, {})]


class Cond:
    """P(byte | key) by counts with absolute discounting: a decodable code.

    `P(b|k) = max(c - D, 0)/T + D * types(k)/T * P_backoff(b)`, where `D` is the
    Good-Turing estimate `n1/(n1+2 n2)` of the table itself. It sums to one, it
    needs no tuned knob, and it discounts rare keys instead of trusting them (an
    add-alpha table over a sparse key is overconfident: measured, it costs
    shakespeare's tail 0.15 bpb).
    """

    def __init__(self, keyfn, root: np.ndarray, back=None):
        self.keyfn = keyfn
        self.root = root
        self.back = back
        self.counts: dict[tuple, collections.Counter] = {}
        self.totals: collections.Counter = collections.Counter()
        self.discount: dict[tuple, float] = {}

    def add(self, p: Piece, j: int) -> None:
        k, b = self.keyfn(p, j), p.raw[j]
        self.counts.setdefault(k, collections.Counter())[b] += 1
        self.totals[k] += 1

    def freeze(self, min_total: int = 2, top: int | None = 4096) -> None:
        """Prune the noise and estimate one discount for the level.

        A key seen once pays its backoff (`min_total`); the table keeps the top
        `top` keys by total, so the charge is bounded. `D = n1/(n1+2 n2)` is the
        Good-Turing estimate over the kept counts, floored to 0.25 and capped at
        0.95: with `D = 0` a key whose counts are all 2 puts all its mass on the
        seen byte, which is overconfident (measured: a zero probability).
        """
        keep = [k for k, t in self.totals.items() if t >= min_total]
        if top is not None and len(keep) > top:
            keep = sorted(keep, key=lambda k: -self.totals[k])[:top]
        keep = set(keep)
        self.counts = {k: cc for k, cc in self.counts.items() if k in keep}
        self.totals = collections.Counter(
            {k: t for k, t in self.totals.items() if k in keep}
        )
        n1 = sum(1 for cc in self.counts.values() for x in cc.values() if x == 1)
        n2 = sum(1 for cc in self.counts.values() for x in cc.values() if x == 2)
        self.discount = {
            "D": min(max(n1 / (n1 + 2 * n2), 0.25), 0.95) if n1 + 2 * n2 else 0.5
        }
        if self.back:
            self.back.freeze(min_total, top)

    def prob(self, p: Piece, j: int) -> float:
        k, b = self.keyfn(p, j), p.raw[j]
        base = self.back.prob(p, j) if self.back else float(self.root[b])
        cc = self.counts.get(k)
        if not cc:
            return base
        t = self.totals[k]
        d = self.discount["D"]
        return max(cc.get(b, 0) - d, 0.0) / t + d * len(cc) / t * base

    def bits(self, p: Piece) -> float:
        return -sum(math.log2(self.prob(p, j)) for j in range(len(p.raw)))

    def size(self) -> int:
        """2 bytes per count and 4 per key; the root is the dictionary's table."""
        n = sum(len(cc) for cc in self.counts.values())
        return 2 * n + 4 * len(self.counts) + (self.back.size() if self.back else 0)


def class_of(bc: np.ndarray) -> np.ndarray:
    """The byte classes by frequency: 32 buckets, so the profile stays small."""
    order = np.argsort(bc)
    cls = np.empty(256, dtype=np.int64)
    cls[order] = np.arange(256) // (256 // CLASSES)
    return cls


def profile(raw: bytes, cls: np.ndarray) -> np.ndarray:
    out = np.zeros(DIMS)
    L = len(raw)
    for j, b in enumerate(raw):
        out[min(j * BINS // L, BINS - 1) * CLASSES + cls[b]] += 1 / L
    return out


def kmeans(x: np.ndarray, k: int, rng: np.random.RandomState, iters: int = 20):
    cen = x[rng.choice(len(x), min(k, len(x)), replace=False)].copy()
    lab = np.zeros(len(x), dtype=np.int64)
    for _ in range(iters):
        d = (x**2).sum(1)[:, None] - 2 * x @ cen.T + (cen**2).sum(1)[None]
        lab = d.argmin(1)
        for i in range(len(cen)):
            m = lab == i
            if m.any():
                cen[i] = x[m].mean(0)
    return cen, lab


def nearest(x: np.ndarray, cen: np.ndarray) -> np.ndarray:
    d = (x**2).sum(1)[:, None] - 2 * x @ cen.T + (cen**2).sum(1)[None]
    return d.argmin(1)


# %%
def build(name: str, rect: dict) -> dict:
    """Rebuild the measured point, then split the val into covered and tail."""
    s = c.split(name, rect["vocab"])
    d, bg, uni = c.rebuild(s, rect["dict"], rect["table"])
    toks_fit = c.viterbi(s["fit"], d, bg, uni)[0]
    toks_val, total = c.viterbi(s["va"], d, bg, uni)
    rank = {tok: i for i, tok in enumerate(s["top"][:64])}
    pf, pv = pieces(toks_fit, rank), pieces(toks_val, rank)
    cls = class_of(s["bc"])
    rng = np.random.RandomState(SEED)
    xf = np.array([profile(p.raw, cls) for p in pf])
    cen = kmeans(xf, CONCEPTS, rng)[0] if len(xf) >= CONCEPTS else xf
    cid_fit = nearest(xf, cen) if len(xf) else np.zeros(0, dtype=np.int64)
    train = [Piece(p.raw, p.prev, int(i)) for p, i in zip(pf, cid_fit)]
    return {
        "s": s,
        "d": d,
        "bg": bg,
        "uni": uni,
        "total": total,
        "cls": cls,
        "cen": cen,
        "train": train,
        "val_raw": pv,
    }


def fit_arms(
    train: list[Piece], root: np.ndarray, top: int | None = 4096
) -> dict[str, Cond]:
    pos = Cond(lambda p, j: (len(p.raw), j), root)
    ar = Cond(lambda p, j: (p.raw[j - 1] if j else -1, len(p.raw), j), root, pos)
    prev = Cond(lambda p, j: (p.prev, len(p.raw), j), root, pos)
    par = Cond(
        lambda p, j: (p.prev, p.raw[j - 1] if j else -1, len(p.raw), j), root, ar
    )
    con = Cond(lambda p, j: (p.cid, len(p.raw), j), root, pos)
    for p in train:
        for arm in (pos, ar, prev, par, con):
            for j in range(len(p.raw)):
                arm.add(p, j)
    for arm in (pos, ar, prev, par, con):
        arm.freeze(2, top)
    return {"pos": pos, "ar": ar, "prev": prev, "prev+ar": par, "concept": con}


def prior_bits(counts: collections.Counter, key: int, n: int, k: int) -> float:
    """-log2 P(key) with add-alpha over a known alphabet of size k."""
    return -math.log2((counts.get(key, 0) + ALPHA) / (n + ALPHA * k))


# %%
def byte_matrix(train: list[Piece], root: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The full byte-context table of the train tail.

    Rows are the previous symbol (256 = start), columns the next byte. The counts
    are the ceiling of a byte bigram; the log probabilities are the reference
    model a shared-parameter one must approximate.
    """
    counts = np.zeros((257, 256))
    for p in train:
        prev = 256
        for b in p.raw:
            counts[prev, b] += 1
            prev = b
    rows = counts.sum(1)
    prob = (counts + ALPHA * root[None, :]) / (rows[:, None] + ALPHA)
    return counts, np.log2(prob)


def count_model(counts: np.ndarray, root: np.ndarray, r: int) -> tuple[np.ndarray, int]:
    """The rank-r factorization of the *count* table: shared parameters.

    The low-rank co-occurrence structure is kept and the rows are renormalized
    with the byte-unigram backoff. It generalizes to unseen pairs by
    construction (the counts cannot), and its size is `(257 + 256) * r * 2` bytes
    in float16: r = 16 is 16.4 kB for the whole alphabet. Factorizing the *log*
    probabilities instead is worse: it is least squares in the wrong domain
    (measured, rank 2 < rank 1).
    """
    u, sv, vt = np.linalg.svd(counts, full_matrices=False)
    c = (u[:, :r] * sv[:r]) @ vt[:r]
    c = np.clip(c, 0.0, None)
    rows = c.sum(1)
    prob = (c + ALPHA * root[None, :]) / (rows[:, None] + ALPHA)
    return np.log2(prob), (257 + 256) * r * 2


def svd_bits(p: Piece, logp: np.ndarray) -> tuple[float, float]:
    """The span's bits under the rank-r model, and the cost of the next byte."""
    bits, prev = 0.0, 256
    for b in p.raw:
        bits -= logp[prev, b]
        prev = b
    return bits, -logp[prev, p.post] if p.post < 256 else 0.0


def measure(name: str, rect: dict) -> dict:
    b = build(name, rect)
    s, d, bc = b["s"], b["d"], b["s"]["bc"]
    va, stream = s["va"], s["stream"]
    covered = b["total"] - sum(float(bc[x]) for p in b["val_raw"] for x in p.raw)
    tail_bytes = sum(len(p.raw) for p in b["val_raw"])
    root = np.exp2(-bc)
    root /= root.sum()
    arms = fit_arms(b["train"], root)
    lens = collections.Counter(len(p.raw) for p in b["train"])
    cids = collections.Counter(p.cid for p in b["train"])
    n_train = len(b["train"])
    table = 6 * sum(len(r) for r in b["bg"].values())

    if b["val_raw"]:
        xv = np.array([profile(p.raw, b["cls"]) for p in b["val_raw"]])
        cid = nearest(xv, b["cen"])
        val = [Piece(p.raw, p.prev, int(i), p.post) for p, i in zip(b["val_raw"], cid)]
    else:
        val = []
    lens_cond: dict[int, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    for p in b["train"]:
        lens_cond[p.prev][len(p.raw)] += 1
    rows = []

    def row(arm, bits, size, calls, lb=0.0):
        total = covered + bits
        rows.append(
            {
                "arm": arm,
                "len_bpb": lb / max(tail_bytes, 1),
                "byte_bpb": (bits - lb) / max(tail_bytes, 1),
                "tail_bpb": bits / max(tail_bytes, 1),
                "rate": total / len(va),
                "charged": total / len(va)
                + 8 * (d.model_bytes + table + size) / stream,
                "calls": calls,
                "model_bytes": d.model_bytes + table + size,
            }
        )

    def cond_len(p: Piece) -> float:
        """H(L | the class before the span): the slow tier shapes the span."""
        cc = lens_cond.get(p.prev)
        tot = sum(cc.values()) if cc else 0
        if cc and tot >= 8:
            return -math.log2((cc.get(len(p.raw), 0) + ALPHA) / (tot + ALPHA * CAP))
        return prior_bits(lens, len(p.raw), n_train, CAP)

    def size_of(arm: str) -> int:
        if arm == "uni":
            return 0
        return 2 * (CAP + 1) + arms[arm].size()

    for arm in ARMS:
        bits = lb = 0.0
        for p in val:
            if arm == "uni":
                bits += sum(float(bc[x]) for x in p.raw)
            else:
                one = prior_bits(lens, len(p.raw), n_train, CAP)
                lb += one
                bits += one
                if arm == "concept":
                    bits += prior_bits(cids, p.cid, n_train, len(b["cen"]))
                    bits += arms["concept"].bits(p)
                else:
                    bits += arms[arm].bits(p)
        size = size_of(arm)
        if arm == "concept":
            size += 2 * DIMS * len(b["cen"])
        row(arm, bits, size, 1 + (tail_bytes if arm == "uni" else len(val)), lb)

    if tail_bytes > 1000:
        counts, _ = byte_matrix(b["train"], root)
        rank = {tok: i for i, tok in enumerate(s["top"][:64])}
        sel = pieces(c.viterbi(s["sel"], d, b["bg"], b["uni"])[0], rank)
        chosen = None
        for r in (1, 2, 4, 8, 16, 32, 64):
            q, size = count_model(counts, root, r)
            bits = lb = 0.0
            follows = []
            for piece in val:
                one, follow = svd_bits(piece, q)
                h = prior_bits(lens, len(piece.raw), n_train, CAP)
                lb += h
                bits += one + h
                if piece.post == 32:
                    follows.append(follow)
            sb = 0.0
            for piece in sel:
                one, _ = svd_bits(piece, q)
                sb += one + prior_bits(lens, len(piece.raw), n_train, CAP)
            # the budget is chosen on the held-out 10% of the train, not the val
            sel_charged = (
                sb / len(s["sel"])
                + 8 * (d.model_bytes + table + size + 2 * (CAP + 1)) / stream
            )
            row(f"svd@{r}", bits, size + 2 * (CAP + 1), 1 + len(val), lb)
            rows[-1]["sel_charged"] = sel_charged
            if chosen is None or sel_charged < rows[chosen]["sel_charged"]:
                chosen = len(rows) - 1
        rows[chosen]["chosen"] = True
        pick = rows[chosen]
        # the same model with the boundary coded as the trailing space, not H(L)
        q, size = count_model(counts, root, int(pick["arm"][4:]))
        follows = []
        for piece in val:
            _, follow = svd_bits(piece, q)
            if piece.post == 32:
                follows.append(follow)
        assert any(piece.post == 32 for piece in val), "no trailing space"
        bits, k = 0.0, 0
        for piece in val:
            one, _ = svd_bits(piece, q)
            if piece.post == 32:
                bits += one + follows[k]
                k += 1
            else:
                bits += one + prior_bits(lens, len(piece.raw), n_train, CAP)
        assert k == len(follows)
        row("svd@space", bits, size + 2 * (CAP + 1), 1 + len(val), 0.0)

    if tail_bytes > 1000:
        for top in (256, 1024, 4096, None):
            cand = fit_arms(b["train"], root, top)
            arm = cand["ar"]
            bits = lb = 0.0
            for q in val:
                one = prior_bits(lens, len(q.raw), n_train, CAP)
                lb += one
                bits += one + arm.bits(q)
            row(
                f"ar@{top if top else 'full'}",
                bits,
                2 * (CAP + 1) + arm.size(),
                1 + len(val),
                lb,
            )
        # the ceiling: the chosen shared model, with the boundary removed
        q, size = count_model(counts, root, int(pick["arm"][4:]))
        bits = sum(svd_bits(piece, q)[0] for piece in val)
        row("ceiling", bits, size + 2 * (CAP + 1), 1 + len(val), 0.0)

    return {
        "stream_bytes": stream,
        "val_bytes": len(va),
        "dict": rect["dict"],
        "table": rect["table"],
        "model_bytes": d.model_bytes + table,
        "covered_bpb": covered / max(len(va) - tail_bytes, 1),
        "tail_bytes": tail_bytes,
        "tail_spans": len(val),
        "bytes_per_span": tail_bytes / max(len(val), 1),
        "arms": rows,
        "best": next(
            (a["arm"] for a in rows if a.get("chosen") and a["arm"] != "svd@space"),
            "uni",
        ),
    }


def gates(res: dict) -> list[dict]:
    """G4 the tail beats the byte unigram; G5 the calls dividend; G6 the total
    beats the current codec; G9 the boundary's cost; the budget is the one the
    held-out 10% of the train chose (`e["best"]`), never the val's optimum."""
    out = []
    for name, e in res["corpora"].items():
        by = {a["arm"]: a for a in e["arms"]}
        uni = by["uni"]
        best = by[e["best"]]
        space = by.get("svd@space")
        out += [
            {
                "id": "G4",
                "corpus": name,
                "value": f"uni={uni['tail_bpb']:.3f} best={best['arm']} "
                f"{best['tail_bpb']:.3f} gain={uni['tail_bpb'] - best['tail_bpb']:+.3f}",
                "status": "pass" if best["tail_bpb"] < uni["tail_bpb"] else "fail",
            },
            {
                "id": "G5",
                "corpus": name,
                "value": f"calls {uni['calls']:.0f} -> {best['calls']:.0f} "
                f"(x{uni['calls'] / max(best['calls'], 1):.1f}, "
                f"{e['bytes_per_span']:.1f} B/span)",
                "status": "pass" if uni["calls"] >= 8 * best["calls"] else "fail",
            },
            {
                "id": "G6",
                "corpus": name,
                "value": f"best={best['charged']:.3f} current={uni['charged']:.3f} "
                f"gain={uni['charged'] - best['charged']:+.3f}",
                "status": "pass" if best["charged"] < uni["charged"] else "fail",
            },
            {
                "id": "G10",
                "corpus": name,
                "value": f"space boundary {space['tail_bpb']:.3f} against "
                f"len boundary {best['tail_bpb']:.3f}"
                if space
                else "no space boundary",
                "status": "pass"
                if space and space["tail_bpb"] < best["tail_bpb"]
                else "fail",
            },
            {
                "id": "G9",
                "corpus": name,
                "value": f"ceiling rate={by['ceiling']['rate']:.3f} "
                f"best rate={best['rate']:.3f} uni rate={uni['rate']:.3f}"
                if "ceiling" in by
                else "no tail: the tail coder is a no-op",
                "status": (
                    "pass"
                    if "ceiling" in by and by["ceiling"]["rate"] < best["rate"] - 1e-9
                    else "fail"
                    if "ceiling" in by
                    else "skip"
                ),
            },
        ]
    return out


def report(res: dict, path: Path) -> None:
    lines = [
        "# H-NCM v7: the tail coder (M5), bits and calls",
        "",
        "The dictionary covers a span exactly; the uncovered spans pay the byte "
        "unigram, one byte at a time. The tail coder codes one call per span. The "
        "count arms use absolute discounting (Good-Turing, floored) and keep their "
        "top 4,096 keys; the `svd@r` arms are the rank-r factorization of the "
        "257x256 byte-context table (shared parameters, 8.2 kB per rank step of "
        "16), and r is chosen on the held-out 10% of the train, never the val. The "
        "cost is the span bytes plus the boundary H(L); `svd@space` codes the "
        "trailing space instead, but the covered token that follows still pays its "
        "index, so that row is the tail side of a space-suffixed tokenizer (a "
        "bound). The `ceiling` arm removes the boundary: the rate a boundary-free "
        "span code would reach.",
        "",
        "| corpus | tail B | spans | B/span | covered bpb | tail bpb |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, e in res["corpora"].items():
        uni = next(a for a in e["arms"] if a["arm"] == "uni")
        lines.append(
            f"| {name} | {e['tail_bytes']} | {e['tail_spans']} | "
            f"{e['bytes_per_span']:.2f} | {e['covered_bpb']:.3f} | "
            f"{uni['tail_bpb']:.3f} |"
        )
    for name, e in res["corpora"].items():
        lines += [
            "",
            f"## {name}",
            "",
            "| arm | len bpb | byte bpb | tail bpb | rate | charged | model kB | calls |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for a in e["arms"]:
            lines.append(
                f"| {a['arm']}{' *' if a.get('chosen') else ''} | "
                f"{a['len_bpb']:.3f} | {a['byte_bpb']:.3f} | "
                f"{a['tail_bpb']:.3f} | {a['rate']:.3f} | {a['charged']:.3f} | "
                f"{a['model_bytes'] / 1e3:.1f} | {a['calls']:.0f} |"
            )
    lines += [
        "",
        "## Gates",
        "",
        "| gate | corpus | value | status |",
        "|---|---|---|---|",
    ]
    for g in gates(res):
        lines.append(
            f"| {g['id']} | {g['corpus']} | {g['value']} | **{g['status']}** |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run(names: list[str], out: Path, rect_path: Path) -> dict:
    rects = json.loads(rect_path.read_text())["corpora"]
    res = {"format": 7, "rect": str(rect_path), "corpora": {}}
    for name in names:
        print(f"{name}:", flush=True)
        e = measure(name, rects[name]["oracle"])
        res["corpora"][name] = e
        for a in e["arms"]:
            print(
                f"  {a['arm']:9s} tail={a['tail_bpb']:.3f} rate={a['rate']:.3f} "
                f"charged={a['charged']:.3f} calls={a['calls']:.0f}",
                flush=True,
            )
    for g in gates(res):
        print(
            f"  {g['id']:3s} {g['status']:4s} {g['corpus']}: {g['value']}", flush=True
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    report(res, out.parent / "TAIL.md")
    return res


# %%
def selftest() -> None:
    bs = b"the cat sat on mats"
    bc = v.byte_costs(bs)
    root = np.exp2(-bc)
    root /= root.sum()
    d = v.build_dict(v.counts_substr(bs), len(bs), bc)
    big, uni = c.bigrams(v.cover(bs, d, 0.0))
    toks, total = c.viterbi(bs, d, big, uni)
    assert b"".join(toks) == bs
    spans = tail(toks)
    assert b"".join(spans) == b"".join(s for s in toks if len(s) == 1)
    assert (
        abs(
            sum(float(bc[x]) for s in spans for x in s)
            - sum(d.cost_of(s) for s in toks if len(s) == 1)
        )
        < 1e-9
    )

    rank = {tok: i for i, tok in enumerate(sorted(d.ids, key=lambda s: -len(s))[:4])}
    ps = pieces(toks, rank)
    assert len(ps) == len(spans) and b"".join(p.raw for p in ps) == b"".join(spans)
    posts, cur = [], bytearray()
    for tok in toks:
        if len(tok) == 1:
            cur += tok
        else:
            if cur:
                posts.append(tok[0])
                cur = bytearray()
    if cur:
        posts.append(256)
    assert [p.post for p in ps] == posts

    arms = fit_arms(ps, root)
    for arm in arms.values():
        for p in ps:
            for j in range(len(p.raw)):
                assert 0 < arm.prob(p, j) <= 1 + 1e-12
    # P(byte | seen key) sums to 1 over the alphabet: the code is complete.
    p = ps[0]
    k = arms["pos"].keyfn(p, 0)
    if arms["pos"].totals[k]:
        s = sum(arms["pos"].prob(Piece(bytes([b]), p.prev), 0) for b in range(256))
        assert abs(s - 1) < 1e-9, s
    assert arms["prev"].size() >= arms["pos"].size()

    # k-means is deterministic and assigns each point its nearest centroid.
    rng = np.random.RandomState(SEED)
    x = np.array([profile(p.raw, class_of(bc)) for p in ps])
    cen, lab = kmeans(x, 2, rng)
    rng2 = np.random.RandomState(SEED)
    cen2, lab2 = kmeans(x, 2, rng2)
    assert np.array_equal(lab, lab2) and np.allclose(cen, cen2)
    assert np.array_equal(lab, nearest(x, cen))

    # a key whose counts are all 2 must not put all its mass on the seen byte
    # (the measured failure of a zero discount, D = 0).
    deg = Cond(lambda p, j: (len(p.raw), j), root)
    for raw in (b"aa", b"aa"):
        for j in range(2):
            deg.add(Piece(raw, 0), j)
    deg.freeze()
    q = Piece(b"aa", 0)
    assert 0 < deg.prob(q, 0) < 1
    assert deg.prob(Piece(b"ab", 0), 1) > 0
    assert abs(deg.prob(q, 0) + deg.prob(Piece(b"ab", 0), 1) - 1) < 0.5

    assert prior_bits(collections.Counter({1: 1}), 1, 1, 2) > 0
    print("selftest: ok")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="H-NCM v7 tail coder (M5)")
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--corpora", default="shakespeare,synseg,syngrid")
    r.add_argument("--out", default="runs/v7/tail.json")
    r.add_argument("--rect", default="runs/v7/concept.json")
    sub.add_parser("selftest")
    a = p.parse_args(argv)
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "run":
        run([x for x in a.corpora.split(",") if x], Path(a.out), Path(a.rect))
    else:
        p.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
