"""N0: the bitstream. A real encoder and decoder for the v7 model.

The measured tables of v7 price a token by its *conditional* count
`c(p, s) / c(p)` and a tail byte by the byte unigram. Those are code lengths of
a model that is not normalized: the row's mass that goes to single-byte
successors is not charged, and an unseen token pays the unigram on top of the
row's mass. This file *is* the judge of that account:

- The model is packed as counts (a log-quantized uint16: 8-bit exponent, 8-bit
  mantissa), and every probability is derived from the packed counts, never
  stored. The dictionary is `1 + len + 2` bytes per multi-byte string plus the
  256-entry byte table; the concept table is `2 + 2 + 2` per entry; the tail is
  two float16 factors (`(257 + 256) * r`) plus the length prior.
- The token alphabet is split into a *type* (a multi-byte token or a tail run)
  and the content: at each token boundary the type says whether a run follows.
  The type probability is the row's mass on single-byte successors `q(p)`, so a
  covered token costs exactly its measured `c(p, s)/c(p)` plus the PPM escape
  when it is unseen. A run costs `-log2 q(p)` (the type) + the length prior +
  the shared byte model. This is one normalized distribution, not two
  codebooks, and the decoder never reruns the DP.
- The CDF is derived from the model: `round(2^24 * p)`, a cumulative array,
  and a 32-bit arithmetic coder.
- `--self` puts the model in the file (the per-file convention); `--shared`
  sends it once (the archive/K-file convention). Both are reconciled to the
  code's entropy; the v7 estimate's gap is the finding (it omits the PPM escape
  and the type mass), not a defect of the coder.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import concept_lm as c  # noqa: E402
import hncm7 as v  # noqa: E402
import tail_coder as t  # noqa: E402

CAP = 16
LEN = CAP + 1  # 1..16 and a bucket for longer runs
TOTAL = 1 << 24
ALPHA = 0.2
TOP = 1 << 32
HALF = 1 << 31
QTR = 1 << 30
MASK = TOP - 1
MODEL_MAGIC = b"HNM8"
FILE_MAGIC = b"HNC8"
HEAD = 17  # magic 4 + length 8 + try 1 + model length 4
SEED = 0
_SPLIT: dict[tuple[str, str], dict] = {}


# %%
def q16(x: float) -> int:
    """A count as uint16: 8-bit exponent, 8-bit mantissa (error < 0.4%)."""
    if x <= 0:
        return 0
    e = min(int(math.floor(math.log2(x))), 19)
    m = int(round(255 * (x / 2**e - 1)))
    m = min(max(m, 1), 255)
    if m == 255:
        e, m = min(e + 1, 19), 0
    return (e << 8) | m


def dq16(q: int) -> float:
    e, m = q >> 8, q & 255
    return 2.0**e * (1 + m / 255)


class CDF:
    """The derived cumulative weights: `w_i = round(TOTAL * p_i)`, sum == TOTAL."""

    __slots__ = ("w", "cum", "total")

    def __init__(self, p: np.ndarray):
        p = np.asarray(p, dtype=np.float64)
        s = p.sum()
        if not np.isfinite(s) or s <= 0:
            raise ValueError("CDF needs a positive finite mass")
        w = np.floor(p / s * TOTAL).astype(np.int64)
        w = np.maximum(w, 0)
        w[(p > 0) & (w == 0)] = 1
        w[int(np.argmax(w))] += TOTAL - int(w.sum())
        assert (w >= 0).all() and int(w.sum()) == TOTAL
        self.w = w
        self.cum = np.zeros(len(w) + 1, dtype=np.int64)
        np.cumsum(w, out=self.cum[1:])
        self.total = TOTAL

    def span(self, i: int) -> tuple[int, int, int]:
        return int(self.cum[i]), int(self.w[i]), TOTAL

    def find(self, target: int) -> int:
        return int(np.searchsorted(self.cum, target, side="right")) - 1


class Encoder:
    def __init__(self) -> None:
        self.low = 0
        self.high = MASK
        self.pending = 0
        self.out = bytearray()
        self.cur = 0
        self.n = 0
        self.bits = 0.0

    def _bit(self, b: int) -> None:
        self.cur = (self.cur << 1) | b
        self.n += 1
        if self.n == 8:
            self.out.append(self.cur & 0xFF)
            self.cur, self.n = 0, 0

    def _emit(self, b: int) -> None:
        self._bit(b)
        while self.pending:
            self._bit(1 - b)
            self.pending -= 1

    def symbol(self, cum: int, freq: int, total: int) -> None:
        self.bits -= math.log2(freq / total)
        r = self.high - self.low + 1
        self.high = self.low + (r * (cum + freq)) // total - 1
        self.low = self.low + (r * cum) // total
        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QTR and self.high < 3 * QTR:
                self.pending += 1
                self.low -= QTR
                self.high -= QTR
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self) -> bytes:
        self.pending += 1
        self._emit(0 if self.low < QTR else 1)
        if self.n:
            self.out.append((self.cur << (8 - self.n)) & 0xFF)
            self.cur, self.n = 0, 0
        return bytes(self.out)


class Decoder:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0
        self.bit = 0
        self.low = 0
        self.high = MASK
        self.code = 0
        for _ in range(32):
            self.code = ((self.code << 1) | self._read()) & MASK
        self.bits = 0.0

    def _read(self) -> int:
        if self.pos >= len(self.data):
            return 0
        b = (self.data[self.pos] >> (7 - self.bit)) & 1
        self.bit += 1
        if self.bit == 8:
            self.pos += 1
            self.bit = 0
        return b

    def target(self, total: int) -> int:
        return ((self.code - self.low + 1) * total - 1) // (self.high - self.low + 1)

    def symbol(self, cum: int, freq: int, total: int) -> None:
        self.bits -= math.log2(freq / total)
        r = self.high - self.low + 1
        self.high = self.low + (r * (cum + freq)) // total - 1
        self.low = self.low + (r * cum) // total
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.code -= HALF
            elif self.low >= QTR and self.high < 3 * QTR:
                self.low -= QTR
                self.high -= QTR
                self.code -= QTR
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.code = ((self.code << 1) | self._read()) & MASK


# %%
def factors(counts: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    """The rank-r factors of the count table, as the float16 the file stores."""
    u, sv, vt = np.linalg.svd(counts, full_matrices=False)
    a = (u[:, :r] * sv[:r]).astype(np.float16)
    b = vt[:r].astype(np.float16)
    return a, b


def logp_of(a: np.ndarray, b: np.ndarray, root: np.ndarray) -> np.ndarray:
    """The same reconstruction the decoder makes: clip, add the root, renormalize."""
    cc = np.clip(a.astype(np.float64) @ b.astype(np.float64), 0.0, None)
    rows = cc.sum(1)
    p = (cc + ALPHA * root[None, :]) / (rows[:, None] + ALPHA)
    return np.log2(p)


def run_bits(raw: bytes, logp: np.ndarray) -> float:
    bits, prev = 0.0, 256
    for b in raw:
        bits -= float(logp[prev, b])
        prev = b
    return bits


def runs(toks: list[bytes]) -> list[tuple[int, int]]:
    """The [start, end) indices of the maximal single-byte runs."""
    out, start = [], None
    for i, s in enumerate(toks):
        if len(s) == 1:
            if start is None:
                start = i
        elif start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(toks)))
    return out


# %%
@dataclass
class Model:
    """The packed model: the multi-byte strings, their counts, the concept
    table, the 256-entry byte table, the length counts, and the tail factors."""

    strings: list[bytes]
    counts: np.ndarray
    root: np.ndarray
    table: dict[int, dict[int, int]]
    rows: dict[int, tuple[float, float]]
    lens: np.ndarray
    r: int
    a: np.ndarray | None
    b: np.ndarray | None

    def __post_init__(self) -> None:
        self.m = len(self.strings)
        self.k = self.m + 256
        self.ids = {s: i for i, s in enumerate(self.strings)}
        for i in range(256):
            self.ids.setdefault(bytes([i]), self.m + i)
        self.u = self.counts / self.counts.sum()
        self.u_m = self.u[: self.m] / self.u[: self.m].sum()
        self.u_single = self.u[self.m :]
        self.q0 = float(self.u_single.sum())
        self.lens_p = (self.lens + ALPHA) / (self.lens.sum() + ALPHA * LEN)
        self._ucdf: CDF | None = None
        self._logp: np.ndarray | None = None
        self._tcache: dict[int, CDF] = {}
        self._ycache: dict[int, CDF] = {}
        self._lcache: CDF | None = None
        self._ucache: CDF | None = None
        self._fcache: dict[int, CDF] = {}
        if self.a is not None:
            self._logp = logp_of(self.a, self.b, self.root / self.root.sum())

    def tok(self, i: int) -> bytes:
        return self.strings[i] if i < self.m else bytes([i - self.m])

    def q_of(self, p: int) -> float:
        """P(a run follows | the previous token p), in (0, 1)."""
        if p >= 0 and p in self.rows:
            c, s = self.rows[p]
            return (s + 0.5) / (c + 1.0)
        return self.q0

    def token_cdf(self, p: int) -> CDF:
        if p in self._tcache:
            return self._tcache[p]
        row = self.table.get(p) if p >= 0 else None
        m = self.m
        w = np.zeros(m)
        if row and p in self.rows:
            c_full, s_full = self.rows[p]
            cm = max(c_full - s_full, 1e-9)
            seen = {s: x for s, x in row.items() if s < m}
            for s, x in seen.items():
                w[s] = x / cm
            unseen = w == 0
            left = max(cm - sum(seen.values()), 0.0) / cm
            mass = min(float(self.u_m[list(seen)].sum()), 1 - 1e-9) if seen else 0.0
            if left > 0 and unseen.any():
                w[unseen] = left * self.u_m[unseen] / (1 - mass)
            elif unseen.any():
                w[unseen] = self.u_m[unseen] / (1 - mass)
        else:
            w = self.u_m.copy()
        cdf = CDF(w)
        self._tcache[p] = cdf
        return cdf

    def full_cdf(self, p: int) -> CDF:
        """The concept arm: the token CDF over the whole vocabulary, single bytes
        included, so every token is coded and the segmentation is implicit."""
        if p in self._fcache:
            return self._fcache[p]
        row = self.table.get(p) if p >= 0 else None
        if row and p in self.rows:
            c_full, _ = self.rows[p]
            w = np.zeros(self.k)
            for s, x in row.items():
                w[s] = x / c_full
            unseen = w == 0
            left = max(c_full - sum(row.values()), 0.0) / c_full
            mass = min(float(self.u[list(row)].sum()), 1 - 1e-9)
            if left > 0 and unseen.any():
                w[unseen] = left * self.u[unseen] / (1 - mass)
            elif unseen.any():
                w[unseen] = self.u[unseen] / (1 - mass)
        else:
            w = self.u.copy()
        cdf = CDF(w)
        self._fcache[p] = cdf
        return cdf

    def type_cdf(self, p: int) -> CDF:
        q = min(max(self.q_of(p), 1e-9), 1 - 1e-9)
        return CDF(np.array([1 - q, q]))

    def len_cdf(self) -> CDF:
        if self._lcache is None:
            self._lcache = CDF(self.lens_p)
        return self._lcache

    def byte_cdf(self) -> CDF:
        if self._ucdf is None:
            self._ucdf = CDF(np.ones(256))
        return self._ucdf

    def tail_cdf(self, prev: int) -> CDF:
        if prev in self._ycache:
            return self._ycache[prev]
        if self._logp is not None:
            cdf = CDF(np.exp2(self._logp[prev]))
        else:
            if self._ucache is None:
                self._ucache = CDF(self.root / self.root.sum())
            cdf = self._ucache
        self._ycache[prev] = cdf
        return cdf


def pack(m: Model) -> bytes:
    """The model as bytes: counts, not costs; the CDF is derived from it."""
    out = bytearray(MODEL_MAGIC)
    out += struct.pack("<IB", m.m, m.r if m.a is not None else 0)
    for q in (q16(x) for x in m.root):
        out += struct.pack("<H", q)
    for s, x in zip(m.strings, m.counts[: m.m]):
        out += struct.pack("<BH", len(s), q16(x)) + s
    entries = [(p, s, x) for p, row in m.table.items() for s, x in row.items()]
    out += struct.pack("<I", len(entries))
    for p, s, x in entries:
        out += struct.pack("<HHH", p, s, q16(x))
    out += struct.pack("<I", len(m.rows))
    for p, (cfull, sfull) in m.rows.items():
        out += struct.pack("<HHH", p, q16(cfull), q16(sfull))
    for q in (q16(x) for x in m.lens):
        out += struct.pack("<H", q)
    if m.a is not None:
        out += m.a.astype("<f2").tobytes()
        out += m.b.astype("<f2").tobytes()
    return bytes(out)


def unpack(data: bytes) -> Model:
    if data[:4] != MODEL_MAGIC:
        raise ValueError("not a v8 model")
    pos = 4
    m, r = struct.unpack_from("<IB", data, pos)
    pos += 5
    root = np.array([dq16(x) for x in struct.unpack_from("<256H", data, pos)])
    pos += 512
    strings, counts = [], []
    for _ in range(m):
        n, q = struct.unpack_from("<BH", data, pos)
        pos += 3
        strings.append(data[pos : pos + n])
        pos += n
        counts.append(dq16(q))
    entries = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    table: dict[int, dict[int, int]] = collections.defaultdict(dict)
    for _ in range(entries):
        p, s, q = struct.unpack_from("<HHH", data, pos)
        pos += 6
        table[p][s] = dq16(q)
    n_rows = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    rows = {}
    for _ in range(n_rows):
        p, c, s = struct.unpack_from("<HHH", data, pos)
        pos += 6
        rows[p] = (dq16(c), dq16(s))
    lens = np.array([dq16(x) for x in struct.unpack_from(f"<{LEN}H", data, pos)])
    pos += 2 * LEN
    a = b = None
    if r:
        n = 257 * r
        a = np.frombuffer(data, "<f2", n, pos).reshape(257, r).copy()
        pos += 2 * n
        n = r * 256
        b = np.frombuffer(data, "<f2", n, pos).reshape(r, 256).copy()
    all_counts = np.concatenate([np.array(counts), root])
    return Model(strings, all_counts, root, dict(table), rows, lens, r, a, b)


# %%
def fit(name: str, vocab: str, rect: dict) -> dict:
    """The packed model at one budget, the reference (v7) costs, and the paths."""
    if (name, vocab) not in _SPLIT:
        _SPLIT[(name, vocab)] = c.split(name, vocab)
    s = _SPLIT[(name, vocab)]
    d, bg, uni = c.rebuild(s, rect["dict"], rect["table"])
    big, uni = bg, uni
    fit_toks = c.viterbi(s["fit"], d, big, uni)[0]
    val_toks, val_cost = c.viterbi(s["va"], d, big, uni)
    full_big, full_uni = c.bigrams(v.cover(s["fit"], d, 0.0))
    rows = {
        d.ids[p]: (
            float(sum(row.values())),
            float(sum(x for k, x in row.items() if len(k) == 1)),
        )
        for p, row in full_big.items()
        if p in big
    }
    pf = t.pieces(fit_toks, {})
    root_raw = np.exp2(s["bc"].max() - s["bc"])
    root = root_raw / root_raw.sum()
    counts, _ = t.byte_matrix(pf, root)
    lens = collections.Counter(len(p.raw) for p in pf)
    lens_counts = np.array(
        [lens.get(i, 0) for i in range(1, CAP + 1)]
        + [sum(c for ell, c in lens.items() if ell > CAP)],
        float,
    )
    raw_all = np.exp2(max(d.cost) - np.array(d.cost))
    raw_all = raw_all / raw_all.min()
    table = {d.ids[p]: {d.ids[k]: x for k, x in row.items()} for p, row in big.items()}
    model = Model(
        [x for x in d.strings if len(x) > 1],
        np.concatenate(
            [raw_all[: len(d.strings) - 256], raw_all[len(d.strings) - 256 :]]
        ),
        raw_all[len(d.strings) - 256 :],
        table,
        rows,
        lens_counts,
        0,
        None,
        None,
    )
    table_bytes = 6 * sum(len(row) for row in table.values()) + 6 * len(rows)
    base = d.model_bytes + table_bytes
    stream = s["stream"]
    best: tuple | None = None
    if pf and sum(len(p.raw) for p in pf) > 1000:
        sel_toks = c.viterbi(s["sel"], d, big, uni)[0]
        sel_pf = t.pieces(sel_toks, {})
        for r in (1, 2, 4, 8, 16, 32, 64):
            a, b = factors(counts, r)
            logp = logp_of(a, b, root)
            sel_bits = sum(run_bits(p.raw, logp) for p in sel_pf) + sum(
                math.log2(1 / model.lens_p[min(len(p.raw), LEN) - 1]) for p in sel_pf
            )
            size = 34 + (257 + 256) * r * 2
            sel_charged = sel_bits / len(s["sel"]) + 8 * (base + size) / stream
            if best is None or sel_charged < best[0]:
                best = (sel_charged, r, a, b, logp, size)
    ref_logp = None
    ref_size = 0
    if best is not None:
        _, r, a, b, ref_logp, ref_size = best
        model.r, model.a, model.b = r, a, b
        model.__post_init__()
    ref_lens = np.array(
        [lens.get(i, 0) + ALPHA for i in range(1, CAP + 1)]
        + [sum(c for ell, c in lens.items() if ell > CAP) + ALPHA],
        float,
    )
    ref_lens = ref_lens / ref_lens.sum()
    prev = [None] + val_toks[:-1]
    cost = [c.cost_of(t, d, big, uni, p) for t, p in zip(val_toks, prev)]
    total = sum(cost)
    single_row = sum(x for t, x in zip(val_toks, cost) if len(t) == 1)
    single_bc = sum(float(d.cost_of(t)) for t in val_toks if len(t) == 1)
    tail_extra = 0.0
    if ref_logp is not None:
        for a0, e in runs(val_toks):
            tail_extra += -math.log2(ref_lens[min(e - a0, LEN) - 1])
            tail_extra += run_bits(b"".join(val_toks[a0:e]), ref_logp)
        ref_tail = total - single_row + tail_extra
        ref_v7 = total - single_bc + tail_extra
    else:
        ref_tail = total - single_row + single_bc
        ref_v7 = total
    concept_model = Model(
        model.strings,
        model.counts,
        model.root,
        model.table,
        model.rows,
        np.zeros(LEN),
        0,
        None,
        None,
    )
    blob = pack(model)
    return {
        "s": s,
        "dict": d,
        "val_toks": val_toks,
        "model": model,
        "pack": blob,
        "pack_concept": pack(concept_model),
        "ref_tail": ref_tail,
        "ref_concept": total,
        "ref_v7": ref_v7,
        "r": model.r,
    }


# %%
def encode(model: Model, toks: list[bytes], arm: str = "tail") -> tuple[bytes, dict]:
    """Encode the DP's token path: a type at each boundary, then the content."""
    enc = Encoder()
    bits = collections.Counter()
    if arm == "concept":
        p = -1
        for tok in toks:
            before = enc.bits
            cdf = model.full_cdf(p)
            enc.symbol(*cdf.span(model.ids[tok]))
            bits["concept"] += enc.bits - before
            p = model.ids[tok]
        data = enc.finish()
        return data, {
            "bits": dict(bits),
            "entropy": enc.bits,
            "bytes": len(data),
            "tokens": len(toks),
            "calls": 1 + sum(len(t) for t in toks if len(t) == 1),
        }
    i, p, after_run = 0, -1, False
    while i < len(toks):
        s = toks[i]
        if len(s) == 1:
            j = i
            while j < len(toks) and len(toks[j]) == 1:
                j += 1
            if not after_run:
                before = enc.bits
                cdf = model.type_cdf(p)
                enc.symbol(*cdf.span(1))
                bits["type"] += enc.bits - before
            before = enc.bits
            ell = j - i
            cdf = model.len_cdf()
            enc.symbol(*cdf.span(min(ell, LEN) - 1))
            if ell > CAP:
                extra = ell - LEN
                cdf = model.byte_cdf()
                while True:
                    digit = min(extra, 255)
                    enc.symbol(*cdf.span(digit))
                    if digit < 255:
                        break
                    extra -= 255
            bits["len"] += enc.bits - before
            prev = 256
            for k in range(i, j):
                before = enc.bits
                cdf = model.tail_cdf(prev)
                b = toks[k][0]
                enc.symbol(*cdf.span(b))
                bits["tail"] += enc.bits - before
                prev = b
            p = model.m + prev
            after_run = True
            i = j
        else:
            if not after_run:
                before = enc.bits
                cdf = model.type_cdf(p)
                enc.symbol(*cdf.span(0))
                bits["type"] += enc.bits - before
            before = enc.bits
            cdf = model.token_cdf(p)
            enc.symbol(*cdf.span(model.ids[s]))
            bits["covered"] += enc.bits - before
            p = model.ids[s]
            after_run = False
            i += 1
    data = enc.finish()
    return data, {
        "bits": dict(bits),
        "entropy": enc.bits,
        "bytes": len(data),
        "tokens": len(toks),
        "calls": 1 + sum(len(s) for s in toks if len(s) == 1),
    }


def decode(model: Model, data: bytes, n: int, arm: str = "tail") -> tuple[bytes, dict]:
    dec = Decoder(data)
    bits = collections.Counter()
    out, p, after_run = bytearray(), -1, False
    if arm == "concept":
        while len(out) < n:
            before = dec.bits
            cdf = model.full_cdf(p)
            i = cdf.find(dec.target(TOTAL))
            dec.symbol(*cdf.span(i))
            bits["concept"] += dec.bits - before
            out += model.tok(i)
            p = i
        return bytes(out), {"bits": dict(bits), "entropy": dec.bits}
    while len(out) < n:
        if not after_run:
            before = dec.bits
            cdf = model.type_cdf(p)
            kind = cdf.find(dec.target(TOTAL))
            dec.symbol(*cdf.span(kind))
            bits["type"] += dec.bits - before
        else:
            kind = 0
        if kind == 0:
            before = dec.bits
            cdf = model.token_cdf(p)
            i = cdf.find(dec.target(TOTAL))
            dec.symbol(*cdf.span(i))
            bits["covered"] += dec.bits - before
            out += model.strings[i]
            p = i
            after_run = False
        else:
            before = dec.bits
            cdf = model.len_cdf()
            li = cdf.find(dec.target(TOTAL))
            dec.symbol(*cdf.span(li))
            ell = li + 1
            if ell > CAP:
                extra, cdf = 0, model.byte_cdf()
                while True:
                    digit = cdf.find(dec.target(TOTAL))
                    dec.symbol(*cdf.span(digit))
                    extra += digit
                    if digit < 255:
                        break
                ell = LEN + extra
            bits["len"] += dec.bits - before
            prev = 256
            for _ in range(ell):
                before = dec.bits
                cdf = model.tail_cdf(prev)
                b = cdf.find(dec.target(TOTAL))
                dec.symbol(*cdf.span(b))
                bits["tail"] += dec.bits - before
                out.append(b)
                prev = b
            p = model.m + prev
            after_run = True
    return bytes(out), {"bits": dict(bits), "entropy": dec.bits}


# %%
def header(n: int, choice: int, model_len: int, arm: str = "tail") -> bytes:
    flag = 0x80 if arm == "concept" else 0
    return FILE_MAGIC + struct.pack("<QBI", n, choice | flag, model_len)


def file_size(
    payload: bytes,
    model: bytes | None,
    n: int,
    choice: int,
    arm: str = "tail",
) -> bytes:
    ml = len(model) if model is not None else 0
    return header(n, choice, ml, arm) + (model or b"") + payload


def load_candidates(path: Path, name: str, k: int) -> list[dict]:
    """The top-k budgets by the held-out charge, one per dictionary size."""
    d = json.loads(path.read_text())["corpora"][name]
    rows = sorted(d["grid"], key=lambda g: g["sel_charged"])
    out, sizes, tables = [], set(), set()

    def add(g: dict) -> None:
        sizes.add(g["dict"])
        tables.add(g["table"])
        out.append(
            {"vocab": d.get("vocab", "substr"), "dict": g["dict"], "table": g["table"]}
        )

    for g in rows:
        if g["dict"] not in sizes:
            add(g)
            if len(out) == k:
                return out
    for g in rows:
        if len(out) == k:
            break
        if g["table"] not in tables:
            add(g)
    return out


def measure(name: str, candidates: list[dict], try_n: int) -> dict:
    """Both codecs at every candidate: the concept arm (every token) and the
    tail arm (a type, a length, the shared byte model). The winner is the
    smallest *charged* total, so the model's size is part of the choice."""
    options = []
    for rect in candidates:
        corp = fit(name, rect["vocab"], rect)
        n = len(corp["s"]["va"])
        for arm in ("tail", "concept"):
            blob = corp["pack"] if arm == "tail" else corp["pack_concept"]
            model = unpack(blob)
            t0 = time.perf_counter()
            payload, enc = encode(model, corp["val_toks"], arm)
            dt = time.perf_counter() - t0
            charged = 8 * len(payload) / n + 8 * len(blob) / corp["s"]["stream"]
            options.append(
                {
                    **rect,
                    "arm": arm,
                    "corp": corp,
                    "model": model,
                    "blob": blob,
                    "payload": payload,
                    "enc": enc,
                    "size": len(payload),
                    "dt": dt,
                    "charged": charged,
                }
            )
            print(
                f"  {name:11s} {arm:7s} dict={rect['dict']:5d} table={rect['table']:5d} "
                f"payload={len(payload):6d} B ({8 * len(payload) / n:.3f} bpb) "
                f"model={len(blob) / 1e3:5.1f} kB charged={charged:.3f}",
                flush=True,
            )
    best = min(options, key=lambda o: o["charged"])
    corp, model, payload, blob, arm = (
        best["corp"],
        best["model"],
        best["payload"],
        best["blob"],
        best["arm"],
    )
    n = len(corp["s"]["va"])
    xb = 8 * len(header(n, 0, 0)) / n + 8 * len(payload) / n
    shared = xb + 8 * len(blob) / corp["s"]["stream"]
    own = xb + 8 * len(blob) / n
    choice = math.log2(max(len(options), 1)) / n
    t0 = time.perf_counter()
    decoded, dec = decode(model, payload, n, arm)
    ddt = time.perf_counter() - t0
    entropy = best["enc"]["entropy"] / n
    ref = (corp["ref_tail"] if arm == "tail" else corp["ref_concept"]) / n
    published = corp["ref_v7"] / n if arm == "tail" else corp["ref_concept"] / n
    return {
        "corpus": name,
        "arm": arm,
        "vocab": best["vocab"],
        "dict": best["dict"],
        "table": best["table"],
        "r": corp["r"],
        "val_bytes": n,
        "stream_bytes": corp["s"]["stream"],
        "payload_bytes": len(payload),
        "model_bytes": len(blob),
        "rate": 8 * len(payload) / n,
        "entropy": entropy,
        "ref": ref,
        "v7": published,
        "shared": shared + choice,
        "self": own + choice,
        "delta_v7": entropy - published,
        "delta_ref": entropy - ref,
        "delta_code": xb + choice - entropy,
        "bits": {k: v / n for k, v in best["enc"]["bits"].items()},
        "calls": best["enc"]["calls"],
        "encode_s": best["dt"],
        "decode_s": ddt,
        "encode_mb_s": n / 1e6 / max(best["dt"], 1e-9),
        "decode_mb_s": n / 1e6 / max(ddt, 1e-9),
        "model_kb": len(blob) / 1e3,
        "roundtrip": decoded == corp["s"]["va"],
        "options": [
            {
                "arm": o["arm"],
                "dict": o["dict"],
                "table": o["table"],
                "rate": 8 * o["size"] / n,
                "model_bytes": len(o["blob"]),
                "charged": o["charged"],
                "calls": o["enc"]["calls"],
            }
            for o in options
        ],
        "corp": corp,
        "payload": payload,
    }


# %%
def gates(res: dict) -> list[dict]:
    out = []
    for name, e in res["corpora"].items():
        out += [
            {
                "id": "G0",
                "corpus": name,
                "value": f"round-trip={'ok' if e['roundtrip'] else 'FAIL'} "
                f"decode={e['decode_mb_s']:.2f} MB/s encode={e['encode_mb_s']:.2f} MB/s",
                "status": "pass" if e["roundtrip"] else "fail",
            },
            {
                "id": "G1",
                "corpus": name,
                "value": f"recorded shared={e['shared']:.4f} "
                f"entropy+charge={e['entropy'] + 8 * e['model_bytes'] / e['stream_bytes']:.4f} "
                f"delta={e['delta_code']:+.4f}",
                "status": "pass" if abs(e["delta_code"]) <= 0.005 else "fail",
            },
            {
                "id": "G1s",
                "corpus": name,
                "value": f"recorded self={e['self']:.4f} "
                f"entropy+charge={e['entropy'] + 8 * e['model_bytes'] / e['val_bytes']:.4f}",
                "status": "pass"
                if abs(
                    e["self"] - (e["entropy"] + 8 * e["model_bytes"] / e["val_bytes"])
                )
                <= 0.005
                else "fail",
            },
            {
                "id": "G2",
                "corpus": name,
                "value": f"published v7={e['v7']:.4f} code={e['entropy']:.4f} "
                f"gap={e['delta_v7']:+.4f}",
                "status": "pass" if abs(e["delta_v7"]) <= 0.02 else "fail",
            },
            {
                "id": "G2r",
                "corpus": name,
                "value": f"correct model={e['ref']:.4f} code={e['entropy']:.4f} "
                f"gap={e['delta_ref']:+.4f}",
                "status": "pass" if abs(e["delta_ref"]) <= 0.05 else "fail",
            },
            {
                "id": "G3",
                "corpus": name,
                "value": f"model={e['model_kb']:.1f} kB arm={e['arm']}",
                "status": "pass" if e["model_kb"] <= 45 else "fail",
            },
        ]
    return out


def report(res: dict, path: Path) -> None:
    lines = [
        "# H-NCM v8: N0, the bitstream",
        "",
        "The model is packed as counts, every CDF is derived (`round(2^24 p)`), and a",
        "32-bit arithmetic coder writes the DP's token path. The decoder reads the",
        "token stream and never reruns the DP. Two codecs are measured at every budget:",
        "`concept` codes every token with the row's joint probability (the segmentation",
        "is implicit); `tail` codes a type, a length and the shared byte model. The",
        "budget is chosen on the held-out 10% and transmitted (`--try`); the winner is",
        "the smallest *charged* total, so the model's size is part of the choice.",
        "",
        "`ref` is the model's correct account (the multi-byte tokens + the tail + the",
        "length). `v7` is the published estimate: it subtracts the byte unigram for",
        "tail bytes that the concept path already priced at the bigram row (~3.8 vs",
        "5.0 bpb), so it double-counts the tail's gain. `gap` is the published gap;",
        "the code is the judge (rule 17).",
        "",
        "| corpus | arm | dict | table | r | payload B | rate | ref | v7 | gap | model kB | shared | self | calls | enc MB/s | dec MB/s | RT |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, e in res["corpora"].items():
        lines.append(
            f"| {name} | {e['arm']} | {e['dict']} | {e['table']} | {e['r']} | "
            f"{e['payload_bytes']} | {e['rate']:.3f} | {e['ref']:.3f} | "
            f"{e['v7']:.3f} | {e['delta_v7']:+.3f} | {e['model_kb']:.1f} | "
            f"{e['shared']:.3f} | {e['self']:.3f} | {e['calls']:.0f} | "
            f"{e['encode_mb_s']:.2f} | {e['decode_mb_s']:.2f} | "
            f"{'ok' if e['roundtrip'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## The arms at the chosen budget",
        "",
        "| corpus | arm | dict | table | rate | model kB | charged | calls |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, e in res["corpora"].items():
        for o in e["options"]:
            lines.append(
                f"| {name} | {o['arm']} | {o['dict']} | {o['table']} | "
                f"{o['rate']:.3f} | {o['model_bytes'] / 1e3:.1f} | "
                f"{o['charged']:.3f} | {o['calls']:.0f} |"
            )
    lines += [
        "",
        "## The bit categories (bpb)",
        "",
        "| corpus | arm | concept | covered | type | len | tail | sum | rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, e in res["corpora"].items():
        b = e["bits"]
        lines.append(
            f"| {name} | {e['arm']} | {b.get('concept', 0):.3f} | "
            f"{b.get('covered', 0):.3f} | {b.get('type', 0):.3f} | "
            f"{b.get('len', 0):.3f} | {b.get('tail', 0):.3f} | "
            f"{sum(b.values()):.3f} | {e['rate']:.3f} |"
        )
    lines += [
        "",
        "## Gates",
        "",
        "G0 the round-trip (the artifact); G1 the recorded size equals the code's",
        "entropy + the charge within 0.005 bpb (shared); G1s the same, self; G2 the",
        "published v7 estimate reconciles within 0.02 bpb; G2r the correct model does;",
        "G3 the model is at most 45 kB.",
        "",
        "| gate | corpus | value | status |",
        "|---|---|---|---|",
    ]
    for g in gates(res):
        lines.append(
            f"| {g['id']} | {g['corpus']} | {g['value']} | **{g['status']}** |"
        )
    lines += [
        "",
        "## Finding",
        "",
        "The published v7 tail formula is `total - sum(byte_unigram) + tail + length`.",
        "The `total` already priced the tail bytes at the concept bigram row (3.83 bpb",
        "on shakespeare's oracle budget), not at the byte unigram (5.05), so the",
        "subtraction is too large and the 2.723 is about 0.50 bpb below the model it",
        "names. The bitstream prices the model it implements: on shakespeare the",
        "concept arm is the best realizable codec at 1 MB, and the tail arm pays the",
        "type on top of a row cost it does not beat. The recorded `rate`/`shared`",
        "columns are the number of record; the v7/v8 estimates are lower bounds.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run(names: list[str], out: Path, rect_path: Path, try_n: int = 3) -> dict:
    res = {"format": 8, "seed": SEED, "try": try_n, "corpora": {}}
    for name in names:
        print(f"{name}:", flush=True)
        cands = load_candidates(rect_path, name, try_n)
        e = measure(name, cands, try_n)
        res["corpora"][name] = {
            k: v for k, v in e.items() if k not in ("corp", "payload")
        }
        print(
            f"{name:11s} rate={e['rate']:.3f} v7={e['v7']:.3f} "
            f"shared={e['shared']:.3f} model={e['model_kb']:.1f} kB "
            f"RT={'ok' if e['roundtrip'] else 'FAIL'}",
            flush=True,
        )
    for g in gates(res):
        print(
            f"  {g['id']:4s} {g['status']:4s} {g['corpus']}: {g['value']}", flush=True
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    report(res, out.parent / "CODEC.md")
    return res


def fuzz(model: Model, raw: bytes, n: int = 200) -> list[dict]:
    """Random slices the model did not fit: the DP, then the round-trip."""
    rng = np.random.RandomState(SEED)
    trie: dict = {}
    for s, i in model.ids.items():
        if len(s) > 1:
            node = trie
            for b in s:
                node = node.setdefault(b, {})
            node["#"] = i
    fake = v.Dict(model.ids, [b""] * model.m, [0.0] * model.k, np.zeros(256), trie, 0)
    out = []
    for _ in range(n):
        a = int(rng.randint(0, max(1, len(raw) - 32)))
        b = a + int(rng.randint(1, min(32, len(raw) - a) + 1))
        bs = raw[a:b]
        toks, _ = c.viterbi(bs, fake, {}, collections.Counter())
        payload, _ = encode(model, toks)
        dec, _ = decode(model, payload, len(bs))
        out.append(dec == bs)
    return out


def selftest() -> None:
    # the count quantization: relative error < 1%, so < 0.015 bits per symbol
    rng = np.random.RandomState(SEED)
    for x in rng.uniform(1, 1e6, 2000):
        assert abs(dq16(q16(x)) - x) / x < 0.01
    # the CDF: sums to TOTAL and find/span agree
    for _ in range(100):
        p = rng.dirichlet(np.ones(int(rng.randint(2, 40))))
        cdf = CDF(p)
        assert int(cdf.w.sum()) == TOTAL
        for i in range(len(p)):
            cum, freq, total = cdf.span(i)
            assert freq >= 1 and cdf.find(cum + freq // 2) == i
    # the coder round-trips random symbols under random CDFs
    for _ in range(50):
        k = int(rng.randint(2, 20))
        cdf = CDF(rng.dirichlet(np.ones(k)))
        syms = [int(rng.randint(k)) for _ in range(int(rng.randint(1, 300)))]
        enc = Encoder()
        for i in syms:
            enc.symbol(*cdf.span(i))
        data = enc.finish()
        dec = Decoder(data)
        got = []
        for _ in syms:
            i = cdf.find(dec.target(TOTAL))
            dec.symbol(*cdf.span(i))
            got.append(i)
        assert got == syms
        assert 8 * len(data) + 40 >= enc.bits
    # a tiny model: pack/unpack is identical, and a 1 kB slice round-trips
    bs = (b"the quick brown fox jumps over the lazy dog. " * 40) + bytes(range(64)) * 8
    bc = np.ones(256) * 8
    d = v.build_dict(v.counts_substr(bs), len(bs), bc, min_count=1)
    toks, _ = c.viterbi(bs, d, {}, collections.Counter())
    big, _ = c.bigrams(toks)
    raw_all = np.exp2(max(d.cost) - np.array(d.cost))
    raw_all /= raw_all.min()
    m = len(d.strings) - 256
    model = Model(
        [x for x in d.strings if len(x) > 1],
        raw_all,
        raw_all[m:],
        {d.ids[p]: {d.ids[k]: x for k, x in row.items()} for p, row in big.items()},
        {
            d.ids[p]: (
                float(sum(row.values())),
                float(sum(x for k, x in row.items() if len(k) == 1)),
            )
            for p, row in big.items()
        },
        np.ones(LEN),
        0,
        None,
        None,
    )
    blob = pack(model)
    m2 = unpack(blob)
    m3 = unpack(pack(m2))
    for p in (-1, 0, m2.ids[b"the"], m2.m + 32):
        assert np.array_equal(m3.token_cdf(p).w, m2.token_cdf(p).w)
    slice_ = bs[100:1124]
    toks, _ = c.viterbi(slice_, d, big, collections.Counter(toks))
    if b"".join(toks) != slice_:
        toks = slice_ and [slice_[i : i + 1] for i in range(len(slice_))]
    for arm in ("tail", "concept"):
        payload, _ = encode(m2, toks, arm)
        got, _ = decode(m2, payload, len(slice_), arm)
        assert got == slice_, (arm, len(got), len(slice_))
    print("selftest: ok")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="H-NCM v8 bitstream")
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--corpora", default="shakespeare,synseg,syngrid")
    r.add_argument("--out", default="runs/v8/codec.json")
    r.add_argument("--rect", default="runs/v7/concept.json")
    r.add_argument("--try", type=int, default=3)
    s = sub.add_parser("check")
    s.add_argument("--corpora", default="shakespeare,synseg,syngrid")
    s.add_argument("--rect", default="runs/v7/concept.json")
    s.add_argument("--try", type=int, default=3)
    e = sub.add_parser("encode")
    e.add_argument("--corpus", default="shakespeare")
    e.add_argument("--rect", default="runs/v7/concept.json")
    e.add_argument("--try", type=int, default=3)
    e.add_argument("--model", default="runs/v8/model.bin")
    e.add_argument("--out", default="/tmp/hncm.bin")
    e.add_argument("--shared", action="store_true")
    d = sub.add_parser("decode")
    d.add_argument("--model", default="runs/v8/model.bin")
    d.add_argument("--in", dest="src", default="/tmp/hncm.bin")
    d.add_argument("--out", default="/tmp/hncm.txt")
    sub.add_parser("selftest")
    a = p.parse_args(argv)
    if a.cmd == "selftest":
        selftest()
    elif a.cmd in ("run", "check"):
        run(
            [x for x in a.corpora.split(",") if x],
            Path("runs/v8/codec.json" if a.cmd == "check" else a.out),
            Path(a.rect),
            getattr(a, "try"),
        )
    elif a.cmd == "encode":
        cands = load_candidates(Path(a.rect), a.corpus, getattr(a, "try"))
        e = measure(a.corpus, cands, getattr(a, "try"))
        corp = e["corp"]
        blob = corp["pack"] if e["arm"] == "tail" else corp["pack_concept"]
        Path(a.model).parent.mkdir(parents=True, exist_ok=True)
        Path(a.model).write_bytes(blob)
        data = file_size(
            e["payload"],
            None if a.shared else blob,
            e["val_bytes"],
            e["dict"],
            e["arm"],
        )
        Path(a.out).write_bytes(data)
        print(f"wrote {a.out} ({len(data)} B), model {a.model}")
    elif a.cmd == "decode":
        model = unpack(Path(a.model).read_bytes())
        data = Path(a.src).read_bytes()
        n, choice, ml = struct.unpack_from("<QBI", data, 4)
        arm = "concept" if choice & 0x80 else "tail"
        choice &= 0x7F
        raw, _ = decode(model, data[HEAD + ml :], n, arm)
        Path(a.out).write_bytes(raw)
        print(f"wrote {a.out} ({len(raw)} B), dict choice {choice}")
    else:
        p.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
