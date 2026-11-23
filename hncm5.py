# %% [markdown]
# # H-NCM v5 — prove the parallelism dividend, not just the codec
#
# Co-primary claims (report all three, including negatives):
#   P1 absolute compression: concept total < unigram - 0.10 bpb; <= byte GPT + 0.05.
#   P2 parallelism dividend: at matched sample quality calls_AR/calls_CF >= 8;
#      boundary chunker not worse than fixed by > 0.02 bpb at equal calls.
#   P3 mechanism: oracle I > 0 (ceiling, controls only); no-oracle I > 0 and
#      >= max(0.5*I_oracle, 2*SEM) on synseg; I ~ 0 and alignment ~ grid on syngrid;
#      align(no-oracle) >= align(grid) + 0.05.
#
# One file, no backwards compatibility. hncm4.py is the reference implementation.
# All tables read only results.json. No placeholders, no silent fallbacks.
#
# CLI:
#   python hncm5.py run --preset smoke --phase 0
#   python hncm5.py run --preset smoke --phase 2 --corpora synseg --chunkers fixed,oracle,rate-only,surprise,oracle-sup
#   python hncm5.py hpo --stage 1 --trials 20
#   python hncm5.py report --results runs/latest/results.json
# Env: HNCM_PRESET, HNCM_RESUME=0|1, HNCM_OUT, HNCM_CFG (JSON).

# %%
import argparse
import datetime
import hashlib
import json
import math
import os
import random
import tempfile
import time
import urllib.request
from dataclasses import asdict, dataclass, field, replace
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LN2 = math.log(2.0)
MAXOFF = 31
FORMAT = 5

PHASE_CORPORA = {
    0: ("synseg", "syngrid"),
    1: ("synseg",),
    2: ("synseg", "syngrid"),
    3: ("shakespeare",),
}
PHASE_CHUNKERS = {
    0: ("fixed", "oracle", "rate-only"),
    1: ("fixed",),
    2: ("fixed", "oracle", "rate-only", "surprise", "oracle-sup"),
    3: ("fixed", "rate-only", "surprise", "bpe"),
}
CHUNKER_NAMES = ("fixed", "oracle", "rate-only", "surprise", "oracle-sup", "bpe")
ORACLE_NAMES = ("oracle", "oracle-sup")


# %%
@dataclass(frozen=True)
class Cfg:
    # geometry / capacity
    L: int = 128
    R: int = 16
    stride: int = 4
    K1: int = 32
    K: int = 64
    Q: int = 3
    beta: float = 0.25
    d: int = 128
    heads: int = 4
    enc: int = 2
    main: int = 2
    dec: int = 2
    gen: int = 4
    eh_layers: int = 2
    base_layers: int = 6
    bs: int = 64
    # optimization
    codec_steps: int = 1500
    prior_steps: int = 1500
    base_steps: int = 1500
    codec_lr: float = 1e-3
    prior_lr: float = 5e-4
    base_lr: float = 1e-3
    # rate control
    rho: float = 3.4
    lam_eta: float = 2.0
    lam_max: float = 8.0
    lam_fixed: float | None = None
    level_dropout: bool = True
    use_eh_rate: bool = True
    match_dynamic: bool = True
    # budgets / evaluation
    cf_k_list: tuple = (1, 2, 4)
    seeds: tuple = (0, 1)
    eval_windows: int = 512
    sample_n: int = 64
    corpora: tuple | None = None
    chunkers: tuple | None = None
    phase: int = 0
    # boundary arm
    boundary_w: float = 1.0
    surprise_layers: int = 1
    surprise_steps: int = 400
    surprise_lr: float = 1e-3
    dsur: int = 64
    bpe_merges: int = 64
    bpe_bytes: int = 100_000
    # gates (one source of truth)
    p1a_margin: float = 0.10
    p1b_margin: float = 0.05
    p2_calls: float = 8.0
    p2_qeps: float = 0.05
    p2_beats: float = 0.02
    p3_alpha: float = 0.5
    p3_align: float = 0.05
    health_util: float = 0.20
    match_rate: float = 0.05
    M: int = field(init=False, default=0)
    G: int = field(init=False, default=0)

    def __post_init__(s):
        object.__setattr__(s, "seeds", tuple(s.seeds))
        object.__setattr__(s, "cf_k_list", tuple(s.cf_k_list))
        if s.corpora is None:
            object.__setattr__(s, "corpora", PHASE_CORPORA[s.phase])
        else:
            object.__setattr__(s, "corpora", tuple(s.corpora))
        if s.chunkers is None:
            object.__setattr__(s, "chunkers", PHASE_CHUNKERS[s.phase])
        else:
            object.__setattr__(s, "chunkers", tuple(s.chunkers))
        object.__setattr__(s, "M", s.L // s.stride)
        object.__setattr__(s, "G", s.L // s.R)
        assert s.L % s.R == 0 and s.L % s.stride == 0 and s.R % s.stride == 0, (
            "geometry must divide L"
        )
        assert s.d % s.heads == 0, "d must divide heads"
        assert 2 <= s.stride <= s.M and s.M >= 2, "need at least two chunks"
        assert s.Q >= 1 and s.K >= 2 and s.K1 >= 2 and s.beta > 0
        assert s.rho >= 0 and s.lam_eta > 0 and 0 <= s.lam_max
        assert s.eval_windows >= 2 and s.sample_n >= 2 and len(s.seeds) >= 1
        assert all(isinstance(k, int) and k >= 1 for k in s.cf_k_list), (
            "CF budgets must be positive ints"
        )
        assert all(c in CHUNKER_NAMES for c in s.chunkers), (
            f"unknown chunker in {s.chunkers}"
        )
        assert all(c in ("synseg", "syngrid", "shakespeare") for c in s.corpora)
        if "shakespeare" in s.corpora:
            assert not any(c in ORACLE_NAMES for c in s.chunkers), (
                "oracle onsets are forbidden on shakespeare"
            )
        assert s.phase in (0, 1, 2, 3)


PRESETS = {
    "smoke": dict(
        L=64,
        d=64,
        heads=4,
        enc=1,
        main=1,
        dec=1,
        gen=2,
        eh_layers=1,
        base_layers=2,
        K1=16,
        K=32,
        Q=2,
        bs=16,
        codec_steps=80,
        prior_steps=80,
        base_steps=80,
        seeds=(0,),
        eval_windows=32,
        sample_n=4,
        cf_k_list=(1, 2),
        rho=1.5,
        phase=0,
        corpora=("synseg", "syngrid"),
        chunkers=("fixed", "oracle", "rate-only"),
        surprise_steps=40,
    ),
    "quick": dict(
        L=128,
        d=96,
        heads=4,
        gen=3,
        eh_layers=2,
        base_layers=4,
        K1=16,
        K=64,
        Q=3,
        bs=32,
        codec_steps=300,
        prior_steps=400,
        base_steps=400,
        seeds=(0,),
        eval_windows=128,
        sample_n=16,
        cf_k_list=(1, 2, 4),
        rho=3.4,
        phase=0,
        surprise_steps=200,
    ),
    "full": dict(
        L=256,
        d=192,
        heads=6,
        gen=6,
        base_layers=8,
        codec_steps=1500,
        prior_steps=1500,
        base_steps=1500,
        seeds=(0, 1, 2),
        eval_windows=2048,
        sample_n=128,
        rho=3.4,
        phase=0,
        surprise_steps=400,
    ),
}


def canon(o):
    return json.loads(json.dumps(o))


def fingerprint(cfg):
    return hashlib.sha256(
        json.dumps(canon(asdict(cfg)), sort_keys=True).encode()
    ).hexdigest()


def build_cfg(preset="smoke", overrides=None):
    if preset not in PRESETS:
        raise SystemExit(f"unknown preset {preset!r}; choose {sorted(PRESETS)}")
    ov = json.loads(os.environ.get("HNCM_CFG", "{}"))
    if overrides:
        ov.update(overrides)
    return Cfg(**{**PRESETS[preset], **ov})


# %%
# ---------------- data: corpus, controls, onsets, oracle snapping -------------------------------
def load_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    raw = urllib.request.urlopen(url, timeout=20).read()
    return raw, "shakespeare"


def _word(rng, lo=2, hi=10):
    letters = np.arange(ord("a"), ord("z") + 1)
    return bytes(int(c) for c in rng.choice(letters, rng.randint(lo, hi)))


def gen_synseg(seed=7, nbytes=1_200_000):
    rng = np.random.RandomState(seed)
    topics = [[_word(rng) for _ in range(40)] for _ in range(8)]
    out = bytearray()
    while len(out) < nbytes:
        t = rng.randint(8)
        for _ in range(rng.randint(8, 21)):
            out += topics[t][rng.randint(40)] + b" "
        out += b"\n"
    return bytes(out), "synseg"


def gen_syngrid(seed=8, nbytes=1_200_000):
    rng = np.random.RandomState(seed)
    topics = [[_word(rng, 2, 4) for _ in range(40)] for _ in range(8)]
    out = bytearray()
    while len(out) < nbytes:
        t = rng.randint(8)
        for _ in range(16):
            for _ in range(4):
                w = topics[t][rng.randint(40)]
                out += w + b"_" * (4 - len(w))
    return bytes(out[: len(out) // 4 * 4]), "syngrid"


def onsets(name, xb):
    if name == "syngrid":
        return np.arange(len(xb)) % 4 == 0
    prev = np.concatenate([[ord("\n")], xb[:-1]])
    return np.isin(prev, [ord(" "), ord("\n")]) & ~np.isin(
        xb, [ord(" "), ord("\n"), ord("_")]
    )


def snap_onsets(ons, stride, L):
    M = L // stride
    off = np.zeros(M, dtype=int)
    seen = np.zeros(M, dtype=bool)
    for p in np.flatnonzero(ons):
        c = int(p) // stride
        if seen[c]:
            continue
        lo, hi = c * stride, (c + 1) * stride
        cand = np.arange(lo, hi)
        off[c] = int(cand[np.argmin(np.abs(cand - int(p)))]) - lo
        seen[c] = True
    return off


def onset_target(name, xb):
    t = np.zeros(len(xb), dtype=np.float32)
    t[np.flatnonzero(onsets(name, xb))] = 1.0
    return t


CORPORA = {}


def get_corpus(name):
    if name in CORPORA:
        return CORPORA[name]
    raw, tag = {
        "shakespeare": load_shakespeare,
        "synseg": gen_synseg,
        "syngrid": gen_syngrid,
    }[name]()
    arr = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()
    n = (int(0.9 * len(arr)) // 4) * 4
    c = dict(
        tr=arr[:n].to(DEVICE),
        va=arr[n:].to(DEVICE),
        name=tag,
        sha=hashlib.sha256(raw).hexdigest(),
        nbytes=len(raw),
    )
    CORPORA[name] = c
    return c


def windows(data, starts, L):
    return data[starts[:, None] + torch.arange(L, device=data.device)[None]]


def make_starts(corpus, L, n, seed=1234):
    rs = np.random.RandomState(seed)
    hi = len(corpus["va"]) - L
    if corpus["name"] == "syngrid":
        return torch.tensor(4 * rs.randint(0, hi // 4, size=n), device=DEVICE)
    return torch.tensor(rs.randint(0, hi, size=n), device=DEVICE)


def train_batch(corpus, L, bs, rng):
    starts = rng.randint(0, len(corpus["tr"]) - L, size=bs)
    return windows(corpus["tr"], torch.tensor(starts, device=DEVICE), L)


def unigram_bits(corpus):
    c = torch.bincount(corpus["tr"], minlength=256).float() + 1
    return (-torch.log2(c / c.sum())).to(DEVICE)


# %%
# ---------------- blocks and shared layout -----------------------------------------------------
class Block(nn.Module):
    def __init__(s, d, h, causal):
        super().__init__()
        s.h, s.causal = h, causal
        s.ln1 = nn.LayerNorm(d)
        s.qkv = nn.Linear(d, 3 * d, bias=False)
        s.o = nn.Linear(d, d, bias=False)
        s.ln2 = nn.LayerNorm(d)
        s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(s, x, past=None):
        B, T, D = x.shape
        q, k, v = s.qkv(s.ln1(x)).view(B, T, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        if past is not None:
            assert T == 1, "incremental mode is single-token"
            k = torch.cat([past[0], k], 2)
            v = torch.cat([past[1], v], 2)
        new = (k, v)
        a = F.scaled_dot_product_attention(
            q, k, v, is_causal=(s.causal and past is None and T > 1)
        )
        x = x + s.o(a.transpose(1, 2).reshape(B, T, D))
        return x + s.mlp(s.ln2(x)), new


class Stack(nn.Module):
    def __init__(s, n, d, h, causal):
        super().__init__()
        s.blocks = nn.ModuleList(Block(d, h, causal) for _ in range(n))
        s.ln = nn.LayerNorm(d)

    def forward(s, x, past=None):
        past = past if past is not None else [None] * len(s.blocks)
        new = []
        for b, p in zip(s.blocks, past):
            x, np_ = b(x, p)
            new.append(np_)
        return s.ln(x), new


def take(x, idx):
    return torch.gather(x, 1, idx[..., None].expand(-1, -1, x.size(-1)))


def geometry(b):
    B, L = b.shape
    dev = b.device
    assert b[:, 0].all(), "first byte must start a chunk"
    counts = b.sum(1)
    M = int(counts.max())
    assert M <= L
    order = torch.argsort((~b).long(), dim=1, stable=True)[:, :M]
    slot = torch.arange(M, device=dev)[None]
    valid = slot < counts[:, None]
    m_idx = b.long().cumsum(1) - 1
    nxt = torch.cat(
        [order[:, 1:], torch.full((B, 1), L, device=dev, dtype=order.dtype)], 1
    )
    nxt = torch.where(slot + 1 < counts[:, None], nxt, torch.full_like(nxt, L))
    length = (nxt - order).clamp(min=0) * valid
    start_b = torch.gather(order, 1, m_idx)
    offset = torch.arange(L, device=dev)[None] - start_b
    return dict(
        order=order,
        valid=valid,
        counts=counts,
        M=M,
        m_idx=m_idx,
        length=length,
        offset=offset,
        valid_chunks=valid,
    )


def layout(cfg, dyn):
    lv = [("c1", 0, cfg.K1, cfg.G)]
    lo = cfg.K1
    if dyn:
        lv.append(("bnd", lo, lo + cfg.stride, cfg.M - 1))
        lo += cfg.stride
    for q in range(cfg.Q):
        lv.append((f"r{q}", lo, lo + cfg.K, cfg.M))
        lo += cfg.K
    ends = []
    a = 0
    for _, _, _, w in lv:
        a += w
        ends.append(a)
    return lv, lo, ends, a


def seq_tokens(tk, cfg, dyn, lv):
    parts = [tk["c1"]]
    if dyn:
        parts.append(tk["offs"][:, 1:] + lv[1][1])
    base = 2 if dyn else 1
    for q in range(cfg.Q):
        parts.append(tk["codes"][..., q] + lv[base + q][1])
    return torch.cat(parts, 1)


def seq_feats(order, length, cfg, dyn, lv):
    B = order.size(0)
    dev = order.device
    pos = [(torch.arange(cfg.G, device=dev) * cfg.R)[None].expand(B, -1)]
    ln = [torch.zeros(B, cfg.G, dtype=torch.long, device=dev)]
    if dyn:
        pos.append(
            (torch.arange(1, cfg.M, device=dev) * cfg.stride)[None].expand(B, -1)
        )
        ln.append(torch.zeros(B, cfg.M - 1, dtype=torch.long, device=dev))
    for _ in range(cfg.Q):
        pos.append(order)
        ln.append(length.clamp(max=MAXOFF))
    return torch.cat(pos, 1), torch.cat(ln, 1)


class TokenFeats(nn.Module):
    def __init__(s, cfg, dyn, V, nlv):
        super().__init__()
        s.pos = nn.Embedding(cfg.L, cfg.d)
        s.ln = nn.Embedding(MAXOFF + 1, cfg.d)
        s.typ = nn.Embedding(nlv, cfg.d)
        s.bos = nn.Parameter(torch.zeros(1, 1, cfg.d))
        s.tok = nn.Embedding(V + 2, cfg.d)
        s.register_buffer(
            "lvl",
            torch.tensor(
                sum(
                    [[i] * w for i, (_, _, _, w) in enumerate(layout(cfg, dyn)[0])], []
                ),
                dtype=torch.long,
            ),
        )

    def feats(s, pos, ln, T):
        return s.pos(pos[:, :T]) + s.ln(ln[:, :T]) + s.typ(s.lvl[:T])[None]


def level_valid(tk, cfg, dyn):
    lv = layout(cfg, dyn)[0]
    counts = tk["counts"]
    B = counts.size(0)
    dev = counts.device
    out = []
    for name, _, _, w in lv:
        if name == "c1":
            out.append(torch.ones(B, w, dtype=torch.bool, device=dev))
        elif name == "bnd":
            out.append((torch.arange(1, w + 1, device=dev)[None] < counts[:, None]))
        else:
            out.append(torch.arange(w, device=dev)[None] < counts[:, None])
    return out


def level_bits(lg, tok, lv, ends, valid):
    out = 0
    for li, (_, lo, hi, w) in enumerate(lv):
        a, e = ends[li] - w, ends[li]
        ce = F.cross_entropy(
            lg[:, a:e, lo:hi].reshape(-1, hi - lo),
            (tok[:, a:e] - lo).reshape(-1),
            reduction="none",
        ).view(tok.size(0), -1)
        out = out + (ce * valid[li]).sum(1)
    return out


def dual_step(lam, ema, rho, eta, lam_max):
    return float(np.clip(lam + eta * (ema - rho), 0.0, lam_max))


# %%
# ---------------- chunkers: fixed / oracle / rate-only / boundary / BPE ------------------------
@dataclass
class Boundary:
    hard: torch.Tensor
    ste: torch.Tensor
    off_soft: torch.Tensor | None
    offs: torch.Tensor | None
    aux: dict | None = None


class ChunkerBase(nn.Module):
    dynamic = True

    def boundaries(s, h, x):
        raise NotImplementedError

    def loss(s, h, x, bnd):
        return h.new_zeros(())


class Chunker(Protocol):
    dynamic: bool

    def boundaries(self, h, x) -> Boundary: ...

    def loss(self, h, x, bnd) -> torch.Tensor: ...


def group_soft(logits, M, stride):
    return F.softmax(logits.view(logits.size(0), M, stride), -1)


def hard_soft(P, stride):
    B, M, _ = P.shape
    idx = P.argmax(-1)
    idx[:, 0] = 0
    P = torch.cat([F.one_hot(idx[:, :1], stride).float(), P[:, 1:]], 1)
    hard = torch.zeros(B, M * stride, dtype=torch.bool, device=P.device)
    hard.scatter_(1, torch.arange(M, device=P.device)[None] * stride + idx, True)
    hard[:, 0] = True
    one = F.one_hot(idx, stride).float()
    off = one + P - P.detach()
    return hard, off.reshape(B, -1), P, idx


class FixedChunker(ChunkerBase):
    dynamic = False

    def __init__(s, cfg):
        super().__init__()
        s.stride = cfg.stride

    def boundaries(s, h, x):
        B, L = x.shape
        b = (torch.arange(L, device=x.device) % s.stride == 0)[None].expand(B, L)
        return Boundary(b, b.float(), None, None)


class OracleChunker(ChunkerBase):
    dynamic = True

    def __init__(s, cfg, name):
        super().__init__()
        assert name != "shakespeare", "oracle onsets are forbidden on shakespeare"
        s.cfg, s.name = cfg, name

    def boundaries(s, h, x):
        B, L = x.shape
        xb = x.detach().cpu().numpy()
        hard = torch.zeros(B, L, dtype=torch.bool, device=x.device)
        offs = torch.full((B, s.cfg.M), -1, dtype=torch.long, device=x.device)
        P = torch.zeros(B, s.cfg.M, s.cfg.stride, device=x.device)
        for i in range(B):
            off = snap_onsets(onsets(s.name, xb[i][:L]), s.cfg.stride, L)
            off[0] = 0
            oo = torch.tensor(off, device=x.device)
            offs[i] = oo
            P[i, torch.arange(s.cfg.M, device=x.device), oo] = 1.0
            hard[i, torch.arange(s.cfg.M, device=x.device) * s.cfg.stride + oo] = True
        return Boundary(hard, hard.float(), P, offs)


class RateOnlyChunker(ChunkerBase):
    dynamic = True

    def __init__(s, cfg):
        super().__init__()
        s.cfg = cfg
        s.ln = nn.LayerNorm(cfg.d)
        s.head = nn.Linear(cfg.d, 1)
        nn.init.zeros_(s.head.weight)
        nn.init.zeros_(s.head.bias)

    def boundaries(s, h, x):
        B, L = x.shape
        P = group_soft(s.head(s.ln(h)).squeeze(-1), s.cfg.M, s.cfg.stride)
        hard, ste, P, idx = hard_soft(P, s.cfg.stride)
        assert hard.size(1) == L
        return Boundary(hard, ste, P, idx)


def learn_bpe(data, max_merges):
    seq = list(data)
    merges, ranks = {}, {}
    for i in range(max_merges):
        pairs = {}
        for a, b in zip(seq[:-1], seq[1:]):
            pairs[(a, b)] = pairs.get((a, b), 0) + 1
        if not pairs:
            break
        (a, b), cnt = max(pairs.items(), key=lambda kv: kv[1])
        if cnt < 2:
            break
        new = 256 + i
        merges[(a, b)] = new
        ranks[(a, b)] = i
        out, j = [], 0
        while j < len(seq):
            if j + 1 < len(seq) and (seq[j], seq[j + 1]) == (a, b):
                out.append(new)
                j += 2
            else:
                out.append(seq[j])
                j += 1
        seq = out
    return merges, ranks


BPE_CACHE = {}


def bpe_table(cfg, name):
    key = (name, cfg.bpe_merges, cfg.bpe_bytes)
    if key in BPE_CACHE:
        return BPE_CACHE[key]
    data = get_corpus(name)["tr"].detach().cpu().numpy().astype(np.uint8).tobytes()
    merges, ranks = learn_bpe(data[: cfg.bpe_bytes], cfg.bpe_merges)
    BPE_CACHE[key] = (merges, ranks)
    return BPE_CACHE[key]


class BpeChunker(ChunkerBase):
    dynamic = True

    def __init__(s, cfg, name):
        super().__init__()
        s.cfg, s.merges, s.ranks = cfg, *bpe_table(cfg, name)

    def token_starts(s, xb):
        syms = list(int(v) for v in xb)
        starts = list(range(len(syms)))
        while True:
            best, bi = None, None
            for j in range(len(syms) - 1):
                r = s.ranks.get((syms[j], syms[j + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, j
            if best is None:
                break
            syms[bi] = s.merges[(syms[bi], syms[bi + 1])]
            del syms[bi + 1]
            del starts[bi + 1]
        return starts

    def boundaries(s, h, x):
        B, L = x.shape
        xb = x.detach().cpu().numpy()
        hard = torch.zeros(B, L, dtype=torch.bool, device=x.device)
        offs = torch.zeros(B, s.cfg.M, dtype=torch.long, device=x.device)
        P = torch.zeros(B, s.cfg.M, s.cfg.stride, device=x.device)
        for i in range(B):
            starts = s.token_starts(xb[i][:L])
            gs = {}
            for p in starts:
                gs.setdefault(p // s.cfg.stride, p)
            pos = []
            for c in range(s.cfg.M):
                p = gs.get(c, c * s.cfg.stride)
                pos.append(p)
                offs[i, c] = p - c * s.cfg.stride
                P[i, c, offs[i, c]] = 1.0
            hard[i, torch.tensor(pos, device=x.device)] = True
        return Boundary(hard, hard.float(), P, offs)


class BoundaryChunker(ChunkerBase):
    dynamic = True

    def __init__(s, cfg, objective, name=None, aux=None):
        super().__init__()
        assert objective in ("surprise", "oracle")
        s.cfg, s.objective, s.name, s.aux = cfg, objective, name, aux
        s.head = nn.Linear(cfg.d, 1)
        nn.init.zeros_(s.head.weight)
        nn.init.zeros_(s.head.bias)

    @torch.no_grad()
    def target(s, x):
        if s.objective == "oracle":
            xb = x.detach().cpu().numpy()
            t = torch.zeros_like(x, dtype=torch.float)
            for i in range(x.size(0)):
                t[
                    i,
                    torch.tensor(
                        np.flatnonzero(onsets(s.name, xb[i])), device=x.device
                    ),
                ] = 1.0
            return t
        lg = s.aux.logits(x)
        ce = F.cross_entropy(
            lg.reshape(-1, 256), x[:, 1:].reshape(-1), reduction="none"
        ).view(x.size(0), -1)
        z = torch.cat([ce.new_zeros(ce.size(0), 1), ce], 1)
        lo = z.min(1, keepdim=True).values
        hi = z.max(1, keepdim=True).values
        return (z - lo) / (hi - lo + 1e-6)

    def boundaries(s, h, x):
        B, L = x.shape
        logits = s.head(h).squeeze(-1)
        P = group_soft(logits, s.cfg.M, s.cfg.stride)
        hard, ste, P, idx = hard_soft(P, s.cfg.stride)
        assert hard.size(1) == L
        return Boundary(hard, ste, P, idx, aux=dict(logits=logits, target=s.target(x)))

    def loss(s, h, x, bnd):
        if bnd.aux is None:
            return h.new_zeros(())
        return F.binary_cross_entropy_with_logits(bnd.aux["logits"], bnd.aux["target"])


def make_chunker(name, cfg, corpus_name, aux=None):
    if name == "fixed":
        return FixedChunker(cfg)
    if name == "oracle":
        return OracleChunker(cfg, corpus_name)
    if name == "rate-only":
        return RateOnlyChunker(cfg)
    if name == "surprise":
        return BoundaryChunker(cfg, "surprise", corpus_name, aux)
    if name == "oracle-sup":
        return BoundaryChunker(cfg, "oracle", corpus_name)
    if name == "bpe":
        return BpeChunker(cfg, corpus_name)
    raise ValueError(name)


# %%
# ---------------- RVQ with soft assignments ---------------------------------------------------
class RVQ(nn.Module):
    def __init__(s, d, K, Q, beta=0.25):
        super().__init__()
        s.K, s.Q, s.beta = K, Q, beta
        s.cb = nn.Parameter(torch.randn(Q, K, d) * 0.1)
        s.register_buffer("usage", torch.zeros(Q, K))
        s.last = [None] * Q
        s._inited = False

    def forward(s, z, keep=None):
        B, M, D = z.shape
        if s.training and not s._inited:
            with torch.no_grad():
                tmp = z.detach().reshape(-1, D)
                for q in range(s.Q):
                    s.cb.data[q] = tmp[torch.randint(len(tmp), (s.K,), device=z.device)]
                    tmp = tmp - s.cb.data[q][torch.cdist(tmp, s.cb.data[q]).argmin(1)]
            s._inited = True
        res, qsum, codes, softs = z, torch.zeros_like(z), [], []
        vq = z.new_zeros(())
        for q in range(s.Q):
            cb = s.cb[q]
            dist = (
                res.pow(2).sum(-1, keepdim=True) - 2 * res @ cb.t() + cb.pow(2).sum(-1)
            )
            idx = dist.argmin(-1)
            e = cb[idx]
            vq = (
                vq
                + ((e - res.detach()) ** 2).mean()
                + s.beta * ((res - e.detach()) ** 2).mean()
            )
            tau = (0.1 * dist.detach().mean()).clamp_min(1e-6)
            softs.append(F.softmax(-dist / tau, -1))
            if s.training:
                with torch.no_grad():
                    s.usage[q] += torch.bincount(idx.reshape(-1), minlength=s.K).float()
                    s.last[q] = res.detach().reshape(-1, D)
            qsum = qsum + (
                e if keep is None else e * keep[:, q, None, None].to(e.dtype)
            )
            res = res - e.detach()
            codes.append(idx)
        return (
            z + (qsum - z).detach(),
            torch.stack(codes, -1),
            vq,
            torch.stack(softs, 2),
        )

    @torch.no_grad()
    def restart(s):
        for q in range(s.Q):
            dead = s.usage[q] < 1
            n = int(dead.sum())
            if n > 0 and s.last[q] is not None:
                src = s.last[q]
                pick = src[torch.randint(len(src), (n,), device=src.device)]
                s.cb.data[q][dead] = pick + 0.01 * torch.randn_like(pick)
        s.usage.zero_()


# %%
# ---------------- codec: bytes -> (coarse cells, boundary offsets, RVQ chunks) -> bytes --------
class Codec(nn.Module):
    def __init__(s, cfg, chunker_name, corpus_name, aux=None):
        super().__init__()
        s.cfg = cfg
        s.dyn = chunker_name != "fixed"
        d, L = cfg.d, cfg.L
        # shared modules first: fixed and dynamic share init from one seed
        s.emb = nn.Embedding(256, d)
        s.pos = nn.Embedding(L, d)
        s.enc = Stack(cfg.enc, d, cfg.heads, False)
        s.cpos = nn.Embedding(cfg.M, d)
        s.main = Stack(cfg.main, d, cfg.heads, False)
        s.grid_proj = nn.Linear(d, d)
        s.vq1 = RVQ(d, cfg.K1, 1, cfg.beta)
        s.proj = nn.Linear(d, d)
        s.rvq = RVQ(d, cfg.K, cfg.Q, cfg.beta)
        s.dpos = nn.Embedding(L, d)
        s.doff = nn.Embedding(MAXOFF + 1, d)
        s.dlen = nn.Embedding(MAXOFF + 1, d)
        s.dec = Stack(cfg.dec, d, cfg.heads, False)
        s.head = nn.Linear(d, 256)
        lv, V, ends, T = layout(cfg, s.dyn)
        s.eh = Stack(cfg.eh_layers, d, cfg.heads, True)
        s.eh_f = TokenFeats(cfg, s.dyn, V, len(lv))
        s.eh_head = nn.Linear(d, V)
        s.lv, s.ends, s.T = lv, ends, T
        s.chunker = make_chunker(chunker_name, cfg, corpus_name, aux)

    def assign(s, g, beta):
        hard = F.one_hot(g["m_idx"], s.cfg.M).float()
        if not s.dyn:
            return hard
        cum = beta.cumsum(1) - 1
        slots = torch.arange(s.cfg.M, device=beta.device, dtype=beta.dtype)
        sa = F.softmax(-((cum[..., None] - slots) ** 2) / (2 * 0.5**2), -1)
        return hard + sa - sa.detach()

    def encode(s, x, keep=None):
        cfg = s.cfg
        B, L = x.shape
        assert L == cfg.L
        h, _ = s.enc(s.emb(x) + s.pos.weight[:L])
        bnd = s.chunker.boundaries(h, x)
        g = geometry(bnd.hard)
        A = s.assign(g, bnd.ste)
        cnt = g["length"].float().clamp_min(1)
        xc = (A.transpose(1, 2) @ h) / cnt[..., None] + s.cpos.weight[None]
        z = s.proj(s.main(xc)[0])
        e1, c1, l1, soft1 = s.vq1(s.grid_proj(h.reshape(B, cfg.G, cfg.R, -1).mean(2)))
        e1c = take(e1, g["order"] // cfg.R)
        zq, codes, l2, soft2 = s.rvq(z - e1c, keep)
        return dict(
            zq=zq,
            e1c=e1c,
            codes=codes,
            c1=c1[..., 0],
            soft1=soft1[..., 0, :],
            soft2=soft2,
            off_soft=bnd.off_soft,
            offs=bnd.offs,
            g=g,
            vq=l1 + l2,
            bnd=bnd,
            h=h,
        )

    def decode(s, zfull, g, L):
        zup = take(zfull, g["m_idx"])
        lenb = torch.gather(g["length"], 1, g["m_idx"])
        din = (
            zup
            + s.dpos.weight[:L]
            + s.doff(g["offset"].clamp(max=MAXOFF))
            + s.dlen(lenb.clamp(max=MAXOFF))
        )
        return s.head(s.dec(din)[0])

    def tokens_from(s, o):
        g = o["g"]
        vc = g["valid_chunks"]
        offs = None
        if s.dyn:
            offs = g["order"] % s.cfg.stride
            offs = torch.where(vc, offs, torch.zeros_like(offs))
        codes = torch.where(vc[..., None], o["codes"], torch.zeros_like(o["codes"]))
        return dict(
            c1=o["c1"],
            offs=offs,
            codes=codes,
            order=g["order"],
            length=g["length"],
            counts=g["counts"],
        )

    def eh_logits(s, tok, pos, ln):
        content = s.eh_f.tok(tok)
        content = torch.cat([s.eh_f.bos.expand(tok.size(0), 1, -1), content[:, :-1]], 1)
        return s.eh_head(s.eh(content + s.eh_f.feats(pos, ln, s.T))[0])

    def loss(s, x, lam):
        cfg = s.cfg
        B, L = x.shape
        keep = None
        if cfg.level_dropout and s.training:
            qk = torch.where(
                torch.rand(B, device=x.device) < 0.5,
                torch.full((B,), cfg.Q, device=x.device, dtype=torch.long),
                torch.randint(1, cfg.Q + 1, (B,), device=x.device),
            )
            keep = torch.arange(cfg.Q, device=x.device)[None] < qk[:, None]
        o = s.encode(x, keep)
        recon = F.cross_entropy(
            s.decode(o["zq"] + o["e1c"], o["g"], L).reshape(-1, 256), x.reshape(-1)
        )
        tk = s.tokens_from(o)
        tok = seq_tokens(tk, cfg, s.dyn, s.lv)
        valid = level_valid(tk, cfg, s.dyn)
        pos, ln = seq_feats(tk["order"], tk["length"], cfg, s.dyn, s.lv)
        lg = s.eh_logits(tok, pos, ln)
        soft_t = (
            [o["soft1"]]
            + ([o["off_soft"][:, 1:]] if s.dyn else [])
            + [o["soft2"][:, :, q, :] for q in range(cfg.Q)]
        )
        rate_soft = x.new_zeros(())
        for li, (_, lo, hi, w) in enumerate(s.lv):
            a, e = s.ends[li] - w, s.ends[li]
            lp = F.log_softmax(lg[:, a:e, lo:hi], -1)
            mask = valid[li].to(lp.dtype)[..., None]
            rate_soft = rate_soft - (soft_t[li] * lp * mask).sum()
        rate_soft = rate_soft / (B * L)
        with torch.no_grad():
            rate_hard = level_bits(lg, tok, s.lv, s.ends, valid).sum() / (B * L)
        r = rate_soft if cfg.use_eh_rate else rate_soft.detach()
        bloss = s.chunker.loss(o["h"], x, o["bnd"])
        loss = recon + lam * r + o["vq"] + cfg.boundary_w * bloss
        return loss, dict(
            recon_bpb=(recon.detach() / LN2),
            rate_bpb=(rate_hard / LN2),
            vq=o["vq"].detach(),
            bloss=bloss.detach(),
        )

    @torch.no_grad()
    def tokenize(s, x):
        return s.tokens_from(s.encode(x, None))

    @torch.no_grad()
    def rate_bits(s, x):
        o = s.encode(x, None)
        tk = s.tokens_from(o)
        tok = seq_tokens(tk, s.cfg, s.dyn, s.lv)
        valid = level_valid(tk, s.cfg, s.dyn)
        pos, ln = seq_feats(tk["order"], tk["length"], s.cfg, s.dyn, s.lv)
        return level_bits(s.eh_logits(tok, pos, ln), tok, s.lv, s.ends, valid) / LN2

    @torch.no_grad()
    def decode_tokens(s, c1, codes, g, temp=1.0):
        zq = sum(s.rvq.cb[q][codes[..., q]] for q in range(s.cfg.Q))
        e1 = s.vq1.cb[0][c1]
        lg = s.decode(zq + take(e1, g["order"] // s.cfg.R), g, s.cfg.L)
        return torch.multinomial(F.softmax(lg / temp, -1).reshape(-1, 256), 1).view(
            c1.size(0), s.cfg.L
        )

    def maintain(s):
        s.rvq.restart()
        s.vq1.restart()


# %%
# ---------------- priors: identical modules; AR causal vs CF masked ----------------------------
class Prior(nn.Module):
    def __init__(s, cfg, dyn, mode):
        super().__init__()
        assert mode in ("ar", "cf")
        s.cfg, s.dyn, s.mode = cfg, dyn, mode
        lv, V, ends, T = layout(cfg, dyn)
        s.lv, s.V, s.ends, s.T = lv, V, ends, T
        s.MASK = V
        s.stack = Stack(cfg.gen, cfg.d, cfg.heads, mode == "ar")
        s.head = nn.Linear(cfg.d, V)
        s.f = TokenFeats(cfg, dyn, V, len(lv))

    def run(s, inp, pos, ln, T, past=None):
        content = s.f.tok(inp[:, :T])
        if s.mode == "ar" and past is None:
            content = torch.cat(
                [s.f.bos.expand(inp.size(0), 1, -1), content[:, :-1]], 1
            )
        out, new = s.stack(content + s.f.feats(pos, ln, T), past)
        return s.head(out), new

    @torch.no_grad()
    def nll(s, tk):
        tok = seq_tokens(tk, s.cfg, s.dyn, s.lv)
        pos, ln = seq_feats(tk["order"], tk["length"], s.cfg, s.dyn, s.lv)
        valid = level_valid(tk, s.cfg, s.dyn)
        B = tok.size(0)
        out = {}
        if s.mode == "ar":
            lg, _ = s.run(tok, pos, ln, s.T)
        for li, (name, lo, hi, w) in enumerate(s.lv):
            a, e = s.ends[li] - w, s.ends[li]
            if s.mode == "cf":
                inp = tok.clone()
                inp[:, a:] = s.MASK
                lg, _ = s.run(inp, pos, ln, s.T)
            ce = F.cross_entropy(
                lg[:, a:e, lo:hi].reshape(-1, hi - lo),
                (tok[:, a:e] - lo).reshape(-1),
                reduction="none",
            ).view(B, -1)
            out[name] = (ce * valid[li]).sum(1)
        return out

    def train_loss(s, tk):
        tok = seq_tokens(tk, s.cfg, s.dyn, s.lv)
        pos, ln = seq_feats(tk["order"], tk["length"], s.cfg, s.dyn, s.lv)
        valid = level_valid(tk, s.cfg, s.dyn)
        B = tok.size(0)
        if s.mode == "ar":
            lg, _ = s.run(tok, pos, ln, s.T)
            return level_bits(lg, tok, s.lv, s.ends, valid).sum() / (B * s.cfg.L)
        t = int(torch.randint(len(s.lv), (1,), device=tok.device))
        _, lo, hi, w = s.lv[t]
        a, e = s.ends[t] - w, s.ends[t]
        keep = torch.zeros(B, s.T, dtype=torch.bool, device=tok.device)
        keep[:, :a] = True
        gamma = 0.05 + 0.95 * torch.rand(B, 1, device=tok.device)
        keep[:, a:] = False
        keep[:, a:e] = torch.rand(B, w, device=tok.device) > gamma
        inp = torch.where(keep, tok, torch.full_like(tok, s.MASK))
        lg, _ = s.run(inp, pos, ln, s.T)
        m = (~keep[:, a:e]) & valid[t]
        ce = F.cross_entropy(
            lg[:, a:e, lo:hi].reshape(-1, hi - lo),
            (tok[:, a:e] - lo).reshape(-1),
            reduction="none",
        ).view(tok.size(0), -1)
        return (ce * m).sum() / m.sum().clamp_min(1)

    @torch.no_grad()
    def sample(s, codec, n, k_rvq=1, temp=1.0):
        cfg = s.cfg
        dev = s.f.tok.weight.device
        L, M, st = cfg.L, cfg.M, cfg.stride
        tok = torch.full(
            (n, s.T), s.MASK if s.mode == "cf" else 0, dtype=torch.long, device=dev
        )
        order = (torch.arange(M, device=dev) * st)[None].expand(n, -1)
        length = torch.full((n, M), st, dtype=torch.long, device=dev)
        pos, ln = seq_feats(order, length, cfg, s.dyn, s.lv)
        calls = 0
        g = None
        past = None
        fts = s.f.feats(pos, ln, s.T)
        for li, (name, lo, hi, w) in enumerate(s.lv):
            a, e = s.ends[li] - w, s.ends[li]
            if s.dyn and name == "r0":
                bl = s.lv[1][1]
                offs = torch.cat(
                    [
                        torch.zeros(n, 1, dtype=torch.long, device=dev),
                        tok[:, s.ends[1] - (M - 1) : s.ends[1]] - bl,
                    ],
                    1,
                )
                starts = torch.arange(M, device=dev)[None] * st + offs
                b = torch.zeros(n, L, dtype=torch.bool, device=dev)
                b.scatter_(1, starts, True)
                g = geometry(b)
                pos, ln = seq_feats(g["order"], g["length"], cfg, s.dyn, s.lv)
                fts = s.f.feats(pos, ln, s.T)
            if s.mode == "ar":
                for t in range(a, e):
                    x = (
                        (s.f.bos.expand(n, 1, -1) + fts[:, :1])
                        if t == 0
                        else (s.f.tok(tok[:, t - 1])[:, None] + fts[:, t : t + 1])
                    )
                    lg, past = s.stack(x, past)
                    lg = s.head(lg)[:, -1, lo:hi]
                    tok[:, t] = (
                        torch.multinomial(F.softmax(lg / temp, -1), 1)[:, 0] + lo
                    )
                    calls += 1
            else:
                kt = 1 if name in ("c1", "bnd") else k_rvq
                remaining = torch.ones(n, w, dtype=torch.bool, device=dev)
                for p in range(kt):
                    lg, _ = s.run(tok, pos, ln, s.T)
                    pr = F.softmax(lg[:, a:e, lo:hi] / temp, -1)
                    samp = torch.multinomial(pr.reshape(-1, hi - lo), 1).view(n, w)
                    conf = pr.gather(-1, samp[..., None])[..., 0]
                    if p < kt - 1:
                        u = torch.rand_like(conf).clamp(1e-6, 1 - 1e-6)
                        conf = conf + (-torch.log(-torch.log(u)))
                        keep_n = int(
                            math.floor(w * math.cos(math.pi / 2 * (p + 1) / kt))
                        )
                        reveal = (remaining.sum(1) - keep_n).clamp_min(0)
                        rank = (
                            conf.masked_fill(~remaining, -1e9)
                            .argsort(1, descending=True)
                            .argsort(1)
                        )
                        chosen = remaining & (rank < reveal[:, None])
                    else:
                        chosen = remaining
                    sub = tok[:, a:e]
                    sub[chosen] = samp[chosen] + lo
                    remaining = remaining & ~chosen
                    calls += 1
        if g is None:
            b = (torch.arange(L, device=dev) % st == 0)[None].expand(n, L).clone()
            g = geometry(b)
        base = 2 if s.dyn else 1
        c1 = tok[:, : cfg.G]
        codes = torch.stack(
            [
                tok[:, s.ends[base + q] - M : s.ends[base + q]] - s.lv[base + q][1]
                for q in range(cfg.Q)
            ],
            -1,
        )
        return codec.decode_tokens(c1, codes, g, temp), calls


# %%
# ---------------- byte GPT baseline ------------------------------------------------------------
class ByteGPT(nn.Module):
    def __init__(s, cfg):
        super().__init__()
        s.cfg = cfg
        s.emb = nn.Embedding(256, cfg.d)
        s.pos = nn.Embedding(cfg.L, cfg.d)
        s.stack = Stack(cfg.base_layers, cfg.d, cfg.heads, True)
        s.head = nn.Linear(cfg.d, 256)

    def logits(s, x):
        return s.head(s.stack(s.emb(x[:, :-1]) + s.pos.weight[: x.size(1) - 1])[0])

    def loss(s, x):
        return F.cross_entropy(s.logits(x).reshape(-1, 256), x[:, 1:].reshape(-1))

    @torch.no_grad()
    def bits_per_window(s, x, uni):
        ce = (
            F.cross_entropy(
                s.logits(x).reshape(-1, 256), x[:, 1:].reshape(-1), reduction="none"
            )
            .view(x.size(0), -1)
            .sum(1)
        )
        return ce / LN2 + uni[x[:, 0]]

    @torch.no_grad()
    def sample(s, n, L, uni, temp=1.0):
        x = torch.multinomial(torch.exp(-uni * LN2), n, replacement=True)[:, None]
        past = None
        for t in range(L - 1):
            h, past = s.stack(s.emb(x[:, -1:]) + s.pos.weight[t : t + 1][None], past)
            x = torch.cat(
                [x, torch.multinomial(F.softmax(s.head(h)[:, -1] / temp, -1), 1)], 1
            )
        return x, L - 1


def nparams(m):
    return sum(p.numel() for p in m.parameters())


# %%
# ---------------- self-tests ------------------------------------------------------------------
def _selftest():
    torch.manual_seed(0)
    c = Cfg(**PRESETS["smoke"])
    assert c.M == c.L // c.stride and c.G == c.L // c.R
    raw, _ = gen_synseg(nbytes=20000)
    a = np.frombuffer(raw, np.uint8)
    assert a.max() < 128 and (a == 0).mean() < 0.01, (
        "synseg must be ASCII without NUL padding"
    )
    ons = onsets("synseg", a[:2000])
    assert ons.sum() > 100
    raw, _ = gen_syngrid(nbytes=20000)
    a = np.frombuffer(raw, np.uint8)
    assert (a == 0).sum() == 0 and len(a) % 4 == 0, "syngrid must be 4-byte aligned"
    assert (onsets("syngrid", a) == (np.arange(len(a)) % 4 == 0)).all()
    # oracle snapping: at most one boundary per group, aligned, first byte forced
    off = snap_onsets(onsets("synseg", a[:64]), 4, 64)
    assert len(off) == 64 // 4 and np.all(off < 4)
    # variable geometry
    b = torch.rand(4, c.L, device=DEVICE) < 0.3
    b[:, 0] = True
    g = geometry(b)
    assert int(g["counts"].max()) == g["M"] and g["valid_chunks"].any()
    assert (g["length"].sum(1) == c.L).all()
    # chunkers: first byte, at most one per group, dense arms exactly M
    x = torch.randint(0, 256, (3, c.L), device=DEVICE)
    h = torch.randn(3, c.L, c.d, device=DEVICE)
    aux = (
        ByteGPT(replace(c, d=c.dsur, heads=4, base_layers=c.surprise_layers))
        .to(DEVICE)
        .eval()
    )
    for p in aux.parameters():
        p.requires_grad_(False)
    for name in CHUNKER_NAMES:
        corpus = "syngrid" if name == "oracle" else "synseg"
        ch = make_chunker(name, c, corpus, aux).to(DEVICE)
        bnd = ch.boundaries(h, x)
        assert bnd.hard[:, 0].all(), f"{name}: first byte must start"
        for i in range(3):
            p = bnd.hard[i].nonzero().flatten().tolist()
            assert len(p) == len({v // c.stride for v in p}), (
                f"{name}: more than one boundary per group"
            )
        if name != "oracle":
            assert bnd.hard.sum(1).eq(c.M).all(), (
                f"{name}: dense arm must have M chunks"
            )
    # oracle on syngrid must equal the grid
    corp = get_corpus("syngrid")
    xg = windows(corp["va"], make_starts(corp, c.L, 2), c.L)
    org = make_chunker("oracle", c, "syngrid").to(DEVICE)
    bg = org.boundaries(h[:2], xg).hard
    grid = (torch.arange(c.L, device=DEVICE) % c.stride == 0)[None].expand(2, c.L)
    assert torch.equal(bg, grid), "oracle on syngrid must reproduce the grid"
    # paired init: fixed and rate-only share all common modules
    torch.manual_seed(7)
    cf = Codec(c, "fixed", "synseg").to(DEVICE)
    torch.manual_seed(7)
    cd = Codec(c, "rate-only", "synseg").to(DEVICE)
    dpar = dict(cd.named_parameters())
    for name, p in cf.named_parameters():
        if (
            name in dpar
            and p.shape == dpar[name].shape
            and not name.startswith(("chunker", "eh_f", "eh_head"))
        ):
            assert torch.equal(p, dpar[name]), f"shared init differs: {name}"
    # codec loss/backward and router/head gradients
    xb = torch.randint(0, 256, (4, c.L), device=DEVICE)
    for name, corpus in (
        ("fixed", "synseg"),
        ("rate-only", "synseg"),
        ("surprise", "synseg"),
        ("oracle-sup", "synseg"),
    ):
        torch.manual_seed(3)
        codec = Codec(c, name, corpus, aux).to(DEVICE).train()
        loss, st = codec.loss(xb, 0.5)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(st["rate_bpb"])
        if name == "rate-only":
            assert codec.chunker.head.weight.grad is not None
            assert codec.chunker.head.weight.grad.abs().sum() > 0
        if name in ("surprise", "oracle-sup"):
            assert codec.chunker.head.weight.grad is not None
            assert codec.chunker.head.weight.grad.abs().sum() > 0
        codec.eval()
        o = codec.encode(xb, None)
        tk = codec.tokens_from(o)
        zq = sum(codec.rvq.cb[q][tk["codes"][..., q]] for q in range(c.Q))
        z2 = zq + take(codec.vq1.cb[0][tk["c1"]], o["g"]["order"] // c.R)
        assert torch.allclose(
            codec.decode(o["zq"] + o["e1c"], o["g"], c.L),
            codec.decode(z2, o["g"], c.L),
            atol=1e-2,
        )
        rb = codec.rate_bits(xb)
        assert rb.shape == (4,) and torch.isfinite(rb).all()
    # priors: identical params/init, KV cache exact, calls match the layout
    tk = codec.tokenize(xb)
    priors = {}
    for mode in ("ar", "cf"):
        torch.manual_seed(11)
        pr = Prior(c, True, mode).to(DEVICE)
        priors[mode] = pr
        with torch.no_grad():
            nl = pr.nll(tk)
        assert all(torch.isfinite(v).all() for v in nl.values())
        pr.train_loss(tk).backward()
        pr.eval()
        torch.manual_seed(12)
        out, calls = pr.sample(codec, 3, k_rvq=2)
        assert out.shape == (3, c.L) and out.min() >= 0 and out.max() < 256
        assert calls == (pr.T if mode == "ar" else 2 + c.Q * 2), (
            f"{mode}: calls {calls}"
        )
    assert nparams(priors["ar"]) == nparams(priors["cf"]), "prior param mismatch"
    assert torch.equal(
        priors["ar"].stack.blocks[0].qkv.weight, priors["cf"].stack.blocks[0].qkv.weight
    ), "AR/CF init must be paired"
    pr = priors["ar"]
    tok = seq_tokens(tk, c, True, pr.lv)
    pos, ln = seq_feats(tk["order"], tk["length"], c, True, pr.lv)
    with torch.no_grad():
        full, _ = pr.run(tok, pos, ln, pr.T)
        past = None
        fts = pr.f.feats(pos, ln, pr.T)
        for t in range(pr.T):
            x = (
                (pr.f.bos.expand(4, 1, -1) + fts[:, :1])
                if t == 0
                else (pr.f.tok(tok[:, t - 1])[:, None] + fts[:, t : t + 1])
            )
            hh, past = pr.stack(x, past)
            inc = pr.head(hh)[:, -1]
            assert torch.allclose(inc, full[:, t], atol=1e-4), "KV cache mismatch"
    # boundary loss target is finite and differentiates
    assert torch.isfinite(
        F.binary_cross_entropy_with_logits(
            torch.zeros(4, c.L, device=DEVICE), torch.rand(4, c.L, device=DEVICE)
        )
    )
    # statistics helpers
    assert dual_step(0.5, 2.0, 1.0, 1.0, 8.0) == 1.5
    assert dual_step(7.9, 9.0, 1.0, 1.0, 8.0) == 8.0
    assert dual_step(0.1, 0.0, 1.0, 2.0, 8.0) == 0.0
    pw = {
        ("fixed", "ar"): np.array([2.0, 2.0]),
        ("fixed", "cf"): np.array([2.5, 2.5]),
        ("dynamic", "ar"): np.array([3.0, 3.0]),
        ("dynamic", "cf"): np.array([3.4, 3.4]),
    }
    assert abs(ixn(pw).mean() - 0.1) < 1e-12
    fake = json.loads(json.dumps({"rd": {0: 9.0, 1: 8.0}, "util": [0.5]}))
    assert normalize_run(dict(fake))["rd"][1] == 8.0
    assert canon(canon(asdict(c))) == canon(asdict(c))
    # resume: changed config or corpus sha refuses the old artifact
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "results.json")
        shas = {n: get_corpus(n)["sha"] for n in c.corpora}
        fake = dict(
            format=FORMAT,
            preset=PRESET,
            cfg=canon(asdict(c)),
            torch=torch.__version__,
            device=DEVICE,
            corpora={n: {"sha": shas[n]} for n in c.corpora},
        )
        with open(path, "w") as f:
            json.dump(fake, f)
        assert load_prev(c, path, shas) is not None
        fake["cfg"] = canon(asdict(replace(c, rho=c.rho + 1)))
        with open(path, "w") as f:
            json.dump(fake, f)
        assert load_prev(c, path, shas) is None
        fake["cfg"] = canon(asdict(c))
        fake["corpora"] = {n: {"sha": "deadbeef"} for n in c.corpora}
        with open(path, "w") as f:
            json.dump(fake, f)
        assert load_prev(c, path, shas) is None
    # BPE reference is deterministic and respects one boundary per group
    merges, ranks = learn_bpe(b"abababab" * 40, 8)
    assert merges and ranks
    bch = BpeChunker(replace(c, bpe_merges=8, bpe_bytes=10_000), "synseg").to(DEVICE)
    bnd = bch.boundaries(h, torch.randint(0, 256, (3, c.L), device=DEVICE))
    assert bnd.hard[:, 0].all() and bnd.hard.sum(1).eq(c.M).all()
    print("self-tests passed")


# %%
# ---------------- training / evaluation utilities ---------------------------------------------
def fit(model, step_fn, steps, lr, name, maintain_every=200, controller=None):
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    warm = max(1, min(100, steps // 5))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda t: (
            min(1.0, (t + 1) / warm)
            * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, t / steps))))
        ),
    )
    model.train()
    t0 = time.time()
    ema = None
    every = max(1, steps // 5)
    for step in range(steps):
        loss, st = step_fn(step)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{name} step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if hasattr(model, "maintain") and step % maintain_every == maintain_every - 1:
            model.maintain()
        if controller is not None:
            controller(step, st)
        v = float(loss.detach())
        ema = v if ema is None else 0.98 * ema + 0.02 * v
        if step % every == 0 or step == steps - 1:
            extra = " ".join(f"{k}={float(x):.3f}" for k, x in st.items())
            print(
                f"  [{name}] {step:5d}/{steps} loss~{ema:.3f} {extra} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    model.eval()
    return time.time() - t0


@torch.no_grad()
def collect(fn, corpus, L, n, bs=128):
    starts = make_starts(corpus, L, n)
    acc = {}
    for i in range(0, n, bs):
        b = min(bs, n - i)
        d = fn(windows(corpus["va"], starts[i : i + b], L))
        for k, v in d.items():
            if v.numel() != b:
                raise RuntimeError(
                    f"{k}: expected {b} per-window values, got {v.numel()}"
                )
            acc.setdefault(k, []).append(v.float().cpu().reshape(-1))
    return {k: torch.cat(v).numpy() for k, v in acc.items()}


@torch.no_grad()
def components(codec, prior, x):
    o = codec.encode(x, None)
    rec = (
        F.cross_entropy(
            codec.decode(o["zq"] + o["e1c"], o["g"], x.size(1)).reshape(-1, 256),
            x.reshape(-1),
            reduction="none",
        )
        .view(x.size(0), -1)
        .sum(1)
        / LN2
    )
    d = {"recon": rec}
    for k, v in prior.nll(codec.tokens_from(o)).items():
        d[k] = v / LN2
    d["prior"] = sum(v for k, v in d.items() if k != "recon")
    d["total"] = d["recon"] + d["prior"]
    return d


@torch.no_grad()
def rd_curve(codec, corpus, n=128):
    cfg = codec.cfg
    x = windows(corpus["va"], make_starts(corpus, cfg.L, n), cfg.L)
    out = {}
    for k in range(cfg.Q + 1):
        keep = (torch.arange(cfg.Q, device=DEVICE)[None] < k).expand(n, cfg.Q)
        o = codec.encode(x, keep)
        out[k] = float(
            F.cross_entropy(
                codec.decode(o["zq"] + o["e1c"], o["g"], cfg.L).reshape(-1, 256),
                x.reshape(-1),
            )
            / LN2
        )
    o = codec.encode(x, None)
    ppl, util = [], []
    for q in range(cfg.Q):
        h = torch.bincount(o["codes"][..., q].reshape(-1), minlength=cfg.K).float()
        p = h / h.sum()
        ppl.append(float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum())))
        util.append(float((h > 0).float().mean()))
    return out, ppl, util


@torch.no_grad()
def alignment(codec, corpus, n=256):
    cfg = codec.cfg
    x = windows(corpus["va"], make_starts(corpus, cfg.L, n), cfg.L)
    o = codec.encode(x, None)
    order = o["g"]["order"].cpu().numpy()
    counts = o["g"]["counts"].cpu().numpy()
    xb = x.cpu().numpy()
    hits = tot = grid_hits = 0
    for i in range(n):
        ons = onsets(corpus["name"], xb[i])
        for j in order[i, 1 : counts[i]]:
            hits += int(ons[j])
            tot += 1
        for j in range(cfg.stride, cfg.L, cfg.stride):
            grid_hits += int(ons[j])
    return hits / max(tot, 1), grid_hits / max(n * (cfg.M - 1), 1)


def timed(fn):
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    r = fn()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return r, time.time() - t0


def show(t, lim=140):
    return bytes(t.tolist())[:lim].decode("latin-1").replace("\n", "⏎")


def text_stats(ts):
    bg = [t[i : i + 2] for t in ts for i in range(len(t) - 1)]
    return len(set(bg)) / max(len(bg), 1), 1 - len(set(bg)) / max(len(bg), 1)


# %%
# ---------------- artifact store and fingerprints ---------------------------------------------
def load_prev(cfg, path, shas):
    if not os.path.exists(path):
        return None
    try:
        old = json.load(open(path))
    except Exception:
        return None
    if old.get("format") != FORMAT or old.get("preset") != PRESET:
        return None
    if canon(old.get("cfg")) != canon(asdict(cfg)):
        return None
    if old.get("torch") != torch.__version__ or old.get("device") != DEVICE:
        return None
    for cname, sha in shas.items():
        if old.get("corpora", {}).get(cname, {}).get("sha") != sha:
            return None
    return old


def normalize_run(r):
    r["rd"] = {int(k): float(v) for k, v in r.get("rd", {}).items()}
    return r


def ixn(pw):
    dyn = next(name for name, _ in pw if name != "fixed")
    return (pw[("fixed", "cf")] - pw[(dyn, "cf")]) - (
        pw[("fixed", "ar")] - pw[(dyn, "ar")]
    )


# %%
# ---------------- experiment -----------------------------------------------------------------
def run_all(cfg, base_out, resume):
    corpora = {n: get_corpus(n) for n in cfg.corpora}
    fp = fingerprint(cfg)
    out = os.path.join(base_out, f"p{cfg.phase}", f"{PRESET}-{fp[:10]}")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "results.json")

    def save():
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(RES, f)
        os.replace(tmp, path)
        latest = os.path.join(base_out, "latest")
        try:
            if os.path.islink(latest) or os.path.exists(latest):
                os.remove(latest)
            os.symlink(os.path.relpath(out, base_out), latest)
        except OSError:
            pass

    def ckpt(name):
        return os.path.join(out, name)

    def load_sd(p):
        try:
            return torch.load(p, map_location=DEVICE, weights_only=True)
        except Exception:
            return None

    def load_json(p):
        try:
            return json.load(open(p))
        except Exception:
            return None

    old = (
        load_prev(cfg, path, {n: c["sha"] for n, c in corpora.items()})
        if resume
        else None
    )
    RES = dict(
        format=FORMAT,
        cfg=canon(asdict(cfg)),
        preset=PRESET,
        phase=cfg.phase,
        device=DEVICE,
        torch=torch.__version__,
        started=datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z"),
        gpu=torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
        runs=list(old["runs"]) if old else [],
        corpora={},
        unigram={},
    )

    def put(row):
        def same(r):
            return all(
                r.get(k) == row[k]
                for k in ("corpus", "kind", "seed", "chunk", "prior", "k")
            )

        RES["runs"] = [r for r in RES["runs"] if not same(r)] + [row]

    def has_row(cname, seed, chunk, prior, k):
        return any(
            r["corpus"] == cname
            and r["seed"] == seed
            and r["chunk"] == chunk
            and r["prior"] == prior
            and r["k"] == k
            and r["kind"] == "concept"
            for r in RES["runs"]
        )

    def has_gpt(cname, seed):
        return any(
            r["corpus"] == cname and r["seed"] == seed and r["kind"] == "gpt"
            for r in RES["runs"]
        )

    def block_done(cname, seed, chunk):
        need = {("ar", None)} | {("cf", k) for k in cfg.cf_k_list}
        got = {
            (r["prior"], r["k"])
            for r in RES["runs"]
            if r["corpus"] == cname
            and r["seed"] == seed
            and r["chunk"] == chunk
            and r["kind"] == "concept"
        }
        return need <= got

    def achieved_fixed(cname, seed):
        for r in RES["runs"]:
            if (
                r["corpus"] == cname
                and r["seed"] == seed
                and r["chunk"] == "fixed"
                and r["kind"] == "concept"
            ):
                return float(r["achieved_rate"])
        return None

    print(f"resume={bool(old)} completed_runs={len(RES['runs'])}")
    for cname in cfg.corpora:
        corpus = corpora[cname]
        uni = unigram_bits(corpus)
        RES["corpora"][cname] = {k: corpus[k] for k in ("sha", "nbytes", "name")}
        if (
            old
            and old.get("corpora", {}).get(cname, {}).get("sha") == corpus["sha"]
            and cname in old.get("unigram", {})
        ):
            RES["unigram"][cname] = old["unigram"][cname]
        else:
            uw = (
                collect(
                    lambda x: {"total": uni[x].sum(1)}, corpus, cfg.L, cfg.eval_windows
                )["total"]
                / cfg.L
            )
            RES["unigram"][cname] = float(uw.mean())
            print(f"  unigram baseline {uw.mean():.3f} bpb")
        print(
            f"\n######## corpus={cname} sha={corpus['sha'][:12]} unigram={RES['unigram'][cname]:.3f} bpb"
        )
        for seed in cfg.seeds:
            gk = ckpt(f"gpt_{cname}_s{seed}.pt")
            gsd = load_sd(gk)
            if old and has_gpt(cname, seed) and gsd is not None:
                gpt = ByteGPT(cfg).to(DEVICE)
                gpt.load_state_dict(gsd)
                gpt.eval()
            else:
                torch.manual_seed(1000 * seed + 1)
                np.random.seed(seed)
                gpt = ByteGPT(cfg).to(DEVICE)
                g_rng = np.random.RandomState(1000 * seed + 3)
                ts = fit(
                    gpt,
                    lambda s: (gpt.loss(train_batch(corpus, cfg.L, cfg.bs, g_rng)), {}),
                    cfg.base_steps,
                    cfg.base_lr,
                    f"gpt {cname} s{seed}",
                )
                torch.save(gpt.state_dict(), gk)
                gb = (
                    collect(
                        lambda x: {"total": gpt.bits_per_window(x, uni)},
                        corpus,
                        cfg.L,
                        cfg.eval_windows,
                    )["total"]
                    / cfg.L
                )
                torch.manual_seed(1000 * seed + 50)
                (smp, calls), dt = timed(lambda: gpt.sample(cfg.sample_n, cfg.L, uni))
                ref = float((gpt.bits_per_window(smp, uni) / cfg.L).mean())
                d2, rep = text_stats([bytes(r.tolist()) for r in smp])
                put(
                    dict(
                        corpus=cname,
                        kind="gpt",
                        seed=seed,
                        chunk="-",
                        prior="ar",
                        k=None,
                        params=nparams(gpt),
                        bpb=float(gb.mean()),
                        per_window=gb.tolist(),
                        calls=calls,
                        sample_sec=dt,
                        ref_bpb=ref,
                        distinct2=d2,
                        repeat=rep,
                        train_sec=ts,
                        samples=[show(smp[i]) for i in range(min(3, len(smp)))],
                    )
                )
                print(f"  byte GPT val bpb={gb.mean():.4f}")
                save()
            aux = None
            if "surprise" in cfg.chunkers:
                sk = ckpt(f"surprise_{cname}_s{seed}.pt")
                ssd = load_sd(sk)
                sc = replace(cfg, d=cfg.dsur, heads=4, base_layers=cfg.surprise_layers)
                torch.manual_seed(1000 * seed + 500)
                aux = ByteGPT(sc).to(DEVICE)
                if resume and ssd is not None:
                    aux.load_state_dict(ssd)
                else:
                    a_rng = np.random.RandomState(1000 * seed + 501)
                    fit(
                        aux,
                        lambda s: (
                            aux.loss(train_batch(corpus, cfg.L, cfg.bs, a_rng)),
                            {},
                        ),
                        cfg.surprise_steps,
                        cfg.surprise_lr,
                        f"surprise {cname} s{seed}",
                    )
                    torch.save(aux.state_dict(), sk)
                aux.eval()
                for p in aux.parameters():
                    p.requires_grad_(False)
            for chunker in cfg.chunkers:
                chunk = chunker
                if old and block_done(cname, seed, chunk):
                    print(f"  {chunk}: complete (resumed)")
                    continue
                ck = ckpt(f"codec_{cname}_{chunk}_s{seed}.pt")
                mk = ckpt(f"codec_meta_{cname}_{chunk}_s{seed}.json")
                csd = load_sd(ck)
                meta = load_json(mk)
                rho = cfg.rho
                if cfg.phase >= 1 and cfg.match_dynamic and chunk != "fixed":
                    af = achieved_fixed(cname, seed)
                    if af is not None:
                        rho = af
                if resume and csd is not None and meta is not None:
                    codec = Codec(cfg, chunk, cname, aux).to(DEVICE)
                    codec.load_state_dict(csd)
                    codec.rvq._inited = codec.vq1._inited = True
                    ctl = dict(lam=meta["lam"], ema=None)
                    tsc = float(meta.get("train_sec", 0.0))
                else:
                    torch.manual_seed(1000 * seed + 2)
                    np.random.seed(seed)
                    c_rng = np.random.RandomState(1000 * seed + 3)
                    codec = Codec(cfg, chunk, cname, aux).to(DEVICE)
                    ctl = dict(
                        lam=0.1,
                        ema=None,
                        fixed=cfg.lam_fixed if cfg.lam_fixed is not None else None,
                    )
                    if cfg.lam_fixed is not None:
                        ctl["lam"] = cfg.lam_fixed

                    def controller(step, st, ctl=ctl, rho=rho):
                        r = float(st["rate_bpb"])
                        ctl["ema"] = (
                            r if ctl["ema"] is None else 0.99 * ctl["ema"] + 0.01 * r
                        )
                        if ctl["fixed"] is None and step % 50 == 49:
                            ctl["lam"] = dual_step(
                                ctl["lam"], ctl["ema"], rho, cfg.lam_eta, cfg.lam_max
                            )

                    tsc = fit(
                        codec,
                        lambda s: codec.loss(
                            train_batch(corpus, cfg.L, cfg.bs, c_rng), ctl["lam"]
                        ),
                        cfg.codec_steps,
                        cfg.codec_lr,
                        f"codec {chunk} {cname} s{seed}",
                        controller=controller,
                    )
                    torch.save(codec.state_dict(), ck)
                    with open(mk, "w") as f:
                        json.dump(dict(lam=ctl["lam"], train_sec=tsc, rho=rho), f)
                codec.eval()
                for p in codec.parameters():
                    p.requires_grad_(False)
                rd, ppl, util = rd_curve(codec, corpus)
                if cname == "shakespeare":
                    al, grid_al = None, None
                else:
                    al, grid_al = alignment(codec, corpus)
                achieved = float(
                    collect(
                        lambda x: {"r": codec.rate_bits(x)},
                        corpus,
                        cfg.L,
                        cfg.eval_windows,
                    )["r"].mean()
                    / cfg.L
                )
                keys = sorted(rd)
                rd_mono = all(rd[a] >= rd[b] - 1e-6 for a, b in zip(keys, keys[1:]))
                codec_healthy = min(util) >= cfg.health_util and rd_mono
                rate_ok = abs(achieved - rho) <= cfg.match_rate
                print(
                    f"  codec {chunk}: rho={rho} achieved={achieved:.3f} lam={ctl['lam']:.2f} "
                    f"RD={[round(rd[k], 3) for k in rd]} util={[round(u, 2) for u in util]}"
                    + (
                        f" align={al:.3f} (grid {grid_al:.3f})"
                        if al is not None
                        else ""
                    )
                )
                if not codec_healthy:
                    print(
                        f"  WARNING: codec unhealthy (util={min(util):.2f}, "
                        f"RD monotone={rd_mono}); stopping the arm before priors"
                    )
                    put(
                        dict(
                            corpus=cname,
                            kind="codec",
                            seed=seed,
                            chunk=chunk,
                            prior="-",
                            k=None,
                            healthy=False,
                            params=nparams(codec),
                            lam=ctl["lam"],
                            train_sec=tsc,
                            util=util,
                            rd={int(kk): float(v) for kk, v in rd.items()},
                            achieved_rate=achieved,
                            align=al,
                            grid_align=grid_al,
                        )
                    )
                    save()
                    continue
                if not rate_ok:
                    print(
                        f"  WARNING: achieved rate mismatch "
                        f"|achieved-rho|={abs(achieved - rho):.2f}; G4 will fail"
                    )
                recon_ref = None
                for mode in ("ar", "cf"):
                    pk = ckpt(f"prior_{cname}_{chunk}_{mode}_s{seed}.pt")
                    psd = load_sd(pk)
                    torch.manual_seed(1000 * seed + 10 + int(codec.dyn))
                    prior = Prior(cfg, codec.dyn, mode).to(DEVICE)
                    if resume and psd is not None:
                        prior.load_state_dict(psd)
                        tsp = 0.0
                    else:
                        torch.manual_seed(1000 * seed + 11 + int(codec.dyn))
                        p_rng = np.random.RandomState(1000 * seed + 3)
                        tsp = fit(
                            prior,
                            lambda s: (
                                prior.train_loss(
                                    codec.tokenize(
                                        train_batch(corpus, cfg.L, cfg.bs, p_rng)
                                    )
                                ),
                                {},
                            ),
                            cfg.prior_steps,
                            cfg.prior_lr,
                            f"{chunk}/{mode} {cname} s{seed}",
                        )
                        torch.save(prior.state_dict(), pk)
                    prior.eval()
                    comps = collect(
                        lambda x: components(codec, prior, x),
                        corpus,
                        cfg.L,
                        cfg.eval_windows,
                    )
                    if mode == "ar":
                        recon_ref = comps["recon"]
                    else:
                        assert np.allclose(recon_ref, comps["recon"], atol=1e-4), (
                            "SHARED-CODEC VIOLATION"
                        )
                    tot = comps["total"] / cfg.L
                    budgets = [None] if mode == "ar" else list(cfg.cf_k_list)
                    for k in budgets:
                        if old and has_row(cname, seed, chunk, mode, k):
                            continue
                        torch.manual_seed(
                            1000 * seed
                            + 100
                            + int(codec.dyn) * 10
                            + (mode == "cf") * 5
                            + (k or 0)
                        )
                        (smp, calls), dt = timed(
                            lambda: prior.sample(codec, cfg.sample_n, k_rvq=k or 1)
                        )
                        refpw = (gpt.bits_per_window(smp, uni) / cfg.L).cpu().numpy()
                        ref = float(refpw.mean())
                        d2, rep = text_stats([bytes(r.tolist()) for r in smp])
                        put(
                            dict(
                                corpus=cname,
                                kind="concept",
                                seed=seed,
                                chunk=chunk,
                                prior=mode,
                                k=k,
                                params=nparams(codec) + nparams(prior),
                                bpb=float(tot.mean()),
                                per_window=tot.tolist(),
                                comps={
                                    kk: float(v.mean() / cfg.L)
                                    for kk, v in comps.items()
                                },
                                calls=calls,
                                sample_sec=dt,
                                ref_bpb=ref,
                                ref_per_window=refpw.tolist(),
                                distinct2=d2,
                                repeat=rep,
                                train_sec=tsc + tsp,
                                rd={int(kk): float(v) for kk, v in rd.items()},
                                code_ppl=ppl,
                                util=util,
                                lam=ctl["lam"],
                                achieved_rate=achieved,
                                align=al,
                                grid_align=grid_al,
                                samples=[show(smp[i]) for i in range(min(3, len(smp)))],
                            )
                        )
                        print(
                            f"  => {chunk}+{mode}"
                            + (f"(k={k})" if k else "")
                            + f" bound={tot.mean():.4f} calls={calls} ref={ref:.3f} distinct2={d2:.3f}"
                        )
                        save()
    RES["finished"] = (
        datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")
    )
    save()
    print("\nsaved", path)
    return RES


# %%
# ---------------- statistics and gates --------------------------------------------------------
def ms(v):
    v = np.array(v, float)
    return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan"))


def boot(x, reps=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (reps, len(x)))
    m = x[idx].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def one_per_seed(rr):
    d = {}
    for r in rr:
        d.setdefault(r["seed"], r)
    return [d[k] for k in sorted(d)]


def pick(runs, **kw):
    return [r for r in runs if all(r.get(k) == v for k, v in kw.items())]


def cell_runs(runs, cname, chunk, prior, k=None):
    return pick(runs, corpus=cname, kind="concept", chunk=chunk, prior=prior, k=k)


def primary_k(cfg):
    return min(cfg.cf_k_list)


def window_cell(runs, cfg, cname, chunk, prior, seed, k=None):
    rr = pick(
        runs, corpus=cname, kind="concept", chunk=chunk, prior=prior, seed=seed, k=k
    )
    return np.array(rr[0]["per_window"], float) if rr else None


def ref_window(runs, cfg, cname, chunk, prior, seed, k=None):
    rr = pick(
        runs, corpus=cname, kind="concept", chunk=chunk, prior=prior, seed=seed, k=k
    )
    return np.array(rr[0]["ref_per_window"], float) if rr else None


def interaction_for(runs, cfg, cname, seed, chunk, k):
    pw = {}
    for fixed, prior in (
        ("fixed", "ar"),
        ("fixed", "cf"),
        (chunk, "ar"),
        (chunk, "cf"),
    ):
        a = window_cell(
            runs, cfg, cname, fixed, prior, seed, k if prior == "cf" else None
        )
        if a is None:
            return None
        pw[(fixed, prior)] = a
    return ixn(pw) / cfg.L


def health(r, cfg):
    rd = r.get("rd", {})
    keys = sorted(rd)
    mono = all(rd[a] >= rd[b] - 1e-6 for a, b in zip(keys, keys[1:]))
    return bool(mono and min(r.get("util", [0.0])) >= cfg.health_util)


def gate_table(runs, cfg, cname):
    k = primary_k(cfg)
    seeds = sorted({r["seed"] for r in runs if r["corpus"] == cname})
    gpt = one_per_seed(pick(runs, corpus=cname, kind="gpt"))
    ar_rows = pick(runs, corpus=cname, kind="concept", chunk="fixed", prior="ar")
    q_ar = ms([r["ref_bpb"] for r in ar_rows])[0] if ar_rows else float("nan")
    total_rows = pick(
        runs, corpus=cname, kind="concept", chunk="fixed", prior="cf", k=k
    )
    totals = [r["bpb"] for r in total_rows]
    gpt_bpb = ms([r["bpb"] for r in gpt])[0] if gpt else float("nan")
    E1 = bool(
        totals and np.mean(totals) < runs_meta(runs, cname)["unigram"] - cfg.p1a_margin
    )
    E2 = bool(totals and np.mean(totals) <= gpt_bpb + cfg.p1b_margin)
    fixed_rows = pick(runs, corpus=cname, kind="concept", chunk="fixed", prior="ar")
    G3 = bool(fixed_rows and all(health(r, cfg) for r in fixed_rows))
    G4 = {}
    for chunk in cfg.chunkers:
        if chunk == "fixed":
            continue
        af = (
            np.mean(
                [
                    r["achieved_rate"]
                    for r in pick(
                        runs, corpus=cname, kind="concept", chunk="fixed", prior="ar"
                    )
                ]
            )
            if fixed_rows
            else float("nan")
        )
        ad = (
            np.mean(
                [
                    r["achieved_rate"]
                    for r in pick(
                        runs, corpus=cname, kind="concept", chunk=chunk, prior="ar"
                    )
                ]
            )
            if pick(runs, corpus=cname, kind="concept", chunk=chunk, prior="ar")
            else float("nan")
        )
        G4[chunk] = bool(abs(af - ad) <= cfg.match_rate)
    e3 = []
    for seed in seeds:
        for chunk in cfg.chunkers:
            u = util_pareto(runs, cfg, cname, chunk, seed)
            if u is not None and not np.isnan(u["ratio"]):
                e3.append(u["ratio"] >= cfg.p2_calls)
    E3 = bool(e3 and all(e3))
    p2b = []
    for chunk in cfg.chunkers:
        if chunk == "fixed":
            continue
        for seed in seeds:
            for k in cfg.cf_k_list:
                fr = pick(
                    runs,
                    corpus=cname,
                    kind="concept",
                    chunk="fixed",
                    prior="cf",
                    seed=seed,
                    k=k,
                )
                dr = pick(
                    runs,
                    corpus=cname,
                    kind="concept",
                    chunk=chunk,
                    prior="cf",
                    seed=seed,
                    k=k,
                )
                if fr and dr:
                    p2b.append(dr[0]["ref_bpb"] - fr[0]["ref_bpb"])
    P2b = bool(p2b and np.mean(p2b) <= cfg.p2_beats)
    return dict(
        E1=E1,
        E2=E2,
        E3=E3,
        P2b=P2b,
        p2b=float(np.mean(p2b)) if p2b else float("nan"),
        G3=G3,
        G4=G4,
        total=float(np.mean(totals)) if totals else float("nan"),
        unigram=runs_meta(runs, cname)["unigram"],
        gpt=gpt_bpb,
        q_ar=q_ar,
        seeds=seeds,
    )


def runs_meta(runs, cname):
    return META.get(
        cname, dict(unigram=float("nan"), gpt=float("nan"), nbytes=0, sha="")
    )


def util_pareto(runs, cfg, cname, chunk, seed):
    ar = pick(runs, corpus=cname, kind="concept", chunk=chunk, prior="ar", seed=seed)
    if not ar:
        return None
    a = ar[0]
    cf_rows = pick(
        runs, corpus=cname, kind="concept", chunk=chunk, prior="cf", seed=seed
    )
    if not cf_rows:
        return None
    q_ar = float(a["ref_bpb"])
    ok = [r for r in cf_rows if r["ref_bpb"] <= q_ar + cfg.p2_qeps]
    if not ok:
        return dict(
            ratio=float("nan"), calls_cf=None, q_ar=q_ar, quality_cf=float("nan")
        )
    best = min(ok, key=lambda r: r["calls"])
    return dict(
        ratio=float(a["calls"]) / max(best["calls"], 1),
        calls_cf=best["calls"],
        q_ar=q_ar,
        quality_cf=float(best["ref_bpb"]),
        calls_ar=int(a["calls"]),
        k=best["k"],
    )


def ratio_boot(runs, cfg, cname, chunk, seed, reps=400):
    ar = pick(runs, corpus=cname, kind="concept", chunk=chunk, prior="ar", seed=seed)
    cfs = pick(runs, corpus=cname, kind="concept", chunk=chunk, prior="cf", seed=seed)
    if not ar or not cfs:
        return None
    arq = np.array(ar[0]["ref_per_window"], float)
    cfq = {r["k"]: np.array(r["ref_per_window"], float) for r in cfs}
    if any(len(v) != len(arq) for v in cfq.values()):
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arq), (reps, len(arq)))
    out = []
    for b in idx:
        qa = arq[b].mean()
        best = None
        for k, v in cfq.items():
            qc = v[b].mean()
            if qc <= qa + cfg.p2_qeps:
                calls = [r["calls"] for r in cfs if r["k"] == k][0]
                if best is None or calls < best[0]:
                    best = (calls, k)
        if best is not None:
            out.append(ar[0]["calls"] / max(best[0], 1))
    if not out:
        return None
    return (
        float(np.quantile(out, 0.025)),
        float(np.quantile(out, 0.975)),
        float(np.median(out)),
    )


def claim_verdicts(runs, cfg, cname):
    k = primary_k(cfg)
    seeds = sorted({r["seed"] for r in runs if r["corpus"] == cname})
    out = {}
    for chunk in ("oracle", "rate-only", "surprise", "oracle-sup"):
        if chunk not in cfg.chunkers:
            continue
        means = []
        per_seed = []
        allpw = []
        for seed in seeds:
            iv = interaction_for(runs, cfg, cname, seed, chunk, k)
            if iv is None:
                continue
            means.append(float(iv.mean()))
            per_seed.append(iv)
            allpw.append(iv)
        if not means:
            continue
        m, s = ms(means)
        if len(allpw) >= 1:
            lo, hi = boot(np.concatenate(allpw)) if len(allpw) > 1 else boot(allpw[0])
        else:
            lo = hi = float("nan")
        out[chunk] = dict(mean=m, sem=s, lo=lo, hi=hi, n=len(means), per_seed=means)
    return out


def write_report(runs, cfg, RES, path):
    lines = []
    add = lines.append
    add("# H-NCM v5 report\n")
    add(
        f"preset `{RES['preset']}` phase `{RES['phase']}` device `{RES['device']}` torch `{RES['torch']}`\n"
    )
    add(f"started `{RES['started']}` finished `{RES.get('finished', '—')}`\n")
    add(f"cfg fingerprint `{fingerprint(cfg)[:16]}`\n")
    add("\n## Claims\n")
    add(
        "P1 absolute compression; P2 parallelism dividend; P3 boundary mechanism.\n"
        "All three are reported, including negative results. Thresholds come from `Cfg`.\n"
    )
    if not DECISIVE:
        add(
            "\n> Diagnostic only: preset != full. Controls are decisive only on the full preset.\n"
        )
    for cname in cfg.corpora:
        if cname not in RES.get("corpora", {}):
            continue
        meta = RES["corpora"][cname]
        add(f"\n## {cname} (sha {meta['sha'][:12]}, {meta['nbytes']:,} bytes)\n")
        add(f"unigram `{RES['unigram'].get(cname, float('nan')):.3f}` bpb\n")
        gpt_rows = one_per_seed(pick(runs, corpus=cname, kind="gpt"))
        gpt_bpb = ms([r["bpb"] for r in gpt_rows])[0] if gpt_rows else float("nan")
        add(f"byte GPT `{gpt_bpb:.3f}` bpb\n")
        add("\n| cell | bound bpb | recon | c1 | bnd | codes | rate | align |")
        add("|---|---:|---:|---:|---:|---:|---:|---:|")
        for chunk in cfg.chunkers:
            for prior in ("ar", "cf"):
                rr = one_per_seed(
                    pick(runs, corpus=cname, kind="concept", chunk=chunk, prior=prior)
                )
                if not rr:
                    add(f"| {chunk}+{prior} | incomplete | | | | | | |")
                    continue
                m, _ = ms([r["bpb"] for r in rr])

                def cm(key):
                    return np.mean([r["comps"].get(key, 0.0) for r in rr])

                codes = sum(
                    cm(key)
                    for key in rr[0]["comps"]
                    if key.startswith("r") and key != "recon"
                )
                al = np.mean(
                    [r["align"] for r in rr if r["align"] is not None] or [float("nan")]
                )
                add(
                    f"| {chunk}+{prior} | {m:.3f} | {cm('recon'):.3f} | {cm('c1'):.3f} | "
                    f"{cm('bnd'):.3f} | {codes:.3f} | {cm('rate'):.3f} | {al:.3f} |"
                )
        chs = pick(runs, corpus=cname, kind="codec")
        if chs:
            add("\n### Codec health (arms stopped before priors)\n")
            add("| chunker | util | RD | achieved | align |")
            add("|---|---:|---|---:|---:|")
            for r in chs:
                add(
                    f"| {r['chunk']} | {[round(u, 2) for u in r['util']]} | "
                    f"{[round(r['rd'][k], 3) for k in sorted(r['rd'])]} | "
                    f"{r['achieved_rate']:.3f} | "
                    f"{r['align'] if r['align'] is not None else float('nan'):.3f} |"
                )
        G = gate_table(runs, cfg, cname)
        add(
            f"\nGates: E1 `{G['E1']}`, E2 `{G['E2']}`, E3 `{G['E3']}`, "
            f"P2b `{G['P2b']}` (mean {G['p2b']:+.3f}), G3 `{G['G3']}`, G4 `{G['G4']}`\n"
        )
        add(
            f"total(fixed+CF) `{G['total']:.3f}` bpb; unigram `{G['unigram']:.3f}`; "
            f"byte GPT `{G['gpt']:.3f}`; AR sample `{G['q_ar']:.3f}`\n"
        )
        V = claim_verdicts(runs, cfg, cname)
        add("\n### Interaction I (primary k = %d)\n" % primary_k(cfg))
        add("| arm | I mean | SEM | CI low | CI high | n |")
        add("|---|---:|---:|---:|---:|---:|")
        for chunk, v in V.items():
            add(
                f"| {chunk} | {v['mean']:+.4f} | {v['sem']:.4f} | {v['lo']:+.4f} | "
                f"{v['hi']:+.4f} | {v['n']} |"
            )
        add("\n### Mechanism verdicts\n")
        add(mechanism_text(runs, cname, V, cfg))
        add("\n### Utility Pareto (sample quality, sequential calls)\n")
        add("| arm | calls | ref bpb | ratio |")
        add("|---|---:|---:|---:|")
        for chunk in cfg.chunkers:
            for seed in G["seeds"]:
                u = util_pareto(runs, cfg, cname, chunk, seed)
                if u is None:
                    continue
                add(
                    f"| {chunk}+CF s{seed} | {u['calls_cf']} | {u['quality_cf']:.3f} | {u['ratio']:.2f} |"
                )
                rb = ratio_boot(runs, cfg, cname, chunk, seed)
                if rb is not None:
                    add(f"| {chunk}+CF s{seed} CI | | | [{rb[0]:.2f}, {rb[1]:.2f}] |")
        add("\n### Decision\n")
        add(decision_text(runs, cfg, cname, V, G))
    add("\n## Provenance\n")
    add(
        "```json\n"
        + json.dumps(
            {
                k: RES[k]
                for k in (
                    "format",
                    "preset",
                    "phase",
                    "device",
                    "torch",
                    "gpu",
                    "started",
                    "finished",
                )
            },
            indent=2,
        )
        + "\n```\n"
    )
    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return text


def mechanism_text(runs, cname, V, cfg):
    e4, e4n, e5 = V.get("oracle"), V.get("rate-only"), V.get("surprise")
    if cname == "synseg":
        out = []
        if e4 is None:
            out.append("E4 oracle: missing; mechanism cannot be evaluated.")
        else:
            out.append(
                f"E4 oracle: `{e4['lo'] > 0}` (I={e4['mean']:+.4f}, "
                f"CI [{e4['lo']:+.4f}, {e4['hi']:+.4f}])."
            )
        if e4n is None:
            out.append("E4n rate-only: missing.")
        else:
            ok = e4n["lo"] <= 0 <= e4n["hi"]
            out.append(
                f"E4n rate-only null: `{ok}` (I={e4n['mean']:+.4f}, "
                f"CI [{e4n['lo']:+.4f}, {e4n['hi']:+.4f}])."
            )
        if e5 is None:
            out.append("E5 boundary: missing (run phase 2).")
        elif e4 is not None:
            bar = max(cfg.p3_alpha * e4["mean"], 2 * e5["sem"])
            ok = e5["mean"] >= bar and e5["lo"] > 0
            out.append(
                f"E5 no-oracle >= max({cfg.p3_alpha}*I_oracle, 2*SEM): `{ok}` "
                f"(I={e5['mean']:+.4f}, bar {bar:+.4f})."
            )
        sp = pick(runs, corpus=cname, kind="concept", chunk="surprise", prior="ar")
        fx = pick(runs, corpus=cname, kind="concept", chunk="fixed", prior="ar")
        if sp and fx:
            asp = np.mean(
                [r["align"] for r in sp if r["align"] is not None] or [float("nan")]
            )
            afx = np.mean(
                [r["align"] for r in fx if r["align"] is not None] or [float("nan")]
            )
            out.append(
                f"P3d align(no-oracle) `{asp:.3f}` vs grid `{afx:.3f}`: "
                f"`{bool(asp >= afx + cfg.p3_align)}`."
            )
        return "\n".join(out)
    if cname == "syngrid":
        if not V:
            return "Negative control has no arm."
        bad = [name for name, v in V.items() if not (v["lo"] <= 0 <= v["hi"])]
        if bad:
            return (
                f"NEGATIVE CONTROL FAILED for {bad}: the I CI excludes 0 on syngrid. "
                "The boundary effect is a rate-distortion shift or noise."
            )
        return (
            "Negative control holds: all arms' I CIs cover 0. "
            "The content-boundary mechanism is not a rate-distortion shift."
        )
    return (
        "Real corpus: report P1 and P2 only. Oracle onsets are forbidden on "
        "shakespeare, so P3 is not claimed here."
    )


def decision_text(runs, cfg, cname, V, G):
    p2 = (
        f"P2a calls ratio `{G['E3']}`; P2b dynamic vs fixed at equal calls "
        f"`{G['P2b']}` (mean {G['p2b']:+.3f})."
    )
    if cname == "synseg":
        ce = V.get("oracle")
        cn = V.get("rate-only")
        cb = V.get("surprise")
        if ce is None:
            return (
                f"Phase 0 oracle arm missing; mechanism claim cannot be evaluated. {p2}"
            )
        if ce["hi"] <= 0:
            return (
                f"Oracle ceiling is zero (E4 fails; I={ce['mean']:+.4f}). No no-oracle arm "
                f"can pass P3. Report P1/P2 if E1/E3 hold; stop router work. {p2}"
            )
        if cn is not None and cn["lo"] <= 0 <= cn["hi"]:
            return (
                f"E4 passes and the rate-only null covers 0 (expected). "
                f"The boundary arm is the deployable test; check E5. {p2}"
            )
        if cb is not None and cb["lo"] > 0:
            return (
                f"Boundary arm positive; check alignment and the syngrid null before "
                f"claiming P3. {p2}"
            )
        return (
            f"Oracle helps but the no-oracle boundary arm does not pass E5; "
            f"report the ceiling and the deployable negative. {p2}"
        )
    if cname == "syngrid":
        cn = V.get("rate-only")
        cb = V.get("surprise")
        bad = [name for name, v in V.items() if not (v["lo"] <= 0 <= v["hi"])]
        if bad:
            return (
                f"NEGATIVE CONTROL FAILED for {bad}: the I CI excludes 0 on syngrid. "
                "The boundary effect is a rate-distortion shift or noise."
            )
        return (
            "Negative control holds (I covers 0). The content-boundary mechanism is not a "
            f"rate-distortion shift. {p2}"
            if cn or cb
            else f"Negative control has no arm. {p2}"
        )
    return (
        f"Real corpus: P1 `{G['E1']}`, P2a `{G['E3']}`, P2b `{G['P2b']}`. "
        "Oracle onsets are forbidden on shakespeare, so P3 is not claimed here."
    )


def PICK_LATEST():
    return os.path.join("runs", "latest", "results.json")


# %%
# ---------------- HPO (stage 1 fixed codec) ---------------------------------------------------
def run_hpo(cfg, base_out, trials, stage=1, seed=0):
    if stage != 1:
        raise SystemExit("only stage 1 is implemented")
    rng = random.Random(seed)
    space = dict(
        stride=[4, 8, 16],
        K=[8, 16, 32, 64],
        Q=[1, 2, 3],
        d=[64, 96, 128],
        eh_layers=[1, 2, 3],
        beta=[0.1, 0.25, 1.0],
        lam_fixed=[0, 0.1, 0.25, 0.5, 1.0],
        level_dropout=[False, True],
    )
    fp = fingerprint(cfg)
    out = os.path.join(base_out, "hpo", f"stage{stage}-{fp[:10]}")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "hpo.json")
    corpus = get_corpus("synseg")
    uni = unigram_bits(corpus)
    unigram_bpb = float(
        collect(lambda x: {"total": uni[x].sum(1)}, corpus, cfg.L, cfg.eval_windows)[
            "total"
        ].mean()
        / cfg.L
    )
    rows = []
    n0 = max(1, trials)
    for t in range(n0):
        ov = {k: rng.choice(v) for k, v in space.items()}
        tcfg = replace(
            cfg,
            phase=1,
            corpora=("synseg",),
            chunkers=("fixed",),
            seeds=(0,),
            codec_steps=max(20, cfg.codec_steps // 2),
            lam_fixed=ov.pop("lam_fixed"),
        )
        tcfg = replace(tcfg, **ov)
        torch.manual_seed(1000 + t)
        codec = Codec(tcfg, "fixed", "synseg").to(DEVICE)
        rng_b = np.random.RandomState(1000 + t)
        ctl = dict(
            lam=tcfg.lam_fixed if tcfg.lam_fixed is not None else 0.1,
            ema=None,
            fixed=True,
        )
        fit(
            codec,
            lambda s: codec.loss(
                train_batch(corpus, tcfg.L, tcfg.bs, rng_b), ctl["lam"]
            ),
            tcfg.codec_steps,
            tcfg.codec_lr,
            f"hpo t{t}",
            maintain_every=max(1, tcfg.codec_steps // 5),
        )
        codec.eval()
        with torch.no_grad():
            for p in codec.parameters():
                p.requires_grad_(False)
            rd, ppl, util = rd_curve(codec, corpus)
            achieved = float(
                collect(
                    lambda x: {"r": codec.rate_bits(x)},
                    corpus,
                    tcfg.L,
                    tcfg.eval_windows,
                )["r"].mean()
                / tcfg.L
            )
        recon = rd[tcfg.Q]
        score = recon + achieved + 10 * max(0.0, cfg.health_util - min(util))
        row = dict(
            trial=t,
            score=score,
            recon=recon,
            achieved=achieved,
            util=util,
            rd=rd,
            cfg={
                k: getattr(tcfg, k)
                for k in (
                    "stride",
                    "K",
                    "Q",
                    "d",
                    "eh_layers",
                    "beta",
                    "lam_fixed",
                    "level_dropout",
                )
            },
        )
        rows.append(row)
        print(
            f"  hpo t{t} score={score:.3f} recon={recon:.3f} achieved={achieved:.3f} util={min(util):.2f}"
        )
        with open(path + ".tmp", "w") as f:
            json.dump(dict(stage=stage, cfg=canon(asdict(cfg)), trials=rows), f)
        os.replace(path + ".tmp", path)
    rows.sort(key=lambda r: r["score"])
    n1 = max(1, len(rows) // 5)
    promote = rows[:n1] + [
        r for r in rows[n1:] if r["score"] < unigram_bpb - cfg.p1a_margin + 0.3
    ]
    for t, row in enumerate(promote):
        tcfg = replace(
            cfg,
            phase=1,
            corpora=("synseg",),
            chunkers=("fixed",),
            seeds=(0,),
            codec_steps=cfg.codec_steps,
            **row["cfg"],
        )
        torch.manual_seed(2000 + t)
        codec = Codec(tcfg, "fixed", "synseg").to(DEVICE)
        rng_b = np.random.RandomState(2000 + t)
        ctl = dict(
            lam=tcfg.lam_fixed if tcfg.lam_fixed is not None else 0.1,
            ema=None,
            fixed=True,
        )
        fit(
            codec,
            lambda s: codec.loss(
                train_batch(corpus, tcfg.L, tcfg.bs, rng_b), ctl["lam"]
            ),
            tcfg.codec_steps,
            tcfg.codec_lr,
            f"hpo2 t{t}",
            maintain_every=max(1, tcfg.codec_steps // 5),
        )
        codec.eval()
        with torch.no_grad():
            for p in codec.parameters():
                p.requires_grad_(False)
            rd, ppl, util = rd_curve(codec, corpus)
            achieved = float(
                collect(
                    lambda x: {"r": codec.rate_bits(x)},
                    corpus,
                    tcfg.L,
                    tcfg.eval_windows,
                )["r"].mean()
                / tcfg.L
            )
        recon = rd[tcfg.Q]
        score = recon + achieved + 10 * max(0.0, cfg.health_util - min(util))
        row["promoted"] = dict(
            score=score, recon=recon, achieved=achieved, util=util, rd=rd
        )
        print(f"  hpo promoted t{t}: score={score:.3f}")
        with open(path + ".tmp", "w") as f:
            json.dump(dict(stage=stage, cfg=canon(asdict(cfg)), trials=rows), f)
        os.replace(path + ".tmp", path)
    print("saved", path)
    return rows


# %%
# ---------------- report ---------------------------------------------------------------------
def report(results_path, out=None):
    global META, DECISIVE
    RES = json.load(open(results_path))
    DECISIVE = RES.get("preset") == "full"
    cfg = Cfg(**{k: v for k, v in RES["cfg"].items() if k not in ("M", "G")})
    runs = [normalize_run(dict(r)) for r in RES["runs"]]
    META = {
        c: dict(
            unigram=RES["unigram"].get(c, float("nan")),
            gpt=ms([r["bpb"] for r in pick(runs, corpus=c, kind="gpt")])[0]
            if pick(runs, corpus=c, kind="gpt")
            else float("nan"),
            nbytes=RES["corpora"][c]["nbytes"],
            sha=RES["corpora"][c]["sha"],
        )
        for c in RES["corpora"]
    }
    outdir = out or os.path.dirname(os.path.abspath(results_path))
    text = write_report(runs, cfg, RES, os.path.join(outdir, "REPORT.md"))
    print(text)
    try:
        import matplotlib.pyplot as plt

        for cname in cfg.corpora:
            if not all(
                pick(runs, corpus=cname, kind="concept", chunk=c, prior=m)
                for c, m in (("fixed", "ar"), ("fixed", "cf"))
            ):
                continue
            fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
            names = [f"{c}+{m}" for c in cfg.chunkers for m in ("ar", "cf")]
            vals = []
            for c in cfg.chunkers:
                for m in ("ar", "cf"):
                    rr = one_per_seed(
                        pick(runs, corpus=cname, kind="concept", chunk=c, prior=m)
                    )
                    vals.append(
                        ms([r["bpb"] for r in rr])
                        if rr
                        else (float("nan"), float("nan"))
                    )
            ax[0].bar(
                names,
                [v[0] for v in vals],
                yerr=[0 if np.isnan(v[1]) else v[1] for v in vals],
                capsize=4,
            )
            ax[0].axhline(
                RES["unigram"].get(cname, float("nan")),
                color="k",
                ls=":",
                label="unigram",
            )
            g = one_per_seed(pick(runs, corpus=cname, kind="gpt"))
            if g:
                ax[0].axhline(
                    ms([r["bpb"] for r in g])[0], color="g", ls="--", label="byte GPT"
                )
            ax[0].set_ylabel("val bpb (two-part bound)")
            ax[0].set_title(cname)
            ax[0].legend(fontsize=8)
            ax[0].tick_params(axis="x", rotation=20)
            for r in [
                r
                for r in runs
                if r["corpus"] == cname
                and r["seed"]
                == sorted({x["seed"] for x in runs if x["corpus"] == cname})[0]
            ]:
                tag = (
                    "GPT"
                    if r["kind"] == "gpt"
                    else f"{r['chunk'][:3]}/{r['prior']}"
                    + (f"k{r['k']}" if r["k"] else "")
                )
                ax[1].scatter(r["calls"], r["ref_bpb"], s=60)
                ax[1].annotate(tag, (r["calls"], r["ref_bpb"]), fontsize=7)
            ax[1].set_xscale("log")
            ax[1].set_xlabel("sequential network calls (log)")
            ax[1].set_ylabel("sample quality: GPT bits/byte")
            ax[1].set_title("quality vs depth")
            bottoms = np.zeros(len(names))
            for lab, keyf, colr in [
                ("recon", lambda k: k == "recon", "#4c72b0"),
                ("c1", lambda k: k == "c1", "#55a868"),
                ("bnd", lambda k: k == "bnd", "#8172b3"),
                (
                    "codes",
                    lambda k: k.startswith("r") and k not in ("recon", "prior"),
                    "#dd8452",
                ),
            ]:
                v = np.array(
                    [
                        np.mean(
                            [
                                sum(
                                    x
                                    for k, x in r["comps"].items()
                                    if k not in ("total", "prior") and keyf(k)
                                )
                                for r in pick(
                                    runs, corpus=cname, kind="concept", chunk=c, prior=m
                                )
                            ]
                        )
                        if pick(runs, corpus=cname, kind="concept", chunk=c, prior=m)
                        else np.nan
                        for c in cfg.chunkers
                        for m in ("ar", "cf")
                    ]
                )
                ax[2].bar(names, v, bottom=bottoms, label=lab, color=colr)
                bottoms += np.nan_to_num(v)
            ax[2].set_ylabel("bits/byte")
            ax[2].set_title("bit budget")
            ax[2].legend(fontsize=7)
            ax[2].tick_params(axis="x", rotation=20)
            plt.tight_layout()
            p = os.path.join(outdir, f"hncm5_{cname}.png")
            plt.savefig(p, dpi=140)
            plt.close(fig)
            print("saved", p)
    except Exception as e:
        print("plot skipped:", e)


META = {}
DECISIVE = False

# %%
# ---------------- CLI ------------------------------------------------------------------------
PRESET = "smoke"


def parse_list(s):
    return tuple(x.strip() for x in s.split(",") if x.strip())


def main(argv=None):
    global PRESET, META
    p = argparse.ArgumentParser(description="H-NCM v5 experiments")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--preset", default=os.environ.get("HNCM_PRESET", "smoke"))
    r.add_argument("--phase", type=int, default=None)
    r.add_argument("--corpora", default=None)
    r.add_argument("--chunkers", default=None)
    r.add_argument("--seeds", default=None)
    r.add_argument("--resume", type=int, default=None)
    r.add_argument("--out", default=os.environ.get("HNCM_OUT", "runs"))
    r.add_argument("--set", default=None)
    h = sub.add_parser("hpo")
    h.add_argument("--stage", type=int, default=1)
    h.add_argument("--trials", type=int, default=20)
    h.add_argument("--preset", default=os.environ.get("HNCM_PRESET", "smoke"))
    h.add_argument("--out", default=os.environ.get("HNCM_OUT", "runs"))
    h.add_argument("--set", default=None)
    q = sub.add_parser("report")
    q.add_argument("--results", default=PICK_LATEST())
    q.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.cmd == "report":
        report(a.results, a.out)
        return
    PRESET = a.preset
    ov = json.loads(a.set) if a.set else {}
    if a.cmd == "run":
        if a.phase is not None:
            ov["phase"] = a.phase
        if a.corpora:
            ov["corpora"] = parse_list(a.corpora)
        if a.chunkers:
            ov["chunkers"] = parse_list(a.chunkers)
        if a.seeds:
            ov["seeds"] = tuple(int(x) for x in parse_list(a.seeds))
        cfg = build_cfg(a.preset, ov)
        resume = (
            (os.environ.get("HNCM_RESUME", "1") != "0")
            if a.resume is None
            else bool(a.resume)
        )
        print(
            f"device={DEVICE} preset={PRESET} phase={cfg.phase} chunkers={cfg.chunkers}"
        )
        print(cfg)
        _selftest()
        run_all(cfg, a.out, resume)
    elif a.cmd == "hpo":
        cfg = build_cfg(a.preset, ov)
        _selftest()
        print(f"device={DEVICE} preset={PRESET} phase=1 hpo stage={a.stage}")
        run_hpo(cfg, a.out, a.trials, a.stage)


if __name__ == "__main__":
    main()
