# %% [markdown]
# # H-NCM v6 - learn the cost, solve the structure
#
# One file. No backwards compatibility. `hncm5.py` is reference only.
#
# Claims (report every result, including negatives):
#   P1 compression on shakespeare: total < unigram - 0.10; <= byte GPT + 0.05.
#   P2 parallelism: calls_AR/calls_DP >= 8 at matched total; DP <= 0.5 AR wall.
#   P3 structure: paid DP beats fixed/BPE/IsoWarp at matched lambda; oracle is
#      a free ceiling; syngrid is the no-oracle null.
#
# CLI:
#   python hncm6.py selftest
#   python hncm6.py run --preset smoke --phase 0 --seeds 0
#   python hncm6.py run --preset quick --phase 1 --seeds 0,1
#   python hncm6.py report --results runs/latest/results.json
# Env: HNCM_CFG (JSON overrides), HNCM_OUT, HNCM_PRESET, HNCM_CACHE.

# %%
import argparse
import datetime
import hashlib
import json
import math
import os
import time
import urllib.request
import zlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LN2 = math.log(2.0)
FORMAT = 6
PHASE_CORPORA = {
    0: ("synseg", "syngrid"),
    1: ("synseg", "syngrid"),
    2: ("shakespeare",),
    3: ("shakespeare",),
}
PHASE_ARMS = {
    0: (
        "fixed4",
        "fixed8",
        "fixed16",
        "constant-code",
        "no-quant",
        "rel-pos",
        "abs-pos",
    ),
    1: ("fixed", "DP", "BPE", "IsoWarp", "oracle", "oracle-costed", "random"),
    2: ("DP", "fixed", "byte-GPT", "byte-AR", "unigram", "gzip"),
    3: ("DP", "fixed", "byte-GPT", "unigram", "gzip"),
}
ARM_NAMES = set().union(*PHASE_ARMS.values()) | {"vq", "meanfield", "shared"}
ORACLE_ARMS = {"oracle", "oracle-costed"}
CAPABILITIES = {
    "codec": "shared; scalar quantization + low-rank prior; span-relative",
    "tiling": "inference; DP paid; fixed/oracle free; BPE/IsoWarp/random controls",
    "generation": "closed-loop, confidence-gated commitment",
    "claims": "P1 compression; P2 calls/wall; P3 paid structure",
    "deleted": "RVQ, commitment, boundary heads, absolute position, free tilings",
}


# %%
@dataclass(frozen=True)
class Cfg:
    preset: str = "smoke"
    phase: int = 0
    # geometry
    L: int = 64
    d: int = 64
    heads: int = 4
    enc: int = 1
    dec: int = 1
    base_layers: int = 2
    d_c: int = 2
    levels: int = 4
    rank: int = 2
    lengths: tuple = (1, 2, 4, 8)
    fixed_len: int = 4
    span_cost: float = 3.0
    # quantizer / training
    mode: str = "scalar"
    pos: str = "span"
    pos_drop: float = 0.5
    bs: int = 16
    codec_steps: int = 40
    cost_steps: int = 40
    base_steps: int = 40
    codec_lr: float = 1e-3
    cost_lr: float = 1e-3
    base_lr: float = 1e-3
    vq_codes: int = 32
    vq_beta: float = 0.25
    # evaluation
    eval_windows: int = 16
    sample_n: int = 4
    seeds: tuple = (0,)
    lam_values: tuple = (0.0, 2.0)
    boot: int = 200
    # gates (one source)
    g0_util: float = 0.20
    g0_info: float = 2.0
    g0_delta: float = 0.10
    p1a: float = 0.10
    p1b: float = 0.05
    p1c: float = 0.10
    p2_calls: float = 8.0
    p2_match: float = 0.05
    p2_wall: float = 0.5
    p3_alpha: float = 0.5
    p3_align: float = 0.05
    g6_pass: float = 0.05
    g6_fail: float = 0.10
    # resolved by phase
    corpora: tuple | None = None
    arms: tuple | None = None

    def __post_init__(s):
        object.__setattr__(s, "seeds", tuple(int(x) for x in s.seeds))
        object.__setattr__(s, "lam_values", tuple(float(x) for x in s.lam_values))
        object.__setattr__(s, "lengths", tuple(sorted(int(x) for x in s.lengths)))
        if s.corpora is None:
            object.__setattr__(s, "corpora", PHASE_CORPORA[s.phase])
        else:
            object.__setattr__(s, "corpora", tuple(s.corpora))
        if s.arms is None:
            object.__setattr__(s, "arms", PHASE_ARMS[s.phase])
        else:
            object.__setattr__(s, "arms", tuple(s.arms))
        assert s.phase in (0, 1, 2, 3)
        assert s.L >= 8 and s.d >= 8 and s.heads >= 1 and s.d % s.heads == 0
        assert s.enc >= 1 and s.dec >= 1 and s.base_layers >= 1
        assert s.d_c >= 1 and s.levels >= 2 and s.rank >= 0
        assert s.lengths and s.lengths[0] == 1 and max(s.lengths) <= s.L
        assert s.L % s.fixed_len == 0 and s.fixed_len in (4, 8, 16)
        assert s.mode in ("scalar", "vq", "constant", "noquant")
        assert s.pos in ("span", "rel", "abs")
        assert s.eval_windows >= 2 and s.bs >= 2 and s.sample_n >= 1
        assert s.seeds and all(x >= 0 for x in s.seeds)
        assert s.lam_values and all(x >= 0 for x in s.lam_values)
        assert all(c in ARM_NAMES for c in s.arms), f"unknown arm {s.arms}"
        assert all(c in ("synseg", "syngrid", "shakespeare") for c in s.corpora)
        if "shakespeare" in s.corpora:
            assert not (ORACLE_ARMS & set(s.arms)), "oracle is forbidden on shakespeare"


PRESETS = {
    "smoke": dict(
        preset="smoke",
        L=64,
        d=64,
        heads=4,
        enc=1,
        dec=1,
        base_layers=2,
        d_c=2,
        levels=4,
        rank=2,
        lengths=(1, 2, 4, 8),
        bs=16,
        codec_steps=40,
        cost_steps=40,
        base_steps=40,
        eval_windows=16,
        sample_n=4,
        seeds=(0,),
        lam_values=(0.0, 2.0),
        boot=100,
    ),
    "quick": dict(
        preset="quick",
        L=128,
        d=96,
        heads=4,
        enc=2,
        dec=2,
        base_layers=4,
        d_c=4,
        levels=8,
        rank=4,
        lengths=(1, 2, 4, 8, 16),
        bs=32,
        codec_steps=300,
        cost_steps=300,
        base_steps=300,
        eval_windows=128,
        sample_n=16,
        seeds=(0, 1),
        lam_values=(0.0, 1.0, 2.0, 4.0),
        boot=500,
    ),
    "full": dict(
        preset="full",
        L=256,
        d=192,
        heads=6,
        enc=3,
        dec=3,
        base_layers=8,
        d_c=4,
        levels=8,
        rank=4,
        lengths=(1, 2, 4, 8, 16),
        bs=64,
        codec_steps=1500,
        cost_steps=1500,
        base_steps=1500,
        eval_windows=2048,
        sample_n=128,
        seeds=(0, 1, 2),
        lam_values=(0.0, 1.0, 2.0, 4.0),
        boot=2000,
    ),
}


def canon(o):
    return json.dumps(o, sort_keys=True, default=str)


def fingerprint(cfg):
    return hashlib.sha256(canon(asdict(cfg)).encode()).hexdigest()


def build_cfg(preset="smoke", phase=0, seeds=None, corpora=None, arms=None):
    if preset not in PRESETS:
        raise SystemExit(f"unknown preset {preset!r}; choose {sorted(PRESETS)}")
    ov = json.loads(os.environ.get("HNCM_CFG", "{}"))
    kw = dict(PRESETS[preset])
    kw.update(phase=phase)
    if seeds:
        kw.update(seeds=tuple(seeds))
    if corpora:
        kw.update(corpora=tuple(corpora))
    if arms:
        kw.update(arms=tuple(arms))
    kw.update(ov)
    return Cfg(**kw)


# %%
# ---------------- data ------------------------------------------------------------------------
def load_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    cache = Path(os.environ.get("HNCM_CACHE", str(Path.home() / ".cache/hncm")))
    path = cache / "shakespeare.txt"
    if path.exists():
        return path.read_bytes(), "shakespeare"
    try:
        raw = urllib.request.urlopen(url, timeout=30).read()
    except Exception as e:
        raise RuntimeError(f"cannot download shakespeare; set HNCM_CACHE: {e}") from e
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
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


def get_corpus(name):
    raw, tag = {
        "shakespeare": load_shakespeare,
        "synseg": gen_synseg,
        "syngrid": gen_syngrid,
    }[name]()
    arr = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()
    n = (int(0.9 * len(arr)) // 4) * 4
    return {
        "tr": arr[:n].to(DEVICE),
        "va": arr[n:].to(DEVICE),
        "name": tag,
        "sha": hashlib.sha256(raw).hexdigest(),
        "nbytes": len(raw),
    }


def onsets(name, xb):
    if name == "syngrid":
        return np.arange(len(xb)) % 4 == 0
    prev = np.concatenate([[ord("\n")], xb[:-1]])
    return np.isin(prev, [ord(" "), ord("\n")]) & ~np.isin(
        xb, [ord(" "), ord("\n"), ord("_")]
    )


def windows(data, starts, L):
    return data[starts[:, None] + torch.arange(L, device=data.device)[None]]


def make_starts(corpus, L, n, seed=1234):
    rs = np.random.RandomState(seed)
    hi = len(corpus["va"]) - L
    if hi <= 0:
        raise RuntimeError("corpus is smaller than L")
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
# ---------------- blocks and pooling -----------------------------------------------------------
class Block(nn.Module):
    def __init__(s, d, h, causal):
        super().__init__()
        s.h, s.causal = h, causal
        s.ln1 = nn.LayerNorm(d)
        s.qkv = nn.Linear(d, 3 * d, bias=False)
        s.o = nn.Linear(d, d, bias=False)
        s.ln2 = nn.LayerNorm(d)
        s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(s, x):
        B, T, D = x.shape
        q, k, v = s.qkv(s.ln1(x)).view(B, T, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=(s.causal and T > 1))
        x = x + s.o(a.transpose(1, 2).reshape(B, T, D))
        return x + s.mlp(s.ln2(x))


class Stack(nn.Module):
    def __init__(s, n, d, h, causal):
        super().__init__()
        s.blocks = nn.ModuleList(Block(d, h, causal) for _ in range(n))
        s.ln = nn.LayerNorm(d)

    def forward(s, x):
        for b in s.blocks:
            x = b(x)
        return s.ln(x)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def pool(H, bnd):
    B, L, D = H.shape
    ids = (torch.cumsum(bnd.long(), 1) - 1).clamp_min(0)
    sums = torch.zeros(B, L, D, device=H.device).scatter_add_(
        1, ids[..., None].expand(-1, -1, D), H
    )
    cnt = torch.zeros(B, L, device=H.device).scatter_add_(
        1, ids, torch.ones_like(ids, dtype=torch.float)
    )
    return sums / cnt.clamp_min(1)[..., None], cnt, ids


def geometry(bnd):
    B, L = bnd.shape
    pos = torch.arange(L, device=bnd.device)[None].expand(B, -1)
    ids = (torch.cumsum(bnd.long(), 1) - 1).clamp_min(0)
    start = torch.where(bnd, pos, torch.zeros_like(pos))
    start = torch.cummax(start, 1).values
    cnt = torch.zeros(B, L, device=bnd.device).scatter_add_(
        1, ids, torch.ones_like(ids, dtype=torch.float)
    )
    plen = cnt.gather(1, ids.clamp(max=L - 1))
    return ids, pos - start, plen, cnt


def fixed_mask(B, L, step):
    return (torch.arange(L, device=DEVICE) % step == 0)[None].expand(B, -1).clone()


def random_masks(B, L, lengths, rng):
    m = torch.zeros(B, L, dtype=torch.bool, device=DEVICE)
    for i in range(B):
        p = 0
        while p < L:
            m[i, p] = True
            cand = [e for e in lengths if e <= L - p]
            p += int(cand[rng.randint(len(cand))])
    return m


def oracle_masks(ons, starts, L):
    B = len(starts)
    m = torch.zeros(B, L, dtype=torch.bool, device=DEVICE)
    for i, s in enumerate(starts):
        seg = ons[s : s + L]
        m[i, 0] = True
        m[i, 1:] = torch.as_tensor(seg[1:], device=DEVICE)
    return m


def global_bits(mask):
    p = mask.float().mean(1)
    h = torch.where(
        (p > 0) & (p < 1),
        -p * torch.log2(p) - (1 - p) * torch.log2(1 - p),
        torch.zeros_like(p),
    )
    return h * mask.size(1)


def dp_bits(mask, b, ll):
    B, L = mask.shape
    out = torch.zeros(B, device=mask.device)
    for i in range(B):
        pos = mask[i].nonzero().flatten().tolist()
        if not pos or pos[0] != 0:
            pos = [0] + pos
        for j, t in enumerate(pos):
            e = pos[j + 1] if j + 1 < len(pos) else L
            li = _lid(int(e - t))
            out[i] = out[i] - F.logsigmoid(b[i, t]) / LN2
            out[i] = out[i] - F.log_softmax(ll[i, t], -1)[li] / LN2
    return out


def align(mask, ref):
    inter = (mask.bool() & ref.bool()).sum(1).float()
    p = inter / mask.bool().sum(1).clamp_min(1)
    r = inter / ref.bool().sum(1).clamp_min(1)
    return 2 * p * r / (p + r).clamp_min(1e-9)


LENGTHS = (1, 2, 4, 8, 16)


def _lid(ell):
    return LENGTHS.index(ell) if ell in LENGTHS else 0


# %%
# ---------------- codec: scalar quantization + low-rank prior + span decoder -------------------
class Encoder(nn.Module):
    def __init__(s, cfg):
        super().__init__()
        s.emb = nn.Embedding(257, cfg.d)
        s.pos = nn.Embedding(cfg.L, cfg.d)
        s.stack = Stack(cfg.enc, cfg.d, cfg.heads, False)

    def forward(s, x):
        return s.stack(s.emb(x) + s.pos.weight[: x.size(1)][None])


class LowRank(nn.Module):
    def __init__(s, cfg):
        super().__init__()
        s.cfg = cfg
        s.mu = nn.Linear(cfg.d, cfg.d_c)
        s.sig = nn.Linear(cfg.d, cfg.d_c)
        s.h = nn.Linear(cfg.d, cfg.rank) if cfg.rank > 0 else None
        s.gain = nn.Linear(cfg.d, cfg.d_c * cfg.rank) if cfg.rank > 0 else None

    def forward(s, z, train):
        B, M, d = z.shape
        mu = s.mu(z)
        sig = F.softplus(s.sig(z)) + 1e-3
        if s.cfg.rank > 0:
            raw = torch.tanh(s.h(z.mean(1)))
            hq = torch.round(raw * 16) / 16
            h = raw + (hq - raw).detach()
            gain = s.gain(z).view(B, M, s.cfg.d_c, s.cfg.rank)
            gfh = (gain @ h[:, None, :, None]).squeeze(-1)
            hrate = (
                (hq**2).sum(1) / (2 * LN2)
                + 0.5 * s.cfg.rank * math.log2(2 * math.pi * math.e)
                + 2.0 * s.cfg.rank
            )
        else:
            h, hq, gfh, hrate = (
                None,
                None,
                torch.zeros_like(mu),
                torch.zeros(B, device=z.device),
            )
        ctx = torch.cat([z, mu, gfh], -1)
        if s.cfg.rank > 0:
            ctx = torch.cat([ctx, hq[:, None].expand(-1, M, -1)], -1)
        return mu, sig, h, gfh, ctx, hrate


class ScalarQ(nn.Module):
    def __init__(s, cfg):
        super().__init__()
        s.cfg = cfg
        s.proj = nn.Linear(cfg.d, cfg.d_c)
        ctx_dim = cfg.d + 2 * cfg.d_c + cfg.rank
        s.ent = nn.Sequential(
            nn.LayerNorm(ctx_dim),
            nn.Linear(ctx_dim, cfg.d),
            nn.GELU(),
            nn.Linear(cfg.d, cfg.d_c * cfg.levels),
        )
        if cfg.mode == "vq":
            s.cb = nn.Parameter(torch.randn(cfg.vq_codes, cfg.d_c) * 0.1)

    def logits(s, ctx):
        B, M, _ = ctx.shape
        return s.ent(ctx).view(B, M, s.cfg.d_c, s.cfg.levels)

    def quantize(s, y, mu, gfh, sig, ctx, train):
        B, M, dc = y.shape
        if s.cfg.mode == "noquant":
            return (
                y,
                None,
                torch.zeros(B, device=y.device),
                torch.zeros((), device=y.device),
            )
        if s.cfg.mode == "constant":
            return (
                torch.zeros_like(y),
                torch.zeros(B, M, dc, dtype=torch.long),
                torch.zeros(B, device=y.device),
                torch.zeros((), device=y.device),
            )
        if s.cfg.mode == "vq":
            d = ((y[..., None, :] - s.cb) ** 2).sum(-1)
            k = d.argmin(-1)
            yq = s.cb[k]
            rate = torch.full((B,), M * math.log2(s.cfg.vq_codes), device=y.device)
            commit = s.cfg.vq_beta * ((yq.detach() - y) ** 2).mean()
            return yq, k, rate, commit
        step = 2.0 / s.cfg.levels
        u = torch.tanh(y)
        if train:
            u = u + (torch.rand_like(u) - 0.5) * step
        k = torch.floor((u + 1.0) / step).clamp(0, s.cfg.levels - 1).long()
        dq = (k.float() + 0.5) * step - 1.0
        yq = u + (dq - u).detach() if train else dq
        lg = s.logits(ctx)
        rate = (
            F.cross_entropy(
                lg.reshape(-1, s.cfg.levels), k.reshape(-1), reduction="none"
            )
            .view(B, M, s.cfg.d_c)
            .sum(-1)
            .sum(1)
            / LN2
        )
        return yq, k, rate, torch.zeros((), device=y.device)


class Decoder(nn.Module):
    def __init__(s, cfg):
        super().__init__()
        s.cfg = cfg
        s.y = nn.Linear(cfg.d_c, cfg.d)
        s.rel = nn.Embedding(cfg.L, cfg.d)
        s.plen = nn.Embedding(cfg.L + 1, cfg.d)
        s.sidx = nn.Embedding(cfg.L, cfg.d)
        s.abs = nn.Embedding(cfg.L, cfg.d)
        s.h = nn.Linear(cfg.rank, cfg.d) if cfg.rank > 0 else None
        s.stack = Stack(cfg.dec, cfg.d, cfg.heads, False)
        s.head = nn.Linear(cfg.d, 256)

    def forward(s, yq, bnd, h, train=False):
        B, L = bnd.shape
        ids, rel, plen, _ = geometry(bnd)
        f = s.y(yq.gather(1, ids[..., None].expand(-1, -1, yq.size(-1))))
        if s.cfg.pos == "span":
            p = (
                s.rel(rel.clamp(max=L - 1))
                + s.plen(plen.clamp(max=L).long())
                + s.sidx(ids % L)
            )
        elif s.cfg.pos == "rel":
            p = s.rel(rel.clamp(max=L - 1))
            if train and s.cfg.pos_drop > 0:
                keep = (
                    torch.rand(B, ids.size(1), device=bnd.device) > s.cfg.pos_drop
                ).float()
                p = p * keep.gather(1, ids)[..., None]
            f = f + p
        else:
            p = s.abs.weight[:L][None].expand(B, -1, -1)
            if train and s.cfg.pos_drop > 0:
                keep = (
                    torch.rand(B, ids.size(1), device=bnd.device) > s.cfg.pos_drop
                ).float()
                p = p * keep.gather(1, ids)[..., None]
            f = f + p
        if s.h is not None and h is not None:
            f = f + s.h(h)[:, None, :]
        return s.head(s.stack(f))


class Codec(nn.Module):
    def __init__(s, cfg):
        super().__init__()
        s.cfg = cfg
        s.enc = Encoder(cfg)
        s.prior = LowRank(cfg)
        s.quant = ScalarQ(cfg)
        s.dec = Decoder(cfg)

    def forward(s, x, bnd, train):
        H = s.enc(x)
        z, _, _ = pool(H, bnd)
        mu, sig, h, gfh, ctx, hrate = s.prior(z, train)
        y = s.quant.proj(z)
        yq, k, rate, commit = s.quant.quantize(y, mu, gfh, sig, ctx, train)
        rec = (
            F.cross_entropy(
                s.dec(yq, bnd, h, train).reshape(-1, 256),
                x.reshape(-1),
                reduction="none",
            )
            .view(x.shape[0], -1)
            .sum(1)
            / LN2
            / x.size(1)
        )
        code = (rate + hrate) / x.size(1)
        return rec + code + commit, rec, code, k, commit

    @torch.no_grad()
    def eval_bits(s, x, bnd):
        H = s.enc(x)
        z, _, _ = pool(H, bnd)
        mu, sig, h, gfh, ctx, hrate = s.prior(z, False)
        y = s.quant.proj(z)
        yq, k, rate, _ = s.quant.quantize(y, mu, gfh, sig, ctx, False)
        rec = (
            F.cross_entropy(
                s.dec(yq, bnd, h, False).reshape(-1, 256),
                x.reshape(-1),
                reduction="none",
            )
            .view(x.shape[0], -1)
            .sum(1)
            / LN2
            / x.size(1)
        )
        return rec, (rate + hrate) / x.size(1), k

    @torch.no_grad()
    def generate(s, cfg, n=4, steps=4, temp=1.0, cf=None):
        canvas = torch.full((n, cfg.L), 256, device=DEVICE)
        fixed = torch.zeros(n, cfg.L, dtype=torch.bool, device=DEVICE)
        fixed[:, :: cfg.fixed_len] = True
        pos = torch.arange(cfg.L, device=DEVICE)[None]
        for it in range(steps):
            bnd = fixed
            if cf is not None:
                H = s.enc(canvas)
                _, _, cost = cf(H, 0.0)
                bnd = dp_tile(cost, cfg.lengths, cfg.L)
            yq = torch.zeros(n, cfg.L, cfg.d_c, device=DEVICE)
            prob = F.softmax(s.dec(yq, bnd, None, False) / temp, -1)
            samp = torch.multinomial(prob.reshape(-1, 256), 1).view(n, cfg.L)
            commit = (prob.max(-1).values > 0.5) | (pos == cfg.L - 1)
            if it == steps - 1:
                commit = torch.ones_like(commit)
            canvas = torch.where(commit, samp, canvas)
        return canvas


def code_info(k, levels):
    if k is None:
        return 1.0, 0.0
    B, M, dc = k.shape
    util = 0.0
    info = 0.0
    for j in range(dc):
        cnt = torch.bincount(k[..., j].reshape(-1), minlength=levels).float()
        p = cnt / cnt.sum().clamp_min(1)
        q = p[p > 0]
        info += float(-(q * torch.log2(q)).sum())
        util += float((cnt > 0).float().mean())
    return util / dc, info * k.size(1)


# %%
# ---------------- cost field and exact tiling --------------------------------------------------
class CostField(nn.Module):
    def __init__(s, cfg):
        super().__init__()
        s.cfg = cfg
        s.nlen = len(cfg.lengths)
        s.inp = nn.LayerNorm(cfg.d)
        s.mlp = nn.Sequential(
            nn.Linear(cfg.d, cfg.d), nn.GELU(), nn.Linear(cfg.d, cfg.d)
        )
        s.bnd = nn.Linear(cfg.d, 1)
        s.ln = nn.Linear(cfg.d, s.nlen)
        s.register_buffer("lens", torch.tensor(cfg.lengths, dtype=torch.float))

    def forward(s, H, lam):
        B, L, _ = H.shape
        f = s.mlp(s.inp(H))
        b = s.bnd(f).squeeze(-1)
        ll = torch.log_softmax(s.ln(f), -1)
        t = torch.arange(L, device=H.device)[None, :, None]
        valid = t + s.lens[None, None, :].long() <= L
        cost = (
            s.cfg.span_cost
            - F.logsigmoid(b)[..., None]
            - ll
            + lam * s.lens[None, None, :]
        ) / LN2
        cost = torch.where(valid, cost, torch.full_like(cost, float("inf")))
        return b, ll, cost


@torch.no_grad()
def dp_tile(cost, lengths, L):
    B = cost.size(0)
    best = torch.full((B, L + 1), float("inf"), device=cost.device)
    best[:, 0] = 0
    back = torch.zeros((B, L + 1), dtype=torch.long, device=cost.device)
    c = cost.detach()
    for t in range(L):
        for li, ell in enumerate(lengths):
            if t + ell > L:
                continue
            cand = best[:, t] + c[:, t, li]
            better = cand < best[:, t + ell]
            best[:, t + ell] = torch.where(better, cand, best[:, t + ell])
            back[:, t + ell] = torch.where(
                better, torch.full_like(back[:, t + ell], li), back[:, t + ell]
            )
    mask = torch.zeros((B, L), dtype=torch.bool, device=cost.device)
    for i in range(B):
        e = L
        while e > 0:
            li = int(back[i, e])
            ell = lengths[li]
            mask[i, e - ell] = True
            e -= ell
    return mask


def brute_cover(cost, lengths, L):
    best = [float("inf")]
    bm = [None]

    def go(p, val, mask):
        if p == L:
            if val < best[0]:
                best[0], bm[0] = val, mask[:]
            return
        for li, ell in enumerate(lengths):
            if p + ell <= L:
                mask[p] = True
                go(p + ell, val + float(cost[p, li]), mask)
                mask[p] = False

    go(0, 0.0, [False] * L)
    return bm[0], best[0]


# %%
# ---------------- BPE control ----------------------------------------------------------------
def learn_bpe(data, merges):
    toks = list(data)
    out = []
    for _ in range(merges):
        pairs = {}
        for a, b in zip(toks, toks[1:]):
            pairs[(a, b)] = pairs.get((a, b), 0) + 1
        if not pairs:
            break
        (a, b), _ = max(pairs.items(), key=lambda kv: kv[1])
        new = 256 + len(out)
        nt = []
        i = 0
        while i < len(toks):
            if i + 1 < len(toks) and toks[i] == a and toks[i + 1] == b:
                nt.append(new)
                i += 2
            else:
                nt.append(toks[i])
                i += 1
        toks = nt
        out.append((a, b))
    return out


def apply_bpe(data, merges):
    toks = list(data)
    lens = [1] * 256
    for mi, (a, b) in enumerate(merges):
        new = 256 + mi
        nt = []
        i = 0
        while i < len(toks):
            if i + 1 < len(toks) and toks[i] == a and toks[i + 1] == b:
                nt.append(new)
                i += 2
            else:
                nt.append(toks[i])
                i += 1
        toks = nt
        lens.append(lens[a] + lens[b])
    starts = np.zeros(len(data), dtype=bool)
    if not toks:
        return starts
    starts[0] = True
    p = lens[toks[0]]
    for t in toks[1:]:
        if p < len(starts):
            starts[p] = True
        p += lens[t]
    return starts


# %%
# ---------------- training -------------------------------------------------------------------
def codec_variant(cfg, arm):
    if arm == "constant-code":
        return replace(cfg, mode="constant", rank=0, pos="rel"), 4
    if arm == "no-quant":
        return replace(cfg, mode="noquant", pos="rel"), 4
    if arm == "abs-pos":
        return replace(cfg, pos="abs"), 4
    if arm == "rel-pos":
        return replace(cfg, pos="rel"), 4
    if arm == "vq":
        return replace(cfg, mode="vq"), cfg.fixed_len
    if arm == "meanfield":
        return replace(cfg, rank=0, mode="scalar"), cfg.fixed_len
    if arm.startswith("fixed") and arm[5:].isdigit():
        step = int(arm[5:])
        return replace(cfg, fixed_len=step), step
    return cfg, cfg.fixed_len


def train_codec(cfg, corpus, seed, arm, random_tilings):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed * 1000 + sum(map(ord, arm)))
    model = Codec(cfg).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.codec_lr)
    _, step = codec_variant(cfg, arm)
    for it in range(cfg.codec_steps):
        x = train_batch(corpus, cfg.L, cfg.bs, rng)
        b = (
            random_masks(cfg.bs, cfg.L, cfg.lengths, rng)
            if random_tilings
            else fixed_mask(cfg.bs, cfg.L, step)
        )
        loss, rec, code, _, _ = model(x, b, True)
        opt.zero_grad()
        loss.mean().backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it == 0 or (it + 1) % max(1, cfg.codec_steps // 4) == 0:
            print(
                f"  codec {arm} {it + 1}/{cfg.codec_steps} rec={rec.mean():.3f} code={code.mean():.3f} total={loss.mean():.3f}"
            )
    return model.eval()


def train_cost(cfg, corpus, codec, seed):
    torch.manual_seed(seed + 17)
    rng = np.random.RandomState(seed * 1000 + 99)
    cf = CostField(cfg).to(DEVICE)
    opt = torch.optim.Adam(cf.parameters(), lr=cfg.cost_lr)
    for p in codec.parameters():
        p.requires_grad_(False)
    codec.eval()
    for it in range(cfg.cost_steps):
        x = train_batch(corpus, cfg.L, cfg.bs, rng)
        lam = float(cfg.lam_values[rng.randint(len(cfg.lam_values))])
        with torch.no_grad():
            H = codec.enc(x)
        b, ll, cost = cf(H, lam)
        mask = dp_tile(cost, cfg.lengths, cfg.L)
        with torch.no_grad():
            rec, code, _ = codec.eval_bits(x, mask)
        bnd = dp_bits(mask, b, ll).mean() / cfg.L
        loss = rec.mean() + code.mean() + bnd
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(cf.parameters(), 1.0)
        opt.step()
        if it == 0 or (it + 1) % max(1, cfg.cost_steps // 4) == 0:
            print(
                f"  cost {it + 1}/{cfg.cost_steps} total={float(loss.detach()):.3f} bnd={float(bnd.detach()):.3f}"
            )
    return cf.eval()


# %%
# ---------------- evaluation ----------------------------------------------------------------
def paid(arm):
    return arm not in (
        "fixed",
        "oracle",
        "fixed4",
        "fixed8",
        "fixed16",
        "rel-pos",
        "abs-pos",
        "constant-code",
        "no-quant",
    )


def mask_for(arm, cfg, codec, cf, x, lam, starts, ons, bpe_ons):
    B, L = x.shape
    if arm in ("fixed", "rel-pos", "abs-pos", "constant-code", "no-quant"):
        return fixed_mask(B, L, cfg.fixed_len), None
    if arm.startswith("fixed") and arm[5:].isdigit():
        return fixed_mask(B, L, int(arm[5:])), None
    if arm == "oracle":
        return oracle_masks(ons, starts, L), None
    if arm in ("oracle-costed",):
        return oracle_masks(ons, starts, L), None
    if arm == "BPE":
        m = torch.zeros(B, L, dtype=torch.bool, device=DEVICE)
        for i, s in enumerate(starts):
            seg = bpe_ons[s : s + L]
            m[i, 0] = True
            m[i, 1:] = torch.as_tensor(seg[1:], device=DEVICE)
        return m, None
    if arm == "random":
        return random_masks(
            B,
            L,
            cfg.lengths,
            np.random.RandomState((int(starts[0]) + int(lam * 1000) + cfg.L) % 2**31),
        ), None
    if arm == "IsoWarp":
        H = codec.enc(x)
        b, _, _ = cf(H, lam)
        m = torch.sigmoid(b) > 0.5
        m[:, 0] = True
        return m, None
    # DP
    H = codec.enc(x)
    b, ll, cost = cf(H, lam)
    m = dp_tile(cost, cfg.lengths, L)
    return m, (b, ll)


def eval_arm(cfg, corpus, model, cf, arm, lam, seed, starts, ons, bpe_ons):
    x = windows(corpus["va"], starts, cfg.L)
    B = x.size(0)
    t0 = time.perf_counter()
    masks, db = mask_for(arm, cfg, model, cf, x, lam, starts, ons, bpe_ons)
    ref = oracle_masks(ons, starts, cfg.L)
    if db is not None:
        bbits = dp_bits(masks, db[0], db[1])
    elif paid(arm):
        bbits = global_bits(masks)
    else:
        bbits = torch.zeros(B, device=DEVICE)
    recs, codes, ks = [], [], []
    for i in range(0, B, 64):
        r, c, k = model.eval_bits(x[i : i + 64], masks[i : i + 64])
        recs.append(r)
        codes.append(c)
        ks.append(k)
    rec = torch.cat(recs)
    code = torch.cat(codes)
    k = None if ks[0] is None else torch.cat(ks)
    bnd = bbits / cfg.L
    total = rec + code + bnd
    wall = (time.perf_counter() - t0) / B
    util, info = code_info(k, cfg.levels) if cfg.mode == "scalar" else (1.0, 0.0)
    if cfg.mode == "vq":
        util, info = 1.0, masks.size(1) * math.log2(cfg.vq_codes)
    if cfg.mode in ("constant", "noquant"):
        util, info = 0.0, 0.0
    return {
        "claim": "structure",
        "corpus": corpus["name"],
        "seed": seed,
        "arm": arm,
        "lambda": float(lam),
        "total_bpb": float(total.detach().mean()),
        "recon_bpb": float(rec.detach().mean()),
        "code_bpb": float(code.detach().mean()),
        "boundary_bpb": float(bnd.detach().mean()),
        "calls": 2.0,
        "wall_sec": wall,
        "NFE": 1.0,
        "align": float(align(masks, ref).mean()),
        "util": util,
        "code_info": info,
        "params": nparams(model) + (nparams(cf) if cf is not None else 0),
        "samples": [float(v) for v in total.detach().tolist()],
        "provenance": {"device": DEVICE, "torch": torch.__version__},
    }


# %%
# ---------------- baselines -----------------------------------------------------------------
class ByteLM(nn.Module):
    def __init__(s, cfg, layers, width):
        super().__init__()
        s.emb = nn.Embedding(256, width)
        s.pos = nn.Embedding(cfg.L, width)
        s.stack = Stack(layers, width, cfg.heads, True)
        s.head = nn.Linear(width, 256)

    def logits(s, x):
        return s.head(s.stack(s.emb(x[:, :-1]) + s.pos.weight[: x.size(1) - 1][None]))

    def loss(s, x):
        return F.cross_entropy(s.logits(x).reshape(-1, 256), x[:, 1:].reshape(-1))

    @torch.no_grad()
    def bits(s, x, uni):
        ce = (
            F.cross_entropy(
                s.logits(x).reshape(-1, 256), x[:, 1:].reshape(-1), reduction="none"
            )
            .view(x.size(0), -1)
            .sum(1)
        )
        return (ce / LN2 + uni[x[:, 0]]) / x.size(1)


def train_lm(cfg, corpus, seed, kind):
    torch.manual_seed(seed + (0 if kind == "byte-GPT" else 31))
    rng = np.random.RandomState(seed * 1000 + (7 if kind == "byte-GPT" else 8))
    layers = cfg.base_layers if kind == "byte-GPT" else max(1, cfg.base_layers // 2)
    width = cfg.d if kind == "byte-GPT" else max(8, cfg.d // 2)
    model = ByteLM(cfg, layers, width).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.base_lr)
    for it in range(cfg.base_steps):
        x = train_batch(corpus, cfg.L, cfg.bs, rng)
        loss = model.loss(x)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it == 0 or (it + 1) % max(1, cfg.base_steps // 4) == 0:
            print(f"  {kind} {it + 1}/{cfg.base_steps} ce={float(loss.detach()):.3f}")
    return model.eval()


def eval_lm(cfg, corpus, model, starts, kind):
    x = windows(corpus["va"], starts, cfg.L)
    uni = unigram_bits(corpus)
    t0 = time.perf_counter()
    bits = []
    for i in range(0, x.size(0), 32):
        bits.append(model.bits(x[i : i + 32], uni))
    bits = torch.cat(bits)
    wall = (time.perf_counter() - t0) / x.size(0)
    return {
        "claim": "compression",
        "corpus": corpus["name"],
        "seed": None,
        "arm": kind,
        "lambda": None,
        "total_bpb": float(bits.mean()),
        "recon_bpb": float(bits.mean()),
        "code_bpb": 0.0,
        "boundary_bpb": 0.0,
        "calls": float(cfg.L - 1),
        "wall_sec": wall,
        "NFE": float(cfg.L - 1),
        "align": None,
        "util": None,
        "code_info": None,
        "params": nparams(model),
        "samples": [float(v) for v in bits.tolist()],
        "provenance": {"device": DEVICE, "torch": torch.__version__},
    }


def eval_unigram(cfg, corpus, starts):
    x = windows(corpus["va"], starts, cfg.L)
    uni = unigram_bits(corpus)
    bits = uni[x].sum(1) / cfg.L
    return {
        "claim": "compression",
        "corpus": corpus["name"],
        "seed": None,
        "arm": "unigram",
        "lambda": None,
        "total_bpb": float(bits.mean()),
        "recon_bpb": float(bits.mean()),
        "code_bpb": 0.0,
        "boundary_bpb": 0.0,
        "calls": 1.0,
        "wall_sec": 0.0,
        "NFE": 1.0,
        "align": None,
        "util": None,
        "code_info": None,
        "params": 0,
        "samples": [float(v) for v in bits.tolist()],
        "provenance": {"device": DEVICE, "torch": torch.__version__},
    }


def eval_gzip(cfg, corpus, starts):
    raw = corpus["va"].cpu().numpy().astype(np.uint8).tobytes()
    rate = len(zlib.compress(raw)) * 8 / len(raw)
    bits = np.full(len(starts), rate)
    return {
        "claim": "compression",
        "corpus": corpus["name"],
        "seed": None,
        "arm": "gzip",
        "lambda": None,
        "total_bpb": float(bits.mean()),
        "recon_bpb": float(bits.mean()),
        "code_bpb": 0.0,
        "boundary_bpb": 0.0,
        "calls": 1.0,
        "wall_sec": 0.0,
        "NFE": 1.0,
        "align": None,
        "util": None,
        "code_info": None,
        "params": 0,
        "samples": [float(v) for v in bits],
        "provenance": {"device": DEVICE, "torch": torch.__version__},
    }


# %%
# ---------------- store, CLI phases ----------------------------------------------------------
class Store:
    def __init__(s, outdir, cfg, shas):
        s.path = Path(outdir) / "results.json"
        s.cfg = cfg
        s.shas = shas
        s.data = None
        if s.path.exists():
            s.data = json.loads(s.path.read_text())
            if s.data.get("format") != FORMAT:
                raise RuntimeError("results format changed; use a new HNCM_OUT")
            if s.data.get("cfg_fingerprint") != fingerprint(cfg):
                raise RuntimeError(
                    "cfg changed; use a new preset directory or HNCM_OUT"
                )
            if s.data.get("corpora") != shas:
                raise RuntimeError("corpus changed; use a new HNCM_OUT")
        else:
            s.data = {
                "format": FORMAT,
                "cfg_fingerprint": fingerprint(cfg),
                "cfg": json.loads(canon(asdict(cfg))),
                "device": DEVICE,
                "torch": torch.__version__,
                "corpora": shas,
                "started": datetime.datetime.now().isoformat(timespec="seconds"),
                "finished": None,
                "runs": [],
            }
        s.seen = {s.key(r) for r in s.data["runs"]}

    @staticmethod
    def key(r):
        return (r["claim"], r["corpus"], r["seed"], r["arm"], r["lambda"])

    def add(s, r):
        if s.key(r) in s.seen:
            return
        s.data["runs"].append(r)
        s.seen.add(s.key(r))
        s.write()

    def write(s):
        s.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(s.path) + ".tmp")
        tmp.write_text(json.dumps(s.data, indent=1))
        os.replace(tmp, s.path)

    def finish(s):
        s.data["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
        s.write()


def run_phase0(cfg, corpora, store, resume):
    for name in cfg.corpora:
        corpus = corpora[name]
        starts = make_starts(corpus, cfg.L, cfg.eval_windows, seed=99 + cfg.seeds[0])
        ons = onsets(name, corpus["va"].cpu().numpy())
        for seed in cfg.seeds:
            base = None
            for arm in cfg.arms:
                if not resume or ("health", name, seed, arm, 0.0) not in store.seen:
                    variant, _ = codec_variant(cfg, arm)
                    print(f"[phase 0] {name} seed={seed} arm={arm}")
                    if arm in ("fixed4", "fixed8", "fixed16", "rel-pos"):
                        if base is None:
                            base = train_codec(cfg, corpus, seed, "shared", True)
                        model = base
                    else:
                        model = train_codec(variant, corpus, seed, arm, True)
                    r = eval_arm(
                        variant, corpus, model, None, arm, 0.0, seed, starts, ons, None
                    )
                    r["claim"] = "health"
                    store.add(r)
            if ("compression", name, None, "unigram", None) not in store.seen:
                store.add(eval_unigram(cfg, corpus, starts))


def run_phase1(cfg, corpora, store, resume):
    for name in cfg.corpora:
        corpus = corpora[name]
        starts = make_starts(corpus, cfg.L, cfg.eval_windows, seed=199 + cfg.seeds[0])
        ons = onsets(name, corpus["va"].cpu().numpy())
        raw = corpus["va"].cpu().numpy().astype(np.uint8).tobytes()
        merges = learn_bpe(
            corpus["tr"][:100_000].cpu().numpy().astype(np.uint8).tobytes(), 64
        )
        bpe_ons = torch.as_tensor(apply_bpe(raw, merges), device=DEVICE)
        for seed in cfg.seeds:
            shared = None
            cost = None
            variant_cache = {}
            for arm in cfg.arms:
                if arm in ("vq", "meanfield"):
                    if arm not in variant_cache:
                        v, _ = codec_variant(cfg, arm)
                        print(f"[phase 1] variant {name} seed={seed} arm={arm}")
                        m = train_codec(v, corpus, seed, arm, True)
                        variant_cache[arm] = (m, train_cost(v, corpus, m, seed), v)
                    m, c, v = variant_cache[arm]
                    for lam in cfg.lam_values:
                        if (
                            resume
                            and ("structure", name, seed, arm, float(lam)) in store.seen
                        ):
                            continue
                        r = eval_arm(
                            v, corpus, m, c, "fixed", lam, seed, starts, ons, bpe_ons
                        )
                        r["arm"] = arm
                        store.add(r)
                    continue
                if shared is None:
                    print(f"[phase 1] shared codec {name} seed={seed}")
                    shared = train_codec(cfg, corpus, seed, "shared", True)
                    print(f"[phase 1] cost field {name} seed={seed}")
                    cost = train_cost(cfg, corpus, shared, seed)
                for lam in cfg.lam_values:
                    if (
                        resume
                        and ("structure", name, seed, arm, float(lam)) in store.seen
                    ):
                        continue
                    r = eval_arm(
                        cfg, corpus, shared, cost, arm, lam, seed, starts, ons, bpe_ons
                    )
                    store.add(r)
            if ("compression", name, None, "unigram", None) not in store.seen:
                store.add(eval_unigram(cfg, corpus, starts))


def run_phase2(cfg, corpora, store, resume):
    for name in cfg.corpora:
        corpus = corpora[name]
        starts = make_starts(corpus, cfg.L, cfg.eval_windows, seed=299 + cfg.seeds[0])
        ons = onsets(name, corpus["va"].cpu().numpy())
        raw = corpus["va"].cpu().numpy().astype(np.uint8).tobytes()
        merges = learn_bpe(
            corpus["tr"][:100_000].cpu().numpy().astype(np.uint8).tobytes(), 64
        )
        bpe_ons = torch.as_tensor(apply_bpe(raw, merges), device=DEVICE)
        for seed in cfg.seeds:
            shared = None
            cost = None
            lm = {}
            for arm in cfg.arms:
                if arm in ("byte-GPT", "byte-AR", "unigram", "gzip"):
                    if (
                        arm == "unigram"
                        and ("compression", name, None, "unigram", None)
                        not in store.seen
                    ):
                        store.add(eval_unigram(cfg, corpus, starts))
                    elif (
                        arm == "gzip"
                        and ("compression", name, None, "gzip", None) not in store.seen
                    ):
                        store.add(eval_gzip(cfg, corpus, starts))
                    elif arm in ("byte-GPT", "byte-AR"):
                        if arm not in lm:
                            print(f"[phase 2] {arm} {name} seed={seed}")
                            lm[arm] = train_lm(cfg, corpus, seed, arm)
                        if (
                            not resume
                            or ("compression", name, None, arm, None) not in store.seen
                        ):
                            r = eval_lm(cfg, corpus, lm[arm], starts, arm)
                            store.add(r)
                    continue
                if shared is None:
                    print(f"[phase 2] shared codec {name} seed={seed}")
                    shared = train_codec(cfg, corpus, seed, "shared", True)
                    print(f"[phase 2] cost field {name} seed={seed}")
                    cost = train_cost(cfg, corpus, shared, seed)
                for lam in cfg.lam_values:
                    if (
                        resume
                        and ("structure", name, seed, arm, float(lam)) in store.seen
                    ):
                        continue
                    r = eval_arm(
                        cfg, corpus, shared, cost, arm, lam, seed, starts, ons, bpe_ons
                    )
                    store.add(r)


def run_all(cfg, resume):
    base = Path(os.environ.get("HNCM_OUT", "runs"))
    fp = fingerprint(cfg)[:8]
    outdir = base / f"p{cfg.phase}" / f"{cfg.preset}-{fp}"
    corpora = {n: get_corpus(n) for n in cfg.corpora}
    shas = {n: c["sha"] for n, c in corpora.items()}
    store = Store(outdir, cfg, shas)
    path = base / "latest"
    try:
        if path.is_symlink() or path.exists():
            path.unlink()
        path.symlink_to(os.path.relpath(outdir, base))
    except OSError:
        pass
    print(f"out={outdir} device={DEVICE} cfg={fp}")
    if cfg.phase == 0:
        run_phase0(cfg, corpora, store, resume)
    elif cfg.phase == 1:
        run_phase1(cfg, corpora, store, resume)
    else:
        run_phase2(cfg, corpora, store, resume)
    store.finish()
    print(f"wrote {store.path}")


# %%
# ---------------- report --------------------------------------------------------------------
def pick(res, claim=None, corpus=None, arm=None, lam=None, seed=None):
    out = []
    for r in res["runs"]:
        if claim is not None and r["claim"] != claim:
            continue
        if corpus is not None and r["corpus"] != corpus:
            continue
        if arm is not None and r["arm"] != arm:
            continue
        if lam is not None and r["lambda"] != lam:
            continue
        if seed is not None and r["seed"] != seed:
            continue
        out.append(r)
    return out


def fmt(v):
    return "" if v is None else f"{v:.3f}"


def mean_ci(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return 0.0, 0.0
    if len(x) == 1:
        return float(x[0]), 0.0
    return float(x.mean()), float(1.96 * x.std(ddof=1) / math.sqrt(len(x)))


def boot_diff(a, b, reps=1000):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    if n < 2:
        return 0.0, 0.0, 0.0
    d = a[:n] - b[:n]
    rs = np.random.RandomState(0)
    m = np.array([d[rs.randint(0, n, n)].mean() for _ in range(reps)])
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def agg(res, claim, corpus, arm, lam=None):
    rr = pick(res, claim, corpus, arm, lam)
    if not rr:
        return None
    tot = [r["total_bpb"] for r in rr]
    return {
        "n": len(rr),
        "total": mean_ci(tot),
        "recon": mean_ci([r["recon_bpb"] for r in rr]),
        "code": mean_ci([r["code_bpb"] for r in rr]),
        "boundary": mean_ci([r["boundary_bpb"] for r in rr]),
        "align": mean_ci([r["align"] for r in rr if r["align"] is not None]),
        "util": mean_ci([r["util"] for r in rr if r["util"] is not None]),
        "info": mean_ci([r["code_info"] for r in rr if r["code_info"] is not None]),
        "calls": mean_ci([r["calls"] for r in rr]),
        "wall": mean_ci([r["wall_sec"] for r in rr]),
        "params": rr[0]["params"],
        "runs": rr,
    }


def gate(gid, claim, value, threshold, status):
    return {
        "id": gid,
        "claim": claim,
        "value": value,
        "threshold": threshold,
        "status": status,
    }


def gates(res):
    out = []
    phase = res["cfg"]["phase"]
    corpora = res["cfg"]["corpora"]
    if phase == 0:
        for name in corpora:
            scalar = agg(res, "health", name, "rel-pos")
            const = agg(res, "health", name, "constant-code")
            uni = agg(res, "compression", name, "unigram")
            if scalar and const and uni:
                delta = const["recon"][0] - scalar["recon"][0]
                ok = (
                    scalar["util"][0] >= res["cfg"]["g0_util"]
                    and delta >= res["cfg"]["g0_delta"]
                    and scalar["info"][0] >= res["cfg"]["g0_info"]
                    and scalar["total"][0] < uni["total"][0]
                )
                out.append(
                    gate(
                        "G0",
                        f"codec health {name}",
                        f"util={scalar['util'][0]:.3f} delta={delta:.3f} H={scalar['info'][0]:.3f} total={scalar['total'][0]:.3f} uni={uni['total'][0]:.3f}",
                        f"util>={res['cfg']['g0_util']}, delta>={res['cfg']['g0_delta']}, H>={res['cfg']['g0_info']}, total<unigram",
                        "pass" if ok else "fail",
                    )
                )
            ab = agg(res, "health", name, "abs-pos")
            rel = agg(res, "health", name, "rel-pos")
            if ab and rel:
                d = rel["recon"][0] - ab["recon"][0]
                st = (
                    "pass"
                    if d <= res["cfg"]["g6_pass"]
                    else ("fail" if d > res["cfg"]["g6_fail"] else "warn")
                )
                out.append(
                    gate(
                        "G6",
                        f"leak {name}",
                        f"delta_recon(rel-abs)={d:.3f}",
                        f"<={res['cfg']['g6_pass']}",
                        st,
                    )
                )
            out.append(
                gate(
                    "G9",
                    f"statistics {name}",
                    f"seeds={len(res['cfg']['seeds'])}",
                    "3 seeds",
                    "pass" if len(res["cfg"]["seeds"]) >= 3 else "warn",
                )
            )
    elif phase in (1, 2):
        for name in corpora:
            uni = agg(res, "compression", name, "unigram")
            gpt = agg(res, "compression", name, "byte-GPT")
            gz = agg(res, "compression", name, "gzip")
            if name == "shakespeare" and uni:
                dp = agg(res, "structure", name, "DP")
                if dp:
                    ok = dp["total"][0] < uni["total"][0] - res["cfg"]["p1a"]
                    out.append(
                        gate(
                            "G1",
                            "P1a",
                            f"DP={dp['total'][0]:.3f} unigram={uni['total'][0]:.3f}",
                            f"DP<uni-{res['cfg']['p1a']}",
                            "pass" if ok else "fail",
                        )
                    )
                    if gpt:
                        ok = dp["total"][0] <= gpt["total"][0] + res["cfg"]["p1b"]
                        out.append(
                            gate(
                                "G2",
                                "P1b",
                                f"DP={dp['total'][0]:.3f} GPT={gpt['total'][0]:.3f}",
                                f"DP<=GPT+{res['cfg']['p1b']}",
                                "pass" if ok else "fail",
                            )
                        )
                    if gz:
                        ok = dp["total"][0] <= gz["total"][0] + res["cfg"]["p1c"]
                        out.append(
                            gate(
                                "G2b",
                                "P1c",
                                f"DP={dp['total'][0]:.3f} gzip={gz['total'][0]:.3f}",
                                f"DP<=gzip+{res['cfg']['p1c']}",
                                "pass" if ok else "fail",
                            )
                        )
            for lam in res["cfg"]["lam_values"]:
                fixed = agg(res, "structure", name, "fixed", lam)
                dp = agg(res, "structure", name, "DP", lam)
                if not fixed or not dp:
                    continue
                if name == "synseg":
                    orc = agg(res, "structure", name, "oracle", lam)
                    if orc:
                        # paired per-window difference
                        i, lo, hi = boot_diff(
                            fixed["runs"][0]["samples"],
                            orc["runs"][0]["samples"],
                            res["cfg"]["boot"],
                        )
                        out.append(
                            gate(
                                "G4",
                                f"oracle ceiling {name} lam={lam}",
                                f"I_oracle={i:.3f} CI=[{lo:.3f},{hi:.3f}]",
                                "I>0, CI low>0",
                                "pass" if i > 0 and lo > 0 else "fail",
                            )
                        )
                else:
                    orc = agg(res, "structure", name, "oracle", lam)
                    if orc:
                        i, lo, hi = boot_diff(
                            fixed["runs"][0]["samples"],
                            orc["runs"][0]["samples"],
                            res["cfg"]["boot"],
                        )
                        out.append(
                            gate(
                                "G4n",
                                f"no-oracle null {name} lam={lam}",
                                f"I_oracle={i:.3f} CI=[{lo:.3f},{hi:.3f}]",
                                "|I|<=2 SEM",
                                "pass"
                                if abs(i) <= 2 * abs(hi - lo) / 3.92 + 1e-9
                                else "fail",
                            )
                        )
                orc = agg(res, "structure", name, "oracle", lam)
                if orc:
                    i, lo, hi = boot_diff(
                        fixed["runs"][0]["samples"],
                        dp["runs"][0]["samples"],
                        res["cfg"]["boot"],
                    )
                    io, _, _ = boot_diff(
                        fixed["runs"][0]["samples"],
                        orc["runs"][0]["samples"],
                        res["cfg"]["boot"],
                    )
                    sem = (hi - lo) / 3.92
                    ok = (
                        i >= max(res["cfg"]["p3_alpha"] * io, 2 * sem)
                        and dp["align"][0] >= fixed["align"][0] + res["cfg"]["p3_align"]
                    )
                    out.append(
                        gate(
                            "G5",
                            f"DP deployable {name} lam={lam}",
                            f"I_DP={i:.3f} I_oracle={io:.3f} align={dp['align'][0]:.3f}/{fixed['align'][0]:.3f}",
                            "I_DP>=max(.5 I_oracle,2SEM); align+0.05",
                            "pass" if ok else "fail",
                        )
                    )
                    if orc:
                        ok = orc["total"][0] < dp["total"][0] < fixed["total"][0]
                        out.append(
                            gate(
                                "G5b",
                                f"DP between {name} lam={lam}",
                                f"oracle={orc['total'][0]:.3f} DP={dp['total'][0]:.3f} fixed={fixed['total'][0]:.3f}",
                                "oracle<DP<fixed",
                                "pass" if ok else "fail",
                            )
                        )
                    for aid in ("BPE", "IsoWarp"):
                        alt = agg(res, "structure", name, aid, lam)
                        if alt:
                            ok = dp["total"][0] <= alt["total"][0]
                            out.append(
                                gate(
                                    "G5c" if aid == "BPE" else "G5d",
                                    f"DP vs {aid} {name} lam={lam}",
                                    f"DP={dp['total'][0]:.3f} {aid}={alt['total'][0]:.3f}",
                                    "DP<=" + aid,
                                    "pass" if ok else "fail",
                                )
                            )
            ar = agg(res, "compression", name, "byte-AR")
            dp = agg(res, "structure", name, "DP")
            if ar and dp:
                pairs = []
                for l1 in res["cfg"]["lam_values"]:
                    d1 = agg(res, "structure", name, "DP", l1)
                    if d1:
                        pairs.append((abs(d1["total"][0] - ar["total"][0]), d1, l1))
                if pairs:
                    dist, d1, l1 = min(pairs, key=lambda z: z[0])
                    if dist <= res["cfg"]["p2_match"]:
                        ratio = ar["calls"][0] / d1["calls"][0]
                        out.append(
                            gate(
                                "G3",
                                f"calls {name}",
                                f"AR/DP={ratio:.1f} at |dTotal|={dist:.3f}",
                                f">={res['cfg']['p2_calls']}",
                                "pass" if ratio >= res["cfg"]["p2_calls"] else "fail",
                            )
                        )
                        ok = d1["wall"][0] <= res["cfg"]["p2_wall"] * ar["wall"][0]
                        out.append(
                            gate(
                                "G3b",
                                f"wall {name}",
                                f"DP={d1['wall'][0]:.4f} AR={ar['wall'][0]:.4f}",
                                f"DP<={res['cfg']['p2_wall']} AR",
                                "pass" if ok else "fail",
                            )
                        )
                    else:
                        out.append(
                            gate(
                                "G3",
                                f"calls {name}",
                                f"best |dTotal|={dist:.3f}",
                                f"<={res['cfg']['p2_match']}",
                                "fail",
                            )
                        )
            vq = agg(res, "structure", name, "vq")
            sc = agg(res, "structure", name, "fixed")
            if vq and sc:
                out.append(
                    gate(
                        "G7",
                        f"quant {name}",
                        f"scalar={sc['total'][0]:.3f} VQ={vq['total'][0]:.3f}",
                        "scalar<=VQ+0.05",
                        "pass" if sc["total"][0] <= vq["total"][0] + 0.05 else "fail",
                    )
                )
            mf = agg(res, "structure", name, "meanfield")
            lr = agg(res, "structure", name, "fixed")
            if mf and lr:
                out.append(
                    gate(
                        "G8",
                        f"prior {name}",
                        f"meanfield={mf['total'][0]:.3f} lowrank={lr['total'][0]:.3f}",
                        "lowrank<=meanfield+0.05",
                        "pass" if lr["total"][0] <= mf["total"][0] + 0.05 else "fail",
                    )
                )
            out.append(
                gate(
                    "G9",
                    f"statistics {name}",
                    f"seeds={len(res['cfg']['seeds'])}",
                    "3 seeds",
                    "pass" if len(res["cfg"]["seeds"]) >= 3 else "warn",
                )
            )
    return out


def claim_verdicts(gs):
    def st(gid):
        return {g["status"] for g in gs if g["id"] == gid}

    def ok(*ids):
        return all("fail" not in st(i) and "pass" in st(i) for i in ids)

    return {
        "P1": ok("G1", "G2", "G2b") if st("G1") else False,
        "P2": ok("G3", "G3b") if st("G3") else False,
        "P3": ok("G4", "G5", "G5b", "G5c", "G5d") if st("G4") else False,
    }


def write_report(res, outdir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gs = gates(res)
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
    lines = [
        "# H-NCM v6 - report",
        "",
        f"format: {res['format']}  cfg: `{res['cfg_fingerprint']}`  device: {res['device']}  torch: {res['torch']}",
        f"corpora: {res['corpora']}",
        "",
        "## Runs",
        "",
        "| claim | corpus | seed | arm | lambda | total | recon | code | boundary | align | util | H(code) | calls | wall |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(
        res["runs"],
        key=lambda z: (
            z["corpus"],
            z["arm"],
            z["lambda"] if z["lambda"] is not None else -1,
            z["seed"] if z["seed"] is not None else -1,
        ),
    ):
        lines.append(
            f"| {r['claim']} | {r['corpus']} | {r['seed']} | {r['arm']} | {r['lambda']} | {r['total_bpb']:.3f} | {r['recon_bpb']:.3f} | {r['code_bpb']:.3f} | {r['boundary_bpb']:.3f} | {fmt(r['align'])} | {fmt(r['util'])} | {fmt(r['code_info'])} | {r['calls']:.0f} | {r['wall_sec']:.4f} |"
        )
    lines += [
        "",
        "## Gates",
        "",
        "| gate | claim | value | threshold | status |",
        "|---|---|---|---|---|",
    ]
    for g in gs:
        lines.append(
            f"| {g['id']} | {g['claim']} | {g['value']} | {g['threshold']} | **{g['status']}** |"
        )
    v = claim_verdicts(gs)
    lines += ["", "## Verdicts", ""]
    for k, ok in v.items():
        lines.append(f"- **{k}: {'supported' if ok else 'not supported'}**")
    lines += ["", "## Decision", ""]
    fails = [g for g in gs if g["status"] == "fail"]
    if not gs:
        lines.append(
            "No gates were computed. Run a phase first; a smoke diagnostic is not evidence."
        )
    elif any(g["id"] == "G0" and g["status"] == "fail" for g in gs):
        lines.append(
            "G0 fails: the codec is unhealthy. Fix the quantizer/prior/decoder before the DP. Do not interpret structure results."
        )
    elif any(g["id"] == "G6" and g["status"] == "fail" for g in gs):
        lines.append(
            "G6 fails: absolute position leaks. Fix the decoder before the DP."
        )
    elif any(g["id"] == "G4" and g["status"] == "fail" for g in gs):
        lines.append(
            "G4 fails: the oracle ceiling is not positive. Report P1/P2 if their gates hold; stop structure work."
        )
    elif any(g["id"] == "G5" and g["status"] == "fail" for g in gs):
        lines.append(
            "G4 passes but G5 fails: the oracle helps, the paid DP does not. Report the ceiling and the negative."
        )
    elif fails:
        lines.append(
            f"{len(fails)} gate(s) fail; report every claim, including negatives. Do not cherry-pick."
        )
    else:
        lines.append(
            "All computed gates pass. Replicate at the pre-registered scale before claiming."
        )
    lines.append("A negative result with clean controls is publishable.")
    (path / "REPORT.md").write_text("\n".join(lines) + "\n")

    for corpus in res["cfg"]["corpora"]:
        rows = []
        for arm in sorted({r["arm"] for r in res["runs"] if r["corpus"] == corpus}):
            a = agg(res, "structure", corpus, arm)
            if a is None:
                a = agg(res, "health", corpus, arm)
            if a is None:
                a = agg(res, "compression", corpus, arm)
            if a is not None:
                rows.append((arm, a["total"][0], a["total"][1]))
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(max(6, len(rows) * 0.6), 3.5))
        ax.bar(
            [r[0] for r in rows],
            [r[1] for r in rows],
            yerr=[r[2] for r in rows],
            capsize=3,
        )
        ax.set_ylabel("total bpb")
        ax.set_title(f"hncm6 {corpus}")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(path / f"hncm6_{corpus}.png", dpi=140)
        plt.close(fig)
    return path / "REPORT.md"


def report(results_path, out=None):
    p = Path(results_path)
    if not p.exists():
        raise SystemExit(f"no results at {p}")
    res = json.loads(p.read_text())
    outdir = Path(out) if out else p.parent
    rp = write_report(res, outdir)
    gs = gates(res)
    print(f"report={rp}")
    print("gate status:")
    for g in gs:
        print(
            f"  {g['id']:4s} {g['status']:4s} {g['claim']}: {g['value']} [{g['threshold']}]"
        )
    print("verdicts:", claim_verdicts(gs))


# %%
# ---------------- self-tests ----------------------------------------------------------------
def _selftest():
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = Cfg(**PRESETS["smoke"])
    assert cfg.L % cfg.fixed_len == 0 and cfg.lengths[0] == 1
    # DP vs brute force
    L = 8
    lengths = (1, 2, 4)
    cost = torch.rand(1, L, len(lengths))
    m = dp_tile(cost, lengths, L)
    bm, best = brute_cover(cost[0], lengths, L)
    assert m[0].tolist() == bm, (m[0].tolist(), bm)
    assert m[0, 0] and int(torch.tensor(bm).sum()) >= 1
    # exact cover
    starts = torch.tensor([0, 4, 8], device=DEVICE)
    ons = np.zeros(64, dtype=bool)
    ons[[0, 5, 9, 21, 40]] = True
    om = oracle_masks(ons, starts, 8)
    assert om[:, 0].all() and om.shape == (3, 8)
    # cumsum pool vs naive
    H = torch.randn(2, 8, 4)
    bnd = torch.tensor(
        [[1, 0, 0, 1, 0, 1, 0, 0], [1, 0, 1, 0, 0, 0, 0, 1]], dtype=torch.bool
    )
    z, cnt, ids = pool(H, bnd)
    for i in range(2):
        p = 0
        j = 0
        while p < 8:
            e = p + 1
            while e < 8 and not bnd[i, e]:
                e += 1
            assert torch.allclose(z[i, j], H[i, p:e].mean(0), atol=1e-5)
            p = e
            j += 1
    # entropy round-trip within a step
    x = torch.rand(4, 16, cfg.d_c, device=DEVICE) * 2 - 1
    q = cfg.levels
    step = 2.0 / q
    mu = torch.zeros_like(x)
    sig = torch.ones_like(x)
    ctx = torch.randn(4, 16, cfg.d + 2 * cfg.d_c + cfg.rank, device=DEVICE)
    yq, k, rate, _ = (
        ScalarQ(cfg).to(DEVICE).quantize(x, mu, torch.zeros_like(x), sig, ctx, False)
    )
    assert yq.shape == x.shape and torch.isfinite(rate).all()
    u = torch.tanh(x)
    k = torch.floor((u + 1) / step).clamp(0, q - 1).long()
    dq = (k.float() + 0.5) * step - 1
    assert float((u - dq).abs().max()) <= step / 2 + 1e-5
    assert k.min() >= 0 and k.max() < q
    # codec shapes and finite loss
    model = Codec(cfg).to(DEVICE)
    x = torch.randint(0, 256, (3, cfg.L), device=DEVICE)
    b = fixed_mask(3, cfg.L, cfg.fixed_len)
    loss, rec, code, k, _ = model(x, b, True)
    assert loss.shape == (3,) and torch.isfinite(loss).all()
    assert rec.shape == (3,) and code.shape == (3,)
    # no absolute position in span mode
    assert model.dec.abs.weight.shape[0] == cfg.L
    # fingerprint changes with cfg
    assert fingerprint(cfg) != fingerprint(replace(cfg, levels=8))
    # masks cover every byte
    rm = random_masks(4, cfg.L, cfg.lengths, np.random.RandomState(1))
    assert rm[:, 0].all() and (rm.sum(1) >= 1).all()
    print("selftest: ok")


# %%
def parse_list(s):
    if s is None:
        return None
    return tuple(x for x in s.split(",") if x != "")


def main(argv=None):
    p = argparse.ArgumentParser(description="H-NCM v6")
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--preset", default=os.environ.get("HNCM_PRESET", "smoke"))
    r.add_argument("--phase", type=int, default=0)
    r.add_argument("--seeds", default=None)
    r.add_argument("--corpora", default=None)
    r.add_argument("--arms", default=None)
    r.add_argument(
        "--resume", type=int, default=int(os.environ.get("HNCM_RESUME", "0"))
    )
    q = sub.add_parser("report")
    q.add_argument("--results", default="runs/latest/results.json")
    q.add_argument("--out", default=None)
    sub.add_parser("selftest")
    a = p.parse_args(argv)
    if a.cmd == "selftest":
        _selftest()
    elif a.cmd == "run":
        cfg = build_cfg(
            a.preset,
            a.phase,
            parse_list(a.seeds),
            parse_list(a.corpora),
            parse_list(a.arms),
        )
        run_all(cfg, bool(a.resume))
    elif a.cmd == "report":
        report(a.results, a.out)
    else:
        p.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
