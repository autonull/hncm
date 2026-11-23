# %% [markdown]
# # H-NCM v2: Hierarchical Neural Concept Modeling (rebuilt from the v1 post-mortems)
#
# **Question.** Does learned, content-adaptive chunking (H-Net idea) help *more* under coarse-to-fine parallel
# decoding of residual-VQ concept codes (NCM idea) than under autoregressive decoding of the same codes?
#
# **What v1 taught us, and what changed here**
#
# | v1 failure | v2 fix |
# |---|---|
# | Codes were near-lossless (5.07 of 5.79 bpb was RVQ codes); generator could not compress them | **Rate-aware codec**: differentiable code-rate term, level dropout (coarse levels must be informative), small codebooks |
# | Different tokenizers *and* different generators per cell, so effects were confounded | **One codec per chunking mode; AR and coarse-to-fine priors are trained on the SAME frozen codes**, same architecture, same token set. Only the attention pattern differs |
# | Dynamic chunking collapsed to ~1.7 bytes/chunk and needed ~1 bpb of boundary bits | **Jittered chunking**: exactly one boundary per stride-cell, at a learned offset. Rate is fixed by construction (no collapse), boundaries are cheap *tokens* (vocab = stride) that the prior generates |
# | Router got almost no learning signal | Chunk vectors are pooled through a **soft-assignment straight-through** map, so gradients reach the boundary choice |
# | ChatGPT v1: 4-example bootstrap, 160 steps, placeholder metrics and invented samples | All tables and plots are computed **only from `results.json`**; no fallbacks. Bootstrap is over held-out windows (paired), plus across-seed spread |
# | No strong reference | Exact-likelihood **byte GPT** baseline and a **unigram** baseline; samples are scored by the byte GPT |
#
# **Metric.** Bits per byte (bpb), lower is better. Byte GPT: exact. Concept models: a *two-part code upper bound*
# `-log p(x) <= bits(codes | prior) + bits(x | codes)` (valid because the tokenizer is a deterministic function of x).
# The within-family contrasts (fixed vs dynamic, AR vs CF) share a codec, so the interaction
# `I = (bpb[fixed,CF] - bpb[dyn,CF]) - (bpb[fixed,AR] - bpb[dyn,AR])` is a like-for-like comparison. `I > 0` means dynamic
# chunking helps more under CF than under AR.
#
# **Honest status.** Written without being executed by its author. The self-tests below run first and raise on failure.
# Treat the first run as a debugging run (`HNCM_PRESET=smoke`), then `quick`, then `full`.

# %%
import os, sys, math, time, json, random, urllib.request, glob
from dataclasses import dataclass, asdict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LN2 = math.log(2.0)
MAXOFF = 31                                   # clamp for in-chunk offsets / chunk lengths

# ---- EDIT ME ---------------------------------------------------------------------------------
PRESET = os.environ.get('HNCM_PRESET', 'quick')     # 'smoke' (minutes, CPU ok) | 'quick' | 'full'
OUT_DIR = os.environ.get('HNCM_OUT', 'hncm_v2_out')
RUN_TESTS = True
# ----------------------------------------------------------------------------------------------

@dataclass
class Cfg:
    L: int = 128            # window length (bytes)
    R: int = 16             # coarse grid cell (bytes); coarse code per cell
    stride: int = 4         # chunk stride: M = L/stride chunks, exactly one per cell (fixed or jittered)
    K1: int = 32            # coarse codebook
    K: int = 64             # RVQ codebook per level
    Q: int = 3              # RVQ levels
    d: int = 128
    heads: int = 4
    enc: int = 2
    main: int = 2
    dec: int = 2
    gen: int = 4            # prior layers
    base_layers: int = 6    # byte GPT layers
    bs: int = 64
    codec_lr: float = 1e-3
    prior_lr: float = 5e-4
    base_lr: float = 1e-3
    codec_steps: int = 2000
    prior_steps: int = 2500
    base_steps: int = 2500
    lam_list: tuple = (0.5,)   # weight of the differentiable rate term in codec training
    lam_warm: float = 0.3      # fraction of codec steps over which lambda ramps from 0
    seeds: tuple = (0, 1)
    eval_windows: int = 512
    sample_n: int = 16

PRESETS = {
    'smoke': dict(d=64, heads=4, enc=1, main=1, dec=1, gen=2, base_layers=2, L=64, R=16, K1=16, K=32, Q=2, bs=16,
                  codec_steps=60, prior_steps=60, base_steps=60, seeds=(0,), eval_windows=64, sample_n=4),
    'quick': dict(),
    'full': dict(seeds=(0, 1, 2), lam_list=(0.25, 1.0), codec_steps=4000, prior_steps=6000, base_steps=6000,
                 eval_windows=1024),
}
CFG = Cfg(**PRESETS[PRESET])
assert CFG.L % CFG.R == 0 and CFG.L % CFG.stride == 0 and CFG.R % CFG.stride == 0 and CFG.d % CFG.heads == 0
os.makedirs(OUT_DIR, exist_ok=True)
if DEVICE == 'cpu' and PRESET != 'smoke':
    print('WARNING: no GPU. In Colab: Runtime > Change runtime type > T4 GPU, or use HNCM_PRESET=smoke.')
print(f'device={DEVICE} preset={PRESET}')
print(CFG)

# %%
# ----------------------------------------------------------------------------------------------
# Data: tiny-shakespeare bytes (fallback: Python stdlib source so the notebook always runs; the
# source is recorded in results.json so a fallback run can never be mistaken for the real one)
# ----------------------------------------------------------------------------------------------
def load_bytes():
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    try:
        raw = urllib.request.urlopen(url, timeout=15).read(); src = 'tinyshakespeare'
    except Exception:
        parts = []
        for f in sorted(glob.glob(os.path.join(os.path.dirname(os.__file__), '*.py'))):
            try:
                with open(f, 'rb') as fh: parts.append(fh.read())
            except Exception:
                pass
        raw = b'\n'.join(parts)[:3_000_000]; src = 'python-stdlib (FALLBACK, not Shakespeare)'
    arr = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()
    n = int(0.9 * len(arr))
    return arr[:n].to(DEVICE), arr[n:].to(DEVICE), src

DATA_TR, DATA_VA, DATA_SRC = load_bytes()
_cnt = torch.bincount(DATA_TR, minlength=256).float() + 1
UNI_BITS = -torch.log2(_cnt / _cnt.sum())                 # -log2 unigram prob per byte
print(f'data={DATA_SRC} train={len(DATA_TR):,} val={len(DATA_VA):,} bytes')

def windows(data, starts, L):
    return data[starts[:, None] + torch.arange(L, device=data.device)[None]]

def train_batch(L, bs):
    return windows(DATA_TR, torch.randint(0, len(DATA_TR) - L, (bs,), device=DEVICE), L)

def val_starts(L, n):
    rs = np.random.RandomState(1234)                       # identical windows for every model and seed (paired)
    return torch.tensor(rs.randint(0, len(DATA_VA) - L, size=n), device=DEVICE)

# %%
# ----------------------------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(s, d, h, causal):
        super().__init__()
        s.h, s.causal = h, causal
        s.ln1 = nn.LayerNorm(d); s.qkv = nn.Linear(d, 3 * d, bias=False); s.o = nn.Linear(d, d, bias=False)
        s.ln2 = nn.LayerNorm(d); s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(s, x):
        B, T, D = x.shape
        q, k, v = s.qkv(s.ln1(x)).view(B, T, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=s.causal)
        x = x + s.o(a.transpose(1, 2).reshape(B, T, D))
        return x + s.mlp(s.ln2(x))

class Stack(nn.Module):
    def __init__(s, n, d, h, causal):
        super().__init__()
        s.blocks = nn.ModuleList(Block(d, h, causal) for _ in range(n)); s.ln = nn.LayerNorm(d)
    def forward(s, x):
        for b in s.blocks: x = b(x)
        return s.ln(x)

def take(x, idx):                      # x (B,T,D), idx (B,N) -> (B,N,D)
    return torch.gather(x, 1, idx[..., None].expand(-1, -1, x.size(-1)))

def geometry(b):
    """b: (B,L) bool, b[:,0]=True. Chunk bookkeeping; all indices are byte positions."""
    B, L = b.shape; dev = b.device
    counts = b.sum(1); M = int(counts.max())
    order = torch.argsort((~b).long(), dim=1, stable=True)[:, :M]            # chunk starts first, in order
    slot = torch.arange(M, device=dev)[None]
    valid = slot < counts[:, None]
    m_idx = b.long().cumsum(1) - 1                                            # chunk id of every byte
    nxt = torch.cat([order[:, 1:], torch.full((B, 1), L, device=dev, dtype=order.dtype)], 1)
    nxt = torch.where(slot + 1 < counts[:, None], nxt, torch.full_like(nxt, L))
    length = (nxt - order).clamp(min=0) * valid
    start_b = torch.gather(order, 1, m_idx)
    offset = torch.arange(L, device=dev)[None] - start_b                      # position inside its chunk
    return dict(order=order, valid=valid, counts=counts, M=M, m_idx=m_idx, length=length, offset=offset)

class RVQ(nn.Module):
    """Residual VQ. Returns straight-through quantization (optionally with level dropout), codes, VQ loss, and a
    differentiable rate proxy: expected code length of the *soft* assignment under a learned per-level marginal."""
    def __init__(s, d, K, Q, beta=0.25):
        super().__init__()
        s.d, s.K, s.Q, s.beta = d, K, Q, beta
        s.cb = nn.Parameter(torch.randn(Q, K, d) * 0.1)
        s.pi = nn.Parameter(torch.zeros(Q, K))
        s.register_buffer('usage', torch.zeros(Q, K))
        s.last = [None] * Q; s._inited = False
    def forward(s, z, keep=None):
        B, M, D = z.shape
        if s.training and not s._inited:
            with torch.no_grad():
                tmp = z.detach().reshape(-1, D)
                for q in range(s.Q):
                    s.cb.data[q] = tmp[torch.randint(len(tmp), (s.K,), device=z.device)]
                    tmp = tmp - s.cb.data[q][torch.cdist(tmp, s.cb.data[q]).argmin(1)]
            s._inited = True
        res, qsum, codes = z, torch.zeros_like(z), []
        vq, rate = z.new_zeros(()), z.new_zeros(())
        for q in range(s.Q):
            cb = s.cb[q]
            dist = res.pow(2).sum(-1, keepdim=True) - 2 * res @ cb.t() + cb.pow(2).sum(-1)     # (B,M,K)
            idx = dist.argmin(-1); e = cb[idx]
            vq = vq + ((e - res.detach()) ** 2).mean() + s.beta * ((res - e.detach()) ** 2).mean()
            tau = (0.1 * dist.detach().mean()).clamp_min(1e-6)
            soft = F.softmax(-dist / tau, -1)
            rate = rate - (soft * F.log_softmax(s.pi[q], -1)).sum(-1).mean()               # nats per code
            if s.training:
                with torch.no_grad():
                    s.usage[q] += torch.bincount(idx.reshape(-1), minlength=s.K).float()
                    s.last[q] = res.detach().reshape(-1, D)
            use = e if keep is None else e * keep[:, q, None, None].to(e.dtype)
            qsum = qsum + use
            res = res - e.detach()
            codes.append(idx)
        zq = z + (qsum - z).detach()                                                         # straight-through
        return zq, torch.stack(codes, -1), vq, rate
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
# ----------------------------------------------------------------------------------------------
# Codec: bytes -> (coarse code per R-byte cell, M chunks, RVQ codes per chunk) -> bytes
# "fixed":  chunk j starts at j*stride.
# "dynamic" (jittered): chunk j starts at j*stride + off_j, off_j in [0,stride) chosen by a learned router.
#           M is constant (no rate collapse); boundaries are generated as tokens by the prior.
# ----------------------------------------------------------------------------------------------
class Codec(nn.Module):
    def __init__(s, cfg, dynamic):
        super().__init__()
        s.cfg, s.dynamic = cfg, dynamic
        d, L = cfg.d, cfg.L
        s.G, s.M = L // cfg.R, L // cfg.stride
        s.emb = nn.Embedding(256, d); s.pos = nn.Embedding(L, d)
        s.enc = Stack(cfg.enc, d, cfg.heads, False)
        if dynamic:
            s.router = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
            nn.init.zeros_(s.router[1].weight); nn.init.zeros_(s.router[1].bias)           # starts == fixed chunking
        s.cpos = nn.Embedding(s.M, d)
        s.main = Stack(cfg.main, d, cfg.heads, False)
        s.grid_proj = nn.Linear(d, d); s.vq1 = RVQ(d, cfg.K1, 1)
        s.proj = nn.Linear(d, d); s.rvq = RVQ(d, cfg.K, cfg.Q)
        s.dpos = nn.Embedding(L, d); s.doff = nn.Embedding(MAXOFF + 1, d); s.dlen = nn.Embedding(MAXOFF + 1, d)
        s.dec = Stack(cfg.dec, d, cfg.heads, False); s.head = nn.Linear(d, 256)

    def chunk(s, h):
        B, L, _ = h.shape; st = s.cfg.stride; C = L // st
        a = s.router(h).squeeze(-1).view(B, C, st)
        soft = F.softmax(a, -1)
        idx = a.argmax(-1); idx[:, 0] = 0                                                   # first chunk starts at byte 0
        hard = F.one_hot(idx, st).float()
        soft = torch.cat([hard[:, :1], soft[:, 1:]], 1)
        ste = hard + soft - soft.detach()                                                   # forward: hard
        return hard.reshape(B, L).bool(), ste.reshape(B, L)

    def assign(s, b, beta, g):
        hard = F.one_hot(g['m_idx'], s.M).float()                                           # (B,L,M)
        if not s.dynamic: return hard
        cum = beta.cumsum(1) - 1                                                            # soft chunk coordinate
        slots = torch.arange(s.M, device=b.device, dtype=beta.dtype)
        sa = F.softmax(-(cum[..., None] - slots) ** 2 / (2 * 0.5 ** 2), -1)
        return hard + sa - sa.detach()

    def encode(s, x, keep=None):
        cfg = s.cfg; B, L = x.shape; R = cfg.R
        h = s.enc(s.emb(x) + s.pos.weight[:L])
        if s.dynamic:
            b, beta = s.chunk(h)
        else:
            b = (torch.arange(L, device=x.device) % cfg.stride == 0)[None].expand(B, L).clone(); beta = b.float()
        g = geometry(b)
        A = s.assign(b, beta, g)
        cnt = g['length'].float().clamp_min(1)
        xc = (A.transpose(1, 2) @ h) / cnt[..., None] + s.cpos.weight[None]                 # mean-pool chunk bytes
        z = s.proj(s.main(xc))
        hc = s.grid_proj(h.reshape(B, s.G, R, -1).mean(2))
        e1, c1, l1, r1 = s.vq1(hc)
        e1c = take(e1, g['order'] // R)
        zq, codes, l2, r2 = s.rvq(z - e1c, keep)
        return dict(zq=zq, codes=codes, c1=c1, e1c=e1c, g=g, vq=l1 + l2, rate1=r1, rate2=r2)

    def decode(s, zfull, g, L):
        zup = take(zfull, g['m_idx'])
        lenb = torch.gather(g['length'], 1, g['m_idx'])
        din = zup + s.dpos.weight[:L] + s.doff(g['offset'].clamp(max=MAXOFF)) + s.dlen(lenb.clamp(max=MAXOFF))
        return s.head(s.dec(din))

    def loss(s, x, lam, level_dropout=True):
        cfg = s.cfg; B, L = x.shape
        keep = None
        if level_dropout and s.training:
            qk = torch.where(torch.rand(B, device=x.device) < 0.5,
                             torch.full((B,), cfg.Q, device=x.device),
                             torch.randint(1, cfg.Q + 1, (B,), device=x.device))
            keep = torch.arange(cfg.Q, device=x.device)[None] < qk[:, None]
        o = s.encode(x, keep)
        logits = s.decode(o['zq'] + o['e1c'], o['g'], L)
        recon = F.cross_entropy(logits.reshape(-1, 256), x.reshape(-1))                    # nats/byte
        rate = o['rate1'] / cfg.R + o['rate2'] / cfg.stride                                 # nats/byte (proxy)
        loss = recon + lam * rate + o['vq']
        return loss, dict(recon_bpb=recon.detach() / LN2, rate_bpb=rate.detach() / LN2)

    def tokens_from(s, o):
        g = o['g']; cfg = s.cfg
        offs = None
        if s.dynamic:
            offs = g['order'] - (torch.arange(s.M, device=g['order'].device) * cfg.stride)[None]
        return dict(c1=o['c1'][..., 0], offs=offs, codes=o['codes'], order=g['order'], length=g['length'])

    @torch.no_grad()
    def tokenize(s, x):
        return s.tokens_from(s.encode(x, None))

    @torch.no_grad()
    def decode_tokens(s, c1, codes, g, temp=1.0):
        cfg = s.cfg; B = c1.size(0); L = cfg.L
        zq = sum(s.rvq.cb[q][codes[..., q]] for q in range(cfg.Q))
        e1 = s.vq1.cb[0][c1]
        logits = s.decode(zq + take(e1, g['order'] // cfg.R), g, L)
        return torch.multinomial(F.softmax(logits / temp, -1).reshape(-1, 256), 1).view(B, L)

    def maintain(s):
        s.rvq.restart(); s.vq1.restart()

# %%
# ----------------------------------------------------------------------------------------------
# Prior over the codes. ONE architecture, ONE token set, two decoding regimes:
#   'ar': causal transformer over the level-major token sequence (exact autoregressive factorization)
#   'cf': coarse-to-fine, one parallel (mean-field) pass per level, conditioned on all coarser levels
# Token sequence (level-major): [c1 cells | boundary offsets (dynamic only) | RVQ level 0 | ... | RVQ level Q-1]
# ----------------------------------------------------------------------------------------------
class Prior(nn.Module):
    def __init__(s, cfg, dynamic, mode):
        super().__init__()
        assert mode in ('ar', 'cf')
        s.cfg, s.dynamic, s.mode = cfg, dynamic, mode
        d, L = cfg.d, cfg.L
        s.G, s.M = L // cfg.R, L // cfg.stride
        lv, lo = [('c1', 0, cfg.K1, s.G)], cfg.K1
        if dynamic:
            lv.append(('bnd', lo, lo + cfg.stride, s.M - 1)); lo += cfg.stride          # offsets of chunks 1..M-1
        for q in range(cfg.Q):
            lv.append((f'r{q}', lo, lo + cfg.K, s.M)); lo += cfg.K
        s.levels, s.V = lv, lo
        s.MASK, s.PAD = s.V, s.V + 1
        s.ends = []; acc = 0
        for (_, _, _, w) in lv: acc += w; s.ends.append(acc)
        s.T = acc
        s.register_buffer('lvl', torch.cat([torch.full((w,), i, dtype=torch.long) for i, (_, _, _, w) in enumerate(lv)]))
        s.tok = nn.Embedding(s.V + 2, d); s.pos = nn.Embedding(L, d)
        s.ln = nn.Embedding(MAXOFF + 1, d); s.typ = nn.Embedding(len(lv), d)
        s.bos = nn.Parameter(torch.zeros(1, 1, d))
        s.stack = Stack(cfg.gen, d, cfg.heads, mode == 'ar'); s.head = nn.Linear(d, s.V)

    def seq_tokens(s, tk):
        parts = [tk['c1']]
        if s.dynamic: parts.append(tk['offs'][:, 1:] + s.levels[1][1])
        base = len(parts)
        for q in range(s.cfg.Q): parts.append(tk['codes'][..., q] + s.levels[base + q][1])
        return torch.cat(parts, 1)

    def seq_feats(s, order, length):
        cfg = s.cfg; B = order.size(0); dev = order.device
        pos = [(torch.arange(s.G, device=dev) * cfg.R)[None].expand(B, -1)]
        ln = [torch.zeros(B, s.G, dtype=torch.long, device=dev)]
        if s.dynamic:
            pos.append((torch.arange(1, s.M, device=dev) * cfg.stride)[None].expand(B, -1))
            ln.append(torch.zeros(B, s.M - 1, dtype=torch.long, device=dev))
        for q in range(cfg.Q): pos.append(order); ln.append(length.clamp(max=MAXOFF))
        return torch.cat(pos, 1), torch.cat(ln, 1)

    def run(s, tok_in, pos, ln, T):
        B = tok_in.size(0)
        feat = s.pos(pos[:, :T]) + s.ln(ln[:, :T]) + s.typ(s.lvl[:T])[None]
        content = s.tok(tok_in[:, :T])
        if s.mode == 'ar':
            content = torch.cat([s.bos.expand(B, 1, -1), content[:, :-1]], 1)           # predict token i from tokens < i
        return s.head(s.stack(content + feat))

    def nll(s, tk):
        """Per-level, per-window code NLL in nats: dict name -> (B,)."""
        tok = s.seq_tokens(tk); pos, ln = s.seq_feats(tk['order'], tk['length']); B = tok.size(0)
        out = {}
        logits = s.run(tok, pos, ln, s.T) if s.mode == 'ar' else None
        for li, (name, lo, hi, w) in enumerate(s.levels):
            a, e = s.ends[li] - w, s.ends[li]
            if s.mode == 'cf':
                tin = tok[:, :e].clone(); tin[:, a:e] = s.MASK                          # level visible only as positions
                lg = s.run(tin, pos, ln, e)[:, a:e, lo:hi]
            else:
                lg = logits[:, a:e, lo:hi]
            ce = F.cross_entropy(lg.reshape(-1, hi - lo), (tok[:, a:e] - lo).reshape(-1), reduction='none')
            out[name] = ce.view(B, -1).sum(1)
        return out

    def loss(s, tk):
        nl = s.nll(tk); B = tk['c1'].size(0)
        tot = sum(nl.values())
        return tot.sum() / (B * s.cfg.L), dict(code_bpb=(tot.sum() / (B * s.cfg.L)).detach() / LN2)

    @torch.no_grad()
    def sample(s, codec, n, temp=1.0):
        """Returns (bytes (n,L), sequential network calls). Prior only; target bytes are never seen."""
        cfg = s.cfg; dev = s.tok.weight.device; L, M, G, st = cfg.L, s.M, s.G, cfg.stride
        tok = torch.zeros(n, s.T, dtype=torch.long, device=dev)
        order = (torch.arange(M, device=dev) * st)[None].expand(n, -1)
        length = torch.full((n, M), st, dtype=torch.long, device=dev)
        pos, ln = s.seq_feats(order, length)
        stageA = 2 if s.dynamic else 1
        calls, g = 0, None
        for li, (name, lo, hi, w) in enumerate(s.levels):
            a, e = s.ends[li] - w, s.ends[li]
            if s.dynamic and li == stageA:                                               # boundaries are known now
                bl, bw = s.levels[1][1], s.levels[1][3]
                offs = torch.cat([torch.zeros(n, 1, dtype=torch.long, device=dev),
                                  tok[:, s.ends[1] - bw:s.ends[1]] - bl], 1)
                starts = torch.arange(M, device=dev)[None] * st + offs
                b = torch.zeros(n, L, dtype=torch.bool, device=dev); b.scatter_(1, starts, True)
                g = geometry(b); pos, ln = s.seq_feats(g['order'], g['length'])
            if s.mode == 'cf':
                tin = tok[:, :e].clone(); tin[:, a:e] = s.MASK
                lg = s.run(tin, pos, ln, e)[:, a:e, lo:hi]; calls += 1
                tok[:, a:e] = torch.multinomial(F.softmax(lg / temp, -1).reshape(-1, hi - lo), 1).view(n, -1) + lo
            else:
                for t in range(a, e):
                    lg = s.run(tok[:, :t + 1], pos, ln, t + 1)[:, t, lo:hi]; calls += 1
                    tok[:, t] = torch.multinomial(F.softmax(lg / temp, -1), 1)[:, 0] + lo
        if g is None:
            b = (torch.arange(L, device=dev) % st == 0)[None].expand(n, L).clone(); g = geometry(b)
        base = 2 if s.dynamic else 1
        c1 = tok[:, :G]
        codes = torch.stack([tok[:, s.ends[base + q] - M:s.ends[base + q]] - s.levels[base + q][1]
                             for q in range(cfg.Q)], -1)
        return codec.decode_tokens(c1, codes, g, temp), calls

# %%
# ----------------------------------------------------------------------------------------------
# Baseline: exact-likelihood byte GPT (reference model and strong comparison)
# ----------------------------------------------------------------------------------------------
class ByteGPT(nn.Module):
    def __init__(s, cfg):
        super().__init__()
        d = cfg.d; s.cfg = cfg
        s.emb = nn.Embedding(256, d); s.pos = nn.Embedding(cfg.L, d)
        s.stack = Stack(cfg.base_layers, d, cfg.heads, True); s.head = nn.Linear(d, 256)
    def forward(s, inp):
        return s.head(s.stack(s.emb(inp) + s.pos.weight[:inp.size(1)]))
    def loss(s, x):
        lg = s(x[:, :-1]); ce = F.cross_entropy(lg.reshape(-1, 256), x[:, 1:].reshape(-1))
        return ce, dict(bpb=ce.detach() / LN2)
    @torch.no_grad()
    def bits_per_window(s, x):
        lg = s(x[:, :-1])
        ce = F.cross_entropy(lg.reshape(-1, 256), x[:, 1:].reshape(-1), reduction='none').view(x.size(0), -1).sum(1)
        return ce / LN2 + UNI_BITS[x[:, 0]]                                                # first byte under unigram
    @torch.no_grad()
    def sample(s, n, L, temp=1.0):
        x = torch.multinomial(torch.exp(-UNI_BITS * LN2), n, replacement=True)[:, None]
        for _ in range(L - 1):
            lg = s(x)[:, -1]
            x = torch.cat([x, torch.multinomial(F.softmax(lg / temp, -1), 1)], 1)
        return x, L - 1

def count_params(m): return sum(p.numel() for p in m.parameters())

# %%
# ----------------------------------------------------------------------------------------------
# Self-tests: run first, raise on failure (cheap; catches shape/causality/geometry bugs before GPU time)
# ----------------------------------------------------------------------------------------------
def run_tests():
    torch.manual_seed(0)
    c = Cfg(**PRESETS['smoke'])
    # geometry tiles the sequence
    b = torch.rand(4, 32, device=DEVICE) < 0.3; b[:, 0] = True
    g = geometry(b)
    assert (g['length'].sum(1) == 32).all() and (g['offset'][b] == 0).all(), 'geometry'
    for dyn in (False, True):
        cd = Codec(c, dyn).to(DEVICE).train()
        x = torch.randint(0, 256, (6, c.L), device=DEVICE)
        loss, st_ = cd.loss(x, 0.5); loss.backward()
        assert torch.isfinite(loss), f'codec loss not finite (dyn={dyn})'
        if dyn:
            gw = cd.router[1].weight.grad
            assert gw is not None and torch.isfinite(gw).all() and gw.abs().sum() > 0, 'router received no gradient'
        cd.eval()
        o = cd.encode(x); tk = cd.tokens_from(o); gg = o['g']
        assert gg['M'] == c.L // c.stride and (gg['length'] >= 1).all() and (gg['length'] <= MAXOFF).all()
        if dyn: assert (tk['offs'] >= 0).all() and (tk['offs'] < c.stride).all() and (tk['offs'][:, 0] == 0).all()
        # token round trip: decoding from tokens reproduces the encoder-path logits
        zq = sum(cd.rvq.cb[q][tk['codes'][..., q]] for q in range(c.Q))
        z2 = zq + take(cd.vq1.cb[0][tk['c1']], gg['order'] // c.R)
        l1 = cd.decode(o['zq'] + o['e1c'], gg, c.L); l2 = cd.decode(z2, gg, c.L)
        assert torch.allclose(l1, l2, atol=1e-2), 'token round trip mismatch'
        for mode in ('ar', 'cf'):
            pr = Prior(c, dyn, mode).to(DEVICE)
            nl = pr.nll(tk)
            assert all(torch.isfinite(v).all() for v in nl.values()), 'prior nll not finite'
            loss, _ = pr.loss(tk); loss.backward()
            if mode == 'ar':                                       # causality: future tokens cannot change past logits
                tok = pr.seq_tokens(tk); pos, ln = pr.seq_feats(tk['order'], tk['length'])
                cut = pr.T // 2; t2 = tok.clone(); t2[:, cut:] = torch.randint(0, pr.V, t2[:, cut:].shape, device=DEVICE)
                with torch.no_grad():
                    la = pr.run(tok, pos, ln, pr.T)[:, :cut]; lb = pr.run(t2, pos, ln, pr.T)[:, :cut]
                assert torch.allclose(la, lb, atol=1e-4), 'AR prior leaks the future'
            pr.eval()
            out, calls = pr.sample(cd, 3)
            assert out.shape == (3, c.L) and out.min() >= 0 and out.max() < 256, 'sample shape'
            exp_calls = (pr.T if mode == 'ar' else len(pr.levels))
            assert calls == exp_calls, f'call count {calls} != {exp_calls}'
    gpt = ByteGPT(c).to(DEVICE)
    xb = torch.randint(0, 256, (4, c.L), device=DEVICE)
    assert gpt.bits_per_window(xb).shape == (4,)
    print('self-tests passed')

if RUN_TESTS: run_tests()

# %%
# ----------------------------------------------------------------------------------------------
# Training / evaluation utilities
# ----------------------------------------------------------------------------------------------
def fit(model, step_fn, steps, lr, name, maintain_every=200):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    warm = max(1, min(100, steps // 5))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, t / steps)))))
    model.train(); t0 = time.time(); ema = None; every = max(1, steps // 5)
    for step in range(steps):
        loss, st = step_fn(step)
        if not torch.isfinite(loss): raise FloatingPointError(f'{name}: non-finite loss at step {step}')
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if hasattr(model, 'maintain') and step % maintain_every == maintain_every - 1: model.maintain()
        v = float(loss.detach()); ema = v if ema is None else 0.98 * ema + 0.02 * v
        if step % every == 0 or step == steps - 1:
            extra = ' '.join(f'{k}={float(v_):.3f}' for k, v_ in st.items())
            print(f'    [{name}] step {step:5d}/{steps} loss~{ema:.3f} {extra} ({time.time() - t0:.0f}s)', flush=True)
    model.eval()
    return time.time() - t0

@torch.no_grad()
def collect(fn, n, L, bs=128):
    """Run fn(x)->dict of (B,) tensors over fixed validation windows; returns dict of numpy arrays (n,)."""
    starts = val_starts(L, n); acc = {}
    for i in range(0, n, bs):
        d = fn(windows(DATA_VA, starts[i:i + bs], L))
        for k, v in d.items(): acc.setdefault(k, []).append(v.float().cpu())
    return {k: torch.cat(v).numpy() for k, v in acc.items()}

@torch.no_grad()
def concept_components(codec, prior, x):
    o = codec.encode(x, None)
    logits = codec.decode(o['zq'] + o['e1c'], o['g'], x.size(1))
    recon = F.cross_entropy(logits.reshape(-1, 256), x.reshape(-1), reduction='none').view(x.size(0), -1).sum(1)
    d = {'recon': recon / LN2}
    for k, v in prior.nll(codec.tokens_from(o)).items(): d[k] = v / LN2
    d['total'] = sum(d.values())
    return d

@torch.no_grad()
def codec_rd(codec, n=128):
    """Reconstruction bpb when only the first k RVQ levels are used (k=0: coarse cells only)."""
    cfg = codec.cfg; codec.eval()
    x = windows(DATA_VA, val_starts(cfg.L, n)[:n], cfg.L); out = {}
    for k in range(cfg.Q + 1):
        keep = (torch.arange(cfg.Q, device=DEVICE)[None] < k).expand(n, cfg.Q)
        o = codec.encode(x, keep)
        lg = codec.decode(o['zq'] + o['e1c'], o['g'], cfg.L)
        out[k] = float(F.cross_entropy(lg.reshape(-1, 256), x.reshape(-1)) / LN2)
    o = codec.encode(x, None); ppl = []
    for q in range(cfg.Q):
        h = torch.bincount(o['codes'][..., q].reshape(-1), minlength=cfg.K).float(); p = h / h.sum()
        ppl.append(float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum())))
    return out, ppl

def show_text(t, lim=160): return bytes(t.tolist())[:lim].decode('latin-1').replace('\n', '⏎')

@torch.no_grad()
def segmentation_string(codec, n_chars=150):
    x = windows(DATA_VA, val_starts(codec.cfg.L, 1)[:1], codec.cfg.L)
    order = codec.encode(x, None)['g']['order'][0].tolist()
    txt = show_text(x[0], codec.cfg.L); starts = set(order)
    return ''.join(('|' if (i in starts and i > 0) else '') + ch for i, ch in enumerate(txt))[:n_chars + 20]

def timed(fn):
    if DEVICE == 'cuda': torch.cuda.synchronize()
    t0 = time.time(); r = fn()
    if DEVICE == 'cuda': torch.cuda.synchronize()
    return r, time.time() - t0

# %%
# ----------------------------------------------------------------------------------------------
# Experiment: for each seed -> byte GPT baseline; for each lambda and chunking -> train codec once,
# then train an AR prior and a CF prior on the SAME frozen codes. Everything is saved to results.json.
# ----------------------------------------------------------------------------------------------
def run_all(cfg=CFG):
    L = cfg.L
    RES = dict(cfg=asdict(cfg), preset=PRESET, data=DATA_SRC, device=DEVICE, runs=[])
    uni = collect(lambda x: {'total': UNI_BITS[x].sum(1)}, cfg.eval_windows, L)['total'] / L
    RES['unigram_bpb'] = float(uni.mean()); RES['unigram_per_window'] = uni.tolist()
    path = os.path.join(OUT_DIR, 'results.json')
    def save():
        with open(path, 'w') as f: json.dump(RES, f)

    for seed in cfg.seeds:
        print(f'\n=== seed {seed}: byte GPT baseline ===')
        torch.manual_seed(1000 * seed + 1); np.random.seed(seed)
        gpt = ByteGPT(cfg).to(DEVICE)
        tsec = fit(gpt, lambda s: gpt.loss(train_batch(L, cfg.bs)), cfg.base_steps, cfg.base_lr, f'gpt s{seed}')
        bw = collect(lambda x: {'total': gpt.bits_per_window(x)}, cfg.eval_windows, L)['total'] / L
        smp, dt = timed(lambda: gpt.sample(cfg.sample_n, L))
        ref = float((gpt.bits_per_window(smp[0]) / L).mean())
        RES['runs'].append(dict(kind='gpt', seed=seed, lam=None, chunk='-', prior='ar', params=count_params(gpt),
                                bpb=float(bw.mean()), per_window=bw.tolist(), comps={'total': float(bw.mean())},
                                calls=smp[1], sample_sec=dt, ref_bpb=ref, train_sec=tsec,
                                samples=[show_text(smp[0][i]) for i in range(min(3, cfg.sample_n))]))
        print(f'  byte GPT: val bpb={bw.mean():.4f}  params={count_params(gpt) / 1e6:.2f}M')
        save()

        for lam in cfg.lam_list:
            for dyn in (False, True):
                chunk = 'dynamic' if dyn else 'fixed'
                print(f'\n=== seed {seed} lam={lam}: codec ({chunk}) ===')
                torch.manual_seed(1000 * seed + 2 + int(dyn)); np.random.seed(seed)
                codec = Codec(cfg, dyn).to(DEVICE)
                warm = max(1, int(cfg.lam_warm * cfg.codec_steps))
                tsec_c = fit(codec, lambda s: codec.loss(train_batch(L, cfg.bs), lam * min(1.0, s / warm)),
                             cfg.codec_steps, cfg.codec_lr, f'codec {chunk} s{seed}')
                rd, ppl = codec_rd(codec)
                seg = segmentation_string(codec)
                print(f'  recon bpb by #RVQ levels used (0..{cfg.Q}): ' + ', '.join(f'{rd[k]:.3f}' for k in range(cfg.Q + 1))
                      + f' | code perplexity/level: {np.round(ppl, 1).tolist()}')
                if dyn: print('  segmentation:', seg)
                torch.save(codec.state_dict(), os.path.join(OUT_DIR, f'codec_{chunk}_lam{lam}_s{seed}.pt'))
                for mode in ('ar', 'cf'):
                    print(f'--- seed {seed} lam={lam}: {chunk} + {mode.upper()} prior ---')
                    torch.manual_seed(1000 * seed + 10 + int(dyn) * 2 + (mode == 'cf')); np.random.seed(seed)
                    prior = Prior(cfg, dyn, mode).to(DEVICE)
                    def step_fn(s):
                        with torch.no_grad(): tk = codec.tokenize(train_batch(L, cfg.bs))
                        return prior.loss(tk)
                    tsec_p = fit(prior, step_fn, cfg.prior_steps, cfg.prior_lr, f'{chunk}/{mode} s{seed}')
                    comps = collect(lambda x: concept_components(codec, prior, x), cfg.eval_windows, L)
                    tot = comps['total'] / L
                    smp, dt = timed(lambda: prior.sample(codec, cfg.sample_n))
                    ref = float((gpt.bits_per_window(smp[0]) / L).mean())
                    RES['runs'].append(dict(
                        kind='concept', seed=seed, lam=lam, chunk=chunk, prior=mode,
                        params=count_params(codec) + count_params(prior), bpb=float(tot.mean()),
                        per_window=tot.tolist(), comps={k: float(v.mean() / L) for k, v in comps.items()},
                        calls=smp[1], sample_sec=dt, ref_bpb=ref, train_sec=tsec_c + tsec_p,
                        rd=rd, code_ppl=ppl, segmentation=seg if dyn else None,
                        samples=[show_text(smp[0][i]) for i in range(min(3, cfg.sample_n))]))
                    c_ = RES['runs'][-1]['comps']
                    print(f'  => total bound bpb={tot.mean():.4f} (recon {c_["recon"]:.3f}, '
                          + ', '.join(f'{k} {v:.3f}' for k, v in c_.items() if k not in ('recon', 'total'))
                          + f') | sequential calls={smp[1]} | ref-bpb of samples={ref:.3f}')
                    save()
    print('\nsaved', path)
    return RES

RESULTS = run_all()

# %%
# ----------------------------------------------------------------------------------------------
# Analysis: reads ONLY results.json (nothing is hard-coded or filled in).
# ----------------------------------------------------------------------------------------------
RES = json.load(open(os.path.join(OUT_DIR, 'results.json')))
runs = RES['runs']; SEEDS = sorted({r['seed'] for r in runs}); LAMS = sorted({r['lam'] for r in runs if r['lam'] is not None})
print(f"data: {RES['data']} | preset: {RES['preset']} | seeds: {SEEDS} | lambdas: {LAMS}")
if 'FALLBACK' in RES['data']: print('!! Fallback corpus was used: these are NOT Shakespeare results.')
CELLS = [('fixed', 'ar'), ('dynamic', 'ar'), ('fixed', 'cf'), ('dynamic', 'cf')]

def pick(**kw): return [r for r in runs if all(r.get(k) == v for k, v in kw.items())]
def ms(vals): v = np.array(vals, dtype=float); return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float('nan'))

gpt_b = [r['bpb'] for r in pick(kind='gpt')]
print(f"\nBaselines (val bpb): unigram {RES['unigram_bpb']:.3f} | byte GPT {ms(gpt_b)[0]:.3f} ± {ms(gpt_b)[1]:.3f} "
      f"({pick(kind='gpt')[0]['params'] / 1e6:.2f}M params)")

for lam in LAMS:
    print(f'\n--- lambda={lam}: two-part-code bound (bpb), mean ± SEM over seeds ---')
    print(f"{'cell':<14}{'bpb':>16}{'recon':>8}{'c1':>8}{'bnd':>8}{'codes':>8}{'calls':>7}{'sample s':>10}{'ref-bpb':>9}{'params':>9}")
    for chunk, mode in CELLS:
        rr = pick(kind='concept', lam=lam, chunk=chunk, prior=mode)
        if not rr: continue
        m, s = ms([r['bpb'] for r in rr]); cm = lambda k: np.mean([r['comps'].get(k, 0.0) for r in rr])
        codes = sum(cm(k) for k in rr[0]['comps'] if k.startswith('r'))
        print(f"{chunk + '+' + mode.upper():<14}{m:>9.3f} ±{s:5.3f}{cm('recon'):>8.3f}{cm('c1'):>8.3f}{cm('bnd'):>8.3f}{codes:>8.3f}"
              f"{rr[0]['calls']:>7}{np.mean([r['sample_sec'] for r in rr]):>10.2f}{np.mean([r['ref_bpb'] for r in rr]):>9.3f}"
              f"{rr[0]['params'] / 1e6:>8.2f}M")
    g_ = pick(kind='gpt')
    print(f"{'byte GPT':<14}{ms(gpt_b)[0]:>9.3f} ±{ms(gpt_b)[1]:5.3f}{'':>32}{g_[0]['calls']:>7}"
          f"{np.mean([r['sample_sec'] for r in g_]):>10.2f}{np.mean([r['ref_bpb'] for r in g_]):>9.3f}")
    print('(ref-bpb = byte-GPT bits/byte on generated samples; real text scores ≈ the byte GPT val bpb. Lower is more text-like.)')

# Codec rate-distortion diagnostic: are coarse levels informative?
print('\nReconstruction bpb when only k RVQ levels are used (k=0 is the coarse cell code alone):')
for lam in LAMS:
    for chunk in ('fixed', 'dynamic'):
        rr = pick(kind='concept', lam=lam, chunk=chunk, prior='ar')
        if rr:
            ks = sorted(rr[0]['rd'], key=int)
            print(f'  lam={lam} {chunk:<8}: ' + '  '.join(f"k={k}: {np.mean([r['rd'][k] for r in rr]):.3f}" for k in ks)
                  + f"   code perplexity/level {np.round(np.mean([r['code_ppl'] for r in rr], 0), 1).tolist()}")

# The interaction, paired over held-out windows
def boot(x, reps=2000, seed=0):
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(x), (reps, len(x)))
    m = x[idx].mean(1); return np.quantile(m, 0.025), np.quantile(m, 0.975)

print('\nInteraction I = (bpb[fixed,CF]-bpb[dyn,CF]) - (bpb[fixed,AR]-bpb[dyn,AR])   (>0: dynamic chunking helps more under CF)')
INTER = {}
for lam in LAMS:
    per_seed = []
    for seed in SEEDS:
        pw = {}
        for chunk, mode in CELLS:
            rr = pick(kind='concept', lam=lam, seed=seed, chunk=chunk, prior=mode)
            if rr: pw[(chunk, mode)] = np.array(rr[0]['per_window'])
        if len(pw) < 4: continue
        gain_ar = pw[('fixed', 'ar')] - pw[('dynamic', 'ar')]; gain_cf = pw[('fixed', 'cf')] - pw[('dynamic', 'cf')]
        I = gain_cf - gain_ar; lo, hi = boot(I)
        per_seed.append(I.mean()); INTER[(lam, seed)] = (I.mean(), lo, hi)
        print(f'  lam={lam} seed={seed}: dyn gain|AR={gain_ar.mean():+.4f}  dyn gain|CF={gain_cf.mean():+.4f}  '
              f'I={I.mean():+.4f}  95% CI over windows [{lo:+.4f}, {hi:+.4f}]')
    if per_seed:
        m, s = ms(per_seed)
        print(f'  lam={lam} across seeds: I = {m:+.4f} ± {s:.4f} (SEM, n={len(per_seed)})')

print('\nPre-registered gates (informational; apply only to a `full` run):')
for lam in LAMS:
    fcf = ms([r['bpb'] for r in pick(kind='concept', lam=lam, chunk='fixed', prior='cf')])[0]
    far = ms([r['bpb'] for r in pick(kind='concept', lam=lam, chunk='fixed', prior='ar')])[0]
    g0 = ms(gpt_b)[0]
    print(f"  lam={lam}: G1 fixed+CF beats unigram: {fcf < RES['unigram_bpb']} ({fcf:.3f} vs {RES['unigram_bpb']:.3f}) | "
          f"G2 fixed+AR within 15% of byte GPT: {far <= 1.15 * g0} ({far:.3f} vs {g0:.3f}) | "
          f"G3 parallel penalty (CF-AR, fixed): {fcf - far:+.3f} bpb")
print('  G4: claim synergy only if the across-seed lower bound of I is > 0 at every lambda and the effect persists at larger sizes.')

# %%
import matplotlib
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
lam0 = LAMS[0]; names = [f'{c}+{m.upper()}' for c, m in CELLS]; cols = ['#8c8c8c', '#4c72b0', '#dd8452', '#c44e52']
vals = [ms([r['bpb'] for r in pick(kind='concept', lam=lam0, chunk=c, prior=m)]) for c, m in CELLS]
ax[0].bar(names, [v[0] for v in vals], yerr=[0 if np.isnan(v[1]) else v[1] for v in vals], color=cols, capsize=4)
ax[0].axhline(RES['unigram_bpb'], color='k', ls=':', label='unigram'); ax[0].axhline(ms(gpt_b)[0], color='g', ls='--', label='byte GPT (exact)')
ax[0].set_ylabel('val bpb (two-part bound for concept models)'); ax[0].set_title(f'2x2 on shared codecs (lambda={lam0})')
ax[0].legend(fontsize=8); ax[0].tick_params(axis='x', rotation=15)
for i, ((lam, seed), (m, lo, hi)) in enumerate(INTER.items()):
    ax[1].errorbar([i], [m], yerr=[[m - lo], [hi - m]], fmt='o', color='k', capsize=4)
ax[1].set_xticks(range(len(INTER))); ax[1].set_xticklabels([f'λ{l}\ns{s}' for (l, s) in INTER], fontsize=8)
ax[1].axhline(0, color='gray'); ax[1].set_ylabel('I (bpb)'); ax[1].set_title('Interaction, 95% CI over windows')
bottoms = np.zeros(4)
for lab, keyf, colr in [('recon', lambda k: k == 'recon', '#4c72b0'), ('coarse c1', lambda k: k == 'c1', '#55a868'),
                        ('boundaries', lambda k: k == 'bnd', '#8172b3'), ('RVQ codes', lambda k: k.startswith('r') and k != 'recon', '#dd8452')]:
    v = np.array([np.mean([sum(x for k, x in r['comps'].items() if k != 'total' and keyf(k))
                           for r in pick(kind='concept', lam=lam0, chunk=c, prior=m)]) for c, m in CELLS])
    ax[2].bar(names, v, bottom=bottoms, label=lab, color=colr); bottoms += v
ax[2].set_ylabel('bits per byte'); ax[2].set_title('Where the bits go'); ax[2].legend(fontsize=8); ax[2].tick_params(axis='x', rotation=15)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, 'hncm_v2_results.png'), dpi=140); plt.show()

# %%
# Qualitative: learned jittered segmentation and samples (straight from results.json)
dyn_runs = pick(kind='concept', chunk='dynamic', prior='cf', seed=SEEDS[0])
if dyn_runs: print('Learned boundaries on a validation window ("|" = chunk start):\n ', dyn_runs[0]['segmentation'])
print('\nSamples (temperature 1.0; first seed; small models, expect rough text):')
for r in [r for r in runs if r['seed'] == SEEDS[0] and (r['lam'] in (None, LAMS[0]))]:
    tag = 'byte GPT' if r['kind'] == 'gpt' else f"{r['chunk']}+{r['prior'].upper()}"
    print(f"  {tag:<14} ref-bpb={r['ref_bpb']:.3f}  {r['samples'][0]!r}")

# %% [markdown]
# ## Reading the result
#
# 1. **Check G1 first.** If fixed+CF does not beat the unigram baseline, the codec/prior pair is still broken and
#    nothing about dynamic chunking can be concluded. Look at the bit-budget panel: if `RVQ codes` dominate, raise λ,
#    lower `K`/`Q`, or raise `stride`.
# 2. **The AR-vs-CF gap on shared codes (G3)** is the cost of mean-field parallel decoding, isolated from the codec.
#    The reconstruction-vs-levels table says whether coarse levels carry information.
# 3. **Only then read `I`.** Positive `I` with a lower bound above 0 across seeds and λ would be preliminary support
#    for super-additivity; a null is equally informative.
#
# ## Known limitations (deliberate, to keep the experiment interpretable)
# * Jittered chunking has bounded adaptivity (offset within a stride-cell); free-form variable-rate chunking needs a
#   generable boundary model and is the next step.
# * The generator does not backprop into the codec; the rate term is a context-free proxy (soft expected code length
#   under learned marginals), not the true prior's code length.
# * Two-part bpb is an upper bound for the concept models; byte GPT is exact. Compare across families with care.
# * Positions within a CF pass are independent (mean-field). Iterative masked refinement is the obvious next upgrade.
# * Tiny models, tiny corpus, fp32, no KV cache in AR sampling (wall-clock for AR is pessimistic; compare call counts).

