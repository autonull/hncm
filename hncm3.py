# %% [markdown]
# # H-NCM v3 — rate-matched shared codecs, prior-in-the-loop tokenizer, budgeted parallel decoding
#
# Thesis: content-adaptive chunking reduces the PARALLEL-DECODING PENALTY of concept codes
# more than it reduces AR NLL. With one frozen codec per chunking mode and two identical
# priors (AR / CF) trained on the same codes:
#   I = (bpb[f,CF]-bpb[d,CF]) - (bpb[f,AR]-bpb[d,AR]) = Gap_parallel(fixed) - Gap_parallel(dynamic)
# Reconstruction cancels EXACTLY (asserted per window). Controls: SYN-SEG (must show I>0),
# SYN-GRID (must show I~0). All tables/plots read ONLY results.json. No fallbacks, no placeholders.
# UNEXECUTED BY AUTHOR. Run smoke -> quick -> full.

# %%
import os, math, time, json, hashlib, datetime, urllib.request
from dataclasses import dataclass, asdict, field
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LN2 = math.log(2.0); MAXOFF = 31
PRESET = os.environ.get('HNCM_PRESET', 'smoke')
OUT = os.environ.get('HNCM_OUT', 'hncm_v3_out'); os.makedirs(OUT, exist_ok=True)

@dataclass
class Cfg:
    L: int = 128; R: int = 16; stride: int = 4
    K1: int = 32; K: int = 64; Q: int = 3
    d: int = 128; heads: int = 4; enc: int = 2; main: int = 2; dec: int = 2
    gen: int = 4; eh_layers: int = 2; base_layers: int = 6
    bs: int = 64
    codec_steps: int = 1500; prior_steps: int = 1500; base_steps: int = 1500
    codec_lr: float = 1e-3; prior_lr: float = 5e-4; base_lr: float = 1e-3
    rho: float = 1.0            # target latent rate (bits/byte), matched across modes via dual ascent
    lam_eta: float = 2.0; lam_max: float = 8.0
    level_dropout: bool = True; use_eh_rate: bool = True   # ablation switches
    cf_k_list: tuple = (1, 2, 4)   # CF passes per RVQ level (budgets)
    seeds: tuple = (0, 1)
    eval_windows: int = 512; sample_n: int = 64
    corpora: tuple = ('synseg', 'syngrid', 'shakespeare')
    def __post_init__(s):
        s.M = s.L // s.stride; s.G = s.L // s.R
        assert s.L % s.R == 0 and s.L % s.stride == 0 and s.R % s.stride == 0 and s.d % s.heads == 0

PRESETS = {
 'smoke': dict(L=64, d=64, enc=1, main=1, dec=1, gen=2, eh_layers=1, base_layers=2, K1=16, K=32, Q=2,
               bs=16, codec_steps=120, prior_steps=120, base_steps=120, seeds=(0,),
               eval_windows=64, sample_n=8, cf_k_list=(1, 2), corpora=('synseg',)),
 'quick': dict(),
 'full':  dict(L=256, d=192, heads=6, gen=6, base_layers=8, codec_steps=4000, prior_steps=5000,
               base_steps=5000, seeds=(0, 1, 2), eval_windows=2048, sample_n=128, rho=1.0),
}
CFG = Cfg(**PRESETS[PRESET])
if DEVICE == 'cpu' and PRESET != 'smoke': print('WARNING: no GPU; use smoke or a T4 runtime.')
print(f'device={DEVICE} preset={PRESET}'); print(CFG)

# %%
# ---------------- data: real corpus + positive/negative controls with known segmentation -------
def load_shakespeare():
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    raw = urllib.request.urlopen(url, timeout=20).read()   # no silent fallback: fail loudly
    return raw, 'shakespeare'

def gen_synseg(seed=7, nbytes=1_200_000):
    rng = np.random.RandomState(seed); letters = np.arange(ord('a'), ord('z') + 1)
    topics = [[bytes(rng.choice(letters, rng.randint(2, 10))) for _ in range(40)] for _ in range(8)]
    out = bytearray()
    while len(out) < nbytes:
        t = rng.randint(8)
        for _ in range(rng.randint(8, 21)):
            out += topics[t][rng.randint(40)] + b' '
        out += b'\n'
    return bytes(out), 'synseg'

def gen_syngrid(seed=8, nbytes=1_200_000):
    rng = np.random.RandomState(seed); letters = np.arange(ord('a'), ord('z') + 1)
    topics = [[bytes(rng.choice(letters, rng.randint(2, 4))) for _ in range(40)] for _ in range(8)]
    out = bytearray()
    while len(out) < nbytes:
        t = rng.randint(8)
        for _ in range(16):                       # topic persists 64 bytes = 4 coarse cells, aligned
            for _ in range(4):                    # 4 words x 4 bytes = one 16-byte cell
                w = topics[t][rng.randint(40)]
                out += w + b'_' * (4 - len(w))    # every word exactly stride bytes
    return bytes(out), 'syngrid'

def onsets(name, xb):                             # word-onset mask for a 1-D byte window (numpy)
    if name == 'syngrid': return (np.arange(len(xb)) % 4 == 0)
    prev = np.concatenate([[ord('\n')], xb[:-1]])
    return np.isin(prev, [ord(' '), ord('\n')]) & ~np.isin(xb, [ord(' '), ord('\n'), ord('_')])

CORPORA = {}
def get_corpus(name):
    if name in CORPORA: return CORPORA[name]
    raw, tag = {'shakespeare': load_shakespeare, 'synseg': gen_synseg, 'syngrid': gen_syngrid}[name]()
    arr = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()
    n = int(0.9 * len(arr))
    c = dict(tr=arr[:n].to(DEVICE), va=arr[n:].to(DEVICE), name=tag,
             sha=hashlib.sha256(raw).hexdigest(), nbytes=len(raw))
    CORPORA[name] = c; return c

def windows(data, starts, L): return data[starts[:, None] + torch.arange(L, device=data.device)[None]]
def val_starts(L, n):
    rs = np.random.RandomState(1234)              # identical windows for every cell/seed (paired)
    return torch.tensor(rs.randint(0, 10**7, size=n)) % max(1, 1)  # placeholder replaced below
def make_starts(corpus, L, n):
    rs = np.random.RandomState(1234)
    return torch.tensor(rs.randint(0, len(corpus['va']) - L, size=n), device=DEVICE)
def train_batch(corpus, L, bs):
    return windows(corpus['tr'], torch.randint(0, len(corpus['tr']) - L, (bs,), device=DEVICE), L)

_cnt = torch.bincount(get_corpus('synseg')['tr'], minlength=256).float() + 1   # unigram per corpus below
def unigram_bits(corpus):
    c = torch.bincount(corpus['tr'], minlength=256).float() + 1
    return (-torch.log2(c / c.sum())).to(DEVICE)

# %%
# ---------------- blocks (KV-cache capable), geometry ------------------------------------------
class Block(nn.Module):
    def __init__(s, d, h, causal):
        super().__init__(); s.h, s.causal = h, causal
        s.ln1 = nn.LayerNorm(d); s.qkv = nn.Linear(d, 3*d, bias=False); s.o = nn.Linear(d, d, bias=False)
        s.ln2 = nn.LayerNorm(d); s.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(s, x, past=None):
        B, T, D = x.shape
        q, k, v = s.qkv(s.ln1(x)).view(B, T, 3, s.h, D//s.h).permute(2, 0, 3, 1, 4)
        if past is not None:
            assert T == 1, 'incremental mode is single-token'
            k = torch.cat([past[0], k], 2); v = torch.cat([past[1], v], 2)
        new = (k, v)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=(s.causal and past is None and T > 1))
        x = x + s.o(a.transpose(1, 2).reshape(B, T, D))
        return x + s.mlp(s.ln2(x)), new

class Stack(nn.Module):
    def __init__(s, n, d, h, causal):
        super().__init__(); s.blocks = nn.ModuleList(Block(d, h, causal) for _ in range(n)); s.ln = nn.LayerNorm(d)
    def forward(s, x, past=None):
        past = past if past is not None else [None]*len(s.blocks); new = []
        for b, p in zip(s.blocks, past):
            x, np_ = b(x, p); new.append(np_)
        return s.ln(x), new

def take(x, idx): return torch.gather(x, 1, idx[..., None].expand(-1, -1, x.size(-1)))

def geometry(b):
    B, L = b.shape; dev = b.device
    counts = b.sum(1); M = int(counts.max())
    order = torch.argsort((~b).long(), dim=1, stable=True)[:, :M]
    slot = torch.arange(M, device=dev)[None]; valid = slot < counts[:, None]
    m_idx = b.long().cumsum(1) - 1
    nxt = torch.cat([order[:, 1:], torch.full((B, 1), L, device=dev, dtype=order.dtype)], 1)
    nxt = torch.where(slot + 1 < counts[:, None], nxt, torch.full_like(nxt, L))
    length = (nxt - order).clamp(min=0) * valid
    start_b = torch.gather(order, 1, m_idx)
    offset = torch.arange(L, device=dev)[None] - start_b
    return dict(order=order, valid=valid, counts=counts, M=M, m_idx=m_idx, length=length, offset=offset)

# ---------------- shared token layout (entropy head and priors use the SAME serialization) -----
def layout(cfg, dyn):
    lv = [('c1', 0, cfg.K1, cfg.G)]; lo = cfg.K1
    if dyn: lv.append(('bnd', lo, lo + cfg.stride, cfg.M - 1)); lo += cfg.stride
    for q in range(cfg.Q): lv.append((f'r{q}', lo, lo + cfg.K, cfg.M)); lo += cfg.K
    ends = []; a = 0
    for _, _, _, w in lv: a += w; ends.append(a)
    return lv, lo, ends, a                        # levels, V, ends, T

def seq_tokens(tk, cfg, dyn, lv):
    parts = [tk['c1']]
    if dyn: parts.append(tk['offs'][:, 1:] + lv[1][1])
    base = 2 if dyn else 1
    for q in range(cfg.Q): parts.append(tk['codes'][..., q] + lv[base + q][1])
    return torch.cat(parts, 1)

def seq_feats(order, length, cfg, dyn, lv):
    B = order.size(0); dev = order.device
    pos = [(torch.arange(cfg.G, device=dev) * cfg.R)[None].expand(B, -1)]
    ln = [torch.zeros(B, cfg.G, dtype=torch.long, device=dev)]
    if dyn:
        pos.append((torch.arange(1, cfg.M, device=dev) * cfg.stride)[None].expand(B, -1))
        ln.append(torch.zeros(B, cfg.M - 1, dtype=torch.long, device=dev))
    for _ in range(cfg.Q): pos.append(order); ln.append(length.clamp(max=MAXOFF))
    return torch.cat(pos, 1), torch.cat(ln, 1)

class TokenFeats(nn.Module):
    def __init__(s, cfg, dyn, V, nlv):
        super().__init__()
        s.tok = nn.Embedding(V + 2, cfg.d); s.pos = nn.Embedding(cfg.L, cfg.d)
        s.ln = nn.Embedding(MAXOFF + 1, cfg.d); s.typ = nn.Embedding(nlv, cfg.d)
        s.bos = nn.Parameter(torch.zeros(1, 1, cfg.d))
        s.register_buffer('lvl', torch.tensor(sum([[i]*w for i, (_, _, _, w) in
                            enumerate(layout(cfg, dyn)[0])], []), dtype=torch.long))
    def feats(s, pos, ln, T): return s.pos(pos[:, :T]) + s.ln(ln[:, :T]) + s.typ(s.lvl[:T])[None]

# %%
# ---------------- RVQ with soft assignments (differentiable rate path) -------------------------
class RVQ(nn.Module):
    def __init__(s, d, K, Q, beta=0.25):
        super().__init__(); s.K, s.Q, s.beta = K, Q, beta
        s.cb = nn.Parameter(torch.randn(Q, K, d) * 0.1)
        s.register_buffer('usage', torch.zeros(Q, K)); s.last = [None]*Q; s._inited = False
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
            dist = res.pow(2).sum(-1, keepdim=True) - 2*res @ cb.t() + cb.pow(2).sum(-1)
            idx = dist.argmin(-1); e = cb[idx]
            vq = vq + ((e - res.detach())**2).mean() + s.beta*((res - e.detach())**2).mean()
            tau = (0.1 * dist.detach().mean()).clamp_min(1e-6)
            softs.append(F.softmax(-dist / tau, -1))
            if s.training:
                with torch.no_grad():
                    s.usage[q] += torch.bincount(idx.reshape(-1), minlength=s.K).float()
                    s.last[q] = res.detach().reshape(-1, D)
            qsum = qsum + (e if keep is None else e * keep[:, q, None, None].to(e.dtype))
            res = res - e.detach(); codes.append(idx)
        return z + (qsum - z).detach(), torch.stack(codes, -1), vq, torch.stack(softs, 2)
    @torch.no_grad()
    def restart(s):
        for q in range(s.Q):
            dead = s.usage[q] < 1; n = int(dead.sum())
            if n > 0 and s.last[q] is not None:
                src = s.last[q]; pick = src[torch.randint(len(src), (n,), device=src.device)]
                s.cb.data[q][dead] = pick + 0.01*torch.randn_like(pick)
        s.usage.zero_()

# %%
# ---------------- codec: bytes -> (coarse cells, jitter offsets, RVQ chunks) -> bytes ----------
class Codec(nn.Module):
    def __init__(s, cfg, dyn):
        super().__init__(); s.cfg, s.dyn = cfg, dyn; d, L = cfg.d, cfg.L
        s.emb = nn.Embedding(256, d); s.pos = nn.Embedding(L, d)
        s.enc = Stack(cfg.enc, d, cfg.heads, False)
        if dyn:
            s.router = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
            nn.init.zeros_(s.router[1].weight); nn.init.zeros_(s.router[1].bias)  # starts == fixed
        s.cpos = nn.Embedding(cfg.M, d); s.main = Stack(cfg.main, d, cfg.heads, False)
        s.grid_proj = nn.Linear(d, d); s.vq1 = RVQ(d, cfg.K1, 1)
        s.proj = nn.Linear(d, d); s.rvq = RVQ(d, cfg.K, cfg.Q)
        s.dpos = nn.Embedding(L, d); s.doff = nn.Embedding(MAXOFF+1, d); s.dlen = nn.Embedding(MAXOFF+1, d)
        s.dec = Stack(cfg.dec, d, cfg.heads, False); s.head = nn.Linear(d, 256)
        lv, V, ends, T = layout(cfg, dyn); s.lv, s.ends, s.T = lv, ends, T
        s.eh_f = TokenFeats(cfg, dyn, V, len(lv))
        s.eh = Stack(cfg.eh_layers, d, cfg.heads, True); s.eh_head = nn.Linear(d, V)

    def chunk(s, h):
        B, L, _ = h.shape; st = s.cfg.stride; C = L // st
        a = s.router(h).squeeze(-1).view(B, C, st)
        soft = F.softmax(a, -1); idx = a.argmax(-1).clone(); idx[:, 0] = 0
        hard = F.one_hot(idx, st).float()
        off_soft = torch.cat([hard[:, :1], soft[:, 1:]], 1)          # first offset forced to 0
        ste = hard + off_soft - off_soft.detach()
        return hard.reshape(B, L).bool(), ste.reshape(B, L), off_soft, idx

    def assign(s, b, beta, g):
        hard = F.one_hot(g['m_idx'], s.cfg.M).float()
        if not s.dyn: return hard
        cum = beta.cumsum(1) - 1
        slots = torch.arange(s.cfg.M, device=b.device, dtype=beta.dtype)
        sa = F.softmax(-(cum[..., None] - slots)**2 / (2*0.5**2), -1)
        return hard + sa - sa.detach()

    def encode(s, x, keep=None):
        cfg = s.cfg; B, L = x.shape
        h, _ = s.enc(s.emb(x) + s.pos.weight[:L])
        if s.dyn: b, beta, off_soft, offs = s.chunk(h)
        else:
            b = (torch.arange(L, device=x.device) % cfg.stride == 0)[None].expand(B, L).clone()
            beta = b.float(); off_soft = None; offs = None
        g = geometry(b); A = s.assign(b, beta, g)
        cnt = g['length'].float().clamp_min(1)
        xc = (A.transpose(1, 2) @ h) / cnt[..., None] + s.cpos.weight[None]
        z = s.proj(s.main(xc)[0])
        e1, c1, l1, soft1 = s.vq1(s.grid_proj(h.reshape(B, cfg.G, cfg.R, -1).mean(2)))
        e1c = take(e1, g['order'] // cfg.R)
        zq, codes, l2, soft2 = s.rvq(z - e1c, keep)
        return dict(zq=zq, e1c=e1c, codes=codes, c1=c1[..., 0], soft1=soft1[..., 0, :],
                    soft2=soft2, off_soft=off_soft, offs=offs, g=g, vq=l1 + l2)

    def decode(s, zfull, g, L):
        zup = take(zfull, g['m_idx'])
        lenb = torch.gather(g['length'], 1, g['m_idx'])
        din = zup + s.dpos.weight[:L] + s.doff(g['offset'].clamp(max=MAXOFF)) + s.dlen(lenb.clamp(max=MAXOFF))
        return s.head(s.dec(din)[0])

    def tokens_from(s, o):
        g = o['g']
        offs = g['order'] - (torch.arange(s.cfg.M, device=g['order'].device)*s.cfg.stride)[None] if s.dyn else None
        return dict(c1=o['c1'], offs=offs, codes=o['codes'], order=g['order'], length=g['length'])

    def eh_logits(s, tok, pos, ln):
        content = s.eh_f.tok(tok)
        content = torch.cat([s.eh_f.bos.expand(tok.size(0), 1, -1), content[:, :-1]], 1)
        return s.eh_head(s.eh(content + s.eh_f.feats(pos, ln, s.T))[0])

    def loss(s, x, lam):
        cfg = s.cfg; B, L = x.shape; keep = None
        if cfg.level_dropout and s.training:
            qk = torch.where(torch.rand(B, device=x.device) < 0.5, torch.full((B,), cfg.Q, device=x.device),
                             torch.randint(1, cfg.Q + 1, (B,), device=x.device))
            keep = torch.arange(cfg.Q, device=x.device)[None] < qk[:, None]
        o = s.encode(x, keep)
        recon = F.cross_entropy(s.decode(o['zq'] + o['e1c'], o['g'], L).reshape(-1, 256), x.reshape(-1))
        tk = s.tokens_from(o); tok = seq_tokens(tk, cfg, s.dyn, s.lv)
        pos, ln = seq_feats(tk['order'], tk['length'], cfg, s.dyn, s.lv)
        lg = s.eh_logits(tok, pos, ln)
        soft_t = [o['soft1']] + ([o['off_soft'][:, 1:]] if s.dyn else []) + [o['soft2'][:, :, q, :] for q in range(cfg.Q)]
        rate_soft = x.new_zeros(()); rate_hard = x.new_zeros(())
        for li, (name, lo, hi, w) in enumerate(s.lv):
            a, e = s.ends[li] - w, s.ends[li]
            lp = F.log_softmax(lg[:, a:e, lo:hi], -1)
            rate_soft = rate_soft - (soft_t[li] * lp).sum()
            with torch.no_grad():
                rate_hard = rate_hard + F.cross_entropy(lg[:, a:e, lo:hi].reshape(-1, hi-lo),
                                                        (tok[:, a:e]-lo).reshape(-1), reduction='sum')
        rate_soft = rate_soft / (B*L); rate_hard = rate_hard / (B*L)
        r = rate_soft if cfg.use_eh_rate else rate_soft.detach()   # ablation: no prior-in-loop gradient
        loss = recon + lam*r + o['vq']
        return loss, dict(recon_bpb=(recon.detach()/LN2), rate_bpb=(rate_hard/LN2), vq=o['vq'].detach())

    @torch.no_grad()
    def tokenize(s, x): return s.tokens_from(s.encode(x, None))
    @torch.no_grad()
    def decode_tokens(s, c1, codes, g, temp=1.0):
        zq = sum(s.rvq.cb[q][codes[..., q]] for q in range(s.cfg.Q))
        e1 = s.vq1.cb[0][c1]
        lg = s.decode(zq + take(e1, g['order'] // s.cfg.R), g, s.cfg.L)
        return torch.multinomial(F.softmax(lg/temp, -1).reshape(-1, 256), 1).view(c1.size(0), s.cfg.L)
    def maintain(s): s.rvq.restart(); s.vq1.restart()

# %%
# ---------------- priors: identical modules; AR causal vs CF masked (MaskGIT within levels) ----
class Prior(nn.Module):
    def __init__(s, cfg, dyn, mode):
        super().__init__(); assert mode in ('ar', 'cf')
        s.cfg, s.dyn, s.mode = cfg, dyn, mode
        lv, V, ends, T = layout(cfg, dyn); s.lv, s.V, s.ends, s.T = lv, V, ends, T
        s.MASK = V
        s.f = TokenFeats(cfg, dyn, V, len(lv))
        s.stack = Stack(cfg.gen, cfg.d, cfg.heads, mode == 'ar'); s.head = nn.Linear(cfg.d, V)

    def run(s, inp, pos, ln, T, past=None):
        content = s.f.tok(inp[:, :T])
        if s.mode == 'ar' and past is None:
            content = torch.cat([s.f.bos.expand(inp.size(0), 1, -1), content[:, :-1]], 1)
        out, new = s.stack(content + s.f.feats(pos, ln, T), past)
        return s.head(out), new

    @torch.no_grad()
    def nll(s, tk):                                # VALID two-part-bound NLLs, nats per window per level
        tok = seq_tokens(tk, s.cfg, s.dyn, s.lv)
        pos, ln = seq_feats(tk['order'], tk['length'], s.cfg, s.dyn, s.lv); B = tok.size(0); out = {}
        if s.mode == 'ar':
            lg, _ = s.run(tok, pos, ln, s.T)
        for li, (name, lo, hi, w) in enumerate(s.lv):
            a, e = s.ends[li] - w, s.ends[li]
            if s.mode == 'cf':                     # one-shot mean-field factorization (normalized joint)
                inp = tok.clone(); inp[:, a:] = s.MASK
                lg, _ = s.run(inp, pos, ln, s.T)
            ce = F.cross_entropy(lg[:, a:e, lo:hi].reshape(-1, hi-lo), (tok[:, a:e]-lo).reshape(-1),
                                 reduction='none')
            out[name] = ce.view(B, -1).sum(1)
        return out

    def train_loss(s, tk):
        tok = seq_tokens(tk, s.cfg, s.dyn, s.lv)
        pos, ln = seq_feats(tk['order'], tk['length'], s.cfg, s.dyn, s.lv); B = tok.size(0)
        if s.mode == 'ar':
            lg, _ = s.run(tok, pos, ln, s.T); tot = 0
            for li, (name, lo, hi, w) in enumerate(s.lv):
                a, e = s.ends[li] - w, s.ends[li]
                tot = tot + F.cross_entropy(lg[:, a:e, lo:hi].reshape(-1, hi-lo),
                                            (tok[:, a:e]-lo).reshape(-1), reduction='sum')
            return tot / (B*s.cfg.L)
        t = int(torch.randint(len(s.lv), (1,)))    # CF: level-prefix visible, current level partially masked
        name, lo, hi, w = s.lv[t]; a, e = s.ends[t] - w, s.ends[t]
        keep = torch.zeros(B, s.T, dtype=torch.bool, device=tok.device); keep[:, :a] = True
        gamma = 0.05 + 0.95*torch.rand(B, 1, device=tok.device)
        keep[:, a:e] = torch.rand(B, w, device=tok.device) > gamma
        inp = torch.where(keep, tok, torch.full_like(tok, s.MASK))
        lg, _ = s.run(inp, pos, ln, s.T)
        m = ~keep[:, a:e]
        return F.cross_entropy(lg[:, a:e, lo:hi][m], (tok[:, a:e]-lo)[m])

    @torch.no_grad()
    def sample(s, codec, n, k_rvq=1, temp=1.0):
        cfg = s.cfg; dev = s.f.tok.weight.device; L, M, st = cfg.L, cfg.M, cfg.stride
        tok = torch.full((n, s.T), s.MASK if s.mode == 'cf' else 0, dtype=torch.long, device=dev)
        order = (torch.arange(M, device=dev)*st)[None].expand(n, -1)
        length = torch.full((n, M), st, dtype=torch.long, device=dev)
        pos, ln = seq_feats(order, length, cfg, s.dyn, s.lv)
        calls = 0; g = None; past = None
        for li, (name, lo, hi, w) in enumerate(s.lv):
            a, e = s.ends[li] - w, s.ends[li]
            if s.dyn and name == 'r0':             # offsets known -> true geometry for RVQ positions
                bl = s.lv[1][1]
                offs = torch.cat([torch.zeros(n, 1, dtype=torch.long, device=dev),
                                  tok[:, s.ends[1]-(M-1):s.ends[1]] - bl], 1)
                starts = torch.arange(M, device=dev)[None]*st + offs
                b = torch.zeros(n, L, dtype=torch.bool, device=dev); b.scatter_(1, starts, True)
                g = geometry(b); pos, ln = seq_feats(g['order'], g['length'], cfg, s.dyn, s.lv)
            if s.mode == 'ar':
                for t in range(a, e):
                    x = (s.f.bos.expand(n, 1, -1) + s.f.feats(pos, ln, s.T)[:, :1]) if t == 0 else \
                        (s.f.tok(tok[:, t-1])[:, None] + s.f.feats(pos, ln, s.T)[:, t:t+1])
                    lg, past = s.stack(x, past); lg = s.head(lg)[:, -1, lo:hi]
                    tok[:, t] = torch.multinomial(F.softmax(lg/temp, -1), 1)[:, 0] + lo; calls += 1
            else:
                kt = 1 if name in ('c1', 'bnd') else k_rvq
                remaining = torch.ones(n, w, dtype=torch.bool, device=dev)
                for p in range(kt):
                    lg, _ = s.run(tok, pos, ln, s.T)
                    pr = F.softmax(lg[:, a:e, lo:hi]/temp, -1)
                    samp = torch.multinomial(pr.reshape(-1, hi-lo), 1).view(n, w)
                    conf = pr.gather(-1, samp[..., None])[..., 0]
                    if p < kt - 1:
                        u = torch.rand_like(conf).clamp(1e-6, 1-1e-6)
                        conf = conf + (-torch.log(-torch.log(u)))          # Gumbel exploration
                        keep_n = int(math.floor(w*math.cos(math.pi/2*(p+1)/kt)))
                        reveal = (remaining.sum(1) - keep_n).clamp_min(0)
                        rank = conf.masked_fill(~remaining, -1e9).argsort(1, descending=True).argsort(1)
                        chosen = remaining & (rank < reveal[:, None])
                    else:
                        chosen = remaining
                    sub = tok[:, a:e]; sub[chosen] = samp[chosen] + lo
                    remaining = remaining & ~chosen; calls += 1
        if g is None:
            b = (torch.arange(L, device=dev) % st == 0)[None].expand(n, L).clone(); g = geometry(b)
        base = 2 if s.dyn else 1
        c1 = tok[:, :cfg.G]
        codes = torch.stack([tok[:, s.ends[base+q]-M:s.ends[base+q]] - s.lv[base+q][1]
                             for q in range(cfg.Q)], -1)
        return codec.decode_tokens(c1, codes, g, temp), calls

# %%
# ---------------- byte-GPT baseline (exact likelihood, KV-cached sampler) ----------------------
class ByteGPT(nn.Module):
    def __init__(s, cfg):
        super().__init__(); s.cfg = cfg
        s.emb = nn.Embedding(256, cfg.d); s.pos = nn.Embedding(cfg.L, cfg.d)
        s.stack = Stack(cfg.base_layers, cfg.d, cfg.heads, True); s.head = nn.Linear(cfg.d, 256)
    def loss(s, x):
        lg = s.head(s.stack(s.emb(x[:, :-1]) + s.pos.weight[:x.size(1)-1])[0])
        return F.cross_entropy(lg.reshape(-1, 256), x[:, 1:].reshape(-1))
    @torch.no_grad()
    def bits_per_window(s, x, uni):
        lg = s.head(s.stack(s.emb(x[:, :-1]) + s.pos.weight[:x.size(1)-1])[0])
        ce = F.cross_entropy(lg.reshape(-1, 256), x[:, 1:].reshape(-1), reduction='none').view(x.size(0), -1).sum(1)
        return ce/LN2 + uni[x[:, 0]]
    @torch.no_grad()
    def sample(s, n, L, uni, temp=1.0):
        x = torch.multinomial(torch.exp(-uni*LN2), n, replacement=True)[:, None]; past = None
        for t in range(L-1):
            h, past = s.stack(s.emb(x[:, -1:]) + s.pos.weight[t:t+1][None], past)
            x = torch.cat([x, torch.multinomial(F.softmax(s.head(h)[:, -1]/temp, -1), 1)], 1)
        return x, L-1

def nparams(m): return sum(p.numel() for p in m.parameters())

# %%
# ---------------- self-tests (raise before spending GPU time) ----------------------------------
def run_tests():
    torch.manual_seed(0); c = Cfg(**PRESETS['smoke'])
    b = torch.rand(4, 32, device=DEVICE) < 0.3; b[:, 0] = True; g = geometry(b)
    assert (g['length'].sum(1) == 32).all() and (g['offset'][b] == 0).all()
    xb = torch.randint(0, 256, (6, c.L), device=DEVICE)
    for dyn in (False, True):
        cd = Codec(c, dyn).to(DEVICE).train()
        loss, st = cd.loss(xb, 0.5); loss.backward()
        assert torch.isfinite(loss)
        if dyn:
            gw = cd.router[1].weight.grad
            assert gw is not None and gw.abs().sum() > 0, 'router got no gradient'
        cd.eval(); o = cd.encode(xb); tk = cd.tokens_from(o)
        zq = sum(cd.rvq.cb[q][tk['codes'][..., q]] for q in range(c.Q))
        z2 = zq + take(cd.vq1.cb[0][tk['c1']], o['g']['order'] // c.R)
        assert torch.allclose(cd.decode(o['zq']+o['e1c'], o['g'], c.L), cd.decode(z2, o['g'], c.L), atol=1e-2)
        priors = {}
        for mode in ('ar', 'cf'):
            pr = Prior(c, dyn, mode).to(DEVICE); priors[mode] = pr
            with torch.no_grad(): nl = pr.nll(tk)
            assert all(torch.isfinite(v).all() for v in nl.values())
            pr.train_loss(tk).backward()
            pr.eval(); out, calls = pr.sample(cd, 3, k_rvq=2)
            assert out.shape == (3, c.L) and out.min() >= 0 and out.max() < 256
            exp = pr.T if mode == 'ar' else (2 if dyn else 1) + c.Q*2
            assert calls == exp, f'{mode} calls {calls} != {exp}'
        assert nparams(priors['ar']) == nparams(priors['cf']), 'prior param mismatch'
        # KV-cache == full-prefix logits (AR)
        pr = priors['ar']
        tok = seq_tokens(tk, c, dyn, pr.lv); pos, ln = seq_feats(tk['order'], tk['length'], c, dyn, pr.lv)
        with torch.no_grad():
            full, _ = pr.run(tok, pos, ln, pr.T); past = None
            for t in range(pr.T):
                x = (pr.f.bos.expand(6, 1, -1) + pr.f.feats(pos, ln, pr.T)[:, :1]) if t == 0 else \
                    (pr.f.tok(tok[:, t-1])[:, None] + pr.f.feats(pos, ln, pr.T)[:, t:t+1])
                h, past = pr.stack(x, past); inc = pr.head(h)[:, -1]
                assert torch.allclose(inc, full[:, t], atol=1e-4), 'KV cache mismatch'
    raw, _ = gen_synseg(nbytes=5000); a = np.frombuffer(raw, np.uint8)
    ons = onsets('synseg', a); assert ons.sum() > 100
    raw, _ = gen_syngrid(nbytes=5000); a = np.frombuffer(raw, np.uint8)
    assert (onsets('syngrid', a) == (np.arange(len(a)) % 4 == 0)).all()
    lam = 0.5
    for ema in (1.5, 0.5): lam2 = float(np.clip(lam + 2.0*(ema - 1.0), 0, 8))
    assert lam2 != lam
    print('self-tests passed')
run_tests()

# %%
# ---------------- training / eval utilities ----------------------------------------------------
def fit(model, step_fn, steps, lr, name, maintain_every=200, controller=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    warm = max(1, min(100, steps//5))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t+1)/warm) *
                                              (0.1 + 0.9*0.5*(1 + math.cos(math.pi*min(1.0, t/steps)))))
    model.train(); t0 = time.time(); ema = None; every = max(1, steps//5)
    for step in range(steps):
        loss, st = step_fn(step)
        if not torch.isfinite(loss): raise FloatingPointError(f'{name} step {step}')
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if hasattr(model, 'maintain') and step % maintain_every == maintain_every-1: model.maintain()
        if controller is not None: controller(step, st)
        v = float(loss.detach()); ema = v if ema is None else 0.98*ema + 0.02*v
        if step % every == 0 or step == steps-1:
            extra = ' '.join(f'{k}={float(x):.3f}' for k, x in st.items())
            print(f'  [{name}] {step:5d}/{steps} loss~{ema:.3f} {extra} ({time.time()-t0:.0f}s)', flush=True)
    model.eval(); return time.time() - t0

@torch.no_grad()
def collect(fn, corpus, L, n, bs=128):
    starts = make_starts(corpus, L, n); acc = {}
    for i in range(0, n, bs):
        d = fn(windows(corpus['va'], starts[i:i+bs], L))
        for k, v in d.items(): acc.setdefault(k, []).append(v.float().cpu())
    return {k: torch.cat(v).numpy() for k, v in acc.items()}

@torch.no_grad()
def components(codec, prior, x):
    o = codec.encode(x, None)
    rec = F.cross_entropy(codec.decode(o['zq']+o['e1c'], o['g'], x.size(1)).reshape(-1, 256),
                          x.reshape(-1), reduction='none').view(x.size(0), -1).sum(1) / LN2
    d = {'recon': rec}
    for k, v in prior.nll(codec.tokens_from(o)).items(): d[k] = v / LN2
    d['prior'] = sum(v for k, v in d.items() if k != 'recon')
    d['total'] = d['recon'] + d['prior']
    return d

@torch.no_grad()
def rd_curve(codec, corpus, n=128):
    cfg = codec.cfg; x = windows(corpus['va'], make_starts(corpus, cfg.L, n), cfg.L); out = {}
    for k in range(cfg.Q+1):
        keep = (torch.arange(cfg.Q, device=DEVICE)[None] < k).expand(n, cfg.Q)
        o = codec.encode(x, keep)
        out[k] = float(F.cross_entropy(codec.decode(o['zq']+o['e1c'], o['g'], cfg.L).reshape(-1, 256),
                                       x.reshape(-1)) / LN2)
    o = codec.encode(x, None); ppl, util = [], []
    for q in range(cfg.Q):
        h = torch.bincount(o['codes'][..., q].reshape(-1), minlength=cfg.K).float(); p = h/h.sum()
        ppl.append(float(torch.exp(-(p[p > 0]*p[p > 0].log()).sum()))); util.append(float((h > 0).float().mean()))
    return out, ppl, util

@torch.no_grad()
def alignment(codec, corpus, n=256):
    cfg = codec.cfg; x = windows(corpus['va'], make_starts(corpus, cfg.L, n), cfg.L)
    order = codec.encode(x, None)['g']['order'].cpu().numpy(); xb = x.cpu().numpy()
    hits = tot = grid_hits = 0
    for i in range(n):
        ons = onsets(corpus['name'], xb[i])
        for j in order[i, 1:]:
            hits += int(ons[j]); tot += 1
        for j in range(cfg.stride, cfg.L, cfg.stride): grid_hits += int(ons[j])
    return hits/max(tot, 1), grid_hits/max(n*(cfg.M-1), 1)

def timed(fn):
    if DEVICE == 'cuda': torch.cuda.synchronize()
    t0 = time.time(); r = fn()
    if DEVICE == 'cuda': torch.cuda.synchronize()
    return r, time.time() - t0

def show(t, lim=140): return bytes(t.tolist())[:lim].decode('latin-1').replace('\n', '⏎')
def text_stats(ts):
    bg = [t[i:i+2] for t in ts for i in range(len(t)-1)]
    return len(set(bg))/max(len(bg), 1), 1 - len(set(bg))/max(len(bg), 1)

# %%
# ---------------- main experiment --------------------------------------------------------------
def run_all(cfg=CFG):
    RES = dict(cfg=asdict(cfg), preset=PRESET, device=DEVICE,
               torch=torch.__version__, started=datetime.datetime.utcnow().isoformat()+'Z',
               gpu=torch.cuda.get_device_name(0) if DEVICE == 'cuda' else None, runs=[])
    path = os.path.join(OUT, 'results.json')
    def save():
        with open(path, 'w') as f: json.dump(RES, f)
    for cname in cfg.corpora:
        corpus = get_corpus(cname); uni = unigram_bits(corpus)
        RES.setdefault('corpora', {})[cname] = {k: corpus[k] for k in ('sha', 'nbytes', 'name')}
        uw = collect(lambda x: {'total': uni[x].sum(1)}, corpus, cfg.L, cfg.eval_windows)['total']/cfg.L
        RES.setdefault('unigram', {})[cname] = float(uw.mean())
        print(f"\n######## corpus={cname} sha={corpus['sha'][:12]} unigram={uw.mean():.3f} bpb")
        for seed in cfg.seeds:
            torch.manual_seed(1000*seed+1); np.random.seed(seed)
            gpt = ByteGPT(cfg).to(DEVICE)
            ts = fit(gpt, lambda s: (gpt.loss(train_batch(corpus, cfg.L, cfg.bs)), {}),
                     cfg.base_steps, cfg.base_lr, f'gpt {cname} s{seed}')
            gb = collect(lambda x: {'total': gpt.bits_per_window(x, uni)}, corpus, cfg.L, cfg.eval_windows)['total']/cfg.L
            (smp, calls), dt = timed(lambda: gpt.sample(cfg.sample_n, cfg.L, uni))
            ref = float((gpt.bits_per_window(smp, uni)/cfg.L).mean())
            d2, rep = text_stats([bytes(r.tolist()) for r in smp])
            RES['runs'].append(dict(corpus=cname, kind='gpt', seed=seed, chunk='-', prior='ar', k=None,
                params=nparams(gpt), bpb=float(gb.mean()), per_window=gb.tolist(), calls=calls,
                sample_sec=dt, ref_bpb=ref, distinct2=d2, repeat=rep, train_sec=ts,
                samples=[show(smp[i]) for i in range(3)]))
            print(f'  byte GPT val bpb={gb.mean():.4f}'); save()
            for dyn in (False, True):
                chunk = 'dynamic' if dyn else 'fixed'
                torch.manual_seed(1000*seed+2+int(dyn)); np.random.seed(seed)
                codec = Codec(cfg, dyn).to(DEVICE)
                ctl = dict(lam=0.1, ema=None)
                def controller(step, st):
                    r = float(st['rate_bpb'])
                    ctl['ema'] = r if ctl['ema'] is None else 0.99*ctl['ema'] + 0.01*r
                    if step % 50 == 49:
                        ctl['lam'] = float(np.clip(ctl['lam'] + cfg.lam_eta*(ctl['ema'] - cfg.rho), 0, cfg.lam_max))
                tsc = fit(codec, lambda s: codec.loss(train_batch(corpus, cfg.L, cfg.bs), ctl['lam']),
                          cfg.codec_steps, cfg.codec_lr, f'codec {chunk} {cname} s{seed}', controller=controller)
                codec.eval()
                for p in codec.parameters(): p.requires_grad_(False)
                rd, ppl, util = rd_curve(codec, corpus)
                al, grid_al = alignment(codec, corpus) if dyn else (None, None)
                rate = float(collect(lambda x: {'r': components(codec, Prior(cfg, dyn, 'ar').to(DEVICE), x)['prior']*0},
                                     corpus, cfg.L, 8)['r'].mean())  # placeholder removed below
                # honest achieved latent rate under the entropy head:
                def eh_rate(x):
                    o = codec.encode(x, None); tk = codec.tokens_from(o)
                    tok = seq_tokens(tk, cfg, dyn, codec.lv)
                    pos, ln = seq_feats(tk['order'], tk['length'], cfg, dyn, codec.lv)
                    lg = codec.eh_logits(tok, pos, ln); tot = 0
                    for li, (name, lo, hi, w) in enumerate(codec.lv):
                        a, e = codec.ends[li]-w, codec.ends[li]
                        tot = tot + F.cross_entropy(lg[:, a:e, lo:hi].reshape(-1, hi-lo),
                                                    (tok[:, a:e]-lo).reshape(-1), reduction='sum')
                    return (tot/(x.size(0)*cfg.L)/LN2).detach()
                achieved = float(collect(lambda x: {'r': eh_rate(x)}, corpus, cfg.L, 256)['r'].mean())
                print(f'  codec {chunk}: rho={cfg.rho} achieved={achieved:.3f} lam={ctl["lam"]:.2f} '
                      f'RD={[round(rd[k],3) for k in rd]} util={[round(u,2) for u in util]}'
                      + (f' align={al:.3f} (grid {grid_al:.3f})' if dyn else ''))
                torch.save(codec.state_dict(), os.path.join(OUT, f'codec_{cname}_{chunk}_s{seed}.pt'))
                recon_ref = None
                for mode in ('ar', 'cf'):
                    torch.manual_seed(1000*seed+10+int(dyn)*2+(mode == 'cf')); np.random.seed(seed)
                    prior = Prior(cfg, dyn, mode).to(DEVICE)
                    tsp = fit(prior, lambda s: (prior.train_loss(codec.tokenize(train_batch(corpus, cfg.L, cfg.bs))), {}),
                              cfg.prior_steps, cfg.prior_lr, f'{chunk}/{mode} {cname} s{seed}')
                    comps = collect(lambda x: components(codec, prior, x), corpus, cfg.L, cfg.eval_windows)
                    if mode == 'ar': recon_ref = comps['recon']
                    else: assert np.allclose(recon_ref, comps['recon'], atol=1e-4), 'SHARED-CODEC VIOLATION'
                    tot = comps['total']/cfg.L
                    budgets = [None] if mode == 'ar' else list(cfg.cf_k_list)
                    for k in budgets:
                        (smp, calls), dt = timed(lambda: prior.sample(codec, cfg.sample_n, k_rvq=k or 1))
                        ref = float((gpt.bits_per_window(smp, uni)/cfg.L).mean())
                        d2, rep = text_stats([bytes(r.tolist()) for r in smp])
                        RES['runs'].append(dict(corpus=cname, kind='concept', seed=seed, chunk=chunk, prior=mode, k=k,
                            params=nparams(codec)+nparams(prior), bpb=float(tot.mean()), per_window=tot.tolist(),
                            comps={kk: float(v.mean()/cfg.L) for kk, v in comps.items()},
                            calls=calls, sample_sec=dt, ref_bpb=ref, distinct2=d2, repeat=rep,
                            train_sec=tsc+tsp, rd=rd, code_ppl=ppl, util=util, lam=ctl['lam'],
                            achieved_rate=achieved, align=al, grid_align=grid_al,
                            samples=[show(smp[i]) for i in range(3)]))
                        print(f'  => {chunk}+{mode}'+(f'(k={k})' if k else '')+
                              f' bound={tot.mean():.4f} calls={calls} ref={ref:.3f} distinct2={d2:.3f}')
                        save()
    RES['finished'] = datetime.datetime.utcnow().isoformat()+'Z'; save()
    print('\nsaved', path); return RES
RESULTS = run_all()

# %%
# ---------------- analysis: reads ONLY results.json; recomputes headlines from raw arrays ------
RES = json.load(open(os.path.join(OUT, 'results.json')))
runs = RES['runs']; L = RES['cfg']['L']
def pick(**kw): return [r for r in runs if all(r.get(k) == v for k, v in kw.items())]
def ms(v):
    v = np.array(v, float); return v.mean(), (v.std(ddof=1)/np.sqrt(len(v)) if len(v) > 1 else float('nan'))
def boot(x, reps=4000, seed=0):
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(x), (reps, len(x)))
    m = x[idx].mean(1); return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))
CELLS = [('fixed', 'ar'), ('dynamic', 'ar'), ('fixed', 'cf'), ('dynamic', 'cf')]
print('provenance:', {k: RES[k] for k in ('preset', 'device', 'torch', 'started')})
for cname, meta in RES['corpora'].items():
    print(f"\n================ {cname} (sha {meta['sha'][:12]}, {meta['nbytes']:,} bytes) ================")
    seeds = sorted({r['seed'] for r in runs if r['corpus'] == cname})
    gpt = pick(corpus=cname, kind='gpt'); g0 = ms([r['bpb'] for r in gpt])[0]
    print(f"unigram {RES['unigram'][cname]:.3f} | byte GPT {g0:.3f} (exact)")
    print(f"{'cell':<14}{'bound bpb':>12}{'recon':>8}{'c1':>7}{'bnd':>7}{'codes':>8}{'rate':>7}{'align':>7}")
    for chunk, mode in CELLS:
        rr = pick(corpus=cname, kind='concept', chunk=chunk, prior=mode)
        if not rr: continue
        m, s = ms([r['bpb'] for r in rr]); cm = lambda k: np.mean([r['comps'].get(k, 0.0) for r in rr])
        codes = sum(cm(k) for k in rr[0]['comps'] if k.startswith('r') and k not in ('recon',))
        al = np.mean([r['align'] for r in rr if r['align'] is not None] or [float('nan')])
        print(f"{chunk+'+'+mode:<14}{m:>8.3f}±{s:4.3f}{cm('recon'):>8.3f}{cm('c1'):>7.3f}{cm('bnd'):>7.3f}"
              f"{codes:>8.3f}{np.mean([r['achieved_rate'] for r in rr]):>7.3f}{al:>7.3f}")
    # gates
    fcf = ms([r['bpb'] for r in pick(corpus=cname, kind='concept', chunk='fixed', prior='cf')])[0]
    far = ms([r['bpb'] for r in pick(corpus=cname, kind='concept', chunk='fixed', prior='ar')])[0]
    rf = np.mean([r['achieved_rate'] for r in pick(corpus=cname, chunk='fixed', prior='ar')])
    rd_ = np.mean([r['achieved_rate'] for r in pick(corpus=cname, chunk='dynamic', prior='ar')])
    rr = pick(corpus=cname, chunk='fixed', prior='ar')[0]
    G = dict(G1=fcf < RES['unigram'][cname], G2=far <= 1.25*g0,
             G3=all(rr['rd'][k] >= rr['rd'][k+1]-1e-6 for k in range(len(rr['rd'])-1)) and min(rr['util']) > 0.2,
             G4=abs(rf-rd_) <= 0.15)
    print('gates:', G, '| all pass:', all(G.values()))
    # interaction (== differential parallelism gap; recon cancels exactly)
    per_seed = []
    for seed in seeds:
        pw = {}
        for cell in CELLS:
            rr = pick(corpus=cname, kind='concept', seed=seed, chunk=cell[0], prior=cell[1])
            if rr: pw[cell] = np.array(rr[0]['per_window'])
        if len(pw) < 4: continue
        I = (pw[('fixed', 'cf')] - pw[('dynamic', 'cf')]) - (pw[('fixed', 'ar')] - pw[('dynamic', 'ar')])
        lo, hi = boot(I); per_seed.append(I.mean())
        print(f'  seed {seed}: I={I.mean():+.4f}  95% paired-window CI [{lo:+.4f}, {hi:+.4f}]')
    if per_seed:
        m, s = ms(per_seed)
        print(f'  across seeds: I={m:+.4f} ± {s:.4f} SEM (n={len(per_seed)}); mean-2SEM={m-2*s:+.4f}')
    # utility Pareto (sample quality vs sequential depth)
    print(f"{'arm':<20}{'calls':>7}{'sec':>8}{'ref-bpb':>9}{'distinct2':>10}{'repeat':>8}")
    for r in [r for r in runs if r['corpus'] == cname and r['seed'] == seeds[0]]:
        tag = 'byte GPT' if r['kind'] == 'gpt' else f"{r['chunk']}+{r['prior']}" + (f"(k={r['k']})" if r['k'] else '')
        print(f"{tag:<20}{r['calls']:>7}{r['sample_sec']:>8.2f}{r['ref_bpb']:>9.3f}{r['distinct2']:>10.3f}{r['repeat']:>8.3f}")
print("""
DECISION RULES (pre-registered):
  synseg  : expect I CI lower bound > 0  (positive control; if not, machinery/hypothesis broken)
  syngrid : expect I CI covers 0         (negative control; if not, CONFOUND ALARM)
  shakespeare: claim mechanism support only if gates pass AND across-seed mean-2SEM > 0.
  Utility claim (independent of I): dynamic+CF ref-bpb < fixed+CF at equal calls, approaching AR at << depth.
""")

# %%
import matplotlib.pyplot as plt
RES = json.load(open(os.path.join(OUT, 'results.json')))
cn = RES['cfg']['corpora'][-1]; seeds = sorted({r['seed'] for r in RES['runs'] if r['corpus'] == cn})
fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
names = [f'{c}+{m}' for c, m in CELLS]
vals = [ms([r['bpb'] for r in pick(corpus=cn, kind='concept', chunk=c, prior=m)]) for c, m in CELLS]
ax[0].bar(names, [v[0] for v in vals], yerr=[0 if np.isnan(v[1]) else v[1] for v in vals], capsize=4)
ax[0].axhline(RES['unigram'][cn], color='k', ls=':', label='unigram')
ax[0].axhline(ms([r['bpb'] for r in pick(corpus=cn, kind='gpt')])[0], color='g', ls='--', label='byte GPT')
ax[0].set_ylabel('val bpb (two-part bound)'); ax[0].set_title(cn); ax[0].legend(fontsize=8)
ax[0].tick_params(axis='x', rotation=15)
for r in [r for r in RES['runs'] if r['corpus'] == cn and r['seed'] == seeds[0]]:
    tag = 'GPT' if r['kind'] == 'gpt' else f"{r['chunk'][:3]}/{r['prior']}" + (f"k{r['k']}" if r['k'] else '')
    ax[1].scatter(r['calls'], r['ref_bpb'], s=60)
    ax[1].annotate(tag, (r['calls'], r['ref_bpb']), fontsize=7)
ax[1].set_xscale('log'); ax[1].set_xlabel('sequential network calls (log)'); ax[1].set_ylabel('sample quality: GPT bits/byte')
ax[1].set_title('quality vs depth (KV-cached AR)')
bottoms = np.zeros(4)
for lab, keyf, colr in [('recon', lambda k: k == 'recon', '#4c72b0'), ('c1', lambda k: k == 'c1', '#55a868'),
                        ('bnd', lambda k: k == 'bnd', '#8172b3'),
                        ('codes', lambda k: k.startswith('r') and k not in ('recon', 'prior'), '#dd8452')]:
    v = np.array([np.mean([sum(x for k, x in r['comps'].items() if k not in ('total', 'prior') and keyf(k))
                           for r in pick(corpus=cn, kind='concept', chunk=c, prior=m)]) for c, m in CELLS])
    ax[2].bar(names, v, bottom=bottoms, label=lab, color=colr); bottoms += v
ax[2].set_ylabel('bits/byte'); ax[2].set_title('bit budget'); ax[2].legend(fontsize=7)
ax[2].tick_params(axis='x', rotation=15)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'hncm_v3.png'), dpi=140); plt.show()
for r in [r for r in RES['runs'] if r['corpus'] == cn and r['seed'] == seeds[0]]:
    tag = 'GPT' if r['kind'] == 'gpt' else f"{r['chunk']}+{r['prior']}" + (f"(k={r['k']})" if r['k'] else '')
    print(f"{tag:<22} {r['samples'][0]!r}")     # verbatim, never edited

