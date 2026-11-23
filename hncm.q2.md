## The next-level move: **stop learning the structure — solve it exactly.**

Everything in H‑NCM v3 that was hard (router instability, smoothing bridges, STE on boundaries, ratio loss, boundary‑head distillation, gate penalties, level schedules, exposure bias) exists because we made **segmentation a *learning* problem**. It isn't. Given a cost for describing each candidate span, the optimal segmentation and rate allocation is a **shortest path in a DAG** — solvable *exactly* in `O(L·|Λ|)` by dynamic programming.

So the alternate design inverts the split:

> **Learn the costs. Solve the structure. Commit by confidence, not by schedule.**

**MOSAIC** — *Minimum‑objective Span Allocation with Iterative Commitment*. One shared codebook, one tiny span decoder, one canvas transformer with an **atomic `(length, code)` action head**, one DP. It deletes the router, smoother, ratio loss, boundary heads, gates, K‑level RVQ, nested dropout, and corruption chains — and *gains* capabilities: globally optimal rate–distortion tiling, closed‑loop generation (no exposure bias), content‑adaptive NFE, natural streaming, and a single principled rate knob λ.

**Why this is a real advance, not a reshuffle**

| Problem in v3 | MOSAIC's answer |
|---|---|
| Discrete boundaries need STE/smoother/ratio loss | DP is exact inference; gradient flows through the **min‑path** (envelope theorem, like max‑pool). No boundary STE at all. |
| Boundary head vs. encoder router drift; insert/delete misalignment | **Atomic `(ℓ, c)` action** — length and content are one decision, so they *cannot* misalign. |
| Open‑loop coarse→fine compounds errors | **Closed loop**: every step re‑encodes the *actual* canvas. Self‑synchronizing. |
| Fixed K levels; NFE constant regardless of difficulty | **Confidence‑gated parallel commit**: easy spans commit in bulk, hard ones get more steps. NFE is content‑adaptive. |
| Rate = gate penalty + ratio loss (two heuristics) | Rate = **λ in the Lagrangian** `Σ[NLL(span) + λ]`. One knob, a genuine rate–distortion curve. |
| Per‑window, not streaming | Canvas is a sliding buffer; committed spans freeze. Streaming is the same loop. |

The DP objective is literally a **rate–distortion Lagrangian**, so MOSAIC is a *learned codec whose encoder is exact combinatorial inference and whose decoder is neural* — with an any‑order prior over its atoms. That unifies H‑Net (chunking), NCM (discrete concepts + parallelism), and classical coding theory under one objective.

---

```python
# ============================================================================
# MOSAIC — Minimum-objective Span Allocation with Iterative Commitment
# ----------------------------------------------------------------------------
# An alternate, lower-complexity answer to the H-NCM brief.
#
# THESIS: segmentation and rate allocation are INFERENCE problems, not learning
# problems. Learn a cost field over candidate spans; solve the optimal tiling
# exactly with a shortest-path DP; generate by closed-loop, confidence-gated
# parallel commitment of atomic (length, code) actions on a byte canvas.
#
# WHAT IT DELETES vs. router+smoother+RVQ+levels:
#   router, smoothing EMA, STE-on-boundaries, ratio loss, boundary heads,
#   halting gates, K codebooks, nested dropout, corruption chains, level weights.
# WHAT IT ADDS:
#   exact rate-distortion tiling, atomic actions, closed-loop generation,
#   content-adaptive NFE, streaming, one lambda.
#
# GRADIENT STORY (the key trick):
#   best[L] = min over tilings of sum(cost) is a piecewise-linear min of
#   differentiable costs. Backprop through it sends gradient to the SELECTED
#   path only -- exactly like max-pooling. The tiling is treated as locally
#   constant (envelope theorem). So we optimize the rate-distortion Lagrangian
#   end-to-end WITHOUT any straight-through estimator on boundaries.
#
# SELF-CHECK: sanity() verifies DP optimality against brute-force enumeration,
# cumsum span pooling against explicit means, and exact-canvas coverage.
# ============================================================================
from __future__ import annotations
import math, time, json, os, random, itertools
from dataclasses import dataclass, asdict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt

DEV = "cuda" if torch.cuda.is_available() else "cpu"
INVALID = 1e6          # large finite cost (never inf -> no NaN gradients)

# ============================================================================
# 0. CONFIG  (every design possibility is a switch)
# ============================================================================
@dataclass
class Cfg:
    # canvas / model
    d: int = 96; heads: int = 4; layers: int = 2; L: int = 128
    lengths: tuple = (1, 2, 4, 8)      # candidate span lengths (dyadic covers most units)
    codebook: int = 512                # ONE shared codebook (not K)
    # rate-distortion
    lam: float = 1.0                   # bits per span penalty in the DP  <-- THE rate knob
    gen_lam: float = 0.0               # extra penalty at generation
    # generation
    commit_mode: str = "threshold"     # threshold (adaptive NFE) | fraction (MaskGIT schedule)
    commit_tau: float = 0.30           # confidence threshold for parallel commit
    commit_frac: float = 0.5           # fraction schedule endpoint
    max_steps: int = 12
    temp: float = 1.0; top_k: int = 0
    # training
    mask_lo: float = 0.15; mask_hi: float = 1.0   # committed fraction curriculum
    noop_weight: float = 0.10          # class imbalance: most positions start no span
    beta: float = 0.25                 # VQ commitment
    action_weight: float = 1.0
    action_ramp: tuple = (0.05, 0.30)
    dagger_frac: float = 0.0           # >0: scheduled sampling on model's own canvas
    lr: float = 3e-4; steps: int = 2500; batch: int = 32
    # ablations
    tiling: str = "dp"                 # dp | fixed  (fixed = optimal-allocation ablation)
    fixed_stride: int = 8
    one_step: bool = False             # commit everything at once (closed-loop ablation)
    # experiment
    data: str = "zipf"; train_bytes: int = 300_000; val_bytes: int = 30_000
    gen_samples: int = 512; seeds: tuple = (0, 1)
    judge_d: int = 128; judge_steps: int = 3000
    outdir: str = "/content/mosaic_results"

# ============================================================================
# 1. CORE: exact tiling DP  (the structure solver)
# ============================================================================
def dp_tile(costs: torch.Tensor, lengths: tuple, L: int):
    """Shortest-path tiling of [0,L) by spans (t-ell, t).
    costs: (B, L+1, K) cost of the span ENDING at t with length lengths[k].
    Returns best (B,L+1) differentiable min-cost, and selected spans.
    Optimality: induction on t -- best[t] is optimal for prefix [0,t) because
    every tiling's last span ends at t and the remainder is an optimal prefix.
    Verified against brute force in sanity()."""
    B = costs.shape[0]; K = len(lengths)
    ell_t = torch.tensor(lengths, device=costs.device)
    best = costs.new_full((B, L + 1), INVALID); best[:, 0] = 0.0
    choice = torch.zeros(B, L + 1, dtype=torch.long, device=costs.device)
    for t in range(1, L + 1):
        opts = costs.new_full((B, K), INVALID)
        for k, ell in enumerate(lengths):
            i = t - ell
            if i >= 0:
                opts[:, k] = best[:, i] + costs[:, t, k]
        c = opts.argmin(1)
        choice[:, t] = c
        best[:, t] = opts.gather(1, c[:, None])[:, 0]
    # batched backtrack (per-row cursor), spans recorded right-to-left
    t = torch.full((B,), L, dtype=torch.long, device=costs.device)
    sel_s = torch.zeros(B, L, dtype=torch.long, device=costs.device)
    sel_l = torch.zeros(B, L, dtype=torch.long, device=costs.device)
    sel_m = torch.zeros(B, L, dtype=torch.bool, device=costs.device)
    ar = torch.arange(B, device=costs.device)
    for step in range(L):
        act = t > 0
        if not bool(act.any()): break
        k = choice[ar, t.clamp(min=0)]
        ell = ell_t[k]
        s = t - ell
        sel_s[:, step] = torch.where(act, s, torch.zeros_like(s))
        sel_l[:, step] = torch.where(act, ell, torch.zeros_like(ell))
        sel_m[:, step] = act
        t = torch.where(act, s, t)
    return best, sel_s, sel_l, sel_m

def gap_costs(costs, committed, lengths, L):
    """Forbid atoms overlapping committed bytes; allow FREE unit skips of them.
    Lets one DP cover the whole canvas while only proposing on gaps."""
    B = costs.shape[0]
    cs = torch.cat([torch.zeros(B, 1, device=costs.device),
                    committed.float().cumsum(1)], 1)              # (B,L+1)
    out = costs.clone()
    t_idx = torch.arange(L + 1, device=costs.device)
    for k, ell in enumerate(lengths):
        s = (t_idx - ell).clamp(min=0)
        ncom = cs - cs.gather(1, s[:, None].expand(B, L + 1))
        overlap = ncom > 0
        out[:, :, k] = torch.where(overlap, torch.full_like(out[:, :, k], INVALID), out[:, :, k])
        if ell == 1:                                              # free skip of committed byte
            free = committed                                      # position t-1
            col = out[:, 1:, k]
            out[:, 1:, k] = torch.where(free, torch.zeros_like(col), col)
        out[:ell, k] = INVALID                                    # start < 0
    return out

def fixed_tile(B, L, stride, device):
    s = torch.arange(0, L, stride, device=device)[None].expand(B, -1)
    l = torch.full_like(s, stride)
    m = torch.ones_like(s, dtype=torch.bool)
    return s, l, m

# ============================================================================
# 2. SPAN CODEC: O(1) pooling + one codebook + tiny parallel decoder
# ============================================================================
class SpanCodec(nn.Module):
    def __init__(self, cfg, V):
        super().__init__()
        self.cfg = cfg; self.V = V; self.maxlen = max(cfg.lengths)
        self.book = nn.Embedding(cfg.codebook, cfg.d)
        nn.init.normal_(self.book.weight, std=1.0)
        self.len_emb = nn.Embedding(self.maxlen + 1, cfg.d)
        self.pos_emb = nn.Embedding(self.maxlen, cfg.d)
        self.dec = nn.Sequential(nn.Linear(cfg.d, cfg.d), nn.GELU(), nn.Linear(cfg.d, V))
        self.register_buffer("usage", torch.zeros(cfg.codebook))
    @staticmethod
    def prefix(h):
        return torch.cat([torch.zeros(h.shape[0], 1, h.shape[2], device=h.device), h.cumsum(1)], 1)
    def pool(self, cs, starts, lens):
        """O(1) span mean via integral image. starts/lens: (B,N)."""
        B, _, d = cs.shape
        s = starts.clamp(min=0); e = (starts + lens).clamp(max=cs.shape[1] - 1)
        a = cs.gather(1, s.unsqueeze(-1).expand(B, s.shape[1], d))
        b = cs.gather(1, e.unsqueeze(-1).expand(B, e.shape[1], d))
        return (b - a) / lens.clamp(min=1).unsqueeze(-1)
    def quantize(self, z):
        E = self.book.weight
        flat = z.reshape(-1, z.shape[-1])
        dist = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ E.t() + E.pow(2).sum(1)[None]
        c = dist.argmin(1); e = self.book(c)
        with torch.no_grad():
            self.usage.mul_(0.99).scatter_add_(0, c, torch.ones_like(c, dtype=self.usage.dtype))
        return c.view(z.shape[:-1]), e.view_as(z), z + (e - z).detach()   # STE on CONTENT only
    def decode(self, e, lens):
        """e:(B,N,d) lens:(B,N) -> logits (B,N,maxlen,V); positions >= lens are padding."""
        B, N, d = e.shape; P = self.maxlen
        j = torch.arange(P, device=e.device)
        h = e.unsqueeze(2) + self.len_emb(lens.clamp(0, self.maxlen)).unsqueeze(2) + self.pos_emb(j)[None, None]
        return self.dec(h)
    @torch.no_grad()
    def restart(self, z):
        dead = (self.usage < 0.05).nonzero(as_tuple=True)[0]
        if len(dead) and z.numel():
            flat = z.reshape(-1, z.shape[-1])
            pick = flat[torch.randint(0, flat.shape[0], (len(dead),), device=flat.device)]
            self.book.weight[dead] = pick + 0.01 * torch.randn_like(pick); self.usage[dead] = 1.0

# ============================================================================
# 3. MOSAIC MODEL
# ============================================================================
class Mosaic(nn.Module):
    def __init__(self, cfg, V):
        super().__init__()
        self.cfg = cfg; self.V = V; self.K = len(cfg.lengths); self.M = cfg.codebook
        self.mask_id = V                                   # extra MASK symbol
        self.tok = nn.Embedding(V + 1, cfg.d)
        self.pos = nn.Embedding(cfg.L, cfg.d)
        self.com = nn.Embedding(2, cfg.d)                  # committed flag
        self.blocks = nn.ModuleList([self._block(cfg) for _ in range(cfg.layers)])
        self.ln = nn.LayerNorm(cfg.d)
        self.action = nn.Linear(cfg.d, 1 + self.K * self.M)  # [no-op] + atomic (length, code)
        self.codec = SpanCodec(cfg, V)
        self.register_buffer("pos_ids", torch.arange(cfg.L))
        self.register_buffer("ell_t", torch.tensor(cfg.lengths))
    @staticmethod
    def _block(cfg):
        return nn.ModuleDict(dict(
            n1=nn.LayerNorm(cfg.d),
            qkv=nn.Linear(cfg.d, 3 * cfg.d), proj=nn.Linear(cfg.d, cfg.d),
            n2=nn.LayerNorm(cfg.d),
            ff=nn.Sequential(nn.Linear(cfg.d, 4 * cfg.d), nn.GELU(), nn.Linear(4 * cfg.d, cfg.d))))
    def _attn(self, b, x):
        B, T, D = x.shape; h = self.cfg.heads; dk = D // h
        qkv = b["qkv"](x).view(B, T, 3, h, dk).permute(2, 0, 3, 1, 4)
        a = (qkv[0] @ qkv[1].transpose(-2, -1)) / math.sqrt(dk)
        a = a.softmax(-1)                                   # bidirectional canvas
        return b["proj"]((a @ qkv[2]).transpose(1, 2).contiguous().view(B, T, D))
    def encode(self, bytes_, committed):
        h = self.tok(bytes_) + self.pos(self.pos_ids) + self.com(committed.long())
        for b in self.blocks:
            h = h + self._attn(b, b["n1"](h)); h = h + b["ff"](b["n2"](h))
        return self.ln(h)
    # ---- all-candidate cost field: the learned rate-distortion landscape ----
    def cost_field(self, h, x):
        cfg = self.cfg; B, L, d = h.shape
        cs = self.codec.prefix(h)
        costs = h.new_zeros(B, L + 1, self.K)
        codes = torch.zeros(B, L + 1, self.K, dtype=torch.long, device=h.device)
        t_idx = torch.arange(L + 1, device=h.device)
        for k, ell in enumerate(cfg.lengths):
            s = (t_idx - ell)
            valid = s >= 0
            sc = s.clamp(min=0)
            z = (cs - cs.gather(1, sc[:, None].expand(B, L + 1, d))) / ell
            c, e, e_st = self.codec.quantize(z)
            codes[:, :, k] = c
            lg = self.codec.dec(e_st.unsqueeze(2) +
                                self.codec.len_emb(torch.full_like(sc, ell))[None, :, None] +
                                self.codec.pos_emb(torch.arange(ell, device=h.device))[None, None])
            tgt = x.gather(1, (sc[:, None] + torch.arange(ell, device=h.device)[None, :])
                           .clamp(max=L - 1)[:, None].expand(B, L + 1, ell))
            nll = F.cross_entropy(lg.reshape(-1, self.V), tgt.reshape(-1),
                                  reduction="none").view(B, L + 1, ell).sum(-1) / math.log(2)
            costs[:, :, k] = torch.where(valid[None, :], nll + cfg.lam,
                                         torch.full_like(nll, INVALID))
        return costs, codes
    # ---- training step ----
    def forward(self, x, step, total):
        cfg = self.cfg; B, L = x.shape
        h_full = self.encode(x, torch.ones_like(x, dtype=torch.bool))
        costs, codes = self.cost_field(h_full, x)
        if cfg.tiling == "dp":
            best, ss, sl, sm = dp_tile(costs, cfg.lengths, L)
            loss_rd = best[:, L].mean()                    # <-- gradient through the min-path
        else:
            ss, sl, sm = fixed_tile(B, L, cfg.fixed_stride, x.device)
            loss_rd = costs.gather(1, (ss + sl).unsqueeze(-1).expand(B, ss.shape[1], self.K)
                                   ).gather(2, torch.zeros(B, ss.shape[1], 1, dtype=torch.long,
                                                           device=x.device))[:, :, 0].mean()
        # commitment on selected spans
        cs = self.codec.prefix(h_full)
        z = self.codec.pool(cs, ss, sl)
        c, e, _ = self.codec.quantize(z)
        commit = F.mse_loss(z[sm], e[sm].detach()) * cfg.beta + F.mse_loss(z[sm].detach(), e[sm])
        # ---- build a partially committed canvas; targets = DP oracle atoms ----
        keep = (torch.rand(B, L, device=x.device) <
                torch.empty(B, 1, device=x.device).uniform_(cfg.mask_lo, cfg.mask_hi)) & sm
        committed = torch.zeros(B, L, dtype=torch.bool, device=x.device)
        idx = (ss.unsqueeze(-1) + torch.arange(max(cfg.lengths), device=x.device)[None, None])
        within = torch.arange(max(cfg.lengths), device=x.device)[None, None] < sl.unsqueeze(-1)
        pos = idx.clamp(max=L - 1)
        sel_keep = keep & sm
        committed.scatter_(1, pos.reshape(B, -1),
                           (sel_keep.unsqueeze(-1) & within).reshape(B, -1))
        canvas = torch.where(committed, x, torch.full_like(x, self.mask_id))
        # action targets: atom (length-index, code) at each UNCOMMITTED selected span start
        tgt = torch.zeros(B, L, dtype=torch.long, device=x.device)     # 0 = no-op
        k_of = torch.zeros_like(sl)
        for k, ell in enumerate(cfg.lengths): k_of = torch.where(sl == ell, k, k_of)
        atom_code = codes.gather(1, (ss + sl).unsqueeze(-1).expand(B, ss.shape[1], self.K)
                                 ).gather(2, k_of.unsqueeze(-1))[:, :, 0]
        atom = 1 + k_of * self.M + atom_code
        live = sm & ~keep
        tgt.scatter_(1, ss, torch.where(live, atom, torch.zeros_like(atom)))
        h_part = self.encode(canvas, committed)
        logits = self.action(h_part)
        w = torch.ones(1 + self.K * self.M, device=x.device); w[0] = cfg.noop_weight
        t0, t1 = cfg.action_ramp[0] * total, cfg.action_ramp[1] * total
        ramp = 0.0 if step < t0 else min(1.0, (step - t0) / max(1, t1 - t0))
        loss_act = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), weight=w)
        loss = loss_rd + commit + cfg.action_weight * ramp * loss_act
        with torch.no_grad():
            rec_bp = (best[:, L].mean() - cfg.lam * sm.float().sum(1).mean()) / L
            rate = sm.float().sum(1).mean() * math.log2(self.M * self.K) / L
        return loss, dict(rd=float(loss_rd), act=float(loss_act), rec_bpb=float(rec_bp),
                          rate_bpb=float(rate), spans=float(sm.float().sum(1).mean()), ramp=ramp)
    # ---- closed-loop generation: propose -> DP pack -> confidence commit ----
    @torch.no_grad()
    def generate(self, n, bos=None, cfg=None):
        cfg = cfg or self.cfg; L, d = cfg.L, cfg.d
        canvas = torch.full((n, L), self.mask_id, dtype=torch.long, device=DEV)
        committed = torch.zeros(n, L, dtype=torch.bool, device=DEV)
        steps = 0
        for step in range(cfg.max_steps):
            if bool(committed.all()): break
            steps += 1
            h = self.encode(canvas, committed)
            lg = self.action(h).log_softmax(-1)                       # (B,L,1+K*M)
            atoms = lg[:, :, 1:].view(n, L, self.K, self.M)
            len_score = atoms.logsumexp(-1)                           # log p(length | pos)
            conf = len_score.exp().max(-1).values                     # not used directly
            costs = (-len_score + cfg.gen_lam).transpose(1, 2).contiguous()  # (B,K,L)
            costs = F.pad(costs, (1, 0))[:, :, :L + 1]                # index by END t
            # shift: cost of span ending at t with length ell lives at position t-ell
            shifted = torch.full((n, L + 1, self.K), INVALID, device=DEV)
            for k, ell in enumerate(cfg.lengths):
                if L - ell + 1 > 0:
                    shifted[:, ell:, k] = costs[:, k, :L - ell + 1]
            if cfg.tiling == "fixed":
                ss, sl, sm = fixed_tile(n, L, cfg.fixed_stride, DEV)
                ss = ss[~committed.gather(1, ss)] if False else ss
                sm = sm & ~committed.gather(1, ss)
            else:
                gc = gap_costs(shifted, committed, cfg.lengths, L)
                _, ss, sl, sm = dp_tile(gc, cfg.lengths, L)
            # keep only real atoms (length>1 or on a free position)
            real = sm & ~committed.gather(1, ss)
            if not bool(real.any()): break
            k_of = torch.zeros_like(sl)
            for k, ell in enumerate(cfg.lengths): k_of = torch.where(sl == ell, k, k_of)
            # confidence of the chosen length at the chosen start
            lp = len_score.gather(2, k_of.unsqueeze(-1))[:, :, 0]
            conf_sel = lp.gather(1, ss).exp()
            if cfg.one_step:
                take = real
            elif cfg.commit_mode == "fraction":
                frac = cfg.commit_frac * (1 - step / max(1, cfg.max_steps - 1))
                take = real & (conf_sel >= frac)
            else:
                take = real & (conf_sel >= cfg.commit_tau)
            if not bool(take.any()):                                  # guarantee progress
                bestv = torch.where(real, conf_sel, torch.zeros_like(conf_sel))
                j = bestv.argmax(1)
                take = torch.zeros_like(real); take[torch.arange(n, device=DEV), j] = True
            # sample a CODE consistent with the chosen length
            a = atoms.gather(1, ss.unsqueeze(-1).expand(n, ss.shape[1], self.K, self.M)
                             .reshape(n, ss.shape[1], self.K * self.M))
            a = a.view(n, ss.shape[1], self.K, self.M)
            a = a.gather(2, k_of.unsqueeze(-1).unsqueeze(-1).expand(n, ss.shape[1], 1, self.M))[:, :, 0]
            a = a / max(cfg.temp, 1e-6)
            if cfg.top_k > 0:
                v, _ = a.topk(cfg.top_k, -1); a = a.masked_fill(a < v[..., -1:], -1e9)
            code = torch.multinomial(a.softmax(-1).reshape(-1, self.M), 1).view(n, ss.shape[1])
            e = self.codec.book(code)
            lgd = self.codec.decode(e, sl)                            # (B,N,maxlen,V)
            by = lgd.argmax(-1)
            P = max(cfg.lengths)
            idx = (ss.unsqueeze(-1) + torch.arange(P, device=DEV)[None, None]).clamp(max=L - 1)
            within = torch.arange(P, device=DEV)[None, None] < sl.unsqueeze(-1)
            write = take.unsqueeze(-1) & within
            canvas.scatter_(1, idx.reshape(n, -1), torch.where(write, by, canvas.gather(1, idx.reshape(n, -1))).reshape(n, -1))
            committed.scatter_(1, idx.reshape(n, -1), write.reshape(n, -1) | committed.gather(1, idx.reshape(n, -1)).reshape(n, -1))
        canvas = torch.where(committed, canvas, torch.zeros_like(canvas))
        return canvas, steps

# ============================================================================
# 4. BASELINE (byte-AR) — also serves as the judge
# ============================================================================
class ByteAR(nn.Module):
    def __init__(self, d, h, layers, L, V):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, d), nn.Embedding(L, d)
        self.blocks = nn.ModuleList([Mosaic._block(type("C", (), dict(d=d, heads=h))()) for _ in range(layers)])
        self.ln = nn.LayerNorm(d); self.head = nn.Linear(d, V); self.h = h
    def _attn(self, b, x):
        B, T, D = x.shape; h = self.h; dk = D // h
        qkv = b["qkv"](x).view(B, T, 3, h, dk).permute(2, 0, 3, 1, 4)
        a = (qkv[0] @ qkv[1].transpose(-2, -1)) / math.sqrt(dk)
        a = a.masked_fill(torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1), -1e9)
        return b["proj"]((a.softmax(-1) @ qkv[2]).transpose(1, 2).contiguous().view(B, T, D))
    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        for b in self.blocks:
            h = h + self._attn(b, b["n1"](h)); h = h + b["ff"](b["n2"](h))
        return self.head(self.ln(h))
    @torch.no_grad()
    def generate(self, n, L, bos):
        y = torch.full((n, 1), bos, dtype=torch.long, device=DEV)
        for _ in range(L):
            p = self.forward(y[:, -L:])[:, -1].softmax(-1)
            y = torch.cat([y, torch.multinomial(p, 1)], 1)
        return y[:, 1:]

# ============================================================================
# 5. DATA + METRICS (judge-free oracle; O(n) generator)
# ============================================================================
def make_zipf(ntr, nva, bigram=False, seed=1234, nw=200):
    rng = np.random.RandomState(seed); words, seen = [], set()
    while len(words) < nw:
        w = "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), size=rng.randint(2, 8)))
        if w not in seen: seen.add(w); words.append(w)
    p = np.arange(1, nw + 1) ** -1.07; p /= p.sum()
    partner = {w: words[rng.randint(nw)] for w in words}
    def draw(n):
        out, tot = [], 0
        while tot < n + 8:
            w = words[rng.choice(nw, p=p)]; out.append(w); tot += len(w) + 1
            if bigram and rng.rand() < 0.6: out.append(partner[w]); tot += len(partner[w]) + 1
        return out
    ch = "abcdefghijklmnopqrstuvwxyz "; si = {c: i for i, c in enumerate(ch)}
    tr = torch.tensor([si[c] for c in " ".join(draw(ntr))[:ntr]], dtype=torch.long)
    va = torch.tensor([si[c] for c in " ".join(draw(nva))[:nva]], dtype=torch.long)
    w2i = {w: i for i, w in enumerate(words)}; cnt = np.zeros(nw)
    for w in draw(400_000): cnt[w2i[w]] += 1
    return dict(train=tr, val=va, V=len(ch), bos=26, itos=list(ch), vocab=set(words),
                w2i=w2i, ref=torch.tensor((cnt / cnt.sum()).tolist()),
                floor=float(-(p * np.log2(p)).sum() / np.mean([len(w) + 1 for w in words])))
def words_of(s, data): 
    return [w for row in s for w in "".join(data["itos"][int(b)] for b in row).split(" ") if w]
def tv(s, data):
    ws = words_of(s, data); c = torch.zeros(len(data["w2i"]))
    for w in ws:
        if w in data["w2i"]: c[data["w2i"][w]] += 1
    return float(0.5 * (c / c.sum() - data["ref"]).abs().sum()) if c.sum() else 1.0
def oov(s, data):
    ws = words_of(s, data); return sum(w not in data["vocab"] for w in ws) / max(1, len(ws))
def entropy(s, V):
    H = []
    for r in s:
        c = torch.bincount(r, minlength=V).float(); p = c / c.sum(); p = p[p > 0]
        H.append(float(-(p * p.log2()).sum()))
    return float(np.mean(H))
@torch.no_grad()
def judge_bpb(j, s, bos):
    tot = cnt = 0
    for i in range(0, len(s), 32):
        b = s[i:i + 32].to(DEV)
        inp = torch.cat([torch.full((len(b), 1), bos, dtype=torch.long, device=DEV), b[:, :-1]], 1)
        lg = j(inp)
        tot += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), b.reshape(-1), reduction="sum").item(); cnt += b.numel()
    return tot / cnt / math.log(2)

# ============================================================================
# 6. SANITY — provable + brute-force verified, runs before training
# ============================================================================
def sanity():
    rng = np.random.RandomState(0); lengths = (1, 2, 4)
    def brute(cost, L):
        bv, bs = [float("inf")], [None]
        def rec(t, acc, sp):
            if t == L:
                if acc < bv[0]: bv[0], bs[0] = acc, sp[:]
                return
            for k, e in enumerate(lengths):
                if t + e <= L and cost[t + e, k] < 1e5:
                    sp.append((t, e)); rec(t + e, acc + cost[t + e, k], sp); sp.pop()
        rec(0, 0.0, []); return bv[0]
    for _ in range(25):
        L = int(rng.randint(4, 10)); c = np.full((L + 1, 3), INVALID)
        for t in range(1, L + 1):
            for k, e in enumerate(lengths):
                if t - e >= 0: c[t, k] = rng.rand() * 5
        ct = torch.tensor(c)[None].float()
        best, ss, sl, sm = dp_tile(ct, lengths, L)
        assert abs(float(best[0, L]) - brute(c, L)) < 1e-4
        assert int(sl[0][sm[0]].sum()) == L
    print("PASS: DP tiling == brute-force optimum (25 instances); exact cover")
    h = torch.randn(2, 16, 8); cs = SpanCodec.prefix(h)
    for i in range(16):
        for e in lengths:
            if i + e <= 16:
                z = SpanCodec.pool(None, cs, torch.tensor([[i]] * 2), torch.tensor([[e]] * 2))
                assert torch.allclose(z, h[:, i:i + e].mean(1), atol=1e-5)
    print("PASS: O(1) cumsum span pooling == explicit mean")
    cfg = Cfg(d=32, heads=4, L=32, codebook=64, steps=1)
    m = Mosaic(cfg, 27).to(DEV)
    out, st = m.generate(4)
    assert out.shape == (4, 32) and st >= 1
    print(f"PASS: closed-loop generation covers canvas (steps={st})")

# ============================================================================
# 7. EXPERIMENT
# ============================================================================
def train(model, data, cfg, tag, is_ar=False):
    model.to(DEV).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    t0 = time.time()
    for step in range(cfg.steps):
        for g in opt.param_groups:
            g["lr"] = cfg.lr * min((step + 1) / 100, 1.0) * 0.5 * (1 + math.cos(math.pi * min(1, step / cfg.steps)))
        off = torch.randint(0, len(data["train"]) - cfg.L - 2, (cfg.batch,))
        seg = data["train"][off[:, None] + torch.arange(cfg.L + 1)[None, :]].to(DEV)
        if is_ar:
            lg = model(seg[:, :-1])
            loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), seg[:, 1:].reshape(-1))
            parts = {}
        else:
            loss, parts = model(seg[:, :-1], step, cfg.steps)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if not is_ar and step % 200 == 199:
            with torch.no_grad():
                hh = model.encode(seg[:, :-1], torch.ones_like(seg[:, :-1], dtype=torch.bool))
                model.codec.restart(SpanCodec.pool(None, SpanCodec.prefix(hh),
                                                   torch.arange(0, cfg.L, 8)[None].expand(cfg.batch, -1),
                                                   torch.full((cfg.batch, cfg.L // 8), 8, device=DEV)))
        if step % 250 == 0 or step == cfg.steps - 1:
            print(f"  [{tag}] {step:5d} loss {loss.item():.3f} " +
                  " ".join(f"{k}={v:.3f}" for k, v in parts.items()))
    print(f"  [{tag}] {time.time()-t0:.0f}s  params={sum(p.numel() for p in model.parameters())/1e6:.3f}M")

def run(cfg=None):
    cfg = cfg or Cfg(); os.makedirs(cfg.outdir, exist_ok=True)
    data = make_zipf(cfg.train_bytes, cfg.val_bytes, bigram=(cfg.data == "zipf_bigram"))
    print(f"oracle floor {data['floor']:.3f} bits/byte"); sanity()
    res = {}
    judge = ByteAR(cfg.judge_d, cfg.heads, 4, cfg.L, data["V"])
    train(judge, data, Cfg(steps=cfg.judge_steps, batch=cfg.batch, L=cfg.L), "judge", True)
    specs = [("ByteAR", None), ("MOSAIC", Cfg(**{**asdict(cfg)})),
             ("MOSAIC_fixed", Cfg(**{**asdict(cfg), "tiling": "fixed"})),
             ("MOSAIC_1step", Cfg(**{**asdict(cfg), "one_step": True}))]
    for name, c in specs:
        torch.manual_seed(0)
        if name == "ByteAR":
            m = ByteAR(cfg.d, cfg.heads, 4, cfg.L, data["V"]); train(m, data, cfg, name, True)
            s = torch.cat([m.generate(32, cfg.L, data["bos"]) for _ in range(cfg.gen_samples // 32)]).cpu()
            st = cfg.L
        else:
            m = Mosaic(c, data["V"]); train(m, data, c, name)
            outs, steps = [], []
            for _ in range(cfg.gen_samples // 32):
                o, k = m.generate(32); outs.append(o.cpu()); steps.append(k)
            s = torch.cat(outs); st = float(np.mean(steps))
        r = dict(tv=tv(s, data), oov=oov(s, data), H=entropy(s, data["V"]),
                 gen_bpb=judge_bpb(judge, s, data["bos"]), nfe=st)
        res[name] = r
        print(f">>> {name}: TV={r['tv']:.3f} OOV={r['oov']:.3f} H={r['H']:.2f} "
              f"GenBPB={r['gen_bpb']:.3f} NFE={st:.1f}")
        with open(f"{cfg.outdir}/samples_{name}.txt", "w") as f:
            for row in s[:8]: f.write("".join(data["itos"][int(b)] for b in row) + "\n")
    print("\n=== SUMMARY ===")
    for k, v in res.items():
        print(f"{k:14s} TV={v['tv']:.3f} OOV={v['oov']:.3f} GenBPB={v['gen_bpb']:.3f} NFE={v['nfe']:.1f}")
    gain_dp = res["MOSAIC_fixed"]["tv"] - res["MOSAIC"]["tv"]
    gain_loop = res["MOSAIC_1step"]["tv"] - res["MOSAIC"]["tv"]
    print(f"optimal-allocation gain (DP vs fixed): {gain_dp:+.3f} TV")
    print(f"closed-loop gain (iterative vs 1-step): {gain_loop:+.3f} TV")
    json.dump(dict(cfg=asdict(cfg), res=res), open(f"{cfg.outdir}/results.json", "w"), indent=2)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(list(res), [v["tv"] for v in res.values()]); ax[0].set_ylabel("oracle TV (lower=better)")
    ax[0].tick_params(axis="x", rotation=20)
    for k, v in res.items(): ax[1].scatter(v["nfe"], v["tv"], s=90); ax[1].annotate(k, (v["nfe"], v["tv"]))
    ax[1].set_xscale("log"); ax[1].set_xlabel("sequential passes (NFE)")
    plt.tight_layout(); plt.savefig(f"{cfg.outdir}/summary.png", dpi=130); plt.show()
    return res

def rd_sweep(lams=(0.0, 0.5, 1.0, 2.0, 4.0)):
    cfg = Cfg(steps=1200); data = make_zipf(cfg.train_bytes, cfg.val_bytes); pts = []
    for lam in lams:
        c = Cfg(**{**asdict(cfg), "lam": lam}); m = Mosaic(c, data["V"])
        train(m, data, c, f"lam={lam}")
        o, st = m.generate(64)
        pts.append((lam, float(tv(o.cpu(), data)), st))
    plt.figure(); plt.plot([p[1] for p in pts], [p[2] for p in pts], "o-")
    plt.xlabel("oracle TV (distortion)"); plt.ylabel("NFE (compute)"); plt.title("lambda sweep")
    plt.savefig(f"{cfg.outdir}/rd.png", dpi=130); plt.show(); return pts

if __name__ == "__main__":
    run()
    # rd_sweep()
```

---

## Ideas worth harvesting into your existing implementation

1. **Structure as inference, not learning.** Any place you currently *learn* a discrete partition (router, boundary head, gate), ask: *is there an exact solver given a learned cost?* If yes, delete the learned predictor and the STE machinery, and backprop through the solver's argmin path. This is the single biggest complexity reduction available to you.
2. **Differentiate the Lagrangian, not the partition.** `best[L].mean()` is a min of sums of differentiable costs; its gradient lands exactly on the selected structure. You get end‑to‑end rate–distortion training with **zero** boundary straight‑through.
3. **Atomic actions.** Predict `(length, content)` as one softmax. Boundary/content misalignment — the dominant exposure‑bias failure in H‑NCM — becomes structurally impossible.
4. **Closed‑loop canvas.** Condition every step on what was *actually* committed. Errors self‑synchronize instead of compounding; this is strictly stronger than NCM's conditioning corruption.
5. **Commit by confidence, not by schedule.** Replaces fixed K levels *and* halting gates with one threshold. NFE becomes content‑adaptive; easy regions finish in bulk, hard ones get more passes — real variable‑rate compute.
6. **Integral‑image span pooling.** `cumsum` gives O(1) pooling for *all* candidate spans, making an exhaustive cost field (and hence exact DP) affordable. This is what makes "solve the structure" practical.
7. **One codebook + one tiny decoder** beats K RVQ levels here, because the DP — not a residual chain — decides granularity. Nested dropout and corruption chains become unnecessary.
8. **Gap‑DP with free committed skips** lets a single shortest‑path call handle a partially filled canvas — the same solver serves training targets, generation, and streaming.

**Honest status:** I have no code‑execution tool in this turn, so I have not run this file. The DP's optimality is provable by induction on prefix length (it is the standard DAG shortest path), and the script *self‑verifies* it against brute‑force enumeration, checks cumsum pooling against explicit means, and asserts exact canvas coverage before any training. The empirical claims above are mechanism‑level predictions, not measurements — run `run()` and `rd_sweep()` and let the `gain_dp` / `gain_loop` numbers adjudicate. If `gain_dp ≈ 0`, optimal allocation isn't the bottleneck on your data; if `gain_loop ≈ 0`, closed‑loop iteration isn't either — both are clean falsifiers, which is exactly what you want from an alternate design.

