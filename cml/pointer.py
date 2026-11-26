"""M3: the pointer head.  Capacity lives in the installed lexicon, not the weights.

The model reads spans by content: a tiny byte encoder turns each installed unit
into `e_c`, and the pointer scores `(W_q h_t) · e_c`.  A new entry costs zero
parameters; install is a forward pass over its bytes.  The coder's alphabet is
the active pack plus one escape; a literal byte is coded by the byte head.

Training windows are 256 elements.  Encoding and decoding both use the same
incremental `block_step` path, so the arithmetic coder sees identical floats on
both sides.  Window boundaries reset the cache; the first element of a window
has a uniform prior.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import codec
from cml import Block, KV, block_step, truncate_cache

BYTE = 256
SEQ = 256


@dataclass(frozen=True, slots=True)
class PointerConfig:
    d: int = 256
    layers: int = 6
    heads: int = 8
    seq: int = SEQ
    active: int = 256
    steps: int = 300
    batch: int = 32
    lr: float = 3e-4
    seed: int = 0


class SpanEncoder(nn.Module):
    """`e_c = Enc(bytes_c)`: a 2-layer byte encoder, pooled over the span."""

    def __init__(self, d: int = 256, w: int = 64):
        super().__init__()
        self.emb = nn.Embedding(BYTE, w)
        self.pos = nn.Embedding(16, w)
        self.fc1 = nn.Linear(w, w)
        self.fc2 = nn.Linear(w, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, spans: list[bytes], device: str) -> torch.Tensor:
        width = max((len(s) for s in spans), default=1)
        dtype = next(self.parameters()).dtype
        x = torch.zeros((len(spans), width), dtype=torch.long, device=device)
        mask = torch.zeros((len(spans), width), dtype=dtype, device=device)
        for i, span in enumerate(spans):
            x[i, : len(span)] = torch.frombuffer(bytearray(span), dtype=torch.uint8)
            mask[i, : len(span)] = 1
        e = self.emb(x) + self.pos.weight[:width]
        e = e * mask[..., None]
        z = e.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.fc2(F.gelu(self.fc1(z)))


class PointerLM(nn.Module):
    def __init__(self, cfg: PointerConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = SpanEncoder(cfg.d)
        self.pos = nn.Embedding(cfg.seq, cfg.d)
        self.blocks = nn.ModuleList(Block(cfg.d, cfg.heads) for _ in range(cfg.layers))
        self.ln = nn.LayerNorm(cfg.d)
        self.wq = nn.Linear(cfg.d, cfg.d, bias=False)
        self.gate = nn.Linear(cfg.d, 1)
        self.byte = nn.Linear(cfg.d, BYTE)
        nn.init.normal_(self.pos.weight, std=0.02)
        nn.init.normal_(self.wq.weight, std=0.02)

    def embeddings(self, units: tuple[bytes, ...], device: str) -> torch.Tensor:
        spans = list(units) + [bytes([b]) for b in range(BYTE)]
        return self.encoder(spans, device)

    def forward(self, ids: torch.Tensor, units: tuple[bytes, ...]) -> tuple:
        b, t = ids.shape
        e = self.embeddings(units, ids.device)
        x = e[ids] + self.pos.weight[:t]
        h = x
        for block in self.blocks:
            h = block(h)
        h = self.ln(h)
        ptr = self.wq(h) @ e[: len(units)].T
        return ptr, self.gate(h).squeeze(-1), self.byte(h)


def _windows(elements: np.ndarray, seq: int, batch: int, rng: np.random.Generator):
    starts = rng.integers(0, len(elements) - seq - 1, size=batch)
    x = np.stack([elements[i : i + seq] for i in starts])
    y = np.stack([elements[i + 1 : i + seq + 1] for i in starts])
    return torch.as_tensor(x), torch.as_tensor(y)


def _loss(model: PointerLM, x: torch.Tensor, y: torch.Tensor, units: tuple[bytes, ...]):
    ptr, gate, byte = model(x, units)
    k = len(units)
    span = y < k
    loss = F.binary_cross_entropy_with_logits(gate, span.float())
    if bool(span.any()):
        loss = loss + F.cross_entropy(ptr[span], y[span])
    lit = ~span
    if bool(lit.any()):
        loss = loss + F.cross_entropy(byte[lit], y[lit] - k)
    return loss


def train_pointer(
    elements: np.ndarray,
    units: tuple[bytes, ...],
    cfg: PointerConfig,
    device: str = "cuda",
) -> tuple[PointerLM, list[float]]:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    model = PointerLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.1)
    rng = np.random.default_rng(cfg.seed)
    losses: list[float] = []
    t0 = time.perf_counter()
    for step in range(cfg.steps):
        x, y = _windows(elements, cfg.seq, cfg.batch, rng)
        loss = _loss(model, x.to(device), y.to(device), units)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if step % 50 == 0 or step == cfg.steps - 1:
            print(
                f"  pointer step {step:4d} loss {float(loss.detach()):.3f} "
                f"({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )
    return model, losses


def pack_streams(
    train: bytes,
    full_units: tuple[bytes, ...],
    active: int = 256,
    packs: int = 4,
    min_count: int = 16,
    cap: int = 16,
) -> list[tuple[np.ndarray, tuple[bytes, ...]]]:
    """The full pack plus `packs` disjoint chunk packs; each is one parse."""
    full = codec.Lexicon(full_units[:active])
    streams = [(elements(train, full), full.units)]
    bounds = np.linspace(0, len(train), packs + 1).astype(int)
    for i in range(packs):
        chunk = train[bounds[i] : bounds[i + 1]]
        lex = codec.build_lexicon(chunk, 8192, min_count, cap)
        units = lex.units[:active]
        streams.append((elements(chunk, codec.Lexicon(units)), units))
    return streams


def train_pointer_streams(
    streams: list[tuple[np.ndarray, tuple[bytes, ...]]],
    cfg: PointerConfig,
    device: str = "cuda",
    full_every: int = 2,
) -> tuple[PointerLM, list[float]]:
    """Rotate parses; the full pack keeps half the steps, chunks take the rest."""
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    model = PointerLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.1)
    rng = np.random.default_rng(cfg.seed)
    losses: list[float] = []
    t0 = time.perf_counter()
    n = len(streams)
    for step in range(cfg.steps):
        if n > 1 and full_every and step % full_every == 0:
            idx = 0
        elif full_every:
            idx = 1 + (step // full_every) % (n - 1)
        else:
            idx = step % n
        els, units = streams[idx]
        x, y = _windows(els, cfg.seq, cfg.batch, rng)
        loss = _loss(model, x.to(device), y.to(device), units)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if step % 50 == 0 or step == cfg.steps - 1:
            print(
                f"  pointer step {step:4d} pack {idx} loss "
                f"{float(loss.detach()):.3f} ({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )
    return model, losses


@torch.no_grad()
def eval_bpb(
    model: PointerLM, data: bytes, lex: codec.Lexicon, seq: int = 256
) -> dict[str, float]:
    """Teacher-forced payload bpb on one pack; header excluded."""
    els = elements(data, lex)
    k = len(lex.units)
    model.eval()
    total = 0.0
    nbytes = 0
    spans = 0
    n = 0
    for a in range(0, len(els), seq):
        chunk = els[a : a + seq]
        t = len(chunk) - 1
        if t <= 0:
            break
        ids = (
            torch.as_tensor(chunk[:-1]).view(1, -1).to(next(model.parameters()).device)
        )
        y = torch.as_tensor(chunk[1:]).to(ids.device)
        ptr, gate, byte = model(ids, lex.units)
        gate = gate.view(-1)
        ptr = ptr.view(t, -1)
        byte = byte.view(t, -1)
        span = y < k
        g = -torch.log2(torch.sigmoid(gate).clamp_min(1e-9))
        e = -torch.log2((1 - torch.sigmoid(gate)).clamp_min(1e-9))
        nll = torch.where(span, g, e)
        if bool(span.any()):
            nll[span] += -torch.log2(
                torch.softmax(ptr[span], -1)
                .gather(1, y[span, None])
                .squeeze(1)
                .clamp_min(1e-12)
            )
        if bool((~span).any()):
            nll[~span] += -torch.log2(
                torch.softmax(byte[~span], -1)
                .gather(1, (y[~span] - k)[:, None])
                .squeeze(1)
                .clamp_min(1e-12)
            )
        total += float(nll.sum())
        spans += int(span.sum())
        n += t
        for elem in y.tolist():
            nbytes += len(lex.units[elem]) if elem < k else 1
    return {
        "payload_bpb": total / max(nbytes, 1),
        "bits": total,
        "nbytes": nbytes,
        "span_frac": spans / max(n, 1),
    }


def elements(data: bytes, lex: codec.Lexicon) -> np.ndarray:
    parse, _ = codec.greedy_parse(data, lex)
    k = len(lex.units)
    return np.asarray([s if s >= 0 else k - 1 - s for s in parse], dtype=np.int64)


def _cdf(q: np.ndarray) -> np.ndarray:
    return np.concatenate(([0], np.cumsum(q)))


def _uniform(n: int) -> np.ndarray:
    return _cdf(np.ones(n, dtype=np.int64))


def _cache(model: PointerLM, device: str) -> list[KV]:
    heads = model.blocks[0].heads
    d = model.cfg.d
    shape = (1, heads, 0, d // heads)
    return [
        KV(torch.empty(shape, device=device), torch.empty(shape, device=device))
        for _ in model.blocks
    ]


class PointerCodec:
    """Encode/decode with the trained pointer; both sides step identically."""

    def __init__(
        self,
        model: PointerLM,
        lex: codec.Lexicon,
        device: str = "cuda",
        adaptive_bytes: bool = False,
        half: bool = False,
    ):
        self.model = model.eval()
        if half:
            self.model = self.model.half()
        self.lex = lex
        self.k = len(lex.units)
        self.seq = model.cfg.seq
        self.device = device
        self.adaptive = adaptive_bytes
        self.emb = model.embeddings(lex.units, device)
        self.first = _uniform(self.k + 1)
        self.bytes = _uniform(BYTE)

    @torch.no_grad()
    def _feed(self, cache: list[KV], elem: int) -> torch.Tensor:
        if cache[0].k.size(2) >= self.seq:
            truncate_cache(cache, self.seq - 1)
        n = cache[0].k.size(2)
        pos = min(n, self.seq - 1)
        h = self.emb[elem].view(1, 1, -1) + self.model.pos.weight[pos]
        for block, kv in zip(self.model.blocks, cache):
            h = block_step(block, h, kv)
        return self.model.ln(h)

    @torch.no_grad()
    def _probs(self, h: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        ptr = (self.model.wq(h) @ self.emb[: self.k].T).reshape(-1)
        gate = float(torch.sigmoid(self.model.gate(h)))
        p = torch.softmax(ptr, -1).cpu().numpy() * gate
        q = codec.quant(np.append(p, 1 - gate))
        bq = codec.quant(torch.softmax(self.model.byte(h), -1).cpu().numpy())
        return q, bq

    def encode(self, data: bytes) -> bytes:
        els = elements(data, self.lex)
        enc = codec.ArithEncoder()
        ba = codec.ByteArm()
        for a in range(0, len(els), self.seq):
            chunk = els[a : a + self.seq]
            cache = _cache(self.model, self.device)
            h = None
            for i, elem in enumerate(chunk):
                if h is None:
                    sym = elem if elem < self.k else self.k
                    c = self.first
                    enc.encode(int(c[sym]), int(c[sym + 1]), int(c[-1]))
                    cb = self.bytes
                else:
                    q, bq = self._probs(h)
                    c = _cdf(q)
                    sym = elem if elem < self.k else self.k
                    enc.encode(int(c[sym]), int(c[sym + 1]), int(c[-1]))
                    cb = _cdf(bq)
                if elem >= self.k:
                    b = elem - self.k
                    if self.adaptive:
                        ba.encode_one(b, enc)
                    else:
                        enc.encode(int(cb[b]), int(cb[b + 1]), int(cb[-1]))
                if i + 1 < len(chunk):
                    h = self._feed(cache, elem)
        return enc.finish()

    def decode(self, blob: bytes, nbytes: int, limit: int | None = None) -> bytes:
        dec = codec.ArithDecoder(blob)
        ba = codec.ByteArm()
        out = bytearray()
        i = 0
        count = 0
        cache: list[KV] | None = None
        h = None
        while len(out) < nbytes and (limit is None or count < limit):
            if i == 0:
                cache = _cache(self.model, self.device)
                h = None
            if h is None:
                c = self.first
                cb = self.bytes
            else:
                q, bq = self._probs(h)
                c = _cdf(q)
                cb = _cdf(bq)
            target = dec.target(int(c[-1]))
            sym = int(np.searchsorted(c, target, side="right")) - 1
            dec.update(int(c[sym]), int(c[sym + 1]), int(c[-1]))
            if sym < self.k:
                out += self.lex.units[sym]
                elem = sym
            else:
                if self.adaptive:
                    b = ba.decode_one(dec)
                else:
                    target = dec.target(int(cb[-1]))
                    b = int(np.searchsorted(cb, target, side="right")) - 1
                    dec.update(int(cb[b]), int(cb[b + 1]), int(cb[-1]))
                out.append(b)
                elem = self.k + b
            if len(out) > nbytes:
                raise ValueError("pointer: overshot the frame length")
            count += 1
            i = (i + 1) % self.seq
            if limit is not None and count >= limit:
                break
            h = self._feed(cache, elem)
        return bytes(out)


def save_model(
    path: Path, model: PointerLM, cfg: PointerConfig, losses: list[float]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": cfg, "state": model.state_dict(), "losses": losses}, path)


def load_model(path: Path, device: str = "cuda") -> PointerLM:
    data = torch.load(path, map_location=device, weights_only=False)
    model = PointerLM(data["cfg"]).to(device)
    model.load_state_dict(data["state"])
    return model
