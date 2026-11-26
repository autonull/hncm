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
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import codec
from cml import Block, KV, block_step

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


def span_inputs(
    spans: list[bytes], device: str, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad spans once; the pack is static, so this is not rebuilt per step."""
    width = max((len(s) for s in spans), default=1)
    x = torch.zeros((len(spans), width), dtype=torch.long, device=device)
    mask = torch.zeros((len(spans), width), dtype=dtype, device=device)
    for i, span in enumerate(spans):
        x[i, : len(span)] = torch.frombuffer(bytearray(span), dtype=torch.uint8)
        mask[i, : len(span)] = 1
    return x, mask


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

    def embed(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        e = self.emb(x) + self.pos.weight[: x.size(1)]
        e = e * mask[..., None]
        z = e.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.fc2(F.gelu(self.fc1(z)))

    def forward(self, spans: list[bytes], device: str) -> torch.Tensor:
        x, mask = span_inputs(spans, device, next(self.parameters()).dtype)
        return self.embed(x, mask)


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
        self._key: tuple | None = None
        self._spans: tuple[torch.Tensor, torch.Tensor] | None = None

    def embeddings(self, units: tuple[bytes, ...], device: str) -> torch.Tensor:
        dtype = next(self.encoder.parameters()).dtype
        key = (units, device, dtype)
        if key != self._key:
            spans = list(units) + [bytes([b]) for b in range(BYTE)]
            self._spans = span_inputs(spans, device, dtype)
            self._key = key
        assert self._spans is not None
        return self.encoder.embed(*self._spans)

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


def _cache(model: PointerLM, device: str, batch: int = 1) -> list[KV]:
    heads = model.blocks[0].heads
    d = model.cfg.d
    shape = (batch, heads, 0, d // heads)
    return [
        KV(torch.empty(shape, device=device), torch.empty(shape, device=device))
        for _ in model.blocks
    ]


class PointerCodec:
    """Encode/decode with the trained pointer; both sides step identically.

    Windows are grouped in batches of `batch`: each group is one segment per
    window, and both sides run the same batched step at each position, so the
    coded floats agree exactly.  The payload carries one (units, bytes) pair
    per window segment; `shared` charges it with the header.
    """

    def __init__(
        self,
        model: PointerLM,
        lex: codec.Lexicon,
        device: str = "cuda",
        adaptive_bytes: bool = False,
        half: bool = False,
        batch: int = 32,
    ):
        self.model = model.eval()
        if half:
            self.model = self.model.half()
        self.lex = lex
        self.k = len(lex.units)
        self.seq = model.cfg.seq
        self.device = device
        self.adaptive = adaptive_bytes
        self.batch = batch
        self.emb = model.embeddings(lex.units, device)
        self.first = _uniform(self.k + 1)
        self.bytes = _uniform(BYTE)

    @torch.no_grad()
    def _step(
        self, cache: list[KV], prev: torch.Tensor, pos: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Feed one element per row; return quantised unit, byte and gate CDFs."""
        h = self.emb[prev] + self.model.pos.weight[min(pos, self.seq - 1)]
        for block, kv in zip(self.model.blocks, cache):
            h = block_step(block, h, kv)
        h = self.model.ln(h)
        ptr = (self.model.wq(h) @ self.emb[: self.k].T)[:, 0]
        gate = torch.sigmoid(self.model.gate(h))[:, 0, 0]
        p = torch.softmax(ptr, -1) * gate[:, None]
        q = torch.cat((p, (1 - gate)[:, None]), 1)
        q = torch.clamp(torch.round(q.float() * codec.QUANT), min=1).to(torch.int64)
        bq = torch.softmax(self.model.byte(h)[:, 0], -1)
        bq = torch.clamp(torch.round(bq.float() * codec.QUANT), min=1).to(torch.int64)
        return (
            q.cpu().numpy(),
            bq.cpu().numpy(),
            gate.cpu().numpy(),
        )

    @staticmethod
    def _cums(q: np.ndarray) -> np.ndarray:
        return np.concatenate(
            (np.zeros((q.shape[0], 1), dtype=np.int64), q.cumsum(1)), 1
        )

    def _code_byte(
        self,
        enc: codec.ArithEncoder,
        b: int,
        cdf: np.ndarray,
        arm: codec.ByteArm | None,
    ) -> None:
        if arm is None:
            enc.encode(int(cdf[b]), int(cdf[b + 1]), int(cdf[-1]))
        else:
            arm.encode_one(b, enc)

    def _take_byte(
        self, dec: codec.ArithDecoder, cdf: np.ndarray, arm: codec.ByteArm | None
    ) -> int:
        if arm is not None:
            return arm.decode_one(dec)
        target = dec.target(int(cdf[-1]))
        b = int(np.searchsorted(cdf, target, side="right")) - 1
        dec.update(int(cdf[b]), int(cdf[b + 1]), int(cdf[-1]))
        return b

    @torch.no_grad()
    def _encode_group(self, chunks: list[np.ndarray]) -> list[bytes]:
        b = len(chunks)
        lens = [len(c) for c in chunks]
        width = max(lens)
        ids = np.full((b, width), self.k, dtype=np.int64)
        for r, c in enumerate(chunks):
            ids[r, : len(c)] = c
        encs = [codec.ArithEncoder() for _ in range(b)]
        arms = [codec.ByteArm() for _ in range(b)] if self.adaptive else [None] * b
        for r in range(b):
            elem = int(ids[r, 0])
            sym = elem if elem < self.k else self.k
            c = self.first
            encs[r].encode(int(c[sym]), int(c[sym + 1]), int(c[-1]))
            if elem >= self.k:
                self._code_byte(encs[r], elem - self.k, self.bytes, arms[r])
        cache = _cache(self.model, self.device, b)
        for i in range(width - 1):
            prev = torch.as_tensor(ids[:, i], device=self.device).view(b, 1)
            q, bq, _ = self._step(cache, prev, i)
            cums = self._cums(q)
            bq = self._cums(bq)
            for r in range(b):
                if i + 1 >= lens[r]:
                    continue
                elem = int(ids[r, i + 1])
                sym = elem if elem < self.k else self.k
                encs[r].encode(
                    int(cums[r, sym]), int(cums[r, sym + 1]), int(cums[r, -1])
                )
                if elem >= self.k:
                    self._code_byte(encs[r], elem - self.k, bq[r], arms[r])
        return [e.finish() for e in encs]

    @torch.no_grad()
    def _decode_group(
        self, lens: list[int], segs: list[bytes]
    ) -> list[tuple[bytes, int, list[tuple[int, int, int]]]]:
        b = len(lens)
        width = max(lens)
        decs = [codec.ArithDecoder(s) for s in segs]
        arms = [codec.ByteArm() for _ in range(b)] if self.adaptive else [None] * b
        outs = [bytearray() for _ in range(b)]
        spans: list[list[tuple[int, int, int]]] = [[] for _ in range(b)]
        cache = _cache(self.model, self.device, b)
        prev = np.zeros(b, dtype=np.int64)
        for r in range(b):
            c = self.first
            target = decs[r].target(int(c[-1]))
            sym = int(np.searchsorted(c, target, side="right")) - 1
            decs[r].update(int(c[sym]), int(c[sym + 1]), int(c[-1]))
            start = len(outs[r])
            if sym < self.k:
                outs[r] += self.lex.units[sym]
                elem = sym
            else:
                byte = self._take_byte(decs[r], self.bytes, arms[r])
                outs[r].append(byte)
                elem = self.k + byte
            prev[r] = elem
            spans[r].append((start, len(outs[r]), elem))
        for i in range(width - 1):
            t = torch.as_tensor(prev, device=self.device).view(b, 1)
            q, bq, _ = self._step(cache, t, i)
            cums = self._cums(q)
            bq = self._cums(bq)
            for r in range(b):
                if i + 1 >= lens[r]:
                    continue
                total = int(cums[r, -1])
                target = decs[r].target(total)
                sym = int(np.searchsorted(cums[r], target, side="right")) - 1
                decs[r].update(int(cums[r, sym]), int(cums[r, sym + 1]), total)
                start = len(outs[r])
                if sym < self.k:
                    outs[r] += self.lex.units[sym]
                    elem = sym
                else:
                    byte = self._take_byte(decs[r], bq[r], arms[r])
                    outs[r].append(byte)
                    elem = self.k + byte
                prev[r] = elem
                spans[r].append((start, len(outs[r]), elem))
        return [(bytes(o), n, s) for o, n, s in zip(outs, lens, spans)]

    def encode(self, data: bytes) -> bytes:
        els = elements(data, self.lex)
        chunks = [els[a : a + self.seq] for a in range(0, len(els), self.seq)]
        segs = []
        for g in range(0, len(chunks), self.batch):
            segs.extend(self._encode_group(chunks[g : g + self.batch]))
        out = bytearray(struct.pack("<I", len(chunks)))
        for chunk, seg in zip(chunks, segs):
            if len(chunk) > 1 << 8 or len(seg) >= 1 << 16:
                raise ValueError("pointer: window frame too large")
            out += struct.pack("<BH", len(chunk) - 1, len(seg))
        for seg in segs:
            out += seg
        return bytes(out)

    def decode(
        self, blob: bytes, nbytes: int, limit: int | None = None, trace: bool = False
    ) -> bytes | tuple[bytes, list[tuple[int, int, int]]]:
        count = struct.unpack_from("<I", blob, 0)[0]
        lens = []
        pos = 4
        for _ in range(count):
            n, ell = struct.unpack_from("<BH", blob, pos)
            pos += 3
            lens.append((n + 1, ell))
        segs = []
        for _, ell in lens:
            segs.append(blob[pos : pos + ell])
            pos += ell
        out = bytearray()
        spans: list[tuple[int, int, int]] = []
        units = 0
        for g in range(0, count, self.batch):
            group = lens[g : g + self.batch]
            got = self._decode_group([n for n, _ in group], segs[g : g + self.batch])
            for chunk, n, chunk_spans in got:
                off = len(out)
                out += chunk
                units += n
                if trace:
                    spans.extend((a + off, b + off, s) for a, b, s in chunk_spans)
            if limit is not None and units >= limit:
                break
        if len(out) > nbytes:
            raise ValueError("pointer: overshot the frame length")
        if trace:
            return bytes(out), spans
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
