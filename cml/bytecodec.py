"""M4 byte baseline: a trained 5M byte trunk and a static arithmetic codec.

The adaptive byte arm is a codec, not the equal-bytes baseline.  This module
trains a byte `Trunk` on the frozen train split, then codes a held-out slice
with its static CDF.  The pointer arm is compared at the same parameter count.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import codec
from cml import KV, Trunk, block_step, new_cache, train_loop, truncate_cache, windows

BYTE = 256


@dataclass(frozen=True, slots=True)
class ByteConfig:
    d: int = 256
    layers: int = 6
    heads: int = 8
    seq: int = 256
    steps: int = 3000
    batch: int = 32
    seed: int = 0


def train_byte_model(
    train: bytes, cfg: ByteConfig, device: str = "cuda"
) -> tuple[Trunk, float]:
    torch.manual_seed(cfg.seed)
    ids = torch.frombuffer(bytearray(train), dtype=torch.uint8).long()
    win = windows(ids, cfg.seq)
    model = Trunk(BYTE, cfg.d, cfg.layers, cfg.heads, cfg.seq).to(device)
    secs = train_loop(model, win, cfg.steps, cfg.batch, 3e-4)
    return model, secs


class StaticByteCodec:
    """Encode/decode with a trained byte trunk; both sides step identically."""

    def __init__(self, model: Trunk, half: bool = True):
        self.model = model.eval()
        if half:
            self.model = self.model.half()
        self.seq = model.pos.num_embeddings

    def _cache(self) -> list[KV]:
        return new_cache(self.model)

    def _feed(self, cache: list[KV], sym: int) -> torch.Tensor:
        if cache[0].k.size(2) >= self.seq:
            truncate_cache(cache, self.seq - 1)
        n = cache[0].k.size(2)
        pos = min(n, self.seq - 1)
        x = self.model.tok(torch.tensor([[sym]], device="cuda"))
        h = x + self.model.pos.weight[pos]
        for block, kv in zip(self.model.blocks, cache):
            h = block_step(block, h, kv)
        return self.model.ln(h)

    @torch.no_grad()
    def _probs(self, h: torch.Tensor) -> np.ndarray:
        logits = F.linear(h, self.model.tok.weight).reshape(-1)
        return codec.quant(torch.softmax(logits, -1).cpu().numpy())

    def encode(self, data: bytes) -> bytes:
        enc = codec.ArithEncoder()
        for a in range(0, len(data), self.seq):
            chunk = data[a : a + self.seq]
            cache = self._cache()
            h = None
            for i, sym in enumerate(chunk):
                if h is None:
                    enc.encode(sym, sym + 1, BYTE)
                else:
                    q = self._probs(h)
                    enc.encode(
                        int(q[:sym].sum()), int(q[: sym + 1].sum()), int(q.sum())
                    )
                if i + 1 < len(chunk):
                    h = self._feed(cache, sym)
        return enc.finish()

    def decode(self, blob: bytes, n: int, limit: int | None = None) -> bytes:
        dec = codec.ArithDecoder(blob)
        out = bytearray()
        i = 0
        count = 0
        cache: list[KV] | None = None
        h = None
        while len(out) < n and (limit is None or count < limit):
            if i == 0:
                cache = self._cache()
                h = None
            if h is None:
                sym = dec.target(BYTE)
                dec.update(sym, sym + 1, BYTE)
            else:
                q = self._probs(h)
                c = np.concatenate(([0], np.cumsum(q)))
                target = dec.target(int(c[-1]))
                sym = int(np.searchsorted(c, target, side="right")) - 1
                dec.update(int(c[sym]), int(c[sym + 1]), int(c[-1]))
            out.append(sym)
            count += 1
            i = (i + 1) % self.seq
            if limit is not None and count >= limit:
                break
            h = self._feed(cache, sym)
        return bytes(out)


def save_model(path: Path, model: Trunk, cfg: ByteConfig, secs: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"cfg": cfg, "state": model.state_dict(), "secs": secs},
        path,
    )


def load_model(path: Path, device: str = "cuda") -> Trunk:
    data = torch.load(path, map_location=device, weights_only=False)
    cfg = data["cfg"]
    model = Trunk(BYTE, cfg.d, cfg.layers, cfg.heads, cfg.seq).to(device)
    model.load_state_dict(data["state"])
    return model


class StreamCoder:
    """One byte at a time from the trained trunk, keeping the full context.

    The static arm resets every `seq` bytes; this arm carries the cache, so
    a literal is coded against the same contiguous stream the trunk trained
    on.  It is the literal coder for the copy arm.
    """

    def __init__(self, model: Trunk, half: bool = True) -> None:
        self.model = model.eval()
        if half:
            self.model = self.model.half()
        self.seq = model.pos.num_embeddings
        self.cache = new_cache(self.model)
        self.h: torch.Tensor | None = None
        self.n = 0

    @torch.no_grad()
    def advance(self, sym: int) -> None:
        pos = self.n % self.seq
        if pos == 0:
            self.cache = new_cache(self.model)
        x = self.model.tok(torch.tensor([[sym]], device="cuda"))
        h = x + self.model.pos.weight[pos]
        for block, kv in zip(self.model.blocks, self.cache):
            h = block_step(block, h, kv)
        self.h = self.model.ln(h)
        self.n += 1

    @torch.no_grad()
    def _probs(self) -> np.ndarray:
        logits = F.linear(self.h, self.model.tok.weight).reshape(-1)
        return codec.quant(torch.softmax(logits, -1).cpu().numpy())

    def encode_one(self, sym: int, enc) -> None:
        if self.h is None:
            enc.encode(sym, sym + 1, BYTE)
        else:
            q = self._probs()
            enc.encode(int(q[:sym].sum()), int(q[: sym + 1].sum()), int(q.sum()))
        self.advance(sym)

    def decode_one(self, dec) -> int:
        if self.h is None:
            sym = dec.target(BYTE)
            dec.update(sym, sym + 1, BYTE)
        else:
            q = self._probs()
            c = np.concatenate(([0], np.cumsum(q)))
            sym = int(np.searchsorted(c, dec.target(int(c[-1])), side="right")) - 1
            dec.update(int(c[sym]), int(c[sym + 1]), int(c[-1]))
        self.advance(sym)
        return sym

    def encode(self, data: bytes) -> bytes:
        enc = codec.ArithEncoder()
        for sym in data:
            self.encode_one(sym, enc)
        return enc.finish()

    def decode(self, blob: bytes, n: int) -> bytes:
        dec = codec.ArithDecoder(blob)
        return bytes(self.decode_one(dec) for _ in range(n))


class CopyCodec:
    """LZ77 matches plus the trained trunk for the literals; exact round-trip.

    Matches cost a token bit, a distance slot, the slot's raw bits and a
    length symbol.  Literals and every copied byte advance the same trunk
    cache, so the decoder reconstructs the trunk context it needs.  This is
    the Stage 3 copy arm: the parsed bytes are the model's own context.
    """

    def __init__(
        self,
        literals: StreamCoder,
        window: int = 65536,
        min_match: int = 8,
        cap: int = 256,
    ) -> None:
        self.literals = literals
        self.window = window
        self.min_match = min_match
        self.cap = cap
        slots = max((window - 1).bit_length(), 1)
        self.tok = codec.Counts(2, init=np.ones(2, dtype=np.int64))
        self.slot = codec.Counts(slots + 1, init=np.ones(slots + 1, dtype=np.int64))
        self.length = codec.Counts(
            cap - min_match + 1, init=np.ones(cap - min_match + 1, dtype=np.int64)
        )

    def encode(self, data: bytes, tokens: list[tuple[int, int]] | None = None) -> bytes:
        if tokens is None:
            tokens = codec.lz_parse(data, self.window, self.min_match, self.cap)
        enc = codec.ArithEncoder()
        lit = StreamCoder(self.literals.model)
        i = 0
        for length, value in tokens:
            if length == 0:
                codec.emit([self.tok], 0, 2, enc)
                lit.encode_one(data[i], enc)
                i += 1
                continue
            codec.emit([self.tok], 1, 2, enc)
            v = value - 1
            slot = v.bit_length()
            codec.emit([self.slot], slot, self.slot.n, enc)
            for shift in range(slot - 2, -1, -1):
                bit = (v >> shift) & 1
                enc.encode(bit, bit + 1, 2)
            codec.emit([self.length], length - self.min_match, self.length.n, enc)
            for sym in data[i : i + length]:
                lit.advance(sym)
            i += length
        return enc.finish()

    def decode(self, blob: bytes, n: int) -> bytes:
        dec = codec.ArithDecoder(blob)
        lit = StreamCoder(self.literals.model)
        out = bytearray()
        while len(out) < n:
            if codec.take([self.tok], 2, dec) == 0:
                out.append(lit.decode_one(dec))
                continue
            slot = codec.take([self.slot], self.slot.n, dec)
            v = 1 if slot else 0
            for _ in range(slot - 1):
                bit = dec.target(2)
                dec.update(bit, bit + 1, 2)
                v = (v << 1) | bit
            offset = v + 1
            length = codec.take([self.length], self.length.n, dec) + self.min_match
            for _ in range(length):
                sym = out[-offset]
                out.append(sym)
                lit.advance(sym)
            if len(out) > n:
                raise ValueError("copy codec overshot the frame")
        return bytes(out)


class FastByteCodec:
    """A size instrument: one forward per 256-byte chunk.

    A chunk's `255` logits predict its bytes `1..255` under the causal mask,
    so the payload length matches the sequential arm on the tested slices.
    Rounding at `quant` can move individual counts, so the bits are not the
    exact stream: `StaticByteCodec` is the codec, and it proves round-trip
    on a slice.  Use this arm only to price a full stream fast.
    """

    def __init__(self, model: Trunk, half: bool = True) -> None:
        self.model = model.eval()
        if half:
            self.model = self.model.half()
        self.seq = model.pos.num_embeddings

    @torch.no_grad()
    def _cums(self, chunk: bytes) -> np.ndarray:
        n = len(chunk) - 1
        x = torch.tensor([list(chunk[:n])], dtype=torch.long, device="cuda")
        logits = self.model(x)[0].float()
        q = codec.quant(torch.softmax(logits, -1).cpu().numpy()).reshape(n, -1)
        return np.cumsum(q, axis=1)

    def encode(self, data: bytes) -> bytes:
        enc = codec.ArithEncoder()
        for base in range(0, len(data), self.seq):
            chunk = data[base : base + self.seq]
            sym = chunk[0]
            enc.encode(sym, sym + 1, BYTE)
            if len(chunk) == 1:
                continue
            cums = self._cums(chunk)
            for j in range(1, len(chunk)):
                sym = chunk[j]
                row = cums[j - 1]
                lo = int(row[sym - 1]) if sym else 0
                enc.encode(lo, int(row[sym]), int(row[-1]))
        return enc.finish()


class FastCopyCodec:
    """The copy arm priced fast: one forward per output chunk, no decode.

    The parse and the output bytes are both known at encode time, so the
    literal counts come from `FastByteCodec` chunk by chunk.  It is a size
    instrument; `CopyCodec` is the exact codec that proves round-trip.
    """

    def __init__(
        self,
        literals: FastByteCodec,
        window: int = 65536,
        min_match: int = 8,
        cap: int = 256,
    ) -> None:
        self.literals = literals
        self.seq = literals.seq
        self.window = window
        self.min_match = min_match
        self.cap = cap
        slots = max((window - 1).bit_length(), 1)
        self.tok = codec.Counts(2, init=np.ones(2, dtype=np.int64))
        self.slot = codec.Counts(slots + 1, init=np.ones(slots + 1, dtype=np.int64))
        self.length = codec.Counts(
            cap - min_match + 1, init=np.ones(cap - min_match + 1, dtype=np.int64)
        )

    def encode(self, data: bytes, tokens: list[tuple[int, int]] | None = None) -> bytes:
        if tokens is None:
            tokens = codec.lz_parse(data, self.window, self.min_match, self.cap)
        enc = codec.ArithEncoder()
        i = 0
        base = -1
        cums = None
        for length, value in tokens:
            if length == 0:
                j = i % self.seq
                if i - j != base:
                    base = i - j
                    cums = self.literals._cums(data[base : base + self.seq])
                sym = value
                if j == 0:
                    enc.encode(sym, sym + 1, BYTE)
                else:
                    row = cums[j - 1]
                    lo = int(row[sym - 1]) if sym else 0
                    enc.encode(lo, int(row[sym]), int(row[-1]))
                i += 1
                continue
            codec.emit([self.tok], 1, 2, enc)
            v = value - 1
            slot = v.bit_length()
            codec.emit([self.slot], slot, self.slot.n, enc)
            for shift in range(slot - 2, -1, -1):
                bit = (v >> shift) & 1
                enc.encode(bit, bit + 1, 2)
            codec.emit([self.length], length - self.min_match, self.length.n, enc)
            i += length
        return enc.finish()
