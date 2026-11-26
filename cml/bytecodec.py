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
