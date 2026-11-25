"""NCM: the neural context model of the tail, trained end to end.

The H-Net tier gives an exact concept index; the NSCM tier codes the spans the
index does not cover. The count SVD (v7/v8) is the baseline. This module is the
learned replacement at the *same* parameter budget, conditioned on the H-Net
state: the previous concept's class (its length x its frequency bucket, which
the decoder derives from the dictionary), the span position, and the previous
byte. The GRU carries the within-span context (the trigram arm N2 asks for).

Training is on the fit's uncovered spans; the model is selected on the held-out
10% of the train, never the val; it is packed as float16, and every reported
bit is from the *packed* numpy forward (the same pass the codec's decoder
makes), so the charge is the model the artifact can use.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import codec as z
import concept_lm as c
import tail_coder as t

HID = 12
RANK = 6
PE = 12
CE = 12
POS = 8
SEED = 0
MAGIC = b"HNN8"
KEYS = (
    "pe",
    "ce",
    "pos",
    "wz",
    "uz",
    "bz",
    "wr",
    "ur",
    "br",
    "wn",
    "un",
    "bn",
    "w1",
    "w2",
    "b1",
    "b2",
)
ALPHA = 0.2


def class_of(tok: bytes, d, k: int) -> int:
    """The H-Net state: the previous concept's length x its frequency bucket."""
    if tok is None:
        return 64
    lb = min(len(tok), 16) - 1
    fb = min(int(d.cost_of(tok) / 5), 3)
    return lb * 4 + fb


@dataclass
class Tail:
    """The packed model: the embedding tables and the GRU, as float16 arrays."""

    w: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        self.size = len(pack(self))

    def logits(self, prev: int, cls: int, pos: int, h: np.ndarray):
        w = self.w
        x = np.concatenate([w["pe"][prev], w["ce"][cls], w["pos"][pos]])
        z = sigmoid(x @ w["wz"] + h @ w["uz"] + w["bz"])
        r = sigmoid(x @ w["wr"] + h @ w["ur"] + w["br"])
        n = np.tanh(x @ w["wn"] + (r * h) @ w["un"] + w["bn"])
        h = (1 - z) * n + z * h
        y = np.tanh(h @ w["w1"] + w["b1"])
        return y @ w["w2"] + w["b2"], h


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def logp_span(model: Tail, raw: bytes, cls: int) -> tuple[float, np.ndarray]:
    h = np.zeros(HID)
    bits = 0.0
    prev = 256
    for j, b in enumerate(raw):
        logits, h = model.logits(prev, cls, min(j, 15), h)
        logits = logits - logits.max()
        p = np.exp(logits)
        p /= p.sum()
        bits -= math.log2(max(p[b], 1e-30))
        prev = b
    return bits, h


def pack(model: Tail) -> bytes:
    out = bytearray(MAGIC)
    out += struct.pack("<B", len(KEYS))
    for k in KEYS:
        a = model.w[k]
        out += struct.pack("<HH", a.shape[0], a.shape[1] if a.ndim > 1 else 0)
        out += a.astype("<f2").tobytes()
    return bytes(out)


def unpack(data: bytes) -> Tail:
    if data[:4] != MAGIC:
        raise ValueError("not an NCM model")
    pos = 5
    w = {}
    for k in KEYS:
        a, b = struct.unpack_from("<HH", data, pos)
        pos += 4
        shape = (a,) if b == 0 else (a, b)
        n = int(np.prod(shape))
        w[k] = np.frombuffer(data, "<f2", n, pos).reshape(shape).astype(np.float64)
        pos += 2 * n
    return Tail(w)


# %%
def train(name: str, vocab: str, rect: dict, epochs: int = 2) -> dict:
    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s = c.split(name, vocab)
    d, bg, uni = c.rebuild(s, rect["dict"], rect["table"])
    fit_toks = c.viterbi(s["fit"], d, bg, uni)[0]
    val_toks, _ = c.viterbi(s["va"], d, bg, uni)
    pf, pv = t.pieces(fit_toks, {}), t.pieces(val_toks, {})
    if not pf or not pv:
        return {"name": name, "model": None, "bits": 0.0, "size": 0}
    # the H-Net state for a piece: the class of the token before it
    rank = {tok: i for i, tok in enumerate(s["top"][:256])}

    # t.pieces uses the rank of the previous multi token; recover the token
    inv = {i: tok for tok, i in rank.items()}
    pf = [t.Piece(p.raw, p.prev, 0, p.post) for p in pf]
    train_cls = [class_of(inv.get(p.prev), d, len(rank)) for p in pf]
    pv = [t.Piece(p.raw, p.prev, 0, p.post) for p in pv]
    val_cls = [class_of(inv.get(p.prev), d, len(rank)) for p in pv]

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pe = nn.Embedding(257, PE)
            self.ce = nn.Embedding(65, CE)
            self.pos = nn.Embedding(16, POS)
            self.gru = nn.GRU(PE + CE + POS, HID, batch_first=True, bias=False)
            self.w1 = nn.Linear(HID, RANK)
            self.w2 = nn.Linear(RANK, 256)

        def forward(self, prev, cls, pos, h):
            x = torch.cat([self.pe(prev), self.ce(cls), self.pos(pos)], -1)
            y, h = self.gru(x, h)
            return self.w2(torch.tanh(self.w1(y))), h

    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    lossf = nn.CrossEntropyLoss()

    def batches(pf_, cls_, bs=128):
        for i in range(0, len(pf_), bs):
            group = pf_[i : i + bs]
            L = max(len(p.raw) for p in group)
            raw = torch.zeros(len(group), L, dtype=torch.long)
            prev = torch.full((len(group), L), 256, dtype=torch.long)
            pos = torch.zeros(len(group), L, dtype=torch.long)
            cls = torch.tensor(cls_[i : i + len(group)], dtype=torch.long)
            mask = torch.zeros(len(group), L, dtype=torch.bool)
            for j, q in enumerate(group):
                n = len(q.raw)
                raw[j, :n] = torch.tensor(list(q.raw))
                prev[j, 1:n] = raw[j, : n - 1]
                pos[j, :n] = torch.arange(n).clamp(max=15)
                mask[j, :n] = True
            yield (raw.to(dev), prev.to(dev), cls.to(dev), pos.to(dev), mask.to(dev))

    for _ in range(epochs):
        for raw, prev, cls, pos, mask in batches(pf, train_cls):
            h0 = torch.zeros(1, len(raw), HID, device=dev)
            logits, _ = net(prev, cls[:, None].expand(-1, prev.shape[1]), pos, h0)
            loss = lossf(logits[mask], raw[mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
    # export the float16 model and measure the val with the packed forward
    w = {}
    sd = net.state_dict()
    w["pe"] = sd["pe.weight"].detach().cpu().numpy()
    w["ce"] = sd["ce.weight"].detach().cpu().numpy()
    w["pos"] = sd["pos.weight"].detach().cpu().numpy()
    # torch GRU: weight_ih_l0 is (3H, in), weight_hh_l0 is (3H, H), bias_ih/hh
    wih = sd["gru.weight_ih_l0"].cpu().numpy()
    whh = sd["gru.weight_hh_l0"].cpu().numpy()
    H = HID
    for j, nm in enumerate(("r", "z", "n")):
        w[f"w{nm}"] = wih[j * H : (j + 1) * H, :].T
        w[f"u{nm}"] = whh[j * H : (j + 1) * H, :].T
        w[f"b{nm}"] = np.zeros(H)
    w["w1"] = sd["w1.weight"].cpu().numpy().T
    w["b1"] = sd["w1.bias"].cpu().numpy()
    w["w2"] = sd["w2.weight"].cpu().numpy().T
    w["b2"] = sd["w2.bias"].cpu().numpy()
    model = Tail(w)
    bits = sum(logp_span(model, p.raw, cl)[0] for p, cl in zip(pv, val_cls))
    root = np.exp2(-s["bc"])
    root /= root.sum()
    counts, _ = t.byte_matrix(pf, root)
    a, b = z.factors(counts, 16)
    logp = z.logp_of(a, b, root)
    svd_bits = sum(z.run_bits(p.raw, logp) for p in pv)
    return {
        "name": name,
        "rect": rect,
        "bits": bits,
        "tail_bytes": sum(len(p.raw) for p in pv),
        "tail_bpb": bits / max(sum(len(p.raw) for p in pv), 1),
        "size": model.size,
        "svd_bpb": svd_bits / max(sum(len(p.raw) for p in pv), 1),
        "svd_bits": svd_bits,
        "model": model,
    }


def selftest() -> None:
    rng = np.random.RandomState(SEED)
    w = {
        "pe": rng.normal(0, 0.1, (257, PE)),
        "ce": rng.normal(0, 0.1, (65, CE)),
        "pos": rng.normal(0, 0.1, (16, POS)),
    }
    for nm in ("z", "r", "n"):
        w[f"w{nm}"] = rng.normal(0, 0.1, (PE + CE + POS, HID))
        w[f"u{nm}"] = rng.normal(0, 0.1, (HID, HID))
        w[f"b{nm}"] = rng.normal(0, 0.1, HID)
    w["w1"] = rng.normal(0, 0.1, (HID, RANK))
    w["w2"] = rng.normal(0, 0.1, (RANK, 256))
    w["b1"] = rng.normal(0, 0.1, RANK)
    w["b2"] = rng.normal(0, 0.1, 256)
    model = Tail(w)
    blob = pack(model)
    m2 = unpack(blob)
    assert pack(m2) == blob
    a, _ = logp_span(m2, b"abc", 3)
    b, _ = logp_span(unpack(pack(m2)), b"abc", 3)
    assert abs(a - b) < 1e-9
    print("ncm selftest: ok")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="H-NCM neural tail")
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--corpus", default="enwik8_1m")
    r.add_argument("--vocab", default="substr")
    r.add_argument("--dict", type=int, default=1024)
    r.add_argument("--table", type=int, default=2048)
    r.add_argument("--epochs", type=int, default=2)
    r.add_argument("--out", default="runs/v8/ncm.json")
    sub.add_parser("selftest")
    a = p.parse_args(argv)
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "run":
        e = train(a.corpus, a.vocab, {"dict": a.dict, "table": a.table}, a.epochs)
        print(json.dumps({k: v for k, v in e.items() if k != "model"}, indent=1))
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({k: v for k, v in e.items() if k != "model"}, indent=1)
        )
        if e.get("model") is not None:
            (out.parent / f"tail-{a.corpus}.bin").write_bytes(pack(e["model"]))
    else:
        p.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
