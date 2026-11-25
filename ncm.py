"""NCM: the neural context model of the tail, trained end to end at 10 MB.

The H-Net gives an exact concept index; the NCM codes the spans it does not
cover. The defect of the first version was the *context*: it saw only the
previous byte and restarted at 256. This version sees the real previous *three*
bytes (the decoder holds them), the H-Net class of the preceding concept, and
the span position; it is trained on the fit's uncovered spans and selected on
the held-out 10% of the train. It is packed as float16 (25 kB at these sizes),
and every reported bit is from the packed numpy forward — the pass the codec's
decoder makes.
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

PE = 8
CE = 8
POS = 4
HID = 24
RANK = 8
SEED = 0
MAGIC = b"HNN9"
ALPHA = 0.2
KEYS = (
    "pe1",
    "pe2",
    "pe3",
    "ce",
    "pos",
    "wr",
    "uz",
    "wz",
    "ur",
    "wn",
    "un",
    "w1",
    "b1",
    "w2",
    "b2",
    "br",
    "bz",
    "bn",
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def ctxb(raw: bytes, ctx: bytes, j: int, k: int) -> int:
    """The k-th byte before the span position j (256 = unknown)."""
    if j >= k:
        return raw[j - k]
    i = len(ctx) - k + j
    return ctx[i] if 0 <= i < len(ctx) else 256


@dataclass
class Tail:
    w: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        self.size = len(pack(self))

    def one(self, x: np.ndarray, h: np.ndarray):
        w = self.w
        zgt = sigmoid(x @ w["wz"] + h @ w["uz"] + w["bz"])
        rgt = sigmoid(x @ w["wr"] + h @ w["ur"] + w["br"])
        n = np.tanh(x @ w["wn"] + (rgt * h) @ w["un"] + w["bn"])
        h = (1 - zgt) * n + zgt * h
        y = np.tanh(h @ w["w1"] + w["b1"])
        return y @ w["w2"] + w["b2"], h

    def input(self, b1: int, b2: int, b3: int, cls: int, pos: int) -> np.ndarray:
        return np.concatenate(
            [
                self.w["pe1"][b1],
                self.w["pe2"][b2],
                self.w["pe3"][b3],
                self.w["ce"][cls],
                self.w["pos"][pos],
            ]
        )


def logp_span(model: Tail, raw: bytes, cls: int, ctx: bytes) -> float:
    h = np.zeros(HID)
    bits = 0.0
    for j, b in enumerate(raw):
        x = model.input(
            ctxb(raw, ctx, j, 1),
            ctxb(raw, ctx, j, 2),
            ctxb(raw, ctx, j, 3),
            cls,
            min(j, 15),
        )
        logits, h = model.one(x, h)
        logits = logits - logits.max()
        p = np.exp(logits)
        p /= p.sum()
        bits -= math.log2(max(p[b], 1e-30))
    return bits


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
def class_of(tok: bytes | None, d) -> int:
    if tok is None:
        return 64
    return (min(len(tok), 16) - 1) * 4 + min(int(d.cost_of(tok) / 5), 3)


def train(name: str, vocab: str, rect: dict, epochs: int = 3) -> dict:
    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s = c.split(name, vocab)
    d, bg, uni = c.rebuild(s, rect["dict"], rect["table"])
    fit_toks = c.viterbi_chunked(s["fit"], d, bg, uni)
    val_toks, _ = c.viterbi(s["va"], d, bg, uni)
    pf, pv = t.pieces(fit_toks, {}), t.pieces(val_toks, {})
    if not pf or not pv:
        return {"name": name, "bits": 0.0, "size": 0, "tail_bytes": 0}
    rank = {tok: i for i, tok in enumerate(s["top"][:256])}
    inv = {i: tok for tok, i in rank.items()}
    tr_cls = [class_of(inv.get(p.prev), d) for p in pf]
    va_cls = [class_of(inv.get(p.prev), d) for p in pv]

    def arrays(pf_, cls_, bs=256):
        for i in range(0, len(pf_), bs):
            group = pf_[i : i + bs]
            L = max(len(q.raw) for q in group)
            raw = torch.zeros(len(group), L, dtype=torch.long)
            b1 = torch.full((len(group), L), 256, dtype=torch.long)
            b2 = torch.full((len(group), L), 256, dtype=torch.long)
            b3 = torch.full((len(group), L), 256, dtype=torch.long)
            pos = torch.zeros(len(group), L, dtype=torch.long)
            cls = torch.tensor(cls_[i : i + len(group)], dtype=torch.long)
            mask = torch.zeros(len(group), L, dtype=torch.bool)
            for j, q in enumerate(group):
                n = len(q.raw)
                raw[j, :n] = torch.tensor(list(q.raw))
                for k, dst in ((1, b1), (2, b2), (3, b3)):
                    dst[j, :n] = torch.tensor(
                        [ctxb(q.raw, q.ctx, p, k) for p in range(n)]
                    )
                pos[j, :n] = torch.arange(n).clamp(max=15)
                mask[j, :n] = True
            yield (
                raw.to(dev),
                b1.to(dev),
                b2.to(dev),
                b3.to(dev),
                cls.to(dev),
                pos.to(dev),
                mask.to(dev),
            )

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pe1 = nn.Embedding(257, PE)
            self.pe2 = nn.Embedding(257, PE)
            self.pe3 = nn.Embedding(257, PE)
            self.ce = nn.Embedding(65, CE)
            self.pos = nn.Embedding(16, POS)
            self.gru = nn.GRU(3 * PE + CE + POS, HID, batch_first=True, bias=False)
            self.w1 = nn.Linear(HID, RANK)
            self.w2 = nn.Linear(RANK, 256)

        def forward(self, b1, b2, b3, cls, pos, h0):
            x = torch.cat(
                [self.pe1(b1), self.pe2(b2), self.pe3(b3), self.ce(cls), self.pos(pos)],
                -1,
            )
            y, _ = self.gru(x, h0)
            return self.w2(torch.tanh(self.w1(y)))

    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for raw, b1, b2, b3, cls, pos, mask in arrays(pf, tr_cls):
            h0 = torch.zeros(1, len(raw), HID, device=dev)
            logits = net(b1, b2, b3, cls[:, None].expand_as(b1), pos, h0)
            loss = lossf(logits[mask], raw[mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
    sd = net.state_dict()
    w = {
        k: sd[k].detach().cpu().numpy()
        for k in ("pe1.weight", "pe2.weight", "pe3.weight", "ce.weight", "pos.weight")
    }
    w = {k.split(".")[0]: v for k, v in w.items()}
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
    bits = sum(logp_span(model, q.raw, cl, q.ctx) for q, cl in zip(pv, va_cls))
    root = np.exp2(-s["bc"])
    root /= root.sum()
    counts, _ = t.byte_matrix(pf, root)
    a, b, scale = z.factors(counts, 32)
    svd = sum(z.run_bits(q.raw, z.logp_of(a, b, root, scale)) for q in pv)
    tb = sum(len(q.raw) for q in pv)
    return {
        "name": name,
        "rect": rect,
        "tail_bytes": tb,
        "tail_bpb": bits / tb,
        "svd_bpb": svd / tb,
        "size": model.size,
        "epochs": epochs,
        "model": model,
    }


def selftest() -> None:
    rng = np.random.RandomState(SEED)
    w = {
        "pe1": rng.normal(0, 0.1, (257, PE)),
        "pe2": rng.normal(0, 0.1, (257, PE)),
        "pe3": rng.normal(0, 0.1, (257, PE)),
        "ce": rng.normal(0, 0.1, (65, CE)),
        "pos": rng.normal(0, 0.1, (16, POS)),
    }
    for nm in ("z", "r", "n"):
        w[f"w{nm}"] = rng.normal(0, 0.1, (3 * PE + CE + POS, HID))
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
    a = logp_span(m2, b"abc", 3, b"xyz")
    b = logp_span(unpack(pack(m2)), b"abc", 3, b"xyz")
    assert abs(a - b) < 1e-9
    assert ctxb(b"abc", b"xyz", 0, 1) == ord("z")
    assert ctxb(b"abc", b"xyz", 1, 1) == ord("a")
    print("ncm selftest: ok")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="H-NCM neural tail")
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--corpus", default="enwik8_10m")
    r.add_argument("--vocab", default="substr")
    r.add_argument("--dict", type=int, default=8192)
    r.add_argument("--table", type=int, default=4096)
    r.add_argument("--epochs", type=int, default=3)
    r.add_argument("--out", default="runs/v8/ncm10m.json")
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
