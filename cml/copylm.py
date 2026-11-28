"""ByteCopy LM: a byte trunk that copies exact spans and computes the rest.

The trunk predicts novel bytes; a copy head points at an earlier position in
the context and a run head names how many bytes to copy.  Copy labels are
self-supervised: the longest prior match of the next bytes.  At inference a
copied span is emitted exactly and its cache is updated in one chunked
forward, so a small byte model spends capacity on novel bytes and delegates
known spans to the exact path.

This is the model-side fix for the A0.2 result: no flat span ids, no pack
charge, no per-copy id entropy.  The pointer is a position in the context,
which the model already represents.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LENS = (8, 16, 32, 64)


@dataclass(frozen=True, slots=True)
class CopyConfig:
    steps: int = 500
    batch: int = 128
    lr: float = 1e-3
    window: int = 2048
    min_len: int = 8
    cap: int = 64
    depth: int = 8
    train_bytes: int = 2_000_000
    eval_bytes: int = 100_000
    windows: int = 160
    seed: int = 0


class CopyHead(nn.Module):
    """A pointer over the context, a copy gate, and a run-length head."""

    def __init__(self, d: int, window: int):
        super().__init__()
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, 1)
        self.run = nn.Linear(d, len(LENS))
        self.rel = nn.Parameter(torch.zeros(window))
        self.scale = d**-0.5

    def rel_bias(self, q: torch.Tensor, t: int) -> torch.Tensor:
        j = torch.arange(t, device=q.device)[None, :]
        d = (q[:, None] - j).clamp(0, self.rel.numel() - 1)
        return self.rel[d]

    def pointer(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return (self.wq(q) @ self.wk(k).T) * self.scale


def match_labels(
    data: bytes, window: int, min_len: int, cap: int, depth: int
) -> tuple[np.ndarray, np.ndarray]:
    """Longest prior match per position: source start and length."""
    n = len(data)
    src = np.full(n, -1, dtype=np.int32)
    ln = np.zeros(n, dtype=np.int32)
    head: dict[int, list[int]] = {}
    for i in range(n):
        key = int.from_bytes(data[i : i + 4], "big") if i + 4 <= n else -1
        cand = head.get(key, [])
        best = 0
        best_src = -1
        for p in reversed(cand[-depth:]):
            if i - p > window:
                break
            lim = min(cap, n - i, n - p)
            if data[p : p + lim] == data[i : i + lim]:
                ell = lim
            else:
                ell = 0
                while ell < lim and data[p + ell] == data[i + ell]:
                    ell += 1
            if ell > best:
                best = ell
                best_src = p
                if best >= cap:
                    break
        if best >= min_len and best_src >= 1:
            src[i] = best_src
            ln[i] = best
        if key >= 0:
            head.setdefault(key, []).append(i)
    return src, ln


def _bucket(length: int) -> int:
    return int(np.searchsorted(LENS, length, side="right") - 1)


def _windows(data: bytes, seq: int) -> list[bytes]:
    return [data[i : i + seq + 1] for i in range(0, len(data) - seq, seq)]


def prepare(
    model, data: bytes, cfg: CopyConfig
) -> list[tuple[torch.Tensor, np.ndarray, np.ndarray]]:
    """Frozen hidden states plus copy labels for a sample of windows."""
    seq = cfg.window
    wins = _windows(data, seq)
    rng = np.random.default_rng(cfg.seed)
    picks = np.sort(
        rng.choice(len(wins), size=min(cfg.windows, len(wins)), replace=False)
    )
    cache = []
    for w in (wins[i] for i in picks.tolist()):
        src, ln = match_labels(w, seq, cfg.min_len, cfg.cap, cfg.depth)
        x = torch.tensor([list(w[:seq])], dtype=torch.long, device="cuda")
        with torch.no_grad():
            h = model.hidden(x)[0].half().cpu()
        cache.append((h, src[1:], ln[1:]))
    return cache


def train_head(
    model, cache: list[tuple[torch.Tensor, np.ndarray, np.ndarray]], cfg: CopyConfig
) -> CopyHead:
    d = model.tok.embedding_dim
    head = CopyHead(d, cfg.window).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=0.01)
    rng = np.random.default_rng(cfg.seed)
    for step in range(cfg.steps):
        h, src, ln = cache[int(rng.integers(len(cache)))]
        copy_idx = np.nonzero((ln >= cfg.min_len) & (src >= 1))[0]
        lit_idx = np.nonzero(ln < cfg.min_len)[0]
        if not len(copy_idx) or not len(lit_idx):
            continue
        take = max(cfg.batch // 2, 1)
        idx = np.concatenate(
            (
                rng.choice(copy_idx, size=min(take, len(copy_idx)), replace=False),
                rng.choice(lit_idx, size=min(take, len(lit_idx)), replace=False),
            )
        )
        idx.sort()
        q = torch.from_numpy(idx + 1).long().cuda()
        h_cuda = h.float().cuda()
        gate = head.gate(h_cuda[q - 1]).squeeze(-1)
        label = torch.from_numpy((ln[idx] >= cfg.min_len).astype(np.float32)).cuda()
        loss = F.binary_cross_entropy_with_logits(gate, label)
        sel_np = (ln[idx] >= cfg.min_len) & (src[idx] >= 1)
        sel = torch.from_numpy(sel_np).cuda()
        if bool(sel.any()):
            logits = head.pointer(h_cuda[q[sel] - 1], h_cuda)
            logits = logits + head.rel_bias(q[sel] - 1, h.size(0))
            j = torch.arange(h.size(0), device="cuda")[None, :]
            logits = logits.masked_fill(j >= (q[sel] - 1)[:, None], -torch.inf)
            target = torch.from_numpy((src[idx][sel_np] - 1).astype(np.int64)).cuda()
            loss = loss + F.cross_entropy(logits, target)
            buckets = torch.tensor(
                [_bucket(int(x)) for x in ln[idx][sel_np]], device="cuda"
            )
            loss = loss + F.cross_entropy(head.run(h_cuda[q[sel] - 1]), buckets)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return head


def evaluate(
    model,
    head: CopyHead,
    data: bytes,
    cfg: CopyConfig,
    thresholds: tuple[float, ...] = (0.5, 0.7, 0.9),
) -> dict:
    seq = cfg.window
    out = {t: {"bytes": 0, "copies": 0, "steps": 0, "attempt": 0} for t in thresholds}
    n = 0
    for w in _windows(data, seq):
        x = torch.tensor([list(w[:seq])], dtype=torch.long, device="cuda")
        with torch.no_grad():
            h = model.hidden(x)[0].float()
            gate = torch.sigmoid(head.gate(h).squeeze(-1)).cpu().numpy()
            q = head.wq(h)
            k = head.wk(h)
            run = head.run(h).argmax(-1).cpu().numpy()
            qt = torch.arange(seq, device="cuda")
            dist = (qt[:, None] - qt[None, :]).clamp(0, cfg.window - 1)
            ptr = (q @ k.T * head.scale + head.rel[dist]).float().cpu().numpy()
        for thr in thresholds:
            t = 1
            copied = 0
            copies = 0
            attempted = 0
            while t < seq:
                if t > 1 and gate[t - 1] >= thr:
                    row = ptr[t - 1, : t - 1]
                    j = int(row.argmax())
                    src = j + 1
                    length = LENS[int(run[t - 1])]
                    attempted += length
                    if src < t and w[src : src + length] == w[t : t + length]:
                        copied += length
                        copies += 1
                        t += length
                        continue
                t += 1
            out[thr]["bytes"] += copied
            out[thr]["copies"] += copies
            out[thr]["attempt"] += attempted
            out[thr]["steps"] += seq - 1 - copied + attempted
        n += seq
    total = max(n, 1)
    rows = []
    for thr in thresholds:
        r = out[thr]
        precision = r["bytes"] / max(r["attempt"], 1)
        coverage = r["bytes"] / max(total, 1)
        steps = max(r["steps"], 1)
        rows.append(
            {
                "threshold": thr,
                "coverage": coverage,
                "precision": precision,
                "copies": r["copies"],
                "speedup": (total / steps),
            }
        )
    return {"rows": rows, "bytes": total}


def run(
    run: str,
    steps: int,
    window: int,
    train_bytes: int,
    eval_bytes: int,
    windows: int,
    seed: int,
) -> int:
    """Train and evaluate the copy head on a frozen byte trunk."""
    import bytecodec
    import cml

    t0 = time.perf_counter()
    cfg = CopyConfig(
        steps=steps,
        window=window,
        train_bytes=train_bytes,
        eval_bytes=eval_bytes,
        windows=windows,
        seed=seed,
    )
    corpus = cml.code_scale_corpus(48_000_000)
    model = bytecodec.load_model(cml.RUNS / run / "model.pt")
    for p in model.parameters():
        p.requires_grad_(False)
    train = corpus.train[: cfg.train_bytes]
    test = corpus.test[: cfg.eval_bytes]
    train_src, train_ln = match_labels(
        train[: cfg.window * 8], cfg.window, cfg.min_len, cfg.cap, cfg.depth
    )
    copy_rate = float((train_ln >= cfg.min_len).mean())
    print(f"train copy rate {copy_rate:.3f} on {len(train_ln):,} positions", flush=True)
    cache = prepare(model, train, cfg)
    head = train_head(model, cache, cfg)
    print(f"trained {cfg.steps} steps on {len(cache)} windows", flush=True)
    metrics = evaluate(model, head, test, cfg)
    secs = time.perf_counter() - t0
    lines = [
        "# ByteCopy LM: copy head on a frozen byte trunk",
        "",
        f"- trunk `{run}`; {cfg.window} B window; "
        f"{cfg.train_bytes:,} B train, {cfg.eval_bytes:,} B held-out",
        f"- self-supervised longest prior match: min {cfg.min_len} B, "
        f"cap {cfg.cap} B, depth {cfg.depth}",
        f"- head: pointer over the context, gate, run length; "
        f"{sum(p.numel() for p in head.parameters()):,} params, frozen trunk",
        "",
        "| gate | coverage | copy precision | copies | bytes/step |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in metrics["rows"]:
        lines.append(
            f"| {r['threshold']:.1f} | {r['coverage']:.3f} | "
            f"{r['precision']:.3f} | {r['copies']:,} | {r['speedup']:.2f}x |"
        )
    best = max(
        metrics["rows"], key=lambda r: (r["coverage"] * r["precision"], r["precision"])
    )
    lines += [
        "",
        "## Read",
        "",
        f"- best gate {best['threshold']:.1f}: copies "
        f"**{best['coverage']:.1%}** of bytes at "
        f"**{best['precision']:.1%}** precision, "
        f"**{best['speedup']:.2f}x** bytes per sequential step.",
        "- a copied span is exact by construction; the trunk only samples "
        "novel bytes. This is the model-side version of the exact memory: "
        "the pointer replaces a flat span id.",
        "",
        f"Took {secs:.0f} s on one GPU.",
        "",
    ]
    text = "\n".join(lines)
    cml.write_text(cml.REPORTS / "COPY-LM.md", text)
    cml.write_json(
        cml.REPORTS / "copy-lm.json",
        {
            "run": run,
            "config": asdict(cfg),
            "rows": metrics["rows"],
            "train_copy_rate": copy_rate,
            "seconds": round(secs, 1),
            "git": cml.git_sha(),
        },
    )
    print(text)
    return 0
