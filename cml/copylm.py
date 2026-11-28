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
    lr: float = 1e-4
    window: int = 2048
    min_len: int = 8
    cap: int = 64
    depth: int = 8
    train_bytes: int = 2_000_000
    eval_bytes: int = 100_000
    windows: int = 160
    seed: int = 0
    joint: bool = False
    joint_batch: int = 4
    joint_positions: int = 64
    block_lr: float = 1e-5
    pointer_scale: float = 1.0


class CopyHead(nn.Module):
    """A pointer over the context, a copy gate, and a run-length head."""

    def __init__(self, d: int, window: int, pointer_scale: float = 1.0):
        super().__init__()
        width = max(1, int(d * pointer_scale))
        self.wq = nn.Linear(d, width, bias=False)
        self.wk = nn.Linear(d, width, bias=False)
        self.gate = nn.Linear(d + 1, 1)
        self.run = nn.Linear(2 * d, len(LENS))
        self.rel = nn.Parameter(torch.zeros(window))
        self.scale = width**-0.5

    def rel_bias(self, q: torch.Tensor, t: int) -> torch.Tensor:
        j = torch.arange(t, device=q.device)[None, :]
        d = (q[:, None] - j).clamp(0, self.rel.numel() - 1)
        return self.rel[d]

    def pointer(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return (self.wq(q) @ self.wk(k).T) * self.scale

    def margin_and_logits(
        self, q: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pointer logits for positions `q`, plus the top-1 minus top-2 margin."""
        logits = self.pointer(h[q - 1], h) + self.rel_bias(q - 1, h.size(0))
        future = torch.arange(h.size(0), device="cuda")[None, :]
        logits = logits.masked_fill(future >= (q - 1)[:, None], -torch.inf)
        top = logits.topk(2, dim=-1).values
        margin = top[:, 0] - top[:, 1]
        margin = torch.nan_to_num(margin, nan=0.0, posinf=0.0, neginf=0.0)
        return margin, logits


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


def _exact_copies(
    w: bytes,
    idx: np.ndarray,
    pred: np.ndarray,
    run: np.ndarray,
    sel: np.ndarray,
) -> np.ndarray:
    """True where the predicted source and run length copy exact bytes."""
    out = np.zeros(len(idx), dtype=bool)
    for k in np.nonzero(sel)[0]:
        i = int(idx[k]) + 1
        start = int(pred[k]) + 1
        length = LENS[int(run[k])]
        out[k] = start < i and w[start : start + length] == w[i : i + length]
    return out


def _match_len(w: bytes, start: int, pos: int, cap: int) -> int:
    """Longest exact match at `start` for the bytes starting at `pos`."""
    if start < 0 or start >= pos:
        return 0
    limit = min(cap, len(w) - pos, len(w) - start)
    length = 0
    while length < limit and w[start + length] == w[pos + length]:
        length += 1
    return length


def _windows(data: bytes, seq: int) -> list[bytes]:
    return [data[i : i + seq + 1] for i in range(0, len(data) - seq, seq)]


def prepare(
    model, data: bytes, cfg: CopyConfig
) -> list[tuple[torch.Tensor, np.ndarray, np.ndarray, bytes]]:
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
        cache.append((h, src[1:], ln[1:], w[:seq]))
    return cache


def _valid_sources(w: bytes, idx: np.ndarray, min_len: int, width: int) -> np.ndarray:
    """Boolean mask over pointer slots: every prior source that matches `min_len` bytes.

    Slot `j` means source byte `j + 1`; a query at byte `i` accepts it when the
    `min_len` bytes at both places are equal.  Multi-positive training then puts
    probability mass on all exact sources, not only the longest one.
    """
    keys = np.lib.stride_tricks.sliding_window_view(
        np.frombuffer(w, dtype=np.uint8), min_len
    )
    mask = np.zeros((len(idx), width), dtype=bool)
    for k, at in enumerate(idx):
        i = int(at) + 1
        if i >= len(keys):
            continue
        limit = min(i, width + 1)
        if limit > 1:
            mask[k, : limit - 1] = (keys[1:limit] == keys[i]).all(axis=1)
    return mask


def train_head(
    model, cache: list[tuple[torch.Tensor, np.ndarray, np.ndarray]], cfg: CopyConfig
) -> CopyHead:
    d = model.tok.embedding_dim
    head = CopyHead(d, cfg.window, cfg.pointer_scale).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=0.01)
    rng = np.random.default_rng(cfg.seed)
    for step in range(cfg.steps):
        h, src, ln, w = cache[int(rng.integers(len(cache)))]
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
        margin, logits = head.margin_and_logits(q, h_cuda)
        gate = head.gate(torch.cat((h_cuda[q - 1], margin[:, None]), dim=-1)).squeeze(
            -1
        )
        sel_np = (ln[idx] >= cfg.min_len) & (src[idx] >= 1)
        sel = torch.from_numpy(sel_np).cuda()
        pred = logits.argmax(-1).detach().cpu().numpy()
        pred_t = torch.from_numpy(pred).long().cuda()
        run_choice = (
            head.run(torch.cat((h_cuda[q - 1], h_cuda[pred_t]), dim=-1))
            .argmax(-1)
            .detach()
            .cpu()
            .numpy()
        )
        correct = torch.from_numpy(
            _exact_copies(w, idx, pred, run_choice, sel_np)
        ).cuda()
        loss = F.binary_cross_entropy_with_logits(gate, correct.float())
        source_ok = np.zeros(len(idx), dtype=bool)
        run_np = np.zeros(len(idx), dtype=np.int64)
        for k in np.nonzero(sel_np)[0]:
            length = _match_len(w, int(pred[k]) + 1, int(idx[k]) + 1, cfg.cap)
            if length >= cfg.min_len:
                source_ok[k] = True
                run_np[k] = _bucket(length)
        if bool(sel.any()):
            valid = torch.from_numpy(
                _valid_sources(w, idx, cfg.min_len, cfg.window)[sel_np]
            ).cuda()
            spread = logits[sel] - torch.logsumexp(logits[sel], dim=-1)[:, None]
            loss = (
                loss
                + (
                    -torch.logsumexp(spread.masked_fill(~valid, -torch.inf), dim=-1)
                ).mean()
            )
        if source_ok.any():
            picked = torch.from_numpy(source_ok).cuda()
            buckets = torch.from_numpy(run_np[source_ok]).cuda()
            loss = loss + F.cross_entropy(
                head.run(
                    torch.cat((h_cuda[q[picked] - 1], h_cuda[pred_t[picked]]), dim=-1)
                ),
                buckets,
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return head


def _dataset(
    data: bytes, cfg: CopyConfig
) -> list[tuple[bytes, np.ndarray, np.ndarray]]:
    wins = _windows(data, cfg.window)
    rng = np.random.default_rng(cfg.seed)
    picks = np.sort(
        rng.choice(len(wins), size=min(cfg.windows, len(wins)), replace=False)
    )
    out = []
    for i in picks.tolist():
        w = wins[i]
        src, ln = match_labels(w, cfg.window, cfg.min_len, cfg.cap, cfg.depth)
        out.append((w[: cfg.window], src[1:], ln[1:]))
    return out


def train_joint(
    model,
    dataset: list[tuple[bytes, np.ndarray, np.ndarray]],
    cfg: CopyConfig,
) -> CopyHead:
    """Unfreeze the last block and train it with the copy head."""
    d = model.tok.embedding_dim
    head = CopyHead(d, cfg.window, cfg.pointer_scale).cuda()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.blocks[-1].parameters():
        p.requires_grad_(True)
    for p in model.ln.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(
        [
            {"params": list(head.parameters()), "lr": cfg.lr},
            {
                "params": [
                    *model.blocks[-1].parameters(),
                    *model.ln.parameters(),
                ],
                "lr": cfg.block_lr,
            },
        ],
        weight_decay=0.01,
    )
    rng = np.random.default_rng(cfg.seed)
    for step in range(cfg.steps):
        picks = rng.choice(len(dataset), size=cfg.joint_batch, replace=False)
        opt.zero_grad(set_to_none=True)
        total = torch.zeros((), device="cuda")
        used = 0
        for pick in picks.tolist():
            w, src, ln = dataset[pick]
            copy_idx = np.nonzero((ln >= cfg.min_len) & (src >= 1))[0]
            lit_idx = np.nonzero(ln < cfg.min_len)[0]
            if not len(copy_idx) or not len(lit_idx):
                continue
            take = max(cfg.joint_positions // 2, 1)
            idx = np.concatenate(
                (
                    rng.choice(copy_idx, size=min(take, len(copy_idx)), replace=False),
                    rng.choice(lit_idx, size=min(take, len(lit_idx)), replace=False),
                )
            )
            idx.sort()
            x = torch.tensor([list(w)], dtype=torch.long, device="cuda")
            h = model.hidden(x)[0].float()
            q = torch.from_numpy(idx + 1).long().cuda()
            margin, logits = head.margin_and_logits(q, h)
            gate = head.gate(torch.cat((h[q - 1], margin[:, None]), dim=-1)).squeeze(-1)
            sel_np = (ln[idx] >= cfg.min_len) & (src[idx] >= 1)
            sel = torch.from_numpy(sel_np).cuda()
            pred = logits.argmax(-1).detach().cpu().numpy()
            pred_t = torch.from_numpy(pred).long().cuda()
            run_choice = (
                head.run(torch.cat((h[q - 1], h[pred_t]), dim=-1))
                .argmax(-1)
                .detach()
                .cpu()
                .numpy()
            )
            correct = torch.from_numpy(
                _exact_copies(w, idx, pred, run_choice, sel_np)
            ).cuda()
            loss = F.binary_cross_entropy_with_logits(gate, correct.float())
            source_ok = np.zeros(len(idx), dtype=bool)
            run_np = np.zeros(len(idx), dtype=np.int64)
            for k in np.nonzero(sel_np)[0]:
                length = _match_len(w, int(pred[k]) + 1, int(idx[k]) + 1, cfg.cap)
                if length >= cfg.min_len:
                    source_ok[k] = True
                    run_np[k] = _bucket(length)
            if bool(sel.any()):
                valid = torch.from_numpy(
                    _valid_sources(w, idx, cfg.min_len, cfg.window)[sel_np]
                ).cuda()
                spread = logits[sel] - torch.logsumexp(logits[sel], dim=-1)[:, None]
                loss = (
                    loss
                    + (
                        -torch.logsumexp(spread.masked_fill(~valid, -torch.inf), dim=-1)
                    ).mean()
                )
            if source_ok.any():
                picked = torch.from_numpy(source_ok).cuda()
                buckets = torch.from_numpy(run_np[source_ok]).cuda()
                loss = loss + F.cross_entropy(
                    head.run(torch.cat((h[q[picked] - 1], h[pred_t[picked]]), dim=-1)),
                    buckets,
                )
            total = total + loss
            used += 1
        if used:
            (total / used).backward()
            opt.step()
        if step % 200 == 0:
            print(f"  joint step {step}/{cfg.steps}", flush=True)
    return head


def evaluate(
    model,
    head: CopyHead,
    data: bytes,
    cfg: CopyConfig,
    thresholds: tuple[float, ...] = (0.5, 0.7, 0.9, 0.95, 0.99),
) -> dict:
    seq = cfg.window
    out = {
        t: {
            "bytes": 0,
            "copies": 0,
            "steps": 0,
            "attempt": 0,
            "tries": 0,
            "source": 0,
        }
        for t in thresholds
    }
    n = 0
    for w in _windows(data, seq):
        x = torch.tensor([list(w[:seq])], dtype=torch.long, device="cuda")
        with torch.no_grad():
            h = model.hidden(x)[0].float()
            qt = torch.arange(seq, device="cuda")
            dist = (qt[:, None] - qt[None, :]).clamp(0, cfg.window - 1)
            ptr = head.wq(h) @ head.wk(h).T * head.scale + head.rel[dist]
            future = qt[None, :] >= qt[:, None]
            top = ptr.masked_fill(future, -torch.inf).topk(2, dim=-1).values
            top[0] = 0
            margin = top[:, 0] - top[:, 1]
            margin = torch.nan_to_num(margin, nan=0.0, posinf=0.0, neginf=0.0)
            gate = (
                torch.sigmoid(
                    head.gate(torch.cat((h, margin[:, None]), dim=-1)).squeeze(-1)
                )
                .cpu()
                .numpy()
            )
            ptr = ptr.float().cpu().numpy()
        for thr in thresholds:
            t = 1
            copied = 0
            copies = 0
            attempted = 0
            tries = 0
            source = 0
            while t < seq:
                if t > 1 and gate[t - 1] >= thr:
                    row = ptr[t - 1, : t - 1]
                    j = int(row.argmax())
                    src = j + 1
                    with torch.no_grad():
                        length = LENS[
                            int(head.run(torch.cat((h[t - 1], h[j]), dim=-1)).argmax())
                        ]
                    attempted += length
                    tries += 1
                    if src < t and w[src : src + cfg.min_len] == w[t : t + cfg.min_len]:
                        source += 1
                    if src < t and w[src : src + length] == w[t : t + length]:
                        copied += length
                        copies += 1
                        t += length
                        continue
                t += 1
            out[thr]["bytes"] += copied
            out[thr]["copies"] += copies
            out[thr]["attempt"] += attempted
            out[thr]["tries"] += tries
            out[thr]["source"] += source
            out[thr]["steps"] += seq - 1 - copied + copies + attempted
        n += seq
    total = max(n, 1)
    baseline = max(n - n // seq, 1)
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
                "source_precision": r["source"] / max(r["tries"], 1),
                "copies": r["copies"],
                "speedup": (baseline / steps),
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
    joint: bool = False,
    lr: float = 1e-4,
    block_lr: float = 1e-5,
    pointer_scale: float = 1.0,
) -> int:
    """Train and evaluate the copy head on a byte trunk."""
    import bytecodec
    import cml

    t0 = time.perf_counter()
    cfg = CopyConfig(
        steps=steps,
        lr=lr,
        window=window,
        train_bytes=train_bytes,
        eval_bytes=eval_bytes,
        windows=windows,
        seed=seed,
        joint=joint,
        block_lr=block_lr,
        pointer_scale=pointer_scale,
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
    if joint:
        dataset = _dataset(train, cfg)
        head = train_joint(model, dataset, cfg)
        print(f"trained {cfg.steps} joint steps on {len(dataset)} windows", flush=True)
    else:
        cache = prepare(model, train, cfg)
        head = train_head(model, cache, cfg)
        print(f"trained {cfg.steps} steps on {len(cache)} windows", flush=True)
    metrics = evaluate(model, head, test, cfg)
    secs = time.perf_counter() - t0
    title = (
        "# ByteCopy LM: joint last block + margin gate"
        if joint
        else "# ByteCopy LM: copy head on a frozen byte trunk"
    )
    trunk = (
        f"{sum(p.numel() for p in head.parameters()):,} head params; "
        "last block + head trained, rest frozen"
        if joint
        else f"{sum(p.numel() for p in head.parameters()):,} params, frozen trunk"
    )
    lines = [
        title,
        "",
        f"- trunk `{run}`; {cfg.window} B window; "
        f"{cfg.train_bytes:,} B train, {cfg.eval_bytes:,} B held-out",
        f"- self-supervised longest prior match: min {cfg.min_len} B, "
        f"cap {cfg.cap} B, depth {cfg.depth}",
        f"- head: {cfg.pointer_scale:.1f}x pointer over the context, "
        f"gate with top-2 margin, run length conditioned on the chosen "
        f"source; {trunk}",
        "",
        "| gate | coverage | span precision | source precision | copies | bytes/step |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in metrics["rows"]:
        lines.append(
            f"| {r['threshold']:.2f} | {r['coverage']:.3f} | "
            f"{r['precision']:.3f} | {r['source_precision']:.3f} | "
            f"{r['copies']:,} | {r['speedup']:.2f}x |"
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
    stem = "COPY-LM-JOINT" if joint else "COPY-LM"
    cml.write_text(cml.REPORTS / f"{stem}.md", text)
    cml.write_json(
        cml.REPORTS / f"{stem.lower()}.json",
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
