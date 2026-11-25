"""N10: enwik8 at 100 MB. The charge and the coverage, on the GPU scan.

The vocabulary is the union vocabulary selected on the 10 MB stream (32k/64k);
its counts are refit on the 90 MB train with an exact GPU scan of the known
tokens, so the costs are the 100 MB train's. The table is fitted on the first
90% of the train, chosen on the held-out 10%, charged over 100 MB. The val is
10 MB. This is the rate/charge measurement; the codec record at this size is
not run (the 10 MB val encode is out of Python's throughput).

The scan counts only the known vocabulary, so it is exact and needs < 1 GB of
GPU memory; it does not hold the millions of repeated substrings the CPU
`counts_substr` would hold at 90 MB.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import concept_lm as c
import corpora
import hncm7 as v
import tail_coder as t

HERE = Path(__file__).resolve().parent
KINDS = (32768, 65536)
TABLES = (16384, 32768)
VAL = 10_000_000
STREAM = 100_000_000
CHUNK = 22_500_000
_CODEC: dict = {}


def hash_bytes(s: bytes) -> int:
    h = 0
    for b in s:
        h = (h * 257 + b) & ((1 << 64) - 1)
    return h


def count_tokens(trb: bytes, toks: list[bytes], chunk: int = CHUNK) -> dict:
    """Exact counts of the known tokens in `trb`, on the GPU.

    For each length, one sorted hash array and a search over the stream. A
    chunk starts at s0 and ends at min(s0 + chunk, N - ell + 1), so every
    start is visited exactly once.
    """
    by_len: dict[int, list[bytes]] = collections.defaultdict(list)
    for s in toks:
        by_len[len(s)].append(s)
    a = torch.as_tensor(np.frombuffer(bytearray(trb), np.uint8), device="cuda")
    out: dict[bytes, int] = {}
    for ell, ts in by_len.items():
        hs = np.array([hash_bytes(s) for s in ts], dtype=np.uint64).view(np.int64)
        order = np.argsort(hs)
        hs = hs[order]
        ts = [ts[i] for i in order]
        cand = torch.as_tensor(hs, device="cuda")
        cnt = np.zeros(len(ts), dtype=np.int64)
        for s0 in range(0, len(trb) - ell + 1, chunk):
            end = min(s0 + chunk, len(trb) - ell + 1)
            m = end - s0
            seg = a[s0 : end + ell - 1]
            h = torch.zeros(m, dtype=torch.int64, device="cuda")
            for j in range(ell):
                h.mul_(257).add_(seg[j : j + m].long())
            idx = torch.searchsorted(cand, h)
            idx2 = idx.clamp(max=len(cand) - 1)
            ok = idx < len(cand)
            ok &= cand[idx2] == h
            cnt += torch.bincount(idx[ok], minlength=len(cand)).cpu().numpy()
        out.update(zip(ts, cnt.tolist()))
    return out


def candidates100(trb: bytes, min_count: int = 32) -> dict[bytes, int]:
    """All substrings with count >= min_count on the 90 MB train, on the GPU.

    One length at a time: a 90M-element rolling hash, `torch.unique` with the
    inverse and the counts, then the count filter and the representative
    position. Peak GPU memory is ~5 GB at length 16; the kept map is exact.
    """
    a = torch.as_tensor(np.frombuffer(bytearray(trb), np.uint8), device="cuda")
    out: dict[bytes, int] = {}
    for ell in range(2, 17):
        m = len(trb) - ell + 1
        h = torch.zeros(m, dtype=torch.int64, device="cuda")
        for j in range(ell):
            h.mul_(257).add_(a[j : j + m].long())
        u, inv, cnt = torch.unique(h, return_inverse=True, return_counts=True)
        first = torch.empty_like(u)
        first.scatter_(0, inv, torch.arange(m, device="cuda"))
        keep = cnt >= min_count
        u, c, p = (
            u[keep].cpu().numpy(),
            cnt[keep].cpu().numpy(),
            first[keep].cpu().numpy(),
        )
        for pos, cc in zip(p.tolist(), c.tolist()):
            out[bytes(trb[pos : pos + ell])] = int(cc)
        del h, u, inv, cnt, first
        torch.cuda.empty_cache()
        print(f"  len={ell:2d} kept={int(keep.sum())} total={len(out)}", flush=True)
    return out


def save_scan(cand: dict[bytes, int], path: Path) -> None:
    keys = list(cand)
    flat = np.frombuffer(b"".join(keys), np.uint8)
    offs = np.zeros(len(keys) + 1, dtype=np.int64)
    np.cumsum([len(k) for k in keys], out=offs[1:])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        flat=flat,
        offs=offs,
        count=np.array([cand[k] for k in keys], dtype=np.int64),
    )


def load_scan(path: Path) -> dict[bytes, int]:
    d = np.load(path)
    flat, offs, count = d["flat"], d["offs"], d["count"]
    return {bytes(flat[o:e]): int(x) for o, e, x in zip(offs[:-1], offs[1:], count)}


def scan(min_count: int, out: Path) -> dict[bytes, int]:
    trb = corpora.fetch_enwik8()[:90_000_000]
    cand = candidates100(trb, min_count)
    save_scan(cand, out)
    print(f"scan: {len(cand)} candidates -> {out}", flush=True)
    return cand


def cover_chunked(bs: bytes, d: v.Dict, chunk: int = 1 << 20) -> list[bytes]:
    out = []
    for a in range(0, len(bs), chunk):
        out.extend(v.cover(bs[a : a + chunk], d, 0.0))
    return out


def _evaluate(
    d: v.Dict, fit: bytes, sel: bytes, va: bytes, k: int, table: int, t0: float
) -> dict:
    bc = v.byte_costs(fit)
    ft = cover_chunked(fit, d)
    pf = t.pieces(ft, {})
    big, uni = c.bigrams(ft)
    del ft
    bg = c.prune(big, table)
    del big
    sel_toks = c.viterbi_chunked(sel, d, bg, uni)
    sel_rate = c.bits_of(sel_toks, d, bg, uni) / len(sel)
    del sel, sel_toks
    entries = sum(len(row) for row in bg.values())
    model = d.model_bytes + 6 * entries
    charge = 8 * model / STREAM
    sel_charged = sel_rate + charge
    vt = c.viterbi_chunked(va, d, bg, uni)
    cost = c.bits_of(vt, d, bg, uni)
    sh = c.shape(vt, len(va), cost)
    root = np.exp2(-bc)
    root /= root.sum()
    pv = t.pieces(vt, {})
    o2 = t.ord2_model(pf, root, 8192) if pf and pv else None
    o2_bits = t.ord2_bits(pv, o2) if o2 else 0.0
    prev = [None] + vt[:-1]
    row_bits = sum(c.cost_of(s, d, bg, uni, p) for s, p in zip(vt, prev) if len(s) == 1)
    o2_size = o2.size() if o2 else 0
    rate = cost / len(va)
    n_val = len(va)
    out = {
        "corpus": "enwik8_100m",
        "dict": k,
        "table": table,
        "stream": STREAM,
        "val_bytes": n_val,
        "train_bytes": 90_000_000,
        "entries": entries,
        "model_bytes": model,
        "charge": charge,
        "coverage": sh["covered"],
        "rate": rate,
        "charged": rate + charge,
        "tokens": sh["tokens"],
        "calls": sh["calls"],
        "sel_rate": sel_rate,
        "sel_charged": sel_charged,
        "tail_bytes": sum(len(s) for s in vt if len(s) == 1),
        "ord2_kB": o2_size / 1e3,
        "ord2_bpb": o2_bits / max(sum(len(p.raw) for p in pv), 1),
        "row_bpb": row_bits / max(sum(len(s) for s in vt if len(s) == 1), 1),
        "ord2_projected": (
            rate - (row_bits - o2_bits) / n_val + 8 * (model + o2_size) / STREAM
        ),
        "seconds": time.time() - t0,
    }
    if "ppmd" not in _CODEC:
        import bz2
        import lzma
        import zlib

        _CODEC["zlib"] = len(zlib.compress(va))
        _CODEC["bz2"] = len(bz2.compress(va, 9))
        _CODEC["lzma"] = len(lzma.compress(va, preset=9 | lzma.PRESET_EXTREME))
        _CODEC["ppmd"] = c._ppmd(va)
    out["codec"] = {name: 8 * x / n_val for name, x in _CODEC.items()}
    print(
        f"  k={k:6d} t={table:5d} cov={out['coverage']:.3f} "
        f"rate={rate:.3f} model={model / 1e3:.1f} kB charged={out['charged']:.3f} "
        f"sel={sel_charged:.3f} ord2={out['ord2_bpb']:.3f} "
        f"row={out['row_bpb']:.3f} ({out['seconds'] / 60:.1f} min)",
        flush=True,
    )
    return out


def measure(k: int, table: int, val_bytes: int) -> dict:
    t0 = time.time()
    raw = corpora.fetch_enwik8()[:STREAM]
    tr = raw[:90_000_000]
    va = raw[90_000_000:][:val_bytes]
    del raw
    s10 = c.split("enwik8_10m", "union")
    toks = s10["top"][:k]
    del s10
    counts = count_tokens(tr, toks)
    torch.cuda.empty_cache()
    usage = collections.Counter({s: counts[s] for s in toks if counts[s] > 0})
    top = sorted(usage, key=lambda s: -usage[s])
    cut = len(tr) // 10 * 9
    fit, sel = tr[:cut], tr[cut:]
    del tr
    bc = v.byte_costs(fit)
    d = c.dict_at(usage, top, k, bc)
    return _evaluate(d, fit, sel, va, k, table, t0)


def measure_full(k: int, table: int, val_bytes: int, scan_path: Path) -> dict:
    """The 100 MB vocabulary: the union seed plus the 90 MB heavy hitters,
    one EM unigram on the fit, then the same evaluation."""
    t0 = time.time()
    raw = corpora.fetch_enwik8()[:STREAM]
    tr = raw[:90_000_000]
    va = raw[90_000_000:][:val_bytes]
    del raw
    cand = load_scan(scan_path)
    s10 = c.split("enwik8_10m", "union")
    seed = s10["top"][:65536]
    del s10
    seedc = count_tokens(tr, seed)
    torch.cuda.empty_cache()
    by_len: dict[int, list[tuple[int, bytes]]] = collections.defaultdict(list)
    for s, n in cand.items():
        by_len[len(s)].append((n, s))
    del cand
    pool = collections.Counter(seedc)
    for items in by_len.values():
        for n, s in sorted(items, reverse=True)[:8192]:
            pool[s] = max(pool[s], n)
    del by_len
    cut = len(tr) // 10 * 9
    fit, sel = tr[:cut], tr[cut:]
    del tr
    bc = v.byte_costs(fit)
    n0 = sum(pool.values())
    keys = list(pool)
    d0 = v.assemble(keys, [-math.log2(pool[s] / n0) for s in keys], bc)
    usage = collections.Counter(cover_chunked(fit, d0))
    del d0
    top = sorted(usage, key=lambda s: -usage[s])
    keep = top[:k]
    n = sum(usage[s] for s in keep)
    d = v.assemble(keep, [-math.log2(usage[s] / n) for s in keep], bc)
    return _evaluate(d, fit, sel, va, k, table, t0)


def report(res: list[dict], path: Path, full: bool) -> None:
    if full:
        voc = [
            "The vocabulary is the union seed (64k, selected on the 10 MB stream)",
            "plus the 90 MB heavy hitters (count >= 32, per-length top 8k), one",
            "EM unigram on the 81 MB fit. The table is fitted on the first 90% of",
            "the train and chosen on the held-out 10%; the charge is over 100 MB;",
            "the val is 10 MB. The `ord2` column is the order-2 tail on the DP",
            "path, charged; `projected` adds it. The codec record at this size is",
            "not run: the 10 MB val encode is out of Python's throughput.",
        ]
    else:
        voc = [
            "The union vocabulary is selected on the 10 MB stream; its counts are",
            "refit on the 90 MB train with an exact GPU scan of the known tokens.",
            "The table is fitted on the first 90% of the train and chosen on the",
            "held-out 10%; the charge is over 100 MB; the val is 10 MB. The `ord2`",
            "column is the order-2 tail on the DP path, charged; `projected` adds it.",
            "The codec record at this size is not run: the 10 MB val encode is out",
            "of Python's throughput. These are rate/charge measurements.",
        ]
    lines = [
        "# H-NCM v9: N10, enwik8 at 100 MB",
        "",
        *voc,
        "",
        "| dict | table | coverage | rate | model kB | charge | charged | sel charged | tail bytes | ord2 bpb | row bpb | ord2 projected | ppmd |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in res:
        lines.append(
            f"| {r['dict']} | {r['table']} | {r['coverage']:.3f} | "
            f"{r['rate']:.3f} | {r['model_bytes'] / 1e3:.1f} | "
            f"{r['charge']:.3f} | {r['charged']:.3f} | {r['sel_charged']:.3f} | "
            f"{r['tail_bytes']} | {r['ord2_bpb']:.3f} | {r['row_bpb']:.3f} | "
            f"{r['ord2_projected']:.3f} | {r['codec']['ppmd']:.3f} |"
        )
    lines += [
        "",
        "The 10 MB targets were 2.94 (recorded); the 100 MB gate is 1.90. PPMd",
        "on the same 10 MB val is measured by `codec._ppmd`.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run(
    kinds: tuple[int, ...],
    tables: tuple[int, ...],
    val: int,
    out: Path,
    full: bool = False,
    scan_path: Path = Path("runs/v9/scan100.npz"),
) -> dict:
    res = []
    for k in kinds:
        for table in tables:
            res.append(
                measure_full(k, table, val, scan_path)
                if full
                else measure(k, table, val)
            )
    best = min(res, key=lambda r: r["sel_charged"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"format": 9, "full": full, "rows": res, "best": best}, indent=1)
    )
    report(res, out.parent / "ENWIK8-100M.md", full)
    return {"rows": res, "best": best}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="H-NCM N10 at 100 MB")
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--dict", default=",".join(str(k) for k in KINDS))
    r.add_argument("--table", default=",".join(str(k) for k in TABLES))
    r.add_argument("--val", type=int, default=VAL)
    r.add_argument("--out", default="runs/v9/enwik8_100m.json")
    r.add_argument("--full", action="store_true")
    r.add_argument("--scan", default="runs/v9/scan100.npz")
    s = sub.add_parser("scan")
    s.add_argument("--min-count", type=int, default=32)
    s.add_argument("--out", default="runs/v9/scan100.npz")
    sub.add_parser("selftest")
    a = p.parse_args(argv)
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "scan":
        scan(a.min_count, Path(a.out))
    elif a.cmd == "run":
        run(
            tuple(int(x) for x in a.dict.split(",")),
            tuple(int(x) for x in a.table.split(",")),
            a.val,
            Path(a.out),
            a.full,
            Path(a.scan),
        )
    else:
        p.print_help()
        return 1
    return 0


def selftest() -> None:
    bs = b"abcabcabcaabcd"
    toks = [b"abc", b"ab", b"bc", b"zzz"]
    c = count_tokens(bs, toks, chunk=5) if torch.cuda.is_available() else {}
    if c:
        assert c[b"abc"] == 4, c
        assert c[b"ab"] == 4, c
        assert c[b"bc"] == 4, c
        assert c[b"zzz"] == 0, c
    print("n10 selftest: ok")


if __name__ == "__main__":
    raise SystemExit(main())
