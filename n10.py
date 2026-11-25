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
import pickle
import time
from pathlib import Path

import numpy as np
import torch

import codec
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


def _evaluate(
    d: v.Dict, fit: bytes, sel: bytes, va: bytes, k: int, table: int, t0: float
) -> dict:
    bc = v.byte_costs(fit)
    ft = c.cover_chunked(fit, d)
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


def build_full(val_bytes: int, scan_path: Path) -> dict:
    """The 100 MB split: the union seed plus the 90 MB heavy hitters, one EM
    unigram on the 81 MB fit. Returns the split `codec.fit_split` consumes."""
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
    usage = collections.Counter(c.cover_chunked(fit, d0))
    del d0
    top = sorted(usage, key=lambda s: -usage[s])
    return {
        "fit": fit,
        "sel": sel,
        "va": va,
        "stream": STREAM,
        "bc": bc,
        "usage": usage,
        "top": top,
    }


def build_seed(val_bytes: int, k: int) -> tuple[dict, v.Dict]:
    """The plan's N10 baseline: the 10 MB union vocabulary, its counts refit
    on the 81 MB fit with one EM round, then the same codec."""
    raw = corpora.fetch_enwik8()[:STREAM]
    tr = raw[:90_000_000]
    va = raw[90_000_000:][:val_bytes]
    del raw
    s10 = c.split("enwik8_10m", "union")
    cut = len(tr) // 10 * 9
    fit, sel = tr[:cut], tr[cut:]
    del tr
    bc = v.byte_costs(fit)
    d0 = c.dict_at(s10["usage"], s10["top"], k, bc)
    del s10
    ft = c.cover_chunked(fit, d0)
    u = collections.Counter(ft)
    n = sum(u[x] for x in d0.strings)
    keep = [x for x in d0.strings if u[x] > 0 or len(x) == 1]
    cost = [float(bc[x[0]]) if len(x) == 1 else -math.log2(u[x] / n) for x in keep]
    s = {"fit": fit, "sel": sel, "va": va, "stream": STREAM, "bc": bc}
    return s, v.assemble(keep, cost, bc)


def fit_dict(s: dict, k: int, table: int) -> v.Dict:
    """One EM round on the fit: the costs from the final tiling, not the pool."""
    d0, _, _ = c.rebuild(s, k, table)
    s.pop("usage", None)
    s.pop("top", None)
    ft = c.cover_chunked(s["fit"], d0)
    u = collections.Counter(ft)
    n = sum(u[x] for x in d0.strings)
    keep = [x for x in d0.strings if u[x] > 0 or len(x) == 1]
    cost = [float(s["bc"][x[0]]) if len(x) == 1 else -math.log2(u[x] / n) for x in keep]
    return v.assemble(keep, cost, s["bc"])


def measure_full(k: int, table: int, val_bytes: int, scan_path: Path) -> dict:
    """The 100 MB vocabulary, then the same evaluation."""
    t0 = time.time()
    s = build_full(val_bytes, scan_path)
    d = fit_dict(s, k, table)
    return _evaluate(d, s["fit"], s["sel"], s["va"], k, table, t0)


def proof(
    val_bytes: int,
    k: int,
    table: int,
    scan_path: Path,
    out: Path,
    seed: bool = False,
) -> dict:
    """The end-to-end artifact: fit on 81/90 MB, encode and decode the val."""
    t0 = time.time()
    mode = "seed" if seed else "full"
    cache = out.parent / f"fit-{mode}-{k}-{table}-{val_bytes}.pkl"
    if cache.exists():
        with cache.open("rb") as fh:
            fitted = pickle.load(fh)
        print("proof: cached fit", flush=True)
    else:
        if seed:
            s, d = build_seed(val_bytes, k)
            print("proof: seed split", flush=True)
        else:
            s = build_full(val_bytes, scan_path)
            print("proof: split built", flush=True)
            d = fit_dict(s, k, table)
            print("proof: dictionary fitted", flush=True)
        corp = codec.fit_split(s, {"vocab": mode, "dict": k, "table": table}, d)
        print("proof: model fitted", flush=True)
        fitted = {
            "blob": corp["pack_concept"],
            "toks": corp["val_toks"],
            "va": s["va"],
            "ref": corp["ref_concept"],
        }
        with cache.open("wb") as fh:
            pickle.dump(fitted, fh)
    blob = fitted["blob"]
    model = codec.unpack(blob)
    n = len(fitted["va"])
    t1 = time.time()
    payload, enc = codec.encode(model, fitted["toks"], "concept")
    enc_s = time.time() - t1
    print("proof: encoded", flush=True)
    t2 = time.time()
    got, dec = codec.decode(model, payload, n, "concept")
    dec_s = time.time() - t2
    print("proof: decoded", flush=True)
    hdr = len(codec.header(n, 0, len(blob), "concept"))
    res = {
        "corpus": "enwik8_100m",
        "dict": k,
        "table": table,
        "val_bytes": n,
        "train_bytes": STREAM - n,
        "payload_bytes": len(payload),
        "model_bytes": len(blob),
        "header_bytes": hdr,
        "rate": 8 * len(payload) / n,
        "entropy": enc["entropy"] / n,
        "ref": fitted["ref"] / n,
        "delta": (enc["entropy"] - fitted["ref"]) / n,
        "shared": 8 * (len(payload) + hdr) / n + 8 * len(blob) / STREAM,
        "self": 8 * (len(payload) + hdr) / n + 8 * len(blob) / n,
        "calls": enc["calls"],
        "tokens": len(fitted["toks"]),
        "roundtrip": got == fitted["va"],
        "encode_s": enc_s,
        "decode_s": dec_s,
        "encode_mb_s": n / 1e6 / max(enc_s, 1e-9),
        "decode_mb_s": n / 1e6 / max(dec_s, 1e-9),
        "seconds": time.time() - t0,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    (out.parent / "PROOF-100M.md").write_text(
        "# H-NCM v9: the end-to-end proof (enwik8_100m)\n\n"
        f"The model is fitted on 81 MB of the 90 MB train; the held-out "
        f"{res['val_bytes']} B val is encoded and decoded by the concept arm. "
        f"The bitstream round-trips: **{res['roundtrip']}**.\n\n"
        f"- payload: {res['payload_bytes']} B = {res['rate']:.3f} bpb\n"
        f"- model: {res['model_bytes'] / 1e3:.1f} kB = "
        f"{8 * res['model_bytes'] / STREAM:.3f} bpb over 100 MB\n"
        f"- recorded shared: **{res['shared']:.3f} bpb**\n"
        f"- calls: {res['calls']}; tokens: {res['tokens']}\n"
        f"- encode/decode: {res['encode_mb_s']:.2f} / "
        f"{res['decode_mb_s']:.2f} MB/s\n"
        f"- model: {res['ref']:.3f}; entropy {res['entropy']:.3f}; "
        f"gap {res['delta']:+.3f}\n"
    )
    print(
        f"proof: k={k} t={table} RT={'ok' if res['roundtrip'] else 'FAIL'} "
        f"payload={len(payload)} B rate={res['rate']:.3f} "
        f"model={len(blob) / 1e3:.1f} kB shared={res['shared']:.3f} "
        f"calls={res['calls']} enc={res['encode_mb_s']:.2f} MB/s "
        f"dec={res['decode_mb_s']:.2f} MB/s ({res['seconds'] / 60:.1f} min)",
        flush=True,
    )
    return res


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
    q = sub.add_parser("proof")
    q.add_argument("--val", type=int, default=1_000_000)
    q.add_argument("--dict", type=int, default=32768)
    q.add_argument("--table", type=int, default=16384)
    q.add_argument("--scan", default="runs/v9/scan100.npz")
    q.add_argument("--seed", action="store_true")
    q.add_argument("--out", default="runs/v9/proof-100m.json")
    sub.add_parser("selftest")
    a = p.parse_args(argv)
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "scan":
        scan(a.min_count, Path(a.out))
    elif a.cmd == "proof":
        proof(a.val, a.dict, a.table, Path(a.scan), Path(a.out), a.seed)
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
