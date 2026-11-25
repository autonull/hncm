"""N1a: the coverage probe. Does the dictionary carry the stream at 10 MB?

Fit the dictionary at 2-4 sizes on the train, tile the val, fit the shared tail
on the train's uncovered spans, and report coverage, the covered rate, the tail
rate, the charge and the predicted total:

    predicted = cov * covered + (1 - cov) * tail + 8 * model / stream

The cover is the unigram DP (`hncm7.cover`), which is a lower bound on the
first-order DP's coverage and a higher bound on its covered rate; the tail is
the rank-r factorization of the byte-context table, with r chosen on the
held-out 10% of the train. Everything is fitted on the train; the val only
reports. The probe is the cheap gate before the 10 MB codec: it decides whether
the dictionary or the tail is the story at scale.
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np

import codec as z
import concept_lm as c
import hncm7 as v
import tail_coder as t

SIZES = (2048, 4096, 8192, 16384)
TAIL = (1, 2, 4, 8, 16, 32, 64)


def tail_bits(pieces: list[t.Piece], logp, lens, n: int) -> float:
    bits = 0.0
    for p in pieces:
        bits += z.run_bits(p.raw, logp)
        bits += t.prior_bits(lens, len(p.raw), n, c.CAP)
    return bits


def tail_model(
    fit_toks: list[bytes],
    sel_toks: list[bytes],
    val_toks: list[bytes],
    root: np.ndarray,
) -> dict:
    """The shared tail fitted on the train's uncovered spans, r chosen on sel.

    Two arms, both charged: `svd`, the rank-r factorization of the 257x256
    byte table (the shared low-rank model), and `ord2`, the pruned order-2
    count table with the order-1 backoff (the real two preceding bytes).
    """
    pf, ps, pv = (
        t.pieces(fit_toks, {}),
        t.pieces(sel_toks, {}),
        t.pieces(val_toks, {}),
    )
    n = sum(len(p.raw) for p in pf)
    lens = collections.Counter(len(p.raw) for p in pf)
    out = {
        "r": 0,
        "size": 0,
        "bits": 0.0,
        "bytes": sum(len(p.raw) for p in pv),
        "ord2": {"top": 0, "size": 0, "bits": 0.0},
    }
    if not pf or n < 1000 or not pv:
        for p in pv:
            out["bits"] += t.prior_bits(lens, len(p.raw), n, c.CAP)
        out["ord2"]["bits"] = out["bits"]
        return out
    best2 = None
    for k in t.ORD2:
        m = t.ord2_model(pf, root, k)
        sb = sum(m.bits(p) + t.prior_bits(lens, len(p.raw), n, c.CAP) for p in ps)
        sel = sb / max(len(sel_toks), 1) + 8 * m.size() / (
            len(fit_toks) + len(sel_toks)
        )
        if best2 is None or sel < best2[0]:
            best2 = (sel, k, m)
    _, k2, m2 = best2
    out["ord2"] = {
        "top": k2,
        "size": m2.size(),
        "bits": sum(m2.bits(p) + t.prior_bits(lens, len(p.raw), n, c.CAP) for p in pv),
    }
    counts, _ = t.byte_matrix(pf, root)
    best = None
    for r in TAIL:
        a, b, scale = z.factors(counts, r)
        logp = z.logp_of(a, b, root, scale)
        sb = tail_bits(ps, logp, lens, n)
        size = 34 + (257 + 256) * r * 2
        sel = sb / max(len(sel_toks), 1) + 8 * size / (len(fit_toks) + len(sel_toks))
        if best is None or sel < best[0]:
            best = (sel, r, logp, size)
    _, r, logp, size = best
    out.update(r=r, size=size, bits=tail_bits(pv, logp, lens, n), lens=lens, n=n)
    return out


def measure(name: str, vocab: str, sizes: tuple[int, ...]) -> dict:
    t0 = time.time()
    s = c.split(name, vocab)
    fit, sel, va, stream = s["fit"], s["sel"], s["va"], s["stream"]
    root = np.exp2(-s["bc"])
    root /= root.sum()
    rows = []
    print(
        f"{name}: fit={len(fit)} sel={len(sel)} va={len(va)} stream={stream} "
        f"split={time.time() - t0:.1f}s",
        flush=True,
    )
    for k in sorted({min(k, len(s["top"])) for k in sizes}):
        t1 = time.time()
        d = c.dict_at(s["usage"], s["top"], k, s["bc"])
        fit_toks = v.cover(fit, d, 0.0)
        sel_toks = v.cover(sel, d, 0.0)
        val_toks = v.cover(va, d, 0.0)
        cov_bytes = sum(len(x) for x in val_toks if len(x) > 1)
        cov = cov_bytes / len(va)
        cov_bits = sum(float(d.cost_of(x)) for x in val_toks if len(x) > 1)
        cov_rate = cov_bits / max(cov_bytes, 1)
        tail = tail_model(fit_toks, sel_toks, val_toks, root)
        tail_rate = tail["bits"] / max(tail["bytes"], 1)
        calls = 1 + len([p for p in t.pieces(val_toks, {})])
        model = d.model_bytes + tail["size"]
        pred = cov * cov_rate + (1 - cov) * tail_rate + 8 * model / stream
        t2 = tail["ord2"]
        tail2_rate = t2["bits"] / max(tail["bytes"], 1) if t2["top"] else float("nan")
        model2 = d.model_bytes + t2["size"]
        pred2 = (
            cov * cov_rate + (1 - cov) * tail2_rate + 8 * model2 / stream
            if t2["top"]
            else float("nan")
        )
        rows.append(
            {
                "dict": k,
                "coverage": cov,
                "covered_bpb": cov_rate,
                "tail_bpb": tail_rate,
                "tail2_bpb": tail2_rate,
                "tail_bytes": tail["bytes"],
                "tail_spans": calls - 1,
                "r": tail["r"],
                "ord2": t2["top"],
                "dict_bytes": d.model_bytes,
                "tail_bytes_model": tail["size"],
                "tail2_bytes_model": t2["size"],
                "model_bytes": model,
                "model2_bytes": model2,
                "predicted": pred,
                "predicted2": pred2,
                "calls": calls,
            }
        )
        print(
            f"  dict={k:6d} cov={cov:.3f} covered={cov_rate:.3f} "
            f"tail={tail_rate:.3f} ord2={tail2_rate:.3f} top={t2['top']:5d} "
            f"model={8 * model / stream:.3f} pred={pred:.3f} pred2={pred2:.3f} "
            f"({time.time() - t1:.1f}s)",
            flush=True,
        )
    return {
        "corpus": name,
        "vocab": vocab,
        "sha": s.get("sha", ""),
        "stream_bytes": stream,
        "val_bytes": len(va),
        "rows": rows,
    }


def gates(res: dict) -> list[dict]:
    out = []
    for name, e in res["corpora"].items():
        best = min(e["rows"], key=lambda r: r["predicted"])
        out.append(
            {
                "id": "C1",
                "corpus": name,
                "value": f"dict={best['dict']} cov={best['coverage']:.3f} "
                f"pred={best['predicted']:.3f} model={best['model_bytes'] / 1e3:.1f} kB",
                "status": "pass" if best["coverage"] > 0.5 else "fail",
            }
        )
        r2 = min(
            (r for r in e["rows"] if r["ord2"]),
            key=lambda r: r["predicted2"],
            default=None,
        )
        if r2:
            out.append(
                {
                    "id": "C2",
                    "corpus": name,
                    "value": f"ord2={r2['tail2_bpb']:.3f} "
                    f"svd={best['tail_bpb']:.3f} pred2={r2['predicted2']:.3f} "
                    f"top={r2['ord2']}",
                    "status": "pass" if r2["tail2_bpb"] < best["tail_bpb"] else "fail",
                }
            )
    return out


def report(res: dict, path: Path) -> None:
    lines = [
        "# H-NCM v9: the coverage probe (N8, N9)",
        "",
        "The dictionary is fitted on the train and the val is tiled with the unigram",
        "cover (a lower bound on the first-order DP's coverage and a higher bound on",
        "its covered rate). Two shared tails are fitted on the train's uncovered",
        "spans and chosen on the held-out 10% of the train: the rank-r",
        "factorization of the byte table, and the pruned order-2 counts with the",
        "order-1 backoff. `predicted = cov * covered + (1-cov) * tail +",
        "8 * model / stream`; `predicted ord2` repeats it with the count tail. The",
        "concept table's charge is not included, so both totals are lower bounds.",
        "",
        "| corpus | dict | coverage | covered bpb | tail bpb | ord2 bpb | ord2 top | tail bytes | r | model kB | charge | predicted | predicted ord2 | calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, e in res["corpora"].items():
        for r in e["rows"]:
            lines.append(
                f"| {name} | {r['dict']} | {r['coverage']:.3f} | "
                f"{r['covered_bpb']:.3f} | {r['tail_bpb']:.3f} | "
                f"{r['tail2_bpb']:.3f} | {r['ord2']} | {r['tail_bytes']} | "
                f"{r['r']} | {r['model_bytes'] / 1e3:.1f} | "
                f"{8 * r['model_bytes'] / e['stream_bytes']:.3f} | "
                f"{r['predicted']:.3f} | {r['predicted2']:.3f} | {r['calls']} |"
            )
    lines += [
        "",
        "## Gates",
        "",
        "| gate | corpus | value | status |",
        "|---|---|---|---|",
    ]
    for g in gates(res):
        lines.append(
            f"| {g['id']} | {g['corpus']} | {g['value']} | **{g['status']}** |"
        )
    lines += [
        "",
        "C1 is the probe's own gate: the best budget covers more than half of the",
        "val. The decision it feeds: if coverage rises with the stream, the",
        "dictionary and the concept LM are the story and N1b runs them at 10 MB; if",
        "coverage stalls, the tail is the story and N2/M5 (the learned tail) runs",
        "first.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run(names: list[str], out: Path, vocab: str, sizes: tuple[int, ...]) -> dict:
    res = {"format": 8, "vocab": vocab, "sizes": list(sizes), "corpora": {}}
    for name in names:
        res["corpora"][name] = measure(name, vocab, sizes)
    for g in gates(res):
        print(f"  {g['id']} {g['status']} {g['corpus']}: {g['value']}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    report(res, out.parent / "COVERAGE.md")
    return res


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="H-NCM coverage probe")
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--corpus", default="enwik8_10m")
    r.add_argument("--vocab", default="substr")
    r.add_argument("--sizes", default="2048,4096,8192,16384")
    r.add_argument("--out", default="runs/v8/coverage.json")
    sub.add_parser("selftest")
    a = p.parse_args(argv)
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "run":
        run(
            [x for x in a.corpus.split(",") if x],
            Path(a.out),
            a.vocab,
            tuple(int(x) for x in a.sizes.split(",")),
        )
    else:
        p.print_help()
        return 1
    return 0


def selftest() -> None:
    toks = [b"a", b"b", b"c"]
    pf = t.pieces(toks, {})
    assert sum(len(p.raw) for p in pf) == 3
    assert z.run_bits(b"ab", np.log2(np.ones((257, 256)) / 256)) == 16.0
    rng = np.random.RandomState(0)
    logp = np.log2(rng.dirichlet(np.ones(256), size=257))
    assert abs(z.run_bits(b"", logp)) < 1e-12
    assert abs(z.run_bits(bytes([0, 1]), logp) + logp[256, 0] + logp[0, 1]) < 1e-9
    print("probe selftest: ok")


if __name__ == "__main__":
    raise SystemExit(main())
