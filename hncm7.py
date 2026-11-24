"""H-NCM v7: the span code is a dictionary index, not a dense latent.

v6's bottleneck is measured, not suspected. A 12-bit scalar quantization of a
pooled span *mean* is permutation-invariant: the mean cannot tell "abab" from
"aabb", so the code carries ~1 bpb while it costs 3.1 bpb. `total = code +
residual` is then monotone in the code rate, and the optimizer's optimum is to
collapse the code to a constant (measured: `total=4.79`, i.e. no code). The
1200-step slope is 0.30, and it holds for every pool, tiling and rate.

v7 changes what the code *is*. The code is the *index* of a span in a *free*
dictionary (part of the model, like the byte unigram table). The decoder writes
the span's exact bytes, so the residual is zero for dictionary spans; spans that
are not in the dictionary pay their bytes at the byte unigram. The only paid bits
are the indices and the fallback bytes.

The tiling is inference: a shortest-path DP solves the exact cover of the window
with the index costs. The greedy arm runs the *same* cost model with a
longest-match cover, so the DP/greedy gap isolates solving from guessing. The
oracle splits at the true word onsets (the ceiling, free). The fixed grid is what
a chunker would do. Every arm pays the same cost model.

No codec training, no GPU, no latent. That is the point: the win is the
representation (a dictionary index), not a bigger net.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import hncm6 as base  # noqa: E402  (corpora and onsets only)

FORMAT = 7
CAP = 16
ARMS = ("DP", "greedy", "oracle", "fixed4", "fixed8", "fixed16", "random")
LAMBDAS = (0.0, 1.0, 2.0, 4.0)


# %%
def spans(bs: bytes, on: np.ndarray) -> list[bytes]:
    s = np.flatnonzero(on)
    e = np.append(s[1:], len(bs))
    return [bs[a:b] for a, b in zip(s, e)]


def byte_costs(tr: bytes) -> np.ndarray:
    c = np.bincount(np.frombuffer(tr, dtype=np.uint8), minlength=256).astype(float) + 1
    return -np.log2(c / c.sum())


# %%
@dataclass(frozen=True)
class Dict:
    """A free span dictionary with a unigram index model.

    `cost[i]` is bits for token `i`. Unknown spans pay their bytes at the byte
    unigram, i.e. the marginal. The dictionary is part of the model.
    """

    ids: dict[bytes, int]
    strings: list[bytes]
    cost: list[float]
    byte_cost: np.ndarray
    trie: dict
    model_bytes: int

    def cost_of(self, s: bytes) -> float:
        i = self.ids.get(s)
        return (
            self.cost[i]
            if i is not None
            else float(self.byte_cost[np.frombuffer(s, np.uint8)].sum())
        )


def build_dict(tr: bytes, on: np.ndarray, bc: np.ndarray, min_count: int = 1) -> Dict:
    """A free dictionary, pruned by train frequency.

    `min_count` trades model bytes for index rate: a singleton word costs 4 bytes
    of model and saves nothing when it recurs once. The single bytes are always
    present, so every span has a token.
    """
    words = spans(tr, on)
    counts = collections.Counter(w for w in words if len(w) <= CAP)
    keep = [w for w in counts if counts[w] >= min_count]
    ids = {w: i for i, w in enumerate(keep)}
    strings = list(keep)
    for b in range(256):
        s = bytes([b])
        if s not in ids:
            ids[s] = len(strings)
            strings.append(s)
    n = sum(counts[w] for w in keep)
    cost = [float(sum(bc[b] for b in s)) for s in strings]
    for w in keep:
        cost[ids[w]] = -math.log2(counts[w] / n)
    trie: dict = {}
    for s, i in ids.items():
        node = trie
        for b in s:
            node = node.setdefault(b, {})
        node["#"] = i
    # the model is data: 1 byte length + 2 bytes count per word string, plus the
    # 256-entry byte table (2 bytes each). Single bytes are that table.
    model_bytes = 512 + sum(len(s) + 3 for s in strings if len(s) > 1)
    return Dict(ids, strings, cost, bc, trie, model_bytes)


# %%
def cover(bs: bytes, d: Dict, lam: float, greedy: bool = False) -> list[bytes]:
    L = len(bs)
    if greedy:
        out, t = [], 0
        while t < L:
            node, hit = d.trie, None
            for e in range(t, min(L, t + CAP)):
                if bs[e] not in node:
                    break
                node = node[bs[e]]
                if "#" in node:
                    hit = e + 1
            ell = hit - t if hit else 1
            out.append(bs[t : t + ell])
            t += ell
        return out
    best = [float("inf")] * (L + 1)
    back = [0] * (L + 1)
    best[0] = 0.0
    for t in range(L):
        if best[t] == float("inf"):
            continue
        node = d.trie
        for e in range(t, min(L, t + CAP)):
            if bs[e] not in node:
                break
            node = node[bs[e]]
            if "#" in node:
                ell = e + 1 - t
                v = best[t] + d.cost_of(bs[t : e + 1]) + lam * ell
                if v < best[e + 1]:
                    best[e + 1], back[e + 1] = v, ell
    out, e = [], L
    while e > 0:
        ell = back[e]
        out.append(bs[e - ell : e])
        e -= ell
    return out[::-1]


def mask_of(toks: list[bytes], L: int) -> list[bool]:
    m, t = [False] * L, 0
    for s in toks:
        m[t] = True
        t += len(s)
    if t != L:
        raise RuntimeError(f"cover failed: {t} != {L}")
    return m


def bits_of(toks: list[bytes], d: Dict) -> float:
    return float(sum(d.cost_of(s) for s in toks))


# %%
def arm_tokens(arm: str, bs: bytes, d: Dict, lam: float) -> list[bytes]:
    L = len(bs)
    if arm == "oracle":
        return spans(bs, base.onsets(NAME, np.frombuffer(bs, np.uint8)))
    if arm.startswith("fixed"):
        step = int(arm[5:])
        return [bs[t : t + step] for t in range(0, L - step + 1, step)]
    if arm == "random":
        rng = np.random.RandomState(0)
        out, t = [], 0
        while t < L:
            ell = int(rng.choice([e for e in LENGTHS if e <= L - t]))
            out.append(bs[t : t + ell])
            t += ell
        return out
    return cover(bs, d, lam, greedy=arm == "greedy")


LENGTHS = (1, 2, 4, 8, 16)
NAME = "synseg"


def evaluate(name: str, va: bytes, d: Dict) -> list[dict]:
    global NAME
    NAME = name
    n = len(va)
    out = []
    for arm in ARMS:
        for lam in (
            LAMBDAS if arm not in ("oracle", "fixed4", "fixed8", "fixed16") else (0.0,)
        ):
            toks = arm_tokens(arm, va, d, lam)
            bits = bits_of(toks, d)
            if arm == "oracle":
                ref = mask_of(toks, n)
            out.append(
                {
                    "arm": arm,
                    "lambda": lam,
                    "total_bpb": bits / n,
                    "tokens": len(toks),
                    "bytes_per_token": n / len(toks),
                    "toks": toks,
                }
            )
    ref = mask_of(next(r for r in out if r["arm"] == "oracle")["toks"], n)
    for r in out:
        m = mask_of(r["toks"], n)
        inter = sum(a and b for a, b in zip(m, ref))
        p, q = inter / max(sum(m), 1), inter / max(sum(ref), 1)
        r["align"] = 2 * p * q / max(p + q, 1e-9)
        del r["toks"]
    return out


# %%
def gates(res: dict) -> list[dict]:
    out = []
    for name, e in res["corpora"].items():
        rows = collections.defaultdict(dict)
        for r in e["runs"]:
            rows[r["arm"]][r["lambda"]] = r
        uni, gz = e["unigram"], e["gzip"]
        for lam in LAMBDAS:
            dp = rows["DP"].get(lam)
            if not dp:
                continue
            fixed = rows["fixed4"][0.0]
            greedy = rows["greedy"].get(lam)
            oracle = rows["oracle"][0.0]
            out.append(
                {
                    "id": "G0",
                    "corpus": name,
                    "value": f"DP={dp['total_bpb']:.3f} unigram={uni:.3f} lam={lam:.1f}",
                    "status": "pass" if dp["total_bpb"] < uni else "fail",
                }
            )
            out.append(
                {
                    "id": "G1",
                    "corpus": name,
                    "value": f"DP={dp['total_bpb']:.3f} gzip={gz:.3f} lam={lam:.1f}",
                    "status": "pass" if dp["total_bpb"] < gz else "fail",
                }
            )
            out.append(
                {
                    "id": "G0b",
                    "corpus": name,
                    "value": f"charged DP={dp['charged_bpb']:.3f} gzip={gz:.3f} "
                    f"(model={e['model_bpb']:.3f} bpb at {e['stream_bytes'] / 1e6:.2f} MB)",
                    "status": "pass" if dp["charged_bpb"] < gz else "fail",
                }
            )
            out.append(
                {
                    "id": "G4",
                    "corpus": name,
                    "value": f"oracle={oracle['total_bpb']:.3f} fixed={fixed['total_bpb']:.3f}",
                    "status": "pass"
                    if oracle["total_bpb"] < fixed["total_bpb"]
                    else "fail",
                }
            )
            out.append(
                {
                    "id": "G5",
                    "corpus": name,
                    "value": f"DP={dp['total_bpb']:.3f} fixed={fixed['total_bpb']:.3f} "
                    f"align={dp['align']:.3f}/{fixed['align']:.3f} lam={lam:.1f}",
                    "status": "pass"
                    if dp["total_bpb"] < fixed["total_bpb"]
                    and dp["align"] > fixed["align"] + 0.05
                    else "fail",
                }
            )
            out.append(
                {
                    "id": "G5c",
                    "corpus": name,
                    "value": f"DP={dp['total_bpb']:.3f} greedy={greedy['total_bpb']:.3f} lam={lam:.1f}",
                    "status": "pass"
                    if dp["total_bpb"] <= greedy["total_bpb"]
                    else "fail",
                }
            )
    return out


def report(res: dict, path: Path) -> None:
    lines = ["# H-NCM v7 - solved tiling over a pruned span dictionary", ""]
    for name, e in res["corpora"].items():
        cross = {
            k: (f"{v / 1e6:.2f} MB" if v else "never")
            for k, v in e["crossing_bytes"].items()
        }
        lines += [
            f"## {name}",
            "",
            f"stream {e['stream_bytes'] / 1e6:.2f} MB; chosen min_count {e['min_count']}; "
            f"dictionary {e['dict']} spans, {e['model_bytes'] / 1e3:.1f} kB "
            f"({e['model_bpb']:.3f} bpb); byte unigram {e['unigram']:.3f} bpb; "
            f"gzip {e['gzip']:.3f} bpb; crossover unigram {cross['unigram']}, "
            f"gzip {cross['gzip']}",
            "",
            "| min_count | dict | model kB | rate | charged |",
            "|---:|---:|---:|---:|---:|",
        ]
        for b in e["budget"]:
            lines.append(
                f"| {b['min_count']} | {b['dict']} | {b['model_bytes'] / 1e3:.1f} | "
                f"{b['rate']:.3f} | {b['charged']:.3f} |"
            )
        lines += [
            "",
            "| arm | lambda | rate | charged | tokens | bytes/token | align |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in sorted(e["runs"], key=lambda r: (r["arm"], r["lambda"])):
            lines.append(
                f"| {r['arm']} | {r['lambda']:.1f} | {r['total_bpb']:.3f} | "
                f"{r['charged_bpb']:.3f} | {r['tokens']} | "
                f"{r['bytes_per_token']:.2f} | {r['align']:.3f} |"
            )
        lines.append("")
    lines += ["## Gates", "", "| gate | corpus | value | status |", "|---|---|---|---|"]
    for g in gates(res):
        lines.append(
            f"| {g['id']} | {g['corpus']} | {g['value']} | **{g['status']}** |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


# %%
def run(names: list[str], out: Path) -> dict:
    res = {"format": FORMAT, "corpora": {}, "runs": []}
    for name in names:
        corpus = base.get_corpus(name)
        tr = corpus["tr"].cpu().numpy().astype(np.uint8).tobytes()
        va = corpus["va"].cpu().numpy().astype(np.uint8).tobytes()
        va = va[: len(va) // 16 * 16]
        bc = byte_costs(tr)
        on = base.onsets(name, np.frombuffer(tr, np.uint8))
        stream = len(tr) + len(va)
        best = None
        budget = []
        for mc in (1, 2, 4, 8, 16, 32):
            d = build_dict(tr, on, bc, mc)
            rate = bits_of(cover(va, d, 0.0), d) / len(va)
            charged = rate + 8 * d.model_bytes / stream
            budget.append(
                {
                    "min_count": mc,
                    "dict": len(d.strings),
                    "model_bytes": d.model_bytes,
                    "rate": rate,
                    "charged": charged,
                }
            )
            if best is None or charged < best[2]:
                best = (mc, d, charged, rate)
            print(
                f"  {name:11s} min_count={mc:2d} dict={len(d.strings):5d} "
                f"model={d.model_bytes / 1e3:7.1f} kB rate={rate:.3f} charged={charged:.3f}",
                flush=True,
            )
        mc, d, charged, rate = best
        runs = evaluate(name, va, d)
        cnt = np.bincount(np.frombuffer(tr, np.uint8), minlength=256) + 1
        pb = cnt / cnt.sum()
        uni = float(-(pb * np.log2(pb)).sum())
        gz = len(zlib.compress(va)) * 8 / len(va)
        mb = 8 * d.model_bytes / stream
        dp = next(r for r in runs if r["arm"] == "DP")
        crossings = {}
        for key, cond in (("unigram", uni), ("gzip", gz)):
            gap = cond - dp["total_bpb"]
            crossings[key] = d.model_bytes * 8 / gap if gap > 1e-9 else None
        for r in runs:
            r["charged_bpb"] = r["total_bpb"] + mb
        res["corpora"][name] = {
            "min_count": mc,
            "dict": len(d.strings),
            "model_bytes": d.model_bytes,
            "stream_bytes": stream,
            "model_bpb": mb,
            "unigram": uni,
            "gzip": gz,
            "crossing_bytes": crossings,
            "budget": budget,
            "runs": runs,
        }
        print(
            f"{name:11s} chosen min_count={mc} dict={len(d.strings):5d} "
            f"model={d.model_bytes / 1e3:7.1f} kB "
            f"({mb:.3f} bpb at {stream / 1e6:.2f} MB) unigram={uni:.3f} gzip={gz:.3f}",
            flush=True,
        )
        for r in runs:
            if r["arm"] in ("DP", "greedy", "oracle", "fixed4") or r["lambda"] == 0.0:
                print(
                    f"  {r['arm']:7s} lam={r['lambda']:.1f} rate={r['total_bpb']:.3f} "
                    f"charged={r['charged_bpb']:.3f} align={r['align']:.3f}",
                    flush=True,
                )
    gs = gates(res)
    for g in gs:
        print(
            f"  {g['id']:5s} {g['status']:4s} {g['corpus']}: {g['value']}", flush=True
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    report(res, out.parent / "REPORT.md")
    return res


def selftest() -> None:
    bs = b"abcdabcdabcd"
    bc = np.ones(256) * 8
    d = build_dict(bs, np.ones(len(bs), dtype=bool), bc)
    toks = cover(bs, d, 0.0)
    assert b"".join(toks) == bs and len(toks) <= len(bs)
    assert mask_of(toks, len(bs))[0] and sum(mask_of(toks, len(bs))) == len(toks)
    g = cover(bs, d, 0.0, greedy=True)
    assert b"".join(g) == bs
    assert bits_of(toks, d) <= bits_of(g, d) + 1e-9
    brute_best = [float("inf")]

    def go(p, val, n):
        if p == len(bs):
            brute_best[0] = min(brute_best[0], val)
            return
        for e in range(p + 1, min(len(bs), p + CAP) + 1):
            go(e, val + d.cost_of(bs[p:e]), n + 1)

    go(0, 0.0, 0)
    assert abs(bits_of(toks, d) - brute_best[0]) < 1e-6, (
        bits_of(toks, d),
        brute_best[0],
    )
    print("selftest: ok")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="H-NCM v7")
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--corpora", default="synseg,syngrid,shakespeare")
    r.add_argument("--out", default="runs/v7/results.json")
    sub.add_parser("selftest")
    a = p.parse_args(argv)
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "run":
        run([x for x in a.corpora.split(",") if x], Path(a.out))
    else:
        p.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
