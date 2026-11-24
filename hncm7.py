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


def counts_onsets(tr: bytes, on: np.ndarray) -> dict[bytes, int]:
    """Supervised candidate counts: the train word spans. The old builder."""
    return dict(collections.Counter(w for w in spans(tr, on) if len(w) <= CAP))


def counts_substr(tr: bytes) -> dict[bytes, int]:
    """Text-only candidate counts: every substring <= CAP bytes, count >= 2.

    One pass per length with a 64-bit rolling hash. Two different strings collide
    with probability ~2^-64; the count is then attributed to the first index.
    """
    a = np.frombuffer(tr, np.uint8).astype(np.uint64)
    n = len(a)
    out: dict[bytes, int] = {}
    for ell in range(2, CAP + 1):
        m = n - ell + 1
        h = np.zeros(m, dtype=np.uint64)
        for j in range(ell):
            h = h * np.uint64(257) + a[j : j + m]
        vals, idx, cnt = np.unique(h, return_index=True, return_counts=True)
        for i, c in zip(idx.tolist(), cnt.tolist()):
            if c >= 2:
                out[bytes(tr[i : i + ell])] = int(c)
    return out


def build_dict(
    counts: dict[bytes, int], n: int, bc: np.ndarray, min_count: int = 1
) -> Dict:
    """A dictionary from candidate counts, pruned by count.

    `cost = -log2(count / n)` is the static substring unigram code length, where
    `n` is the number of start positions. It over-prices by ~log2(n/tokens), so
    the rate is a conservative bound. `min_count` trades model bytes for rate: a
    singleton costs 4 model bytes and saves nothing. The single bytes are always
    present, so every span has a token and the decoder needs no switch.
    """
    keep = [w for w in counts if counts[w] >= min_count]
    ids = {w: i for i, w in enumerate(keep)}
    strings = list(keep)
    for b in range(256):
        s = bytes([b])
        if s not in ids:
            ids[s] = len(strings)
            strings.append(s)
    cost = [float(sum(bc[b] for b in s)) for s in strings]
    for w in keep:
        cost[ids[w]] = -math.log2(counts[w] / n)
    trie: dict = {}
    for s, i in ids.items():
        node = trie
        for b in s:
            node = node.setdefault(b, {})
        node["#"] = i
    # the model is data: 1 byte length + 2 bytes count per string, plus the
    # 256-entry byte table (2 bytes each). Single bytes are that table.
    model_bytes = 512 + sum(len(s) + 3 for s in strings if len(s) > 1)
    return Dict(ids, strings, cost, bc, trie, model_bytes)


def assemble(strings: list[bytes], cost: list[float], bc: np.ndarray) -> Dict:
    """A dictionary from strings and costs, with the single bytes always present."""
    strings = list(strings)
    cost = list(cost)
    ids = {s: i for i, s in enumerate(strings)}
    for b in range(256):
        s = bytes([b])
        if s not in ids:
            ids[s] = len(strings)
            strings.append(s)
            cost.append(float(bc[b]))
    trie: dict = {}
    for s, i in ids.items():
        node = trie
        for b in s:
            node = node.setdefault(b, {})
        node["#"] = i
    model_bytes = 512 + sum(len(s) + 3 for s in strings if len(s) > 1)
    return Dict(ids, strings, cost, bc, trie, model_bytes)


def em_dict(
    counts: dict[bytes, int], tr: bytes, bc: np.ndarray, rounds: int = 2
) -> tuple[Dict, list[dict]]:
    """One cost model for every vocabulary: a token-usage unigram fitted by EM.

    The initial costs are the candidate counts; the DP re-tokenizes the train; the
    usage counts become the new costs; unused strings are dropped. The model is
    charged at every round, so the vocabulary is selected by the model/rate trade,
    not by a hand-tuned `min_count`, and both vocabularies use the same model.
    """
    total = sum(counts.values())
    strings = list(counts)
    cost = [-math.log2(counts[s] / total) for s in strings]
    d = assemble(strings, cost, bc)
    hist = [
        {
            "round": 0,
            "dict": len(d.strings),
            "model_bytes": d.model_bytes,
            "rate": bits_of(cover(tr, d, 0.0), d) / len(tr),
        }
    ]
    for r in range(1, rounds + 1):
        usage = collections.Counter(cover(tr, d, 0.0))
        used = [s for s in d.strings if usage.get(s, 0) > 0 or len(s) == 1]
        n = sum(usage.get(s, 0) for s in used)
        cost = [
            float(bc[s[0]]) if len(s) == 1 else -math.log2(usage[s] / n) for s in used
        ]
        d = assemble(used, cost, bc)
        hist.append(
            {
                "round": r,
                "dict": len(d.strings),
                "model_bytes": d.model_bytes,
                "rate": bits_of(cover(tr, d, 0.0), d) / len(tr),
            }
        )
    return d, hist, usage


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
    """The gates that are meaningful for a dictionary code: charged rate, tiling
    value, solver value. The oracle is a baseline, not a ceiling: with a substring
    vocabulary the words are not special, and the DP is the optimal cover by
    construction, so `DP <= oracle` is an invariant, not a test.
    """
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
            out.append(
                {
                    "id": "G0",
                    "corpus": name,
                    "value": f"DP charged={dp['charged_bpb']:.3f} gzip={gz:.3f} "
                    f"(model={e['model_bpb']:.3f} bpb)",
                    "status": "pass" if dp["charged_bpb"] < gz else "fail",
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
                    "id": "G2",
                    "corpus": name,
                    "value": f"DP charged={dp['charged_bpb']:.3f} unigram={uni:.3f}",
                    "status": "pass" if dp["charged_bpb"] < uni else "fail",
                }
            )
            out.append(
                {
                    "id": "G5",
                    "corpus": name,
                    "value": f"DP={dp['total_bpb']:.3f} fixed={fixed['total_bpb']:.3f} lam={lam:.1f}",
                    "status": "pass"
                    if dp["total_bpb"] < fixed["total_bpb"]
                    else "fail",
                }
            )
            out.append(
                {
                    "id": "G5c",
                    "corpus": name,
                    "value": f"DP={dp['total_bpb']:.3f} greedy={greedy['total_bpb']:.3f} "
                    f"gain={greedy['total_bpb'] - dp['total_bpb']:+.3f} lam={lam:.1f}",
                    "status": "pass"
                    if dp["total_bpb"] <= greedy["total_bpb"]
                    else "fail",
                }
            )
    return out


def report(res: dict, path: Path) -> None:
    lines = [
        f"# H-NCM v7 - solved tiling over a pruned span dictionary ({res['vocab']})",
        "",
    ]
    for name, e in res["corpora"].items():
        cross = {
            k: (f"{v / 1e6:.2f} MB" if v else "never")
            for k, v in e["crossing_bytes"].items()
        }
        lines += [
            f"## {name}",
            "",
            f"stream {e['stream_bytes'] / 1e6:.2f} MB; chosen vocab {e['min_count']}; "
            f"dictionary {e['dict']} spans, {e['model_bytes'] / 1e3:.1f} kB "
            f"({e['model_bpb']:.3f} bpb); byte unigram {e['unigram']:.3f} bpb; "
            f"gzip {e['gzip']:.3f} bpb; crossover unigram {cross['unigram']}, "
            f"gzip {cross['gzip']}",
            "",
            "| vocab | dict | model kB | rate | charged |",
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
def run(names: list[str], out: Path, vocab: str = "substr") -> dict:
    res = {"format": FORMAT, "vocab": vocab, "corpora": {}, "runs": []}
    for name in names:
        corpus = base.get_corpus(name)
        tr = corpus["tr"].cpu().numpy().astype(np.uint8).tobytes()
        va = corpus["va"].cpu().numpy().astype(np.uint8).tobytes()
        va = va[: len(va) // 16 * 16]
        bc = byte_costs(tr)
        on = base.onsets(name, np.frombuffer(tr, np.uint8))
        counts = counts_onsets(tr, on) if vocab == "onsets" else counts_substr(tr)
        stream = len(tr) + len(va)
        d, rounds, usage = em_dict(counts, tr, bc)
        top = sorted((s for s in usage if len(s) > 1), key=lambda s: -usage[s])
        best = None
        budget = []
        for k in (256, 512, 1024, 2048, 4096, 8192, 16384, 32768):
            keep = top[:k]
            n = sum(usage[s] for s in keep)
            dk = assemble(
                keep,
                [
                    float(bc[s[0]]) if len(s) == 1 else -math.log2(usage[s] / n)
                    for s in keep
                ],
                bc,
            )
            rate = bits_of(cover(va, dk, 0.0), dk) / len(va)
            charged = rate + 8 * dk.model_bytes / stream
            budget.append(
                {
                    "min_count": k,
                    "dict": len(dk.strings),
                    "model_bytes": dk.model_bytes,
                    "rate": rate,
                    "charged": charged,
                }
            )
            if best is None or charged < best[0]:
                best = (charged, k, dk, rate)
            print(
                f"  {name:11s} vocab={k:5d} dict={len(dk.strings):6d} "
                f"model={dk.model_bytes / 1e3:7.1f} kB rate={rate:.3f} charged={charged:.3f}",
                flush=True,
            )
        charged, mc, d, rate = best
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
    d = build_dict(counts_onsets(bs, np.ones(len(bs), dtype=bool)), len(bs), bc)
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
    r.add_argument("--vocab", default="substr", choices=("substr", "onsets"))
    sub.add_parser("selftest")
    a = p.parse_args(argv)
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "run":
        run([x for x in a.corpora.split(",") if x], Path(a.out), a.vocab)
    else:
        p.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
