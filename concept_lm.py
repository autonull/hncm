"""Probe: a first-order concept LM over dictionary indices, measured through the charge.

v7 prices each concept by its *unigram* (no context). The concept alphabet is
predictable, and the previous concept is already decoded, so a bigram over the
indices is a decodable cost model. The DP state becomes the previous concept,
which makes the tiling a first-order Viterbi: exact, and no longer myopic.

Two measured facts make this the solution path, not a guess:

- The iterated bigram is *worse* than the unigram (3.128 vs 2.905 on
  shakespeare). Its costs come from the previous token of the *unigram* tiling,
  so the DP picks tokens under a model built for a different path. The model's
  cost must condition on the actual state of the path.
- The first-order Viterbi is *better*: 2.490 vs 3.003 on shakespeare and 0.886 vs
  1.283 on synseg, with a plan-aligned first token. The count bigram is the
  ceiling; a neural bigram replaces its table (a training step) and the tail
  coder covers spans the dictionary lacks.

This file measures the claim through its charge:

- M1: the charged budget curve (pruned bigram entries x dictionary size), the
  charged total versus the real codecs, and the crossing in MB.
- M2: the solver gain on the *same* bigram costs (exact DP vs greedy, vs a
  fixed-4 grid, vs the unigram).
- M3: zlib, gzip, bz2, lzma, zstd -19 and brotli -q11 on the same val bytes.

The dictionary and the bigram table are fitted on the first 90% of the train;
the budget (vocabulary, dict size, table size) is chosen by a rule that sees the
last 10% of the train, which the model never fitted, and never the val. That is
rule 9. `oracle` is the best grid point on the val plus the header that
transmits the choice: a lower bound for any rule, and the gap from the rule to
it is the cost of the rule.
"""

from __future__ import annotations

import argparse
import bz2
import os
import collections
import gzip
import json
import lzma
import math
import sys
import zlib
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import corpora  # noqa: E402
import hncm7 as v  # noqa: E402

CAP = 16
DICTS = (512, 1024, 2048, 4096, 8192, 16384)
PRUNES = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, None)
ENTRY_BYTES = 6


def codecs() -> dict[str, Callable[[bytes], int]]:
    """The real codecs on the same bytes, as their compressed size.

    Each is self-contained: its model is in the stream, so it is already charged.
    zstd, brotli and PPMd are optional. PPMd (`7z -m0=PPMd -mx9`) is the text bar:
    measured 2.504 bpb on shakespeare, 1.500 on synseg, 2.063 on syngrid.
    """
    out: dict[str, Callable[[bytes], int]] = {
        "zlib": lambda bs: len(zlib.compress(bs)),
        "gzip": lambda bs: len(gzip.compress(bs, 9)),
        "bz2": lambda bs: len(bz2.compress(bs, 9)),
        "lzma": lambda bs: len(lzma.compress(bs, preset=9 | lzma.PRESET_EXTREME)),
    }
    try:
        import zstandard

        out["zstd"] = lambda bs: len(zstandard.ZstdCompressor(level=19).compress(bs))
    except ImportError:
        pass
    try:
        import brotli

        out["brotli"] = lambda bs: len(brotli.compress(bs, quality=11))
    except ImportError:
        pass
    if shutil.which("7z"):
        out["ppmd"] = lambda bs: _ppmd(bs)
    return out


def _ppmd(bs: bytes) -> int:
    """PPMd via 7z: the archive's size on disk, the header included."""
    with tempfile.TemporaryDirectory(prefix="hncm-") as d:
        src = Path(d) / "x"
        src.write_bytes(bs)
        arc = Path(d) / "p.7z"
        subprocess.run(
            ["7z", "a", "-m0=PPMd", "-mx9", str(arc), str(src)],
            check=True,
            capture_output=True,
        )
        return arc.stat().st_size


# %%
def bigrams(toks: list[bytes]) -> tuple[dict, collections.Counter]:
    """The first-order counts of a tiling: P(s | p) = count(p, s) / count(p)."""
    big: dict[bytes, collections.Counter] = collections.defaultdict(collections.Counter)
    for a, b in zip(toks, toks[1:]):
        big[a][b] += 1
    return dict(big), collections.Counter(toks)


def prune(big: dict, n: int | None) -> dict:
    """Keep the top-n entries by count; absent entries back off to the unigram.

    Absence is the model (the table is data), so a pruned entry costs no extra
    bits to signal: the decoder reads the same table.
    """
    if n is None:
        return big
    items = sorted(
        ((a, b, c) for a, row in big.items() for b, c in row.items()),
        key=lambda e: -e[2],
    )[:n]
    out: dict[bytes, collections.Counter] = collections.defaultdict(collections.Counter)
    for a, b, c in items:
        out[a][b] = c
    return dict(out)


def cost_of(
    s: bytes, d: v.Dict, big: dict, uni: collections.Counter, p: bytes | None
) -> float:
    """Bits for `s` given the concept that just ended, with a unigram backoff."""
    if p is None:
        return d.cost_of(s)
    row = big.get(p)
    if row and row.get(s, 0):
        return -math.log2(row[s] / uni[p])
    return d.cost_of(s)


def viterbi(
    bs: bytes, d: v.Dict, big: dict, uni: collections.Counter
) -> tuple[list[bytes], float]:
    """Exact first-order DP: the state is the concept that just ended."""
    L = len(bs)
    best: list[dict] = [dict() for _ in range(L + 1)]
    best[0][None] = (0.0, None)
    for t in range(L):
        if not best[t]:
            continue
        for p, (cst, _) in best[t].items():
            node, e = d.trie, t
            while e < min(L, t + CAP) and bs[e] in node:
                node = node[bs[e]]
                if "#" in node:
                    s = bs[t : e + 1]
                    nc = cst + cost_of(s, d, big, uni, p)
                    cur = best[e + 1].get(s)
                    if cur is None or nc < cur[0]:
                        best[e + 1][s] = (nc, p)
                e += 1
    if not best[L]:
        return [], float("inf")
    s, (cost, _) = min(best[L].items(), key=lambda kv: kv[1][0])
    toks, e = [], L
    while e > 0:
        toks.append(s)
        p = best[e][s][1]
        e -= len(s)
        s = p
    return toks[::-1], cost


def viterbi_chunked(
    bs: bytes, d: v.Dict, big: dict, uni: collections.Counter, chunk: int = 1 << 20
) -> list[bytes]:
    """The first-order DP in 1 MB chunks: the state resets at each boundary.

    The 10 MB `viterbi` allocates L+1 dictionaries (~1.6 GB); the chunked
    version is ~80 MB and the boundary costs a handshake of a few bits. The
    codec resets the state at the same boundaries, so the charge is exact.
    """
    out = []
    for a in range(0, len(bs), chunk):
        out.extend(viterbi(bs[a : a + chunk], d, big, uni)[0])
    return out


def greedy1(bs: bytes, d: v.Dict, big: dict, uni: collections.Counter) -> list[bytes]:
    """The same cost model, myopic: longest dictionary span at each position."""
    out, t = [], 0
    while t < len(bs):
        node, hit = d.trie, None
        for e in range(t, min(len(bs), t + CAP)):
            if bs[e] not in node:
                break
            node = node[bs[e]]
            if "#" in node:
                hit = e + 1
        s = bs[t:hit] if hit else bs[t : t + 1]
        out.append(s)
        t += len(s)
    return out


def fixed(bs: bytes, step: int) -> list[bytes]:
    return [bs[t : t + step] for t in range(0, len(bs), step)]


def bits_of(toks: list[bytes], d: v.Dict, big: dict, uni: collections.Counter) -> float:
    prev = [None] + toks[:-1]
    return float(sum(cost_of(s, d, big, uni, p) for s, p in zip(toks, prev)))


def em_bigram(
    bs: bytes,
    d: v.Dict,
    big: dict,
    uni: collections.Counter,
    n: int | None,
    rounds: int,
) -> list[tuple[dict, collections.Counter]]:
    """Refit the table on the path the DP searches: the first-order EM step.

    The counts from the unigram tiling are an initialisation, not a converged
    model: its costs condition on a path the DP does not take (rule 10). Each
    round re-tokenizes `bs` with the current table and counts that path.
    """
    out = [(big, uni)]
    for _ in range(rounds):
        toks = viterbi(bs, d, big, uni)[0]
        b2, u2 = bigrams(toks)
        big, uni = prune(b2, n), u2
        out.append((big, uni))
    return out


def shape(toks: list[bytes], n: int, bits: float) -> dict:
    single = sum(len(s) for s in toks if len(s) == 1)
    return {
        "tokens": len(toks),
        "bytes_per_token": n / max(len(toks), 1),
        "covered": 1 - single / n,
        "calls": 1 + single,
        "rate": bits / n,
    }


def dict_at(usage: collections.Counter, top: list[bytes], k: int, bc) -> v.Dict:
    keep = top[:k]
    n = sum(usage[s] for s in keep)
    return v.assemble(
        keep,
        [float(bc[s[0]]) if len(s) == 1 else -math.log2(usage[s] / n) for s in keep],
        bc,
    )


# %%
def _cache(name: str, vocab: str, sha: str) -> Path:
    base = Path(os.environ.get("HNCM_CACHE", str(Path.home() / ".cache/hncm")))
    return base / f"counts-{name}-{vocab}-{sha[:12]}.npz"


def split(name: str, vocab: str) -> dict:
    """The protocol: fit on the first 90% of the train, select on the last 10%,
    truncate the val to 16 bytes. Returns the EM usage and its ranked candidates
    (the one model both vocabularies share). The candidate scan is cached by
    corpus x vocab x sha, so a rerun is seconds, not the 9 min scan.

    `union` merges the two EM tilings' used tokens: one candidate map, one
    unigram (the usage counts), one DP. It is the coverage lever, and it costs
    no new scan because both maps are cached."""
    if vocab == "union":
        a, b = split(name, "substr"), split(name, "onsets")
        usage = a["usage"] + b["usage"]
        return {
            **a,
            "vocab": "union",
            "usage": usage,
            "top": sorted((s for s in usage if len(s) > 1), key=lambda s: -usage[s]),
        }
    corpus = corpora.get(name)
    tr = corpus["tr"].cpu().numpy().astype(np.uint8).tobytes()
    va = corpus["va"].cpu().numpy().astype(np.uint8).tobytes()
    va = va[: len(va) // 16 * 16]
    cut = len(tr) // 10 * 9
    fit, sel = tr[:cut], tr[cut:]
    bc = v.byte_costs(fit)
    path = _cache(name, vocab, corpus["sha"])
    if path.exists():
        d = np.load(path, allow_pickle=True)
        flat, offs, ucount = d["flat"], d["offs"], d["ucount"]
        top = [bytes(flat[o:e]) for o, e in zip(offs[:-1], offs[1:])]
        usage = collections.Counter(dict(zip(top, ucount.tolist())))
        return {
            "tr": tr,
            "fit": fit,
            "sel": sel,
            "va": va,
            "bc": bc,
            "usage": usage,
            "top": top,
            "stream": len(tr) + len(va),
            "sha": corpus["sha"],
            "name": corpus["name"],
        }
    xb = np.frombuffer(fit, np.uint8)
    counts = (
        v.counts_onsets(fit, v.base.onsets(name, xb))
        if vocab == "onsets"
        else v.counts_substr(fit)
    )
    _, _, usage = v.em_dict(counts, fit, bc)
    top = sorted((s for s in usage if len(s) > 1), key=lambda s: -usage[s])
    blob = b"".join(top)
    offs = np.zeros(len(top) + 1, dtype=np.int64)
    np.cumsum([len(s) for s in top], out=offs[1:])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        flat=np.frombuffer(blob, np.uint8),
        offs=offs,
        ucount=np.array([usage[s] for s in top], dtype=np.int64),
        bc=bc,
    )
    return {
        "tr": tr,
        "fit": fit,
        "sel": sel,
        "va": va,
        "bc": bc,
        "usage": usage,
        "top": top,
        "stream": len(tr) + len(va),
        "sha": corpus["sha"],
        "name": corpus["name"],
    }


def rebuild(s: dict, k: int, table: int) -> tuple[v.Dict, dict, collections.Counter]:
    """One measured grid point, rebuilt: the dictionary at `k` and the table
    pruned to `table` entries. The costs are the unigram-tiling table (the grid),
    not the EM refit, so the tail is exactly the tail of the reported codec."""
    d = dict_at(s["usage"], s["top"], k, s["bc"])
    big, uni = bigrams(v.cover(s["fit"], d, 0.0))
    return d, prune(big, table), uni


def measure(
    name: str,
    vocab: str,
    dicts: tuple[int, ...] = DICTS,
    prunes: tuple[int | None, ...] = PRUNES,
) -> dict:
    s = split(name, vocab)
    fit, sel, va, bc = s["fit"], s["sel"], s["va"], s["bc"]
    usage, top, stream = s["usage"], s["top"], s["stream"]
    codec = {c: 8 * f(va) / len(va) for c, f in codecs().items()}

    grid = []
    for k in sorted({min(k, len(top)) for k in dicts}):
        d = dict_at(usage, top, k, bc)
        fit_toks = v.cover(fit, d, 0.0)
        big, uni = bigrams(fit_toks)
        uni_rate = v.bits_of(v.cover(va, d, 0.0), d) / len(va)
        for n in prunes:
            bg = prune(big, n)
            model_bytes = d.model_bytes + ENTRY_BYTES * sum(len(r) for r in bg.values())
            model_bpb = 8 * model_bytes / stream
            sel_rate = viterbi(sel, d, bg, uni)[1] / len(sel)
            toks, cost = viterbi(va, d, bg, uni)
            row = {
                "dict": k,
                "prune": n,
                "table": sum(len(r) for r in bg.values()),
                "model_bytes": model_bytes,
                "model_bpb": model_bpb,
                "unigram": uni_rate,
                "sel_rate": sel_rate,
                "sel_charged": sel_rate + model_bpb,
                "charged": cost / len(va) + model_bpb,
                "crossing": {
                    c: (
                        model_bytes * 8 / (r - cost / len(va))
                        if r > cost / len(va)
                        else None
                    )
                    for c, r in codec.items()
                },
                **shape(toks, len(va), cost),
            }
            grid.append(row)
        print(
            f"  {name:11s} {vocab:6s} dict={k:6d} unigram={uni_rate:.3f} "
            f"full={grid[-1]['rate']:.3f} sel={min(g['sel_charged'] for g in grid):.3f}",
            flush=True,
        )

    chosen = min(grid, key=lambda g: g["sel_charged"])
    d = dict_at(usage, top, chosen["dict"], bc)
    fit_toks = v.cover(fit, d, 0.0)
    big, uni = bigrams(fit_toks)
    bg = prune(big, chosen["prune"])
    model_bpb = 8 * chosen["model_bytes"] / stream
    chain = em_bigram(fit, d, bg, uni, chosen["prune"], 2)
    em = min(chain, key=lambda kv: viterbi(sel, d, *kv)[1] / len(sel))
    arms = []
    for arm, toks, b_arm, u_arm, mb in (
        ("DP", viterbi(va, d, bg, uni)[0], bg, uni, chosen["model_bytes"]),
        ("greedy", greedy1(va, d, bg, uni), bg, uni, chosen["model_bytes"]),
        ("fixed4", fixed(va, 4), bg, uni, chosen["model_bytes"]),
        ("unigram", v.cover(va, d, 0.0), {}, {}, d.model_bytes),
        ("DP-em", viterbi(va, d, *em)[0], em[0], em[1], chosen["model_bytes"]),
        ("greedy-em", greedy1(va, d, *em), em[0], em[1], chosen["model_bytes"]),
    ):
        bits = bits_of(toks, d, b_arm, u_arm)
        arms.append(
            {
                "arm": arm,
                "model_bytes": mb,
                "charged": bits / len(va) + 8 * mb / stream,
                **shape(toks, len(va), bits),
            }
        )
    for a in arms:
        if a["arm"] in ("DP-em", "greedy-em"):
            a["crossing"] = {
                c: (
                    chosen["model_bytes"] * 8 / (r - a["rate"])
                    if r > a["rate"]
                    else None
                )
                for c, r in codec.items()
            }

    best = min(codec, key=codec.get)
    fitted = next(a for a in arms if a["arm"] == "DP-em")
    return {
        "stream_bytes": stream,
        "train_bytes": len(s["tr"]),
        "sha": s.get("sha", ""),
        "val_bytes": len(va),
        "dict": chosen["dict"],
        "table": chosen["table"],
        "model_bytes": chosen["model_bytes"],
        "model_bpb": model_bpb,
        "unigram": chosen["unigram"],
        "codec": codec,
        "best_codec": best,
        "best_codec_bpb": codec[best],
        "chosen": chosen,
        "fitted": fitted,
        "grid": grid,
        "arms": arms,
    }


def gates(res: dict) -> list[dict]:
    """The gates of the claim: the charge against the codecs, and the solver.

    G1 is the concept LM (fitted table, budget chosen by the held-out rule)
    against zlib; G1a against the best real codec; G2 the exact DP against a
    greedy on the *same* first-order costs; G3 the first-order rate against the
    fixed grid and the unigram codec. G1/G1a are graded on the rule's point (it
    never sees the val); the envelope over the grid is reported with it.
    """
    out = []
    for name, e in res["corpora"].items():
        fit, orc, best = e["fitted"], e["oracle"], e["best_codec"]
        arms = {a["arm"]: a for a in e["arms"]}
        out += [
            {
                "id": "G1",
                "corpus": name,
                "value": f"rule={fit['charged']:.3f} oracle={orc['charged']:.3f} "
                f"zlib={e['codec']['zlib']:.3f}",
                "status": "pass" if fit["charged"] < e["codec"]["zlib"] else "fail",
            },
            {
                "id": "G1a",
                "corpus": name,
                "value": f"rule={fit['charged']:.3f} oracle={orc['charged']:.3f} "
                f"{best}={e['best_codec_bpb']:.3f}",
                "status": "pass" if fit["charged"] <= e["best_codec_bpb"] else "fail",
            },
            {
                "id": "G2",
                "corpus": name,
                "value": f"DP={arms['DP-em']['rate']:.3f} "
                f"greedy={arms['greedy-em']['rate']:.3f} "
                f"gain={arms['greedy-em']['rate'] - arms['DP-em']['rate']:+.3f}",
                "status": "pass"
                if arms["DP-em"]["rate"] <= arms["greedy-em"]["rate"] + 1e-9
                else "fail",
            },
            {
                "id": "G3",
                "corpus": name,
                "value": f"DP={arms['DP-em']['rate']:.3f} "
                f"fixed4={arms['fixed4']['rate']:.3f} "
                f"unigram={arms['unigram']['rate']:.3f}",
                "status": "pass"
                if arms["DP-em"]["rate"] <= arms["fixed4"]["rate"] + 1e-9
                and arms["DP-em"]["rate"] < arms["unigram"]["rate"]
                else "fail",
            },
        ]
    return out


def report(res: dict, path: Path) -> None:
    lines = [
        "# H-NCM v7: the concept LM, charged (M1-M3)",
        "",
        "The vocabulary (word onsets or text substrings) and the budget (dict size x "
        "table size) are chosen by the *rule*: the charged total on the last 10% of the "
        "train, which the model never fitted, never on the val. `oracle` is the best "
        "grid point on the val plus the header that transmits the choice (a lower bound "
        "for any budget rule; the gap to `rule` is the cost of the rule). The table is "
        "refitted on the path the DP searches (2 EM rounds, chosen on the same 10%). "
        f"The model is charged {ENTRY_BYTES} bytes per bigram entry and amortised over "
        "train+val. The rate is the exact first-order Viterbi.",
        "",
    ]
    names = list(next(iter(res["corpora"].values()))["codec"])
    cols = (
        ["corpus", "vocab", "stream"]
        + names
        + [
            "rule",
            "oracle (vocab, dict, table)",
            "oracle",
            "vs zlib",
            "vs best",
            "x zlib",
        ]
    )
    align = ["---", "---", "---:"] + ["---:"] * len(names) + ["---:"] * 5
    lines += ["| " + " | ".join(cols) + " |", "|" + "|".join(align) + "|"]
    for name, e in res["corpora"].items():
        c, fit, orc = e["codec"], e["fitted"], e["oracle"]
        x = orc["crossing"]
        zc = f"{x['zlib'] / 1e6:.2f} MB" if x["zlib"] else "never"
        lines.append(
            f"| {name} | {e['vocab']} | {e['stream_bytes'] / 1e6:.2f} MB | "
            + " | ".join(f"{c[w]:.3f}" for w in names)
            + f" | {fit['charged']:.3f} | {orc['vocab']}, {orc['dict']}, {orc['table']} | "
            f"{orc['charged']:.3f} | {orc['charged'] - c['zlib']:+.3f} | "
            f"{orc['charged'] - e['best_codec_bpb']:+.3f} | {zc} |"
        )
    for name, e in res["corpora"].items():
        lines += [
            "",
            f"## {name} (vocab {e['vocab']}, dict {e['dict']}, {e['table']} bigram entries)",
            "",
            "| dict | table | model kB | unigram | rate | charged | sel rate | sel charged | zlib crossing | tokens | covered | calls |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for g in e["grid"]:
            zc = g["crossing"]["zlib"]
            lines.append(
                f"| {g['dict']} | {g['table']} | {g['model_bytes'] / 1e3:.1f} | "
                f"{g['unigram']:.3f} | {g['rate']:.3f} | {g['charged']:.3f} | "
                f"{g['sel_rate']:.3f} | {g['sel_charged']:.3f} | "
                f"{f'{zc / 1e6:.2f} MB' if zc else 'never'} | {g['tokens']} | "
                f"{g['covered']:.3f} | {g['calls']:.0f} |"
            )
        lines += [
            "",
            "| vocab | dict | table | model kB | rate | charged | sel charged | oracle charged |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for w, a in e["alt"].items():
            lines.append(
                f"| {w} | {a['dict']} | {a['table']} | {a['model_bytes'] / 1e3:.1f} | "
                f"{a['rate']:.3f} | {a['charged']:.3f} | {a['sel_charged']:.3f} | "
                f"{a['oracle_charged']:.3f} |"
            )
        lines += [
            "",
            f"### arms at dict {e['dict']}, {e['table']} entries",
            "",
            "| arm | rate | charged | tokens | bytes/token | covered | calls |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for a in e["arms"]:
            lines.append(
                f"| {a['arm']} | {a['rate']:.3f} | {a['charged']:.3f} | {a['tokens']} | "
                f"{a['bytes_per_token']:.2f} | {a['covered']:.3f} | {a['calls']:.0f} |"
            )
        lines += [
            "",
            "crossings of the envelope (MB of stream, versus each codec):",
            "",
        ]
        for c, nbytes in e["oracle"]["crossing"].items():
            lines.append(f"- {c}: {f'{nbytes / 1e6:.2f} MB' if nbytes else 'never'}")
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
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run(
    names: list[str],
    out: Path,
    vocabs: tuple[str, ...] = ("substr", "onsets"),
    dicts: tuple[int, ...] = DICTS,
    prunes: tuple[int | None, ...] = PRUNES,
) -> dict:
    res = {
        "format": 7,
        "vocabs": list(vocabs),
        "dicts": list(dicts),
        "prunes": list(prunes),
        "corpora": {},
    }
    for name in names:
        print(f"{name}:", flush=True)
        cand = {w: measure(name, w, dicts, prunes) for w in vocabs}
        vocab = min(cand, key=lambda w: cand[w]["chosen"]["sel_charged"])
        e = cand[vocab]
        e["vocab"] = vocab
        e["alt"] = {
            w: {
                "dict": c["dict"],
                "table": c["table"],
                "model_bytes": c["model_bytes"],
                "rate": c["fitted"]["rate"],
                "charged": c["fitted"]["charged"],
                "sel_charged": c["chosen"]["sel_charged"],
                "oracle_charged": min(g["charged"] for g in c["grid"]),
            }
            for w, c in cand.items()
        }
        rows = [(w, g) for w in vocabs for g in cand[w]["grid"]]
        ow, og = min(rows, key=lambda wg: wg[1]["charged"])
        header = math.ceil(math.log2(max(len(rows), 1))) / e["stream_bytes"]
        e["oracle"] = {
            "vocab": ow,
            "dict": og["dict"],
            "table": og["table"],
            "model_bytes": og["model_bytes"],
            "rate": og["rate"],
            "header_bpb": header,
            "charged": og["charged"] + header,
            "crossing": og["crossing"],
        }
        res["corpora"][name] = e
        fit = e["fitted"]
        print(
            f"{name:11s} vocab={vocab:6s} dict={e['dict']:5d} table={e['table']:6d} "
            f"model={e['model_bytes'] / 1e3:6.1f} kB rate={fit['rate']:.3f} "
            f"charged={fit['charged']:.3f} zlib={e['codec']['zlib']:.3f} "
            f"{e['best_codec']}={e['best_codec_bpb']:.3f}",
            flush=True,
        )
        for a in e["arms"]:
            print(
                f"  {a['arm']:9s} rate={a['rate']:.3f} charged={a['charged']:.3f}",
                flush=True,
            )
    for g in gates(res):
        print(
            f"  {g['id']:4s} {g['status']:4s} {g['corpus']}: {g['value']}", flush=True
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    report(res, out.parent / "CONCEPT.md")
    return res


# %%
def selftest() -> None:
    bs = b"abcabcabcaabcd"
    bc = np.ones(256) * 8
    d = v.build_dict(v.counts_substr(bs + bs), len(bs + bs), bc, min_count=1)
    toks, cost = viterbi(bs, d, {}, collections.Counter())
    assert b"".join(toks) == bs
    assert abs(cost - v.bits_of(v.cover(bs, d, 0.0), d)) < 1e-9

    big, uni = bigrams(toks)
    toks, cost = viterbi(bs, d, big, uni)
    assert b"".join(toks) == bs
    assert abs(cost - bits_of(toks, d, big, uni)) < 1e-9
    assert cost <= bits_of(greedy1(bs, d, big, uni), d, big, uni) + 1e-9
    assert cost <= bits_of(fixed(bs, 4), d, big, uni) + 1e-9

    # entropy round-trip: the code length is the negative log of the model's
    # probability along the path, so P must be <= 1 and the counts must match.
    prev = [None] + toks[:-1]
    for s, p in zip(toks, prev):
        if p is None:
            continue
        row = big[p]
        q = row[s] / uni[p]
        assert 0 < q <= 1 + 1e-12, (s, p, q)
        assert abs(cost_of(s, d, big, uni, p) + math.log2(q)) < 1e-12

    # DP optimality: brute-force every tiling with the first-order costs.
    best = [float("inf")]

    def go(t, p, acc):
        if t == len(bs):
            best[0] = min(best[0], acc)
            return
        for e in range(t + 1, min(len(bs), t + CAP) + 1):
            s = bs[t:e]
            if s not in d.ids:
                continue
            go(e, s, acc + cost_of(s, d, big, uni, p))

    go(0, None, 0.0)
    assert abs(cost - best[0]) < 1e-9, (cost, best[0])

    # pruning keeps the top counts, and the table is a subset of the full one.
    bg = prune(big, 2)
    assert sum(len(r) for r in bg.values()) <= 2
    assert all(b in big[a] for a, row in bg.items() for b in row)
    pruned = bits_of(viterbi(bs, d, bg, uni)[0], d, bg, uni)
    assert pruned >= cost
    print("selftest: ok")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="H-NCM v7 concept LM")
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--corpora", default="shakespeare,synseg,syngrid")
    r.add_argument("--out", default="runs/v7/concept.json")
    r.add_argument("--vocab", default="substr,onsets")
    r.add_argument("--dicts", default=",".join(str(x) for x in DICTS))
    r.add_argument(
        "--prunes", default=",".join("none" if x is None else str(x) for x in PRUNES)
    )
    sub.add_parser("selftest")
    a = p.parse_args(argv)
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "run":
        run(
            [x for x in a.corpora.split(",") if x],
            Path(a.out),
            tuple(x for x in a.vocab.split(",") if x),
            tuple(int(x) for x in a.dicts.split(",") if x),
            tuple(None if x == "none" else int(x) for x in a.prunes.split(",") if x),
        )
    else:
        p.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
