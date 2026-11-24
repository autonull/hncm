"""The amortization: one model, K files. The charge is per model, not per
file.

bz2 and lzma pay their model inside every file. A learned vocabulary and a
concept LM are fitted once and reused, so their charge is paid once: the
K-shard stream pays `8 * model / total`, and the per-file charge `8 * model /
shard` is what a generic codec cannot avoid. The val is split into K contiguous
shards; every codec compresses each shard on its own (that is its per-file
model), and the concept LM tiles each shard with the *same* dictionary and
table.

So the claim is not a bpb race at one size: it is that the concept LM's curve is
flat in K and the codecs' curves rise, and the crossing is where the codec's
mean bpb exceeds the shared charge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import concept_lm as c  # noqa: E402

KS = (1, 2, 4, 8, 16)


def shards(bs: bytes, k: int) -> list[bytes]:
    """K contiguous shards; the first k-1 are equal, the last takes the rest."""
    step = len(bs) // k
    return [bs[i * step : (i + 1) * step if i < k - 1 else len(bs)] for i in range(k)]


# %%
def measure(name: str, rect: dict, ks: tuple[int, ...] = KS) -> dict:
    s = c.split(name, rect["vocab"])
    d, bg, uni = c.rebuild(s, rect["dict"], rect["table"])
    va, stream = s["va"], s["stream"]
    model = d.model_bytes + 6 * sum(len(r) for r in bg.values())
    codec = c.codecs()
    rows = []
    for k in ks:
        parts = shards(va, k)
        rates = {w: sum(f(p) for p in parts) * 8 / len(va) for w, f in codec.items()}
        bits = sum(c.viterbi(p, d, bg, uni)[1] for p in parts)
        rate = bits / len(va)
        row = {
            "k": k,
            "shard_bytes": len(va) / k,
            "rate": rate,
            "shared": rate + 8 * model / len(va),
            "per_file": rate + 8 * model / (len(va) / k),
            "archive": rate + 8 * model / stream,
            "codec": rates,
            "crossing": {
                w: (model * 8 / (r - rate) if r > rate else None)
                for w, r in rates.items()
            },
        }
        rows.append(row)
    return {
        "stream_bytes": stream,
        "val_bytes": len(va),
        "dict": rect["dict"],
        "table": rect["table"],
        "model_bytes": model,
        "rows": rows,
    }


def gates(res: dict) -> list[dict]:
    """G6:f the shared charge is flat in K and beats the per-file charge; G7:f
    the codec's mean rises with K (a generic codec cannot share)."""
    out = []
    for name, e in res["corpora"].items():
        rows = e["rows"]
        first, last = rows[0], rows[-1]
        out += [
            {
                "id": "G6",
                "corpus": name,
                "value": f"shared {first['shared']:.3f} -> {last['shared']:.3f} "
                f"(K {first['k']}->{last['k']}); per-file "
                f"{first['per_file']:.3f} -> {last['per_file']:.3f}",
                "status": "pass"
                if abs(last["shared"] - first["shared"]) < 0.01
                else "fail",
            },
            {
                "id": "G7",
                "corpus": name,
                "value": f"bz2 {first['codec']['bz2']:.3f} -> "
                f"{last['codec']['bz2']:.3f} (K {first['k']}->{last['k']})",
                "status": "pass"
                if last["codec"]["bz2"] > first["codec"]["bz2"]
                else "fail",
            },
        ]
        cross = next((r["k"] for r in rows if r["codec"]["bz2"] > r["archive"]), None)
        out.append(
            {
                "id": "G8",
                "corpus": name,
                "value": f"archive {first['archive']:.3f}; bz2 crosses at K="
                f"{cross if cross else 'never'}",
                "status": "pass" if cross else "fail",
            }
        )
    return out


def report(res: dict, path: Path) -> None:
    lines = [
        "# H-NCM v7: one model, K files (the amortization)",
        "",
        "The val is split into K contiguous shards. Every codec compresses each "
        "shard on its own; the concept LM tiles each shard with the same "
        "dictionary and table. `archive` charges the model over train + the K "
        "shards (the v7 convention); `shared` over the K shards only (a deploy); "
        "`per_file` makes each shard pay it alone (what a generic codec cannot "
        "avoid).",
        "",
    ]
    for name, e in res["corpora"].items():
        lines += [
            f"## {name} (dict {e['dict']}, table {e['table']}, "
            f"model {e['model_bytes'] / 1e3:.1f} kB)",
            "",
        ]
        names = list(next(iter(res["corpora"].values()))["rows"][0]["codec"])
        cols = ["K", "shard B"] + names + ["rate", "archive", "shared", "per-file"]
        align = ["---:", "---:"] + ["---:"] * len(names) + ["---:"] * 4
        lines += ["| " + " | ".join(cols) + " |", "|" + "|".join(align) + "|"]
        for r in e["rows"]:
            cc = r["codec"]
            lines.append(
                f"| {r['k']} | {r['shard_bytes']:.0f} | "
                + " | ".join(f"{cc[w]:.3f}" for w in names)
                + f" | {r['rate']:.3f} | {r['archive']:.3f} | "
                f"{r['shared']:.3f} | {r['per_file']:.3f} |"
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


def run(names: list[str], out: Path, rect_path: Path) -> dict:
    rects = json.loads(rect_path.read_text())["corpora"]
    res = {"format": 7, "rect": str(rect_path), "corpora": {}}
    for name in names:
        print(f"{name}:", flush=True)
        e = measure(name, rects[name]["oracle"])
        res["corpora"][name] = e
        for r in e["rows"]:
            print(
                f"  K={r['k']:2d} shard={r['shard_bytes']:7.0f} rate={r['rate']:.3f} "
                f"archive={r['archive']:.3f} shared={r['shared']:.3f} "
                f"per-file={r['per_file']:.3f} "
                f"bz2={r['codec']['bz2']:.3f}",
                flush=True,
            )
    for g in gates(res):
        print(
            f"  {g['id']:3s} {g['status']:4s} {g['corpus']}: {g['value']}", flush=True
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    report(res, out.parent / "TRANSFER.md")
    return res


# %%
def selftest() -> None:
    bs = b"abcdefghijklmnopqrstuvwxyz"
    for k in (1, 2, 4, 8, 16):
        parts = shards(bs, k)
        assert b"".join(parts) == bs
        assert len(parts) == k
        assert min(len(p) for p in parts) > 0
    assert shards(bs, 1) == [bs]
    # the charge invariant: the per-file charge is K times the shared one.
    model, n = 1000, 20000
    for k in (1, 2, 4, 8, 16):
        shared = 8 * model / n
        per_file = 8 * model / (n / k)
        assert abs(per_file - k * shared) < 1e-9
    print("selftest: ok")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="H-NCM v7 transfer (M6)")
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--corpora", default="shakespeare,synseg,syngrid")
    r.add_argument("--out", default="runs/v7/transfer.json")
    r.add_argument("--rect", default="runs/v7/concept.json")
    sub.add_parser("selftest")
    a = p.parse_args(argv)
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "run":
        run([x for x in a.corpora.split(",") if x], Path(a.out), Path(a.rect))
    else:
        p.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
