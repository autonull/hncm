"""Datasets: one cache, one deterministic split, one dict per corpus.

`enwik8_1m` / `enwik8_10m` / `enwik8_100m` are prefixes of the 100 MB dump
(cached at `$HNCM_CACHE/enwik8`, downloaded from mattmahoney.net). The split is
the same as the v6/v7 corpora: the first 90% is the train, the last 10% is the
val, and the train is cut 90/10 into the fit and the selection block by
`concept_lm.split`. Every corpus carries its SHA-256, so a run has provenance.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
import zipfile
from pathlib import Path

import torch

CACHE = Path(os.environ.get("HNCM_CACHE", str(Path.home() / ".cache/hncm")))
URL = "https://mattmahoney.net/dc/enwik8.zip"
SIZES = {"enwik8_1m": 1_000_000, "enwik8_10m": 10_000_000, "enwik8_100m": 100_000_000}


def fetch_enwik8() -> bytes:
    path = CACHE / "enwik8"
    if path.exists():
        return path.read_bytes()
    CACHE.mkdir(parents=True, exist_ok=True)
    zpath = CACHE / "enwik8.zip"
    if not zpath.exists():
        with urllib.request.urlopen(URL, timeout=120) as r:
            zpath.write_bytes(r.read())
    with zipfile.ZipFile(zpath) as z:
        data = z.read("enwik8")
    path.write_bytes(data)
    return data


def get(name: str) -> dict:
    """The same shape as `hncm6.get_corpus`: tr/va tensors, name, sha, nbytes."""
    if name not in SIZES:
        import hncm6

        return hncm6.get_corpus(name)
    raw = fetch_enwik8()[: SIZES[name]]
    arr = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()
    n = (int(0.9 * len(arr)) // 4) * 4
    return {
        "tr": arr[:n],
        "va": arr[n:],
        "name": name,
        "sha": hashlib.sha256(raw).hexdigest(),
        "nbytes": len(raw),
    }


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="H-NCM corpora")
    sub = p.add_subparsers(dest="cmd")
    f = sub.add_parser("fetch")
    f.add_argument("--corpus", default="enwik8_10m")
    a = p.parse_args(argv)
    if a.cmd == "fetch":
        n = SIZES.get(a.corpus, 100_000_000)
        raw = fetch_enwik8()[:n]
        print(
            f"{a.corpus}: {len(raw)} B sha={hashlib.sha256(raw).hexdigest()[:16]} "
            f"cache={CACHE}"
        )
        return 0
    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
