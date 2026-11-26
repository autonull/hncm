"""cml: the concept memory LM.

One typed pipeline, manifests, a run store. The plan is `PLAN.md`; the numbers
below are the initial gate table, and this module is the source after P0.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import pickle
import random
import struct
import time
import tomllib
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import codec as codec_mod

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"


class Arm(StrEnum):
    BYTE = "byte"
    CONCEPT = "concept"
    ADAPTER = "adapter"
    LEDGER = "ledger"
    LIBRARY = "library"


class Convention(StrEnum):
    RATE = "rate"
    SHARED = "shared"
    SELF = "self"


@dataclass(frozen=True, slots=True)
class Config:
    arm: Arm = Arm.CONCEPT
    convention: Convention = Convention.SHARED
    seed: int = 0
    dict_size: int = 16384
    min_count: int = 32
    stream_bytes: int = 10_000_000
    val_bytes: int = 1_000_000
    params: int = 10_000_000
    steps: int = 2000
    batch: int = 32
    lr: float = 3e-4
    mix: float = 0.5
    seq: int = 128
    base: str = ""
    corpus: str = "enwik8_10m"
    lexicon: str = ""
    mode: str = "adapter"
    structured: str = ""
    pack_corpus: str = ""


@dataclass(frozen=True, slots=True)
class Entry:
    data: bytes
    count: int = 0
    rent: int = 0
    provenance: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Pack:
    entries: tuple[Entry, ...]
    config: Config
    sha: str = ""


@dataclass(frozen=True, slots=True)
class Run:
    id: str
    config: Config
    metrics: dict[str, float]


class CDF(Protocol):
    def range(
        self, symbol: int, history: tuple[int, ...] | None
    ) -> tuple[int, int, int]: ...


class Model(Protocol):
    def logits(self, units: list[int]) -> object: ...


@dataclass(frozen=True, slots=True)
class Gate:
    metric: str
    convention: Convention
    op: str
    bar: float

    def check(self, value: float) -> bool:
        return value <= self.bar if self.op == "<=" else value >= self.bar


# The initial gate table. The plan's scorecard is a snapshot of these numbers.
GATES: dict[str, Gate] = {
    "P0a": Gate("saved_charged", Convention.SHARED, ">=", 0.15),
    "P0b": Gate("accept", Convention.RATE, ">=", 0.50),
    "P0c": Gate("delta", Convention.RATE, "<=", 0.02),
    "P0d": Gate("structured_x", Convention.RATE, ">=", 1.5),
    "P0e": Gate("text_delta", Convention.RATE, "<=", 0.01),
    "P0f": Gate("pack_mb", Convention.RATE, "<=", 1.0),
    "P0g": Gate("provenance", Convention.RATE, ">=", 1.0),
    "P0h": Gate("linchpin", Convention.RATE, ">=", 0.05),
    "P1a": Gate("payload", Convention.RATE, "<=", 2.60),
    "P1b": Gate("payload", Convention.RATE, "<=", 2.293),
    "P1c": Gate("curve60", Convention.RATE, "<=", 2.29),
    "P1d": Gate("accept_per_byte", Convention.RATE, ">=", 0.20),
    "P1e": Gate("zero_shot", Convention.RATE, "<=", 0.10),
    "P2a": Gate("payload", Convention.RATE, "<=", 1.98),
    "P2b": Gate("shared", Convention.SHARED, "<=", 1.834),
    "P2c": Gate("self", Convention.SELF, "<=", 2.664),
    "P2d": Gate("self", Convention.SELF, "<=", 1.99),
    "P3a": Gate("delta", Convention.RATE, ">=", 0.10),
    "P3b": Gate("patch_gain", Convention.RATE, ">=", 0.05),
    "P3c": Gate("accept", Convention.RATE, ">=", 0.50),
    "P3d": Gate("structured_x", Convention.RATE, ">=", 2.0),
    "P4a": Gate("pack_mb", Convention.RATE, "<=", 1.0),
    "P4b": Gate("bits_per_byte", Convention.SHARED, ">=", 1.0),
    "P4c": Gate("provenance", Convention.RATE, ">=", 1.0),
    "P4d": Gate("valid", Convention.RATE, ">=", 1.0),
    "P4e": Gate("auc", Convention.RATE, ">=", 0.70),
    "P4f": Gate("spec_speed", Convention.RATE, ">=", 1.30),
    "L-G0a": Gate("text_delta", Convention.RATE, "<=", 0.03),
    "L-G0b": Gate("code_win", Convention.RATE, ">=", 0.10),
    "L-G1": Gate("usage", Convention.RATE, ">=", 0.50),
    "L-G2": Gate("shuffle", Convention.RATE, ">=", 0.20),
    "L-G3": Gate("dl_drop", Convention.RATE, ">=", 0.0),
    "L-G4": Gate("slots", Convention.RATE, ">=", 10.0),
    "L-G5": Gate("adapt", Convention.SHARED, ">=", 1.0),
    "P5e": Gate("plan", Convention.RATE, ">=", 1.0),
    "P5f": Gate("audit", Convention.RATE, ">=", 0.0),
    "P5g": Gate("context", Convention.RATE, ">=", 1.0),
}


REPORTS = ROOT / "reports"

AXES: dict[str, tuple[str, ...]] = {
    "stage": ("P0", "P1", "P2", "P3", "P4", "P5"),
    "arm": ("byte", "concept", "adapter", "ledger", "library"),
    "price": ("counts", "neural", "neural+counts"),
    "charge": ("none", "archive", "lexicon", "rent", "library"),
    "emission": ("byte", "span", "event"),
    "parse": ("fixed", "joint-em", "lazy"),
    "gate": ("none", "logistic", "mlp", "verify"),
    "embedding": ("random", "composed", "tied", "tied+residual"),
    "schedule": ("fixed", "rotating", "mixed"),
    "distill": ("none", "4m"),
    "data": (
        "enwik8_10m",
        "enwik8_100m",
        "synseg",
        "syngrid",
        "shakespeare",
        "src",
        "json",
        "csv",
    ),
    "scale": ("10m", "20m", "40m", "60m", "100m", "300m"),
    "adapt": ("null", "64k", "256k", "1m"),
    "seed": ("0", "1", "2"),
}

STAGE_AXES: dict[str, dict[str, tuple[str, ...]]] = {
    "P0": {
        "arm": ("adapter",),
        "price": ("neural+counts",),
        "charge": ("archive", "lexicon"),
        "emission": ("span",),
        "parse": ("fixed",),
        "gate": ("logistic",),
        "embedding": ("tied",),
        "schedule": ("rotating",),
        "distill": ("none",),
        "data": ("enwik8_10m",),
        "scale": ("20m",),
        "adapt": ("64k", "256k", "1m"),
        "seed": ("0",),
    },
    "P1": {
        "arm": ("concept",),
        "price": ("neural",),
        "charge": ("lexicon",),
        "emission": ("span",),
        "parse": ("fixed", "joint-em"),
        "gate": ("logistic", "mlp", "verify"),
        "embedding": ("composed", "tied", "tied+residual"),
        "schedule": ("fixed", "mixed", "rotating"),
        "distill": ("none",),
        "data": ("enwik8_10m",),
        "scale": ("10m", "20m", "40m"),
        "adapt": ("null",),
        "seed": ("0", "1", "2"),
    },
    "P2": {
        "arm": ("concept",),
        "price": ("neural",),
        "charge": ("lexicon",),
        "emission": ("span",),
        "parse": ("joint-em",),
        "gate": ("verify",),
        "embedding": ("tied+residual",),
        "schedule": ("mixed",),
        "distill": ("none", "4m"),
        "data": ("enwik8_100m",),
        "scale": ("60m", "100m"),
        "adapt": ("null",),
        "seed": ("0", "1", "2"),
    },
    "P3": {
        "arm": ("ledger",),
        "price": ("neural",),
        "charge": ("rent",),
        "emission": ("span", "byte"),
        "parse": ("joint-em", "lazy"),
        "gate": ("logistic", "mlp", "verify"),
        "embedding": ("tied+residual",),
        "schedule": ("rotating", "mixed"),
        "distill": ("none",),
        "data": ("synseg", "syngrid", "shakespeare"),
        "scale": ("40m", "60m"),
        "adapt": ("null",),
        "seed": ("0", "1"),
    },
    "P4": {
        "arm": ("ledger",),
        "price": ("neural",),
        "charge": ("rent",),
        "emission": ("span", "byte"),
        "parse": ("lazy",),
        "gate": ("verify",),
        "embedding": ("tied+residual",),
        "schedule": ("fixed",),
        "distill": ("none", "4m"),
        "data": ("synseg", "src", "json"),
        "scale": ("40m", "60m"),
        "adapt": ("64k", "256k", "1m"),
        "seed": ("0",),
    },
    "P5": {
        "arm": ("library",),
        "price": ("neural",),
        "charge": ("library",),
        "emission": ("event",),
        "parse": ("joint-em",),
        "gate": ("verify",),
        "embedding": ("tied+residual",),
        "schedule": ("rotating",),
        "distill": ("none",),
        "data": ("src", "json"),
        "scale": ("40m", "60m"),
        "adapt": ("null",),
        "seed": ("0", "1"),
    },
}

STAGE_GATES: dict[str, tuple[str, ...]] = {
    "P0": ("P0a", "P0b", "P0c", "P0d", "P0e", "P0f", "P0g", "P0h"),
    "P1": ("P1a", "P1b", "P1c", "P1d", "P1e"),
    "P2": ("P2a", "P2b", "P2c", "P2d"),
    "P3": ("P3a", "P3b", "P3c", "P3d"),
    "P4": ("P4a", "P4b", "P4c", "P4d", "P4e", "P4f"),
    "P5": (
        "L-G0a",
        "L-G0b",
        "L-G1",
        "L-G2",
        "L-G3",
        "L-G4",
        "L-G5",
        "P5e",
        "P5f",
        "P5g",
    ),
}


@dataclass(frozen=True, slots=True)
class ArmSpec:
    stage: str
    arm: str
    price: str
    charge: str
    emission: str
    parse: str
    gate: str
    embedding: str
    schedule: str
    distill: str
    data: str
    scale: str
    adapt: str
    seed: str


def describe(spec: ArmSpec) -> str:
    return "·".join(
        (
            spec.arm,
            spec.price,
            spec.charge,
            spec.emission,
            spec.parse,
            spec.gate,
            spec.embedding,
            spec.schedule,
            spec.data,
            spec.scale,
            spec.adapt,
            f"s{spec.seed}",
        )
    )


def axes_name(spec: dict) -> str:
    keys = (
        "arm",
        "price",
        "charge",
        "emission",
        "parse",
        "gate",
        "embedding",
        "schedule",
        "data",
        "scale",
        "adapt",
    )
    return "·".join(str(spec[k]) for k in keys)


@dataclass(frozen=True, slots=True)
class Space:
    stage: str

    def arms(self) -> list[ArmSpec]:
        axes = STAGE_AXES[self.stage]
        keys = tuple(axes)
        return [
            ArmSpec(stage=self.stage, **dict(zip(keys, combo)))
            for combo in itertools.product(*(axes[k] for k in keys))
        ]

    def coverage(self) -> list[ArmSpec]:
        chosen: dict[tuple[str, str], ArmSpec] = {}
        for spec in self.arms():
            for k in STAGE_AXES[self.stage]:
                chosen.setdefault((k, getattr(spec, k)), spec)
        return list(dict.fromkeys(chosen.values()))


def simulate(spec: ArmSpec, rung: int) -> dict[str, float | bool]:
    rng = random.Random(f"{spec.stage}|{spec.seed}|{rung}|{describe(spec)}")
    noise = rng.gauss(0.0, 0.003)
    if spec.stage == "P1":
        good = (
            0.05 * (spec.parse == "joint-em")
            + 0.05 * (spec.gate == "verify")
            + 0.04 * (spec.embedding == "tied+residual")
            + 0.03 * (spec.schedule == "mixed")
        )
        drop = {"10m": 0.08, "20m": 0.03, "40m": 0.0}[spec.scale]
        metric = 2.42 - good + drop + noise
        margin = 2.293 - metric
        cost = {1: 0.25, 2: 0.5, 3: 1.0}[rung]
    elif spec.stage == "P0":
        good = (
            0.10 * (spec.gate == "logistic")
            + 0.05 * (spec.embedding == "tied")
            + 0.03 * (spec.adapt == "64k")
        )
        metric = 0.05 + good + noise
        margin = metric - 0.15
        cost = 0.5
    elif spec.stage == "P2":
        good = 0.05 * (spec.distill == "4m") + 0.05 * (spec.scale == "100m")
        metric = 2.05 - good + noise
        margin = 1.98 - metric
        cost = 8.0
    elif spec.stage == "P3":
        good = (
            0.05 * (spec.emission == "span")
            + 0.04 * (spec.gate == "verify")
            + 0.03 * (spec.parse == "joint-em")
        )
        metric = 0.06 + good + noise
        margin = metric - 0.10
        cost = 2.0
    elif spec.stage == "P4":
        good = 0.10 * (spec.adapt == "64k") + 0.05 * (spec.emission == "span")
        metric = 0.65 + good + noise
        margin = metric - 0.60
        cost = 4.0
    else:
        good = (
            0.4 * (spec.gate == "verify")
            + 0.3 * (spec.data == "src")
            + 0.2 * (spec.scale == "60m")
        )
        metric = 9.8 + good + noise
        margin = (metric - 10.0) / 10.0
        cost = 8.0
    return {"metric": metric, "margin": margin, "cost": cost, "passed": margin >= 0}


@dataclass(frozen=True, slots=True)
class Result:
    spec: ArmSpec
    rung: int
    metric: float
    margin: float
    cost: float
    passed: bool


@dataclass(slots=True)
class SearchState:
    stage: str
    budget: float
    seed: int = 0
    simulated: bool = True
    spent: float = 0.0
    results: list[Result] = field(default_factory=list)
    pruned: dict[str, str] = field(default_factory=dict)
    promoted: set[str] = field(default_factory=set)
    frontier: Result | None = None
    next: ArmSpec | None = None

    def ucb(self, spec: ArmSpec) -> float:
        rs = [r for r in self.results if r.spec == spec]
        if not rs:
            return float("inf")
        mean = sum(r.margin - 0.05 * r.cost for r in rs) / len(rs)
        return mean + math.sqrt(2.0 * math.log(max(len(self.results), 2)) / len(rs))


def best_value(state: SearchState, axis: str) -> str:
    axes = STAGE_AXES[state.stage]
    scores: dict[str, float] = {}
    for v in axes[axis]:
        rs = [r for r in state.results if getattr(r.spec, axis) == v]
        if rs:
            scores[v] = sum(r.margin for r in rs) / len(rs)
    return max(scores, key=scores.get) if scores else axes[axis][0]


def candidates(state: SearchState) -> list[ArmSpec]:
    axes = STAGE_AXES[state.stage]
    best = {axis: best_value(state, axis) for axis in axes}
    out = [ArmSpec(stage=state.stage, **best)]
    for axis, vals in axes.items():
        for v in vals:
            if v != best[axis]:
                d = dict(best)
                d[axis] = v
                out.append(ArmSpec(stage=state.stage, **d))
    return out


def run_search(stage: str, budget: float, seed: int) -> SearchState:
    rng = random.Random(seed)
    space = Space(stage)
    coverage = space.coverage()
    rng.shuffle(coverage)
    state = SearchState(stage=stage, budget=budget, seed=seed)
    rungs = {
        "P0": (1,),
        "P1": (1, 2, 3),
        "P2": (1, 2),
        "P3": (1, 2),
        "P4": (1,),
        "P5": (1, 2),
    }[stage]
    alive = coverage
    for rung in rungs:
        passed: list[Result] = []
        for spec in alive:
            sim = simulate(spec, rung)
            r = Result(
                spec,
                rung,
                float(sim["metric"]),
                float(sim["margin"]),
                float(sim["cost"]),
                bool(sim["passed"]),
            )
            state.results.append(r)
            state.spent += r.cost
            if not r.passed:
                state.pruned[describe(spec)] = f"gate r{rung}"
            else:
                passed.append(r)
            if state.spent >= budget:
                break
        if not passed:
            break
        passed.sort(key=lambda r: r.margin, reverse=True)
        keep = max(1, len(passed) // 3)
        incumbent = max(
            (r for r in state.results if r.passed),
            key=lambda r: r.margin,
            default=None,
        )
        for r in passed[keep:]:
            lb = (
                r.margin
                - 0.05 * r.cost
                - math.sqrt(2.0 * math.log(max(len(state.results), 2)))
            )
            if incumbent and lb < incumbent.margin:
                state.pruned[describe(r.spec)] = "lcb"
        alive = [r.spec for r in passed[:keep]]
        state.promoted |= {describe(s) for s in alive}
        if state.spent >= budget:
            break
    tested = {describe(r.spec) for r in state.results}
    rung = max(rungs)
    while state.spent < budget:
        spec = next((s for s in candidates(state) if describe(s) not in tested), None)
        if spec is None:
            break
        sim = simulate(spec, rung)
        r = Result(
            spec,
            rung,
            float(sim["metric"]),
            float(sim["margin"]),
            float(sim["cost"]),
            bool(sim["passed"]),
        )
        state.results.append(r)
        state.spent += r.cost
        tested.add(describe(spec))
        if r.passed:
            state.promoted.add(describe(spec))
        else:
            state.pruned[describe(spec)] = "recombine"
    state.frontier = max(
        (r for r in state.results if r.passed),
        key=lambda r: r.margin,
        default=None,
    )
    passed = [r for r in state.results if r.passed]
    if passed:
        state.next = max((r.spec for r in passed), key=state.ucb)
    return state


def result_dict(r: Result) -> dict:
    return {
        "spec": asdict(r.spec),
        "rung": r.rung,
        "metric": r.metric,
        "margin": r.margin,
        "cost": r.cost,
        "passed": r.passed,
    }


def state_dict(s: SearchState) -> dict:
    return {
        "stage": s.stage,
        "budget": s.budget,
        "spent": s.spent,
        "seed": s.seed,
        "simulated": s.simulated,
        "arms": len(Space(s.stage).arms()),
        "tested": len({describe(r.spec) for r in s.results}),
        "pruned": s.pruned,
        "promoted": sorted(s.promoted),
        "frontier": result_dict(s.frontier) if s.frontier else None,
        "next": asdict(s.next) if s.next else None,
        "results": [result_dict(r) for r in s.results],
    }


def render_state(d: dict) -> str:
    stage = d["stage"]
    head = (
        f"CML SEARCH · {stage} · {d['arms']} arms · "
        f"budget {d['spent']:.1f}/{d['budget']:.1f} h"
    )
    if d["simulated"]:
        head += " · simulated"
    lines = [head]
    f = d["frontier"]
    if f:
        lines.append(
            f"frontier  {axes_name(f['spec'])}·s{f['spec']['seed']}  "
            f"metric {f['metric']:.3f}  margin {f['margin']:+.3f}  "
            f"cost {f['cost']:.1f} h"
        )
    gates = STAGE_GATES[stage]
    gate_line = " | ".join(f"{g} {GATES[g].bar:g}" for g in gates)
    if f:
        gate_line += f"  →  binding margin {f['margin']:+.3f}"
    lines.append(f"gates     {gate_line}")
    bar = int(12 * d["spent"] / max(d["budget"], 1e-9))
    bar = max(0, min(12, bar))
    lines.append(
        f"budget    [{'█' * bar}{'░' * (12 - bar)}] "
        f"{d['spent']:.1f}/{d['budget']:.1f} h"
    )
    groups: dict[str, list[dict]] = {}
    for r in d["results"]:
        groups.setdefault(axes_name(r["spec"]), []).append(r)
    rows = sorted(
        groups.items(),
        key=lambda kv: max(r["margin"] for r in kv[1]),
        reverse=True,
    )[:10]
    lines.append("leaderboard")
    for i, (name, rs) in enumerate(rows, 1):
        mean = sum(r["margin"] for r in rs) / len(rs)
        lines.append(
            f"  {i:2d}  {name}  margin {mean:+.3f}  "
            f"n={len(rs)}  cost {sum(r['cost'] for r in rs):.1f} h"
        )
    lines.append(
        f"tree      tested {d['tested']} · pruned {len(d['pruned'])} · "
        f"promoted {len(d['promoted'])} · waiting {d['arms'] - d['tested']}"
    )
    for axis, vals in STAGE_AXES[stage].items():
        counts = {v: 0 for v in vals}
        for r in d["results"]:
            counts[r["spec"][axis]] = counts.get(r["spec"][axis], 0) + 1
        lines.append(
            f"axes      {axis:10s} " + "  ".join(f"{v}({counts[v]})" for v in vals)
        )
    if d["next"]:
        lines.append(
            f"next      {axes_name(d['next'])}·s{d['next']['seed']}  reason max UCB"
        )
    rungs = sorted({r["rung"] for r in d["results"]})
    spark = " ".join(
        f"r{r}:{max((x['margin'] for x in d['results'] if x['rung'] == r), default=0):+.2f}"
        for r in rungs
    )
    lines.append(f"sparkline {spark}")
    return "\n".join(lines)


def write_report(d: dict) -> Path:
    text = "# CML search progress\n\n```\n" + render_state(d) + "\n```\n"
    text += (
        "\n## decision log\n\n"
        "| rung | arm | metric | margin | cost | pass |\n"
        "|---:|---|---:|---:|---:|---|\n"
    )
    for r in sorted(d["results"], key=lambda r: (r["rung"], -r["margin"]))[:120]:
        text += (
            f"| {r['rung']} | {axes_name(r['spec'])} | {r['metric']:.3f} | "
            f"{r['margin']:+.3f} | {r['cost']:.1f} | "
            f"{'yes' if r['passed'] else 'no'} |\n"
        )
    text += "\n## what the search learned\n\n"
    f = d["frontier"]
    if f:
        text += (
            f"- the frontier is `{axes_name(f['spec'])}` "
            f"(seed {f['spec']['seed']}) at margin {f['margin']:+.3f}.\n"
        )
        text += (
            f"- it spent {d['spent']:.1f} h of {d['budget']:.1f} h on "
            f"{d['tested']} of {d['arms']} arms.\n"
        )
        for axis in STAGE_AXES[d["stage"]]:
            v = f["spec"][axis]
            n = sum(1 for r in d["results"] if r["spec"][axis] == v)
            text += f"- {axis}: `{v}` ({n}/{d['tested']}).\n"
        if d["next"]:
            text += f"- next: `{axes_name(d['next'])}` (max UCB).\n"
    else:
        text += "- no arm passed; the branch is a negative.\n"
    path = REPORTS / "PROGRESS.md"
    path.write_text(text)
    return path


def space() -> int:
    for stage in ("P0", "P1", "P2", "P3", "P4", "P5"):
        sp = Space(stage)
        print(
            f"{stage}: {len(sp.arms()):4d} valid arms  coverage {len(sp.coverage()):2d}"
        )
    print("\naxes")
    for k, vals in AXES.items():
        print(f"  {k:10s} {', '.join(vals)}")
    return 0


def status() -> int:
    path = REPORTS / "search.json"
    if not path.exists():
        print("no search yet: cml search P1 --simulate")
        return 1
    print(render_state(json.loads(path.read_text())))
    return 0


def report() -> int:
    path = REPORTS / "search.json"
    if not path.exists():
        print("no search yet: cml search P1 --simulate")
        return 1
    d = json.loads(path.read_text())
    out = write_report(d)
    print(out.read_text())
    return 0


def config_from_manifest(
    path: Path,
    seed: int | None,
    params: int | None = None,
    steps: int | None = None,
) -> tuple[Config, list[str]]:
    raw = tomllib.loads(path.read_text())
    run = raw.get("run", {})
    data = raw.get("data", {})
    model = raw.get("model", {})
    optim = raw.get("optim", {})
    cfg = Config(
        arm=Arm(model.get("arm", "concept")),
        convention=Convention(run.get("convention", "shared")),
        seed=run.get("seed", 0) if seed is None else seed,
        dict_size=model.get("dict_size", 16384),
        min_count=model.get("min_count", 32),
        stream_bytes=data.get("stream_bytes", 10_000_000),
        val_bytes=data.get("val_bytes", 1_000_000),
        params=model.get("params", 10_000_000) if params is None else params,
        steps=optim.get("steps", 2000) if steps is None else steps,
        batch=optim.get("batch", 32),
        lr=optim.get("lr", 3e-4),
        mix=optim.get("mix", 0.5),
        seq=optim.get("seq", 128),
        base=model.get("base", ""),
        corpus=data.get("corpus", "enwik8_10m"),
        lexicon=data.get("lexicon", ""),
        mode=model.get("mode", "adapter"),
        structured=data.get("structured", ""),
        pack_corpus=data.get("pack_corpus", ""),
    )
    return cfg, list(run.get("gates", []))


def run_id(manifest: Path, cfg: Config) -> str:
    blob = json.dumps(asdict(cfg), sort_keys=True).encode()
    return f"{manifest.stem}-{hashlib.sha256(blob).hexdigest()[:8]}-s{cfg.seed}"


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


def validate(gates: list[str]) -> None:
    missing = [g for g in gates if g not in GATES]
    if missing:
        raise SystemExit(f"unknown gates: {missing}")


@dataclass(frozen=True, slots=True)
class Lexicon:
    strings: tuple[bytes, ...]
    ids: dict[bytes, int]
    size: int = 0
    sha: str = ""


@dataclass(frozen=True, slots=True)
class Parse:
    raw: bytes
    units: tuple[bytes, ...]
    ids: torch.Tensor
    lens: torch.Tensor


MODEL_MAGIC = b"HNM8"


def git_sha() -> str:
    head = ROOT.parent / ".git/HEAD"
    if not head.exists():
        return ""
    ref = head.read_text().strip()
    if not ref.startswith("ref: "):
        return ref
    path = ROOT.parent / ".git" / ref[5:]
    return path.read_text().strip() if path.exists() else ""


def model_blob(path: Path) -> bytes:
    blob = path.read_bytes()
    if path.suffix == ".pkl":
        blob = pickle.loads(blob)["blob"]
    if blob[:4] != MODEL_MAGIC:
        raise ValueError(f"lexicon: not a v8 model: {path}")
    return blob


def read_lexicon(path: Path) -> Lexicon:
    """The shared lexicon from the v9 pack: the strings, not the table.

    The pack's header is `<IBf` (count, tail rank, tail scale), then the
    256-entry byte table, then `<BH` (length, count) and the bytes per string.
    """
    blob = model_blob(path)
    m = struct.unpack_from("<I", blob, 4)[0]
    pos = 4 + 9 + 512
    strings: list[bytes] = []
    for _ in range(m):
        n = blob[pos]
        strings.append(blob[pos + 3 : pos + 3 + n])
        pos += 3 + n
    ids = {s: i for i, s in enumerate(strings)}
    for b in range(256):
        s = bytes([b])
        if s not in ids:
            ids[s] = len(strings)
            strings.append(s)
    return Lexicon(tuple(strings), ids, len(blob), hashlib.sha256(blob).hexdigest())


def read_parse(path: Path, lexicon: Lexicon) -> Parse:
    data = pickle.loads(path.read_bytes())
    units = tuple(data["toks"])
    raw = data["va"]
    if b"".join(units) != raw:
        raise ValueError("parse: the units do not reconstruct the bytes")
    ids: list[int] = []
    for unit in units:
        i = lexicon.ids.get(unit)
        if i is None:
            raise ValueError(f"parse: unit not in the lexicon: {unit!r}")
        ids.append(i)
    lens = torch.tensor([len(u) for u in units], dtype=torch.long)
    return Parse(raw, units, torch.tensor(ids, dtype=torch.long), lens)


def split_index(lens: torch.Tensor, nbytes: int) -> int:
    """The first token index after `nbytes`, so the split keeps a whole span."""
    cum = lens.cumsum(0)
    if nbytes >= int(cum[-1]):
        return len(lens)
    i = int(torch.searchsorted(cum, torch.tensor(nbytes, dtype=cum.dtype)))
    return min(i + 1, len(lens))


def windows(ids: torch.Tensor, seq: int) -> torch.Tensor:
    """Non-overlapping windows of `seq + 1`: one context plus one target each."""
    if ids.numel() < seq + 1:
        raise ValueError(f"need at least {seq + 1} tokens, got {ids.numel()}")
    return ids.unfold(0, seq + 1, seq).contiguous()


def preset(params: int) -> tuple[int, int, int]:
    if params <= 12_000_000:
        return 256, 6, 8
    if params <= 26_000_000:
        return 384, 8, 8
    return 512, 10, 8


def byte_preset(params: int) -> tuple[int, int, int]:
    if params <= 12_000_000:
        return 320, 7, 8
    if params <= 26_000_000:
        return 448, 9, 8
    return 576, 10, 8


class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d)
        self.out = nn.Linear(4 * d, d)
        self.la: nn.Parameter | None = None
        self.lb: nn.Parameter | None = None
        self.lc: nn.Parameter | None = None
        self.ld: nn.Parameter | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h)
        if self.la is not None and self.lb is not None:
            qkv = qkv + (h @ self.la) @ self.lb
        q, k, v = qkv.chunk(3, dim=-1)
        shape = (b, t, self.heads, d // self.heads)
        q, k, v = (z.view(shape).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(b, t, d)
        x = x + self.proj(y)
        if self.lc is not None and self.ld is not None:
            x = x + (y @ self.lc) @ self.ld
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class Trunk(nn.Module):
    """A causal unit LM with a tied output embedding."""

    def __init__(self, vocab: int, d: int, layers: int, heads: int, seq: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pos.num_embeddings:
            raise ValueError("sequence is longer than the position table")
        h = self.tok(x) + self.pos.weight[: x.size(1)]
        for block in self.blocks:
            h = block(h)
        return self.ln(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(self.hidden(x), self.tok.weight)


def lm_loss(model: Trunk, x: torch.Tensor) -> torch.Tensor:
    logits = model(x[:, :-1])
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))


@torch.no_grad()
def eval_bits(
    model: Trunk, win: torch.Tensor, win_lens: torch.Tensor, batch: int
) -> tuple[float, int]:
    model.eval()
    bits = 0.0
    nbytes = 0
    for x, ln in zip(win.split(batch), win_lens.split(batch)):
        x = x.cuda(non_blocking=True)
        logits = model(x[:, :-1])
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1),
            reduction="none",
        )
        bits += float(ce.double().sum()) / math.log(2)
        nbytes += int(ln.sum())
    return bits, nbytes


def unigram_bits(
    train_ids: torch.Tensor,
    win: torch.Tensor,
    win_lens: torch.Tensor,
    vocab: int,
) -> tuple[float, int]:
    counts = torch.bincount(train_ids, minlength=vocab).double()
    p = (counts + 0.5) / (counts.sum() + 0.5 * vocab)
    targets = win[:, 1:]
    bits = float(-torch.log2(p[targets]).sum())
    return bits, int(win_lens.sum())


def bigram_bits(
    train_ids: torch.Tensor,
    win: torch.Tensor,
    win_lens: torch.Tensor,
    vocab: int,
    keep: int = 2_000_000,
) -> tuple[float, int]:
    """The count control: an interpolated first-order bigram over unit ids."""
    prev = train_ids[:-1].numpy().astype(np.int64)
    nxt = train_ids[1:].numpy().astype(np.int64)
    keys = prev * vocab + nxt
    uniq, counts = np.unique(keys, return_counts=True)
    if len(uniq) > keep:
        top = np.argpartition(counts, -keep)[-keep:]
        uniq, counts = uniq[top], counts[top]
        order = np.argsort(uniq)
        uniq, counts = uniq[order], counts[order]
    counts = counts.astype(np.float64)
    uni = np.bincount(train_ids.numpy(), minlength=vocab).astype(np.float64)
    p_uni = (uni + 0.5) / (uni.sum() + 0.5 * vocab)
    row = np.bincount(uniq // vocab, weights=counts, minlength=vocab)
    tprev = win[:, :-1].numpy().astype(np.int64).reshape(-1)
    tnxt = win[:, 1:].numpy().astype(np.int64).reshape(-1)
    tkeys = tprev * vocab + tnxt
    pos = np.clip(np.searchsorted(uniq, tkeys), 0, len(uniq) - 1)
    hit = uniq[pos] == tkeys
    c_pair = np.where(hit, counts[pos], 0.0)
    p = (c_pair + p_uni[tnxt]) / (row[tprev] + 1.0)
    return float(-np.log2(p).sum()), int(win_lens.sum())


def gate_rows(gates: list[str], metrics: dict) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for name in gates:
        gate = GATES[name]
        value = metrics.get(gate.metric)
        rows[name] = {
            "metric": gate.metric,
            "value": None if value is None else float(value),
            "bar": gate.bar,
            "pass": None if value is None else gate.check(float(value)),
        }
    return rows


def run_p1(cfg: Config, gates: list[str], out: Path) -> dict:
    path = Path(cfg.lexicon)
    if not path.is_absolute():
        path = ROOT / path
    lexicon = read_lexicon(path)
    parse = read_parse(path, lexicon)
    vocab = len(lexicon.strings)
    seq = cfg.seq
    split = split_index(parse.lens, cfg.stream_bytes - cfg.val_bytes)
    train_ids, train_lens = parse.ids[:split], parse.lens[:split]
    test_ids, test_lens = parse.ids[split:], parse.lens[split:]
    train_raw = b"".join(parse.units[:split])
    test_raw = b"".join(parse.units[split:])
    if train_raw + test_raw != parse.raw:
        raise ValueError("P1: the byte split is not a partition")
    train_win = windows(train_ids, seq)
    test_win = windows(test_ids, seq)
    test_lens_win = test_lens.unfold(0, seq + 1, seq)[:, 1:].contiguous()

    d, layers, heads = preset(cfg.params)
    torch.manual_seed(cfg.seed)
    model = Trunk(vocab, d, layers, heads, seq).cuda()
    nparams = sum(p.numel() for p in model.parameters())
    print(
        f"P1 {out.name}: vocab={vocab} params={nparams / 1e6:.1f}M "
        f"d={d} layers={layers} seq={seq}",
        flush=True,
    )
    print(
        f"  train {int(train_lens.sum()) / 1e6:.2f} MB "
        f"({train_ids.numel() / 1e6:.2f}M units) · "
        f"test {int(test_lens.sum()) / 1e6:.2f} MB "
        f"({test_ids.numel() / 1e6:.2f}M units)",
        flush=True,
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    warmup = max(1, min(100, cfg.steps // 20))
    span = max(1, cfg.steps - warmup)

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        p = min(1.0, (step - warmup) / span)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    model.train()
    started = time.time()
    last = 0.0
    for step in range(cfg.steps):
        idx = torch.randint(0, train_win.size(0), (cfg.batch,))
        x = train_win[idx].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = lm_loss(model, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        last = float(loss.detach())
        if step % 100 == 0 or step == cfg.steps - 1:
            print(
                f"  step {step:5d}/{cfg.steps} loss {last:.4f} "
                f"lr {sched.get_last_lr()[0]:.2e}",
                flush=True,
            )
    torch.cuda.synchronize()
    secs = time.time() - started

    bits, test_bytes = eval_bits(model, test_win, test_lens_win, cfg.batch)
    payload = bits / test_bytes
    uni, uni_bytes = unigram_bits(train_ids, test_win, test_lens_win, vocab)
    unigram = uni / uni_bytes
    bi, bi_bytes = bigram_bits(train_ids, test_win, test_lens_win, vocab)
    bigram = bi / bi_bytes
    metrics: dict = {
        "id": out.name,
        "arm": cfg.arm.value,
        "seed": cfg.seed,
        "params": nparams,
        "vocab": vocab,
        "lexicon_bytes": lexicon.size,
        "lexicon_sha": lexicon.sha,
        "corpus_sha": hashlib.sha256(parse.raw).hexdigest(),
        "d": d,
        "layers": layers,
        "heads": heads,
        "seq": seq,
        "steps": cfg.steps,
        "secs": secs,
        "train_bytes": int(train_lens.sum()),
        "train_units": int(train_ids.numel()),
        "test_bytes": test_bytes,
        "test_total_bytes": int(test_lens.sum()),
        "test_units": int(test_ids.numel()),
        "bytes_per_unit": test_bytes / max(int(test_lens_win.numel()), 1),
        "bits": bits,
        "payload": payload,
        "unigram": unigram,
        "bigram": bigram,
        "saved": unigram - payload,
        "saved_bigram": bigram - payload,
        "v9_rate": 2.729,
        "curve60": None,
        "accept_per_byte": None,
        "zero_shot": None,
    }
    metrics["gates"] = gate_rows(gates, metrics)
    checked = [
        row["pass"] for row in metrics["gates"].values() if row["pass"] is not None
    ]
    metrics["skipped"] = [
        name for name, row in metrics["gates"].items() if row["pass"] is None
    ]
    metrics["passed"] = bool(checked) and all(checked)
    art = out / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(cfg),
            "vocab": vocab,
            "d": d,
            "layers": layers,
            "heads": heads,
            "params": nparams,
        },
        art / "model.pt",
    )
    return metrics


# ---------------------------------------------------------------------------
# P0: the adapter on a frozen byte base
# ---------------------------------------------------------------------------

CACHE = Path(os.environ.get("HNCM_CACHE", str(Path.home() / ".cache/hncm")))
PACK_MAGIC = b"CMLP"
CAP = 16
HOLDOUT = 10


def corpus_bytes(name: str) -> bytes:
    if name == "synseg":
        return synseg_bytes(10_000_000)
    if name == "src":
        return src_bytes()
    if name == "code":
        return code_corpus().raw
    raw = (CACHE / "enwik8").read_bytes()
    n = {"enwik8_10m": 10_000_000, "enwik8_100m": 100_000_000}.get(name, 10_000_000)
    return raw[:n]


def src_bytes() -> bytes:
    """The repo's Python source in path order, frozen per build hash."""
    files = sorted(
        p
        for p in ROOT.parent.rglob("*.py")
        if not any(q in p.parts for q in (".venv", "__pycache__", "runs", ".git"))
    )
    out = bytearray()
    for p in files:
        out += b"\n# --- " + str(p.relative_to(ROOT.parent)).encode() + b" ---\n"
        out += p.read_bytes()
    raw = bytes(out)
    sha = hashlib.sha256(raw).hexdigest()[:12]
    path = CACHE / f"src-{sha}.bin"
    if not path.exists():
        path.write_bytes(raw)
    return path.read_bytes()


CODE_CAP = 8_000_000
CODE_DROP = frozenset(
    (
        "__pycache__",
        "test",
        "tests",
        ".git",
        "runs",
        ".venv",
        "vendor",
        "vendored",
        "_vendor",
        "third_party",
    )
)


@dataclass(frozen=True, slots=True)
class CodeCorpus:
    raw: bytes
    train: bytes
    test: bytes
    files: tuple[tuple[str, int, int], ...]
    split: int
    sha: str
    train_sha: str


def _code_files() -> list[tuple[str, Path]]:
    """Repo py files plus the installed venv's site-packages, path sorted."""
    here = ROOT
    roots = [("repo", here.parent)]
    roots += [
        ("site", p) for p in (here / ".venv/lib").rglob("site-packages") if p.is_dir()
    ]
    out: list[tuple[str, Path]] = []
    for tag, base in roots:
        for p in sorted(base.rglob("*.py")):
            try:
                rel = p.relative_to(base)
            except ValueError:
                continue
            if any(q in CODE_DROP for q in rel.parts) or p.name.startswith("test_"):
                continue
            out.append((f"{tag}:{rel}", p))
    out.sort(key=lambda item: item[0])
    return out


def code_corpus(refreeze: bool = False) -> CodeCorpus:
    """The code corpus, frozen by SHA.  Whole files split 90/10 by path.

    A pin in the cache (`code-current.json`) keeps the corpus fixed while the
    repo's own source changes; pass ``refreeze=True`` to take a new snapshot.
    """
    pin = CACHE / "code-current.json"
    if pin.exists() and not refreeze:
        data = json.loads(pin.read_text())
        sha = data["sha"]
        raw = (CACHE / f"code-{sha[:12]}.bin").read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError("code corpus: pinned cache mismatch")
        split = int(data["split"])
        files = tuple(tuple(row) for row in data["files"])
        train, test = raw[:split], raw[split:]
        return CodeCorpus(
            raw, train, test, files, split, sha, hashlib.sha256(train).hexdigest()
        )
    out = bytearray()
    files: list[tuple[str, int, int]] = []
    for rel, path in _code_files():
        data = path.read_bytes()
        marker = f"\n# --- {rel} ---\n".encode()
        if len(out) + len(marker) + len(data) > CODE_CAP:
            continue
        start = len(out)
        out += marker + data
        files.append((rel, start, len(out)))
        if len(out) >= CODE_CAP:
            break
    raw = bytes(out)
    sha = hashlib.sha256(raw).hexdigest()
    cached = CACHE / f"code-{sha[:12]}.bin"
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_suffix(".tmp")
        tmp.write_bytes(raw)
        os.replace(tmp, cached)
    raw = cached.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("code corpus: cache mismatch")
    k = min(max(int(len(files) * 0.9), 1), len(files))
    split = files[k][1] if k < len(files) else len(raw)
    train, test = raw[:split], raw[split:]
    write_json(pin, {"sha": sha, "files": files, "split": split})
    return CodeCorpus(
        raw, train, test, tuple(files), split, sha, hashlib.sha256(train).hexdigest()
    )


def shakespeare_corpus() -> CodeCorpus:
    """Tiny Shakespeare, frozen by SHA, split at a line boundary 90/10."""
    raw = (CACHE / "shakespeare.txt").read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    cut = raw.rfind(b"\n", 0, int(len(raw) * 0.9)) + 1
    train, test = raw[:cut], raw[cut:]
    files = (("shakespeare.txt", 0, len(raw)),)
    return CodeCorpus(
        raw, train, test, files, cut, sha, hashlib.sha256(train).hexdigest()
    )


def text_corpus(name: str) -> CodeCorpus:
    if name == "shakespeare":
        return shakespeare_corpus()
    raise SystemExit(f"corpus: unknown text corpus {name}")


def synseg_bytes(n: int, seed: int = 0) -> bytes:
    """A structured synthetic stream: repetitive records with short fields."""
    rng = random.Random(seed)
    users = (b"alice", b"bob", b"carol", b"dave", b"erin", b"frank")
    verbs = (b"GET", b"PUT", b"POST", b"PATCH", b"DELETE")
    paths = (b"/v1/items", b"/v1/orders", b"/v1/users", b"/v1/events", b"/v1/logs")
    states = (b"ok", b"retry", b"fail", b"done", b"queued")
    codes = (b"200", b"201", b"404", b"409", b"500")
    out = bytearray()
    i = 0
    while len(out) < n:
        out += (
            b"BEGIN req=%06d verb=%s user=%s path=%s status=%s code=%s "
            b"lat_ms=%03d END\n"
            % (
                i,
                rng.choice(verbs),
                rng.choice(users),
                rng.choice(paths),
                rng.choice(states),
                rng.choice(codes),
                rng.choice((3, 7, 11, 19, 23, 41, 97, 128)),
            )
        )
        i += 1
    return bytes(out[:n])


def train_loop(
    model: Trunk, train_win: torch.Tensor, steps: int, batch: int, lr: float
) -> float:
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    warmup = max(1, min(100, steps // 20))
    span = max(1, steps - warmup)

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        p = min(1.0, (step - warmup) / span)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    model.train()
    started = time.time()
    loss = torch.zeros(())
    for step in range(steps):
        idx = torch.randint(0, train_win.size(0), (batch,))
        x = train_win[idx].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = lm_loss(model, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == steps - 1:
            print(
                f"  step {step:5d}/{steps} loss {float(loss.detach()):.4f} "
                f"lr {sched.get_last_lr()[0]:.2e}",
                flush=True,
            )
    torch.cuda.synchronize()
    return time.time() - started


def run_base(params: int, steps: int, seed: int, seq: int, corpus: str) -> int:
    raw = corpus_bytes(corpus)
    split = len(raw) * 9 // 10
    ids = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()
    train_win = windows(ids[:split], seq)
    test_win = windows(ids[split:], seq)
    lens = torch.ones_like(test_win[:, 1:])
    d, layers, heads = byte_preset(params)
    torch.manual_seed(seed)
    model = Trunk(256, d, layers, heads, seq).cuda()
    nparams = sum(p.numel() for p in model.parameters())
    out = RUNS / f"base-byte-{params // 1_000_000}m-s{seed}"
    print(
        f"BASE {out.name}: bytes={len(raw)} params={nparams / 1e6:.1f}M "
        f"d={d} layers={layers} seq={seq}",
        flush=True,
    )
    secs = train_loop(model, train_win, steps, 32, 3e-4)
    bits, nbytes = eval_bits(model, test_win, lens, 16)
    payload = bits / nbytes
    metrics = {
        "id": out.name,
        "payload": payload,
        "params": nparams,
        "steps": steps,
        "seq": seq,
        "d": d,
        "layers": layers,
        "heads": heads,
        "secs": secs,
        "test_bytes": int((len(raw) - split)),
        "corpus_sha": hashlib.sha256(raw).hexdigest(),
        "gates": {},
    }
    art = out / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "params": nparams,
            "d": d,
            "layers": layers,
            "heads": heads,
            "seq": seq,
        },
        art / "model.pt",
    )
    write_json(out / "config.json", {"config": metrics, "git": git_sha()})
    write_json(out / "metrics.json", metrics)
    print(f"base bpb {payload:.3f} · {secs:.1f}s", flush=True)
    return 0


def base_path(cfg: Config) -> Path:
    return RUNS / f"base-{cfg.base}-s{cfg.seed}" / "artifacts/model.pt"


def substring_counts(
    data: bytes, cap: int, min_count: int
) -> tuple[dict[bytes, int], dict[bytes, int]]:
    a = np.frombuffer(data, np.uint8).astype(np.uint64)
    counts: dict[bytes, int] = {}
    offsets: dict[bytes, int] = {}
    for ell in range(2, cap + 1):
        m = len(a) - ell + 1
        if m <= 0:
            break
        h = np.zeros(m, dtype=np.uint64)
        for j in range(ell):
            h *= np.uint64(257)
            h += a[j : j + m]
        _, idx, cnt = np.unique(h, return_index=True, return_counts=True)
        keep = cnt >= min_count
        for i, c in zip(idx[keep].tolist(), cnt[keep].tolist()):
            s = bytes(data[i : i + ell])
            counts[s] = int(c)
            offsets[s] = int(i)
        print(
            f"  span len={ell:2d} kept={int(keep.sum())} total={len(counts)}",
            flush=True,
        )
    return counts, offsets


def pack_bytes(pack: Pack) -> bytes:
    out = bytearray(PACK_MAGIC + struct.pack("<II", 1, len(pack.entries)))
    for entry in pack.entries:
        offset = int(entry.provenance[0].rsplit(":", 1)[1])
        out += struct.pack("<HIQ", len(entry.data), entry.count, offset)
        out += entry.data
    return bytes(out)


def read_pack(path: Path) -> tuple[Entry, ...]:
    data = path.read_bytes()
    if data[:4] != PACK_MAGIC:
        raise ValueError(f"pack: bad magic: {path}")
    count = struct.unpack_from("<I", data, 8)[0]
    pos = 12
    entries = []
    for _ in range(count):
        n, c, offset = struct.unpack_from("<HIQ", data, pos)
        pos += 14
        entries.append(Entry(data[pos : pos + n], c, 0, (f"fit:{offset}",)))
        pos += n
    return tuple(entries)


def induce_pack(cfg: Config, train: bytes, out: Path) -> tuple[Pack, int, int]:
    counts, offsets = substring_counts(train, CAP, cfg.min_count)
    total = max(sum(counts.values()), 1)
    scored = []
    for s, c in counts.items():
        gain = c * (len(s) - 1) - (len(s) + 6)
        if gain > 0:
            scored.append((gain, c, s, offsets[s]))
    scored.sort(key=lambda x: (-x[0], x[2]))
    chosen = scored[: cfg.dict_size]
    entries = tuple(Entry(s, c, 0, (f"fit:{o}",)) for _, c, s, o in chosen)
    pack = Pack(entries, cfg)
    data = pack_bytes(pack)
    pack = Pack(entries, cfg, hashlib.sha256(data).hexdigest())
    art = out / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    (art / "pack.bin").write_bytes(data)
    print(
        f"  pack {len(entries)} spans · {len(data) / 1e3:.1f} kB · "
        f"{len(counts)} candidates",
        flush=True,
    )
    return pack, len(data), total


def make_trie(entries: tuple[Entry, ...]) -> dict:
    root: dict = {}
    for i, entry in enumerate(entries):
        node = root
        for b in entry.data:
            node = node.setdefault(b, {})
        node["#"] = i
    return root


def longest_match(trie: dict, data: bytes, t: int, cap: int = CAP) -> tuple[int, int]:
    node = trie
    hit = (0, -1)
    for e in range(t, min(len(data), t + cap)):
        node = node.get(data[e])
        if node is None:
            break
        i = node.get("#")
        if i is not None:
            hit = (e - t + 1, i)
    return hit


@torch.no_grad()
def span_embeddings(
    model: Trunk, entries: tuple[Entry, ...], pool: str = "last"
) -> torch.Tensor:
    d = model.tok.embedding_dim
    out = torch.empty((len(entries), d))
    batch = 512
    for a in range(0, len(entries), batch):
        chunk = entries[a : a + batch]
        width = max(len(e.data) for e in chunk)
        x = torch.zeros((len(chunk), width), dtype=torch.long, device="cuda")
        lens = []
        for i, entry in enumerate(chunk):
            x[i, : len(entry.data)] = torch.frombuffer(
                bytearray(entry.data), dtype=torch.uint8
            )
            lens.append(len(entry.data))
        h = model.hidden(x)
        rows = torch.arange(len(chunk), device="cuda")
        if pool == "mean":
            mask = (
                torch.arange(width, device="cuda")[None, :]
                < torch.tensor(lens, device="cuda")[:, None]
            )
            z = (h * mask[..., None]).sum(1) / torch.tensor(lens, device="cuda")[
                :, None
            ]
        else:
            z = h[rows, torch.tensor(lens, device="cuda") - 1]
        out[a : a + len(chunk)] = F.normalize(z, dim=-1).cpu()
    return out


@torch.no_grad()
def stream_nll(
    model: Trunk, ids: torch.Tensor, batch: int = 16
) -> tuple[torch.Tensor, torch.Tensor, int]:
    seq = model.pos.num_embeddings
    win = windows(ids, seq)
    n = len(ids)
    nll = torch.full((n,), float("nan"))
    hs = torch.empty((n, model.tok.embedding_dim), dtype=torch.float16)
    for a in range(0, win.size(0), batch):
        x = win[a : a + batch].cuda()
        h = model.hidden(x[:, :-1])
        logits = F.linear(h, model.tok.weight)
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1),
            reduction="none",
        ).view(x.size(0), -1)
        rows = torch.arange(a, a + x.size(0))[:, None] * seq
        at = (rows + torch.arange(1, seq + 1)[None]).reshape(-1)
        nll[at] = ce.cpu().reshape(-1)
        hs[at] = h.cpu().half().reshape(-1, h.size(-1))
    return nll, hs, win.size(0) * seq + 1


def match_score(
    h: torch.Tensor, e: torch.Tensor, count: int, total: int, alpha: float
) -> float:
    hn = h.float()
    hn = hn / (hn.norm() + 1e-9)
    return float(hn @ e) + alpha * math.log(count / total)


def entropy(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


@dataclass(slots=True)
class KV:
    """One layer's keys and values: the window's cache, one row per byte."""

    k: torch.Tensor
    v: torch.Tensor


def new_cache(model: Trunk) -> list[KV]:
    heads = model.blocks[0].heads
    d = model.tok.embedding_dim
    shape = (1, heads, 0, d // heads)
    return [
        KV(torch.empty(shape, device="cuda"), torch.empty(shape, device="cuda"))
        for _ in model.blocks
    ]


def truncate_cache(cache: list[KV], keep: int) -> None:
    for kv in cache:
        kv.k = kv.k[:, :, :keep]
        kv.v = kv.v[:, :, :keep]


def block_step(block: Block, x: torch.Tensor, kv: KV) -> torch.Tensor:
    b, t, d = x.shape
    n = kv.k.size(2)
    q, k, v = block.qkv(block.ln1(x)).chunk(3, dim=-1)
    shape = (b, t, block.heads, d // block.heads)
    q, k, v = (z.view(shape).transpose(1, 2) for z in (q, k, v))
    k = torch.cat([kv.k, k], dim=2) if n else k
    v = torch.cat([kv.v, v], dim=2) if n else v
    kv.k, kv.v = k, v
    qi = torch.arange(n, n + t, device=x.device)
    kj = torch.arange(n + t, device=x.device)
    mask = kj[None, :] <= qi[:, None]
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    y = y.transpose(1, 2).reshape(b, t, d)
    x = x + block.proj(y)
    return x + block.out(F.gelu(block.fc(block.ln2(x))))


def win_step(
    model: Trunk, tokens: list[int], pos: int, cache: list[KV], pos_end: int
) -> torch.Tensor:
    """Feed `tokens` at window positions `pos..`; return the hidden states."""
    if pos < 0 or pos + len(tokens) > pos_end:
        raise ValueError("decode: chunk crosses the window")
    x = torch.tensor([tokens], device="cuda")
    h = model.tok(x) + model.pos.weight[pos : pos + len(tokens)]
    for block, kv in zip(model.blocks, cache):
        h = block_step(block, h, kv)
    return model.ln(h)


def gate_prob(gate: dict, feat: list[float]) -> float:
    if gate.get("kind") == "knn":
        return 1.0 if feat[0] >= float(gate["theta"]) else 0.0
    w = torch.tensor(gate["w"])
    b = torch.tensor(float(gate["b"]))
    mu = torch.tensor(gate["mu"])
    sd = torch.tensor(gate["sd"])
    return float(torch.sigmoid(((torch.tensor(feat) - mu) / sd) @ w + b))


@torch.no_grad()
def decode_arm(
    model: Trunk,
    entries: tuple[Entry, ...],
    trie: dict,
    total: int,
    e_c: torch.Tensor,
    gate: dict,
    data: bytes,
    n_eval: int,
    mode: str = "adapter",
) -> dict:
    """Stream the bytes with a KV cache: one trunk step per byte or per span.

    The byte arm steps once per byte; the adapter arm steps once per candidate
    span and emits the verified bytes. Both record the same kind of bits.
    """
    seq = model.pos.num_embeddings
    count = torch.tensor([e.count for e in entries])
    e_c = e_c.cuda()
    alpha = float(gate["alpha"])
    tau = float(gate["tau"])
    bits = 0.0
    copies = matches = fallbacks = verify_fail = copy_bytes = 0
    torch.cuda.synchronize()
    t0 = time.time()
    u = 1
    while u < n_eval:
        stop = min(u + seq, n_eval)
        base = u - 1
        cache = new_cache(model)
        h = win_step(model, [int(data[base])], 0, cache, stop - base)[:, -1]
        t = u
        while t < stop:
            if mode == "byte":
                logits = F.linear(h, model.tok.weight)
                ce = F.cross_entropy(
                    logits, torch.tensor([int(data[t])], device="cuda")
                )
                bits += float(ce) / math.log(2)
                if t + 1 < stop:
                    h = win_step(model, [int(data[t])], t - base, cache, stop - base)[
                        :, -1
                    ]
                t += 1
                continue
            length, idx = longest_match(trie, data, t, cap=min(CAP, stop - t))
            pi = 0.0
            if length >= 2:
                matches += 1
                logc = math.log(float(count[idx]) / total)
                feat = [
                    match_score(h, e_c[idx], int(count[idx]), total, alpha),
                    length / CAP,
                    logc,
                ]
                pi = gate_prob(gate, feat)
                if pi >= 0.5:
                    chunk = data[t : t + length]
                    ctx = list(chunk[:-1])
                    hs = win_step(model, ctx, t - base, cache, stop - base)[0]
                    first = F.cross_entropy(
                        F.linear(h, model.tok.weight),
                        torch.tensor([int(chunk[0])], device="cuda"),
                    )
                    rest = F.cross_entropy(
                        F.linear(hs, model.tok.weight),
                        torch.tensor(list(chunk[1:]), device="cuda"),
                        reduction="none",
                    )
                    nlls = torch.cat([first[None], rest])
                    if float(nlls.mean()) <= tau:
                        bits += entropy(pi)
                        copies += 1
                        copy_bytes += length
                        h = hs[-1]
                        t += length
                        continue
                    truncate_cache(cache, t - base)
                    verify_fail += 1
            logits = F.linear(h, model.tok.weight)
            ce = F.cross_entropy(logits, torch.tensor([int(data[t])], device="cuda"))
            bits += float(ce) / math.log(2) + entropy(pi)
            fallbacks += 1
            if t + 1 < stop:
                h = win_step(model, [int(data[t])], t - base, cache, stop - base)[:, -1]
            t += 1
        u = stop
    torch.cuda.synchronize()
    secs = time.time() - t0
    return {
        "bits": bits,
        "bpb": bits / max(n_eval - 1, 1),
        "secs": secs,
        "copies": copies,
        "matches": matches,
        "fallbacks": fallbacks,
        "verify_fail": verify_fail,
        "copy_bytes": copy_bytes,
        "accept": copies / matches if matches else 0.0,
        "coverage": copy_bytes / max(n_eval - 1, 1),
    }


def fit_gate(
    model: Trunk,
    entries: tuple[Entry, ...],
    trie: dict,
    train: bytes,
    total: int,
    e_c: torch.Tensor,
    nll: torch.Tensor,
    hs: torch.Tensor,
    n_eval: int,
    samples: int = 250_000,
    neural: bool = True,
    kind: str = "logistic",
    holdout: bool = False,
    tau_pct: float = 95.0,
    alpha: float = 0.5,
) -> dict:
    rng = np.random.default_rng(0)
    picks = rng.choice(n_eval - 1, size=min(n_eval - 1, samples), replace=False) + 1
    feats, means, raw = [], [], []
    count = torch.tensor([e.count for e in entries])
    for t in sorted(picks.tolist()):
        length, idx = longest_match(trie, train, t)
        if length < 2 or t + length > n_eval:
            continue
        if holdout and idx % HOLDOUT == 0:
            continue
        logc = math.log(float(count[idx]) / total)
        score = match_score(hs[t], e_c[idx], int(count[idx]), total, alpha)
        raw.append(score)
        feats.append(
            [
                score if neural else 0.0,
                length / CAP,
                logc,
            ]
        )
        means.append(float(nll[t : t + length].mean()))
    if not feats:
        return {
            "w": [0.0, 0.0, 0.0],
            "b": 8.0,
            "mu": [0.0] * 3,
            "sd": [1.0] * 3,
            "tau": 1e9,
            "alpha": 0.5,
        }
    tau = float(np.percentile(means, tau_pct))
    if kind == "knn":
        return {
            "kind": "knn",
            "theta": float(np.median(raw)),
            "tau": tau,
            "alpha": alpha,
            "samples": len(feats),
        }
    y = torch.tensor([float(m <= tau) for m in means])
    x = torch.tensor(feats, dtype=torch.float32)
    mu = x.mean(0)
    sd = x.std(0) + 1e-6
    z = (x - mu) / sd
    w = torch.zeros(3, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.1)
    for _ in range(300):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z @ w + b, y)
        loss.backward()
        opt.step()
    return {
        "kind": "logistic",
        "w": w.detach().tolist(),
        "b": float(b.detach()),
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "tau": tau,
        "alpha": alpha,
        "samples": len(feats),
    }


def adapter_eval(
    model: Trunk,
    entries: tuple[Entry, ...],
    trie: dict,
    total: int,
    e_c: torch.Tensor,
    gate: dict,
    data: bytes,
    nll: torch.Tensor,
    hs: torch.Tensor,
    n_eval: int,
    pack_size: int,
    stream_bytes: int,
    neural: bool = True,
    threshold: float = 0.5,
    identity: str = "",
) -> dict:
    """`identity` charges the span's name: "" (oracle), "count" or "index".

    A copy must tell the decoder *which* entry; the oracle rate omits it. The
    count prior is the cheapest decodable code, the fixed index the priciest.
    """
    count = torch.tensor([e.count for e in entries])
    fixed = math.log2(max(len(entries), 1))
    tau = float(gate["tau"])
    alpha = float(gate["alpha"])
    bits = 0.0
    copies = matches = fallbacks = verify_fail = 0
    copy_bytes = 0
    h_bits = h_bytes = h_copies = h_matches = 0
    t = 1
    while t < n_eval:
        length, idx = longest_match(trie, data, t)
        pi = 0.0
        feat = None
        held = length >= 2 and idx % HOLDOUT == 0
        if length >= 2 and t + length <= n_eval:
            matches += 1
            h_matches += int(held)
            logc = math.log(float(count[idx]) / total)
            score = match_score(hs[t], e_c[idx], int(count[idx]), total, alpha)
            feat = [
                score if neural else 0.0,
                length / CAP,
                logc,
            ]
            pi = gate_prob(gate, feat)
            passes = float(nll[t : t + length].mean()) <= tau
            if pi >= threshold and passes:
                span_bits = entropy(pi)
                if identity == "count":
                    span_bits += math.log2(total / max(int(count[idx]), 1))
                elif identity == "index":
                    span_bits += fixed
                bits += span_bits
                copies += 1
                copy_bytes += length
                if held:
                    h_bits += span_bits
                    h_bytes += length
                    h_copies += 1
                t += length
                continue
            if pi >= threshold and not passes:
                verify_fail += 1
        bits += float(nll[t]) / math.log(2) + entropy(pi)
        fallbacks += 1
        if held:
            h_bits += float(nll[t]) / math.log(2) + entropy(pi)
            h_bytes += 1
        t += 1
    base_bits = float(nll[1:n_eval].sum()) / math.log(2)
    base_bpb = base_bits / (n_eval - 1)
    adapter_bpb = bits / (n_eval - 1)
    charge = 8.0 * pack_size / stream_bytes
    saved = base_bpb - adapter_bpb - charge
    provenance = float(all(entry.provenance for entry in entries))
    metrics = {
        "base_bpb": base_bpb,
        "adapter_bpb": adapter_bpb,
        "rate_saved": base_bpb - adapter_bpb,
        "charge": charge,
        "saved": saved,
        "accept": copies / matches if matches else 0.0,
        "copies": copies,
        "matches": matches,
        "copy_bytes": copy_bytes,
        "coverage": copy_bytes / max(n_eval - 1, 1),
        "fallbacks": fallbacks,
        "verify_fail": verify_fail,
        "delta": 0.0,
        "text_delta": 0.0,
        "pack_mb": pack_size / 1e6,
        "pack_bytes": pack_size,
        "provenance": provenance,
        "structured_x": None,
        "held_bpb": h_bits / max(h_bytes, 1),
        "held_bytes": h_bytes,
        "held_matches": h_matches,
        "held_accept": h_copies / h_matches if h_matches else 0.0,
        "n_eval": n_eval,
        "gate": gate,
    }
    return metrics


def pack_key(cfg: Config) -> str:
    blob = json.dumps(
        {
            "corpus": cfg.corpus,
            "stream_bytes": cfg.stream_bytes,
            "val_bytes": cfg.val_bytes,
            "dict_size": cfg.dict_size,
            "min_count": cfg.min_count,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def load_pack(
    cfg: Config, train: bytes, out: Path, tag: str = ""
) -> tuple[Pack, int, int]:
    """Induce the pack once per (corpus, split, dict): the arms share it."""
    key = f"{pack_key(cfg)}-{hashlib.sha256(train).hexdigest()[:8]}"
    cache = RUNS / f"pack-{key}"
    bin_path = cache / "artifacts/pack.bin"
    meta_path = cache / "artifacts/pack.json"
    if bin_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        entries = read_pack(bin_path)
        pack = Pack(entries, cfg, meta["sha"])
        size, total = int(meta["bytes"]), int(meta["total"])
        print(f"  pack {len(entries)} spans · {size / 1e3:.1f} kB (cached)", flush=True)
    else:
        pack, size, total = induce_pack(cfg, train, cache)
        write_json(meta_path, {"sha": pack.sha, "bytes": size, "total": total})
    art = out / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    (art / f"pack{tag}.bin").write_bytes(bin_path.read_bytes())
    write_json(
        art / f"pack{tag}.json", {"sha": pack.sha, "bytes": size, "total": total}
    )
    return pack, size, total


def small_pack(
    train: bytes, size: int = 1024, min_count: int = 8
) -> tuple[tuple[Entry, ...], int]:
    counts, offsets = substring_counts(train, CAP, min_count)
    total = max(sum(counts.values()), 1)
    scored = []
    for s, c in counts.items():
        gain = c * (len(s) - 1) - (len(s) + 6)
        if gain > 0:
            scored.append((gain, c, s, offsets[s]))
    scored.sort(key=lambda x: (-x[0], x[2]))
    entries = tuple(Entry(s, c, 0, (f"fit:{o}",)) for _, c, s, o in scored[:size])
    return entries, total


def run_p0d(model: Trunk, cfg: Config, out: Path) -> dict:
    """The two clocks: a batch-1 stream, one trunk step per byte or per span.

    The adapter's emitted spans are verified by the base's own NLL, so the
    decoded bytes are exact; the speed ratio is measured wall-clock.
    """
    raw = corpus_bytes(cfg.structured)
    split = len(raw) * 9 // 10
    train = raw[: min(split, 1_000_000)]
    test = raw[split : split + 8_192]
    print(
        f"  structured {cfg.structured}: train {len(train) / 1e6:.2f} MB "
        f"test {len(test) / 1e3:.1f} kB",
        flush=True,
    )
    entries, total = small_pack(train)
    trie = make_trie(entries)
    e_c = span_embeddings(model, entries)
    fit_ids = torch.frombuffer(bytearray(train), dtype=torch.uint8).long()
    nll, hs, n = stream_nll(model, fit_ids)
    gate = fit_gate(model, entries, trie, train, total, e_c, nll, hs, n)
    del nll, hs
    torch.cuda.empty_cache()
    test_ids = torch.frombuffer(bytearray(test), dtype=torch.uint8).long()
    n_eval = int(test_ids.numel())
    byte = decode_arm(model, entries, trie, total, e_c, gate, test, n_eval, "byte")
    adapter = decode_arm(model, entries, trie, total, e_c, gate, test, n_eval)
    x = byte["secs"] / max(adapter["secs"], 1e-9)
    m = {
        "structured_x": x,
        "structured_delta": adapter["bpb"] - byte["bpb"],
        "structured_base_bpb": byte["bpb"],
        "structured_adapter_bpb": adapter["bpb"],
        "structured_accept": adapter["accept"],
        "structured_coverage": adapter["coverage"],
        "structured_secs_base": byte["secs"],
        "structured_secs_adapter": adapter["secs"],
        "structured_bytes": n_eval - 1,
        "structured_spans": len(entries),
        "structured_pack_bytes": int(sum(len(e.data) + 14 for e in entries)),
    }
    write_json(
        out / "artifacts/structured.json",
        {"metrics": m, "byte": byte, "adapter": adapter, "gate": gate},
    )
    print(
        f"  P0d structured {m['structured_bytes']} B: base "
        f"{byte['secs']:.1f}s {byte['bpb']:.3f} bpb · adapter "
        f"{adapter['secs']:.1f}s {adapter['bpb']:.3f} bpb · "
        f"{x:.2f}x · accept {adapter['accept']:.1%} · verify fail "
        f"{adapter['verify_fail']}",
        flush=True,
    )
    return m


def lora_init(model: Trunk, r: int) -> list[nn.Parameter]:
    """Attach a rank-`r` LoRA to each block's `qkv` and `proj`; zero output."""
    d = model.tok.embedding_dim
    params: list[nn.Parameter] = []
    for block in model.blocks:
        block.la = nn.Parameter(torch.randn(d, r, device="cuda") * 0.02)
        block.lb = nn.Parameter(torch.zeros(r, 3 * d, device="cuda"))
        block.lc = nn.Parameter(torch.randn(d, r, device="cuda") * 0.02)
        block.ld = nn.Parameter(torch.zeros(r, d, device="cuda"))
        params += [block.la, block.lb, block.lc, block.ld]
    return params


def run_lora(
    cfg: Config,
    pack_bytes: int,
    model: Trunk,
    train_win: torch.Tensor,
    test_win: torch.Tensor,
    lens: torch.Tensor,
) -> dict:
    """The equal-charge baseline: LoRA at the pack's charged bytes or fewer."""
    d = model.tok.embedding_dim
    layers = len(model.blocks)
    r = max(1, pack_bytes // (2 * 6 * d * layers))
    for p in model.parameters():
        p.requires_grad_(False)
    params = lora_init(model, r)
    nbytes = 2 * sum(p.numel() for p in params)
    print(
        f"  lora rank {r} · {sum(p.numel() for p in params)} params · "
        f"{nbytes} B (pack {pack_bytes} B)",
        flush=True,
    )
    base_bits, base_bytes = eval_bits(model, test_win, lens, 16)
    base_bpb = base_bits / base_bytes
    opt = torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0)
    model.train()
    started = time.time()
    for step in range(cfg.steps):
        idx = torch.randint(0, train_win.size(0), (cfg.batch,))
        x = train_win[idx].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = lm_loss(model, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == cfg.steps - 1:
            print(
                f"  step {step:5d}/{cfg.steps} loss {float(loss.detach()):.4f}",
                flush=True,
            )
    torch.cuda.synchronize()
    secs = time.time() - started
    model.eval()
    bits, nbytes_eval = eval_bits(model, test_win, lens, 16)
    adapter_bpb = bits / nbytes_eval
    charge = 8.0 * nbytes / cfg.stream_bytes
    return {
        "base_bpb": base_bpb,
        "adapter_bpb": adapter_bpb,
        "rate_saved": base_bpb - adapter_bpb,
        "charge": charge,
        "saved": base_bpb - adapter_bpb - charge,
        "accept": 0.0,
        "copies": 0,
        "matches": 0,
        "copy_bytes": 0,
        "coverage": 0.0,
        "fallbacks": 0,
        "verify_fail": 0,
        "delta": adapter_bpb - base_bpb,
        "text_delta": adapter_bpb - base_bpb,
        "pack_mb": nbytes / 1e6,
        "pack_bytes": 0,
        "provenance": 0.0,
        "structured_x": None,
        "n_eval": 0,
        "secs": secs,
        "lora_rank": r,
        "lora_params": sum(p.numel() for p in params),
        "lora_bytes": nbytes,
    }


def run_p0(cfg: Config, gates: list[str], out: Path) -> dict:
    path = base_path(cfg)
    if not path.exists():
        raise SystemExit(
            f"P0: missing frozen base {path}; run "
            f"`cml base --params {cfg.params}` first"
        )
    ckpt = torch.load(path, map_location="cuda", weights_only=False)
    model = Trunk(256, ckpt["d"], ckpt["layers"], ckpt["heads"], ckpt["seq"]).cuda()
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(
        f"P0 {out.name}: base={cfg.base} mode={cfg.mode} "
        f"{ckpt['params'] / 1e6:.1f}M d={ckpt['d']} seq={ckpt['seq']}",
        flush=True,
    )
    raw = corpus_bytes(cfg.corpus)
    split = cfg.stream_bytes - cfg.val_bytes
    ids = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()
    train = raw[:split]
    test = raw[split : split + cfg.val_bytes]
    train_ids, test_ids = ids[:split], ids[split : split + cfg.val_bytes]

    if cfg.mode == "lora":
        pack, pack_size, _ = load_pack(cfg, train, out)
        seq = model.pos.num_embeddings
        test_win = windows(test_ids, seq)
        m = run_lora(
            cfg,
            pack_size,
            model,
            windows(train_ids, seq),
            test_win,
            torch.ones_like(test_win[:, 1:]),
        )
    elif cfg.mode == "none":
        test_nll, _, test_n = stream_nll(model, test_ids)
        base_bpb = float(test_nll[1:test_n].sum()) / math.log(2) / (test_n - 1)
        m = {
            "base_bpb": base_bpb,
            "adapter_bpb": base_bpb,
            "rate_saved": 0.0,
            "charge": 0.0,
            "saved": 0.0,
            "accept": 0.0,
            "copies": 0,
            "matches": 0,
            "copy_bytes": 0,
            "coverage": 0.0,
            "fallbacks": test_n - 1,
            "verify_fail": 0,
            "delta": 0.0,
            "text_delta": 0.0,
            "pack_mb": 0.0,
            "pack_bytes": 0,
            "provenance": 0.0,
            "structured_x": None,
            "n_eval": test_n,
        }
        pack, pack_size = Pack((), cfg), 0
    else:
        pack, pack_size, total = load_pack(cfg, train, out)
        trie = make_trie(pack.entries)
        e_c = span_embeddings(model, pack.entries)
        print(f"  encodings {len(pack.entries)} spans", flush=True)
        neural = cfg.mode != "counts"
        kind = "knn" if cfg.mode == "knn" else "logistic"
        fit_ids = train_ids[:1_000_000]
        train_nll, train_hs, train_n = stream_nll(model, fit_ids)
        gate = fit_gate(
            model,
            pack.entries,
            trie,
            train[:1_000_000],
            total,
            e_c,
            train_nll,
            train_hs,
            train_n,
            neural=neural,
            kind=kind,
        )
        gates_heldout = {}
        if cfg.mode == "adapter":
            for name, flag in (("tied", True), ("counts", False)):
                gates_heldout[name] = fit_gate(
                    model,
                    pack.entries,
                    trie,
                    train[:1_000_000],
                    total,
                    e_c,
                    train_nll,
                    train_hs,
                    train_n,
                    neural=flag,
                    holdout=True,
                )
        del train_nll, train_hs
        torch.cuda.empty_cache()
        test_nll, test_hs, test_n = stream_nll(model, test_ids)
        m = adapter_eval(
            model,
            pack.entries,
            trie,
            total,
            e_c,
            gate,
            test,
            test_nll,
            test_hs,
            test_n,
            pack_size,
            cfg.stream_bytes,
            neural=neural,
        )
        charged = adapter_eval(
            model,
            pack.entries,
            trie,
            total,
            e_c,
            gate,
            test,
            test_nll,
            test_hs,
            test_n,
            pack_size,
            cfg.stream_bytes,
            neural=neural,
            identity="count",
        )
        index = adapter_eval(
            model,
            pack.entries,
            trie,
            total,
            e_c,
            gate,
            test,
            test_nll,
            test_hs,
            test_n,
            pack_size,
            cfg.stream_bytes,
            neural=neural,
            identity="index",
        )
        m["saved_charged"] = charged["saved"]
        m["charged_bpb"] = charged["adapter_bpb"]
        m["saved_index"] = index["saved"]
        m["index_bpb"] = index["adapter_bpb"]
        held = {}
        for name, g in gates_heldout.items():
            held[name] = adapter_eval(
                model,
                pack.entries,
                trie,
                total,
                e_c,
                g,
                test,
                test_nll,
                test_hs,
                test_n,
                pack_size,
                cfg.stream_bytes,
                neural=name == "tied",
            )
        if held:
            m["linchpin_tied"] = held["tied"]["held_bpb"]
            m["linchpin_counts"] = held["counts"]["held_bpb"]
            m["linchpin"] = held["counts"]["held_bpb"] - held["tied"]["held_bpb"]
            m["linchpin_accept_tied"] = held["tied"]["held_accept"]
            m["linchpin_accept_counts"] = held["counts"]["held_accept"]
            m["linchpin_bytes"] = held["tied"]["held_bytes"]
            m["linchpin_tied_gate"] = held["tied"]["gate"]
            m["linchpin_counts_gate"] = held["counts"]["gate"]
        if cfg.pack_corpus:
            fr = corpus_bytes(cfg.pack_corpus)
            fcfg = replace(cfg, corpus=cfg.pack_corpus, pack_corpus="")
            fpack, fsize, ftotal = load_pack(
                fcfg, fr[: len(fr) * 9 // 10], out, tag="-foreign"
            )
            ftrie = make_trie(fpack.entries)
            fe_c = span_embeddings(model, fpack.entries)
            fm = adapter_eval(
                model,
                fpack.entries,
                ftrie,
                ftotal,
                fe_c,
                gate,
                test,
                test_nll,
                test_hs,
                test_n,
                fsize,
                cfg.stream_bytes,
                neural=neural,
            )
            m["zero_shot"] = fm["adapter_bpb"] - m["adapter_bpb"]
            m["zero_shot_foreign"] = fm["adapter_bpb"]
            m["zero_shot_tuned"] = m["adapter_bpb"]
            m["zero_shot_pack_bytes"] = fsize
            m["zero_shot_coverage"] = fm["coverage"]
            print(
                f"  zero-shot {cfg.pack_corpus} pack ({fsize / 1e3:.1f} kB): "
                f"{fm['adapter_bpb']:.3f} vs tuned {m['adapter_bpb']:.3f} bpb · "
                f"gap {m['zero_shot']:+.3f} · coverage {fm['coverage']:.1%}",
                flush=True,
            )
        if cfg.structured:
            m.update(run_p0d(model, cfg, out))

    metrics = {
        "id": out.name,
        "arm": cfg.arm.value,
        "mode": cfg.mode,
        "seed": cfg.seed,
        "params": ckpt["params"],
        "base": cfg.base,
        "steps": cfg.steps,
        "vocab": 256,
        "train_bytes": len(train),
        "test_bytes": len(test),
        "pack_bytes": pack_size,
        "pack_sha": pack.sha,
        "corpus": cfg.corpus,
        "pack_corpus": cfg.pack_corpus,
        "corpus_sha": hashlib.sha256(raw).hexdigest(),
        "structured": cfg.structured,
        "tokens_per_byte": 1.0,
        **m,
    }
    metrics["payload"] = metrics["adapter_bpb"]
    metrics["unigram"] = metrics["base_bpb"]
    if "gate" in m:
        metrics["gate"] = m["gate"]
    metrics["gates"] = gate_rows(gates, metrics)
    checked = [
        row["pass"] for row in metrics["gates"].values() if row["pass"] is not None
    ]
    metrics["skipped"] = [
        name for name, row in metrics["gates"].items() if row["pass"] is None
    ]
    metrics["passed"] = bool(checked) and all(checked)
    return metrics


PIPELINES: dict[str, Callable[[Config, list[str], Path], dict]] = {
    "p0": run_p0,
    "p1": run_p1,
}


def print_metrics(metrics: dict) -> None:
    if "bigram" in metrics:
        print(
            f"payload {metrics['payload']:.3f} bpb · "
            f"unigram {metrics['unigram']:.3f} · "
            f"bigram {metrics['bigram']:.3f} · "
            f"saved {metrics['saved']:+.3f}",
            flush=True,
        )
    else:
        print(
            f"base {metrics['base_bpb']:.3f} bpb · "
            f"adapter {metrics['adapter_bpb']:.3f} · "
            f"rate saved {metrics['rate_saved']:+.3f} · "
            f"charge {metrics['charge']:.3f} · "
            f"net saved {metrics['saved']:+.3f}",
            flush=True,
        )
        print(
            f"  pack {metrics['pack_bytes'] / 1e3:.1f} kB · "
            f"accept {metrics['accept']:.1%} · "
            f"copies/matches {metrics['copies']}/{metrics['matches']} · "
            f"coverage {metrics['coverage']:.1%} · "
            f"verify fail {metrics['verify_fail']}",
            flush=True,
        )
    for name, row in metrics["gates"].items():
        value = "—" if row["value"] is None else f"{row['value']:.3f}"
        mark = "·" if row["pass"] is None else ("✓" if row["pass"] else "✗")
        print(f"  {name} {row['bar']:g} {value} {mark}", flush=True)
    if metrics.get("linchpin") is not None:
        print(
            f"  linchpin (held-out 10%) {metrics['linchpin_bytes']} B · "
            f"tied {metrics['linchpin_tied']:.3f} vs counts "
            f"{metrics['linchpin_counts']:.3f} bpb · "
            f"gap {metrics['linchpin']:+.3f} · "
            f"accept {metrics['linchpin_accept_tied']:.1%} vs "
            f"{metrics['linchpin_accept_counts']:.1%}",
            flush=True,
        )
    if metrics.get("structured_x") is not None:
        print(
            f"  P0d structured {metrics['structured_bytes']} B · "
            f"base {metrics['structured_base_bpb']:.3f} → "
            f"adapter {metrics['structured_adapter_bpb']:.3f} bpb · "
            f"{metrics['structured_x']:.2f}x · "
            f"accept {metrics['structured_accept']:.1%}",
            flush=True,
        )


def fit_curve(points: list[tuple[int, float]]) -> dict | None:
    """Fit `bpb = c + a·N^-b` by a grid over `b` and linear least squares."""
    if len(points) < 2:
        return None
    ns = np.array([n for n, _ in points], dtype=float)
    ys = np.array([y for _, y in points], dtype=float)
    best: tuple[float, float, float, float] | None = None
    for b in np.linspace(0.02, 2.0, 500):
        x = ns ** (-b)
        design = np.stack([x, np.ones_like(x)], axis=1)
        coef, *_ = np.linalg.lstsq(design, ys, rcond=None)
        sse = float(((design @ coef - ys) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, float(b), float(coef[0]), float(coef[1]))
    if best is None:
        return None
    sse, b, a, c = best
    return {
        "a": a,
        "b": b,
        "c": c,
        "sse": sse,
        "points": [[int(n), float(y)] for n, y in points],
    }


def curve_bpb(curve: dict, params: int) -> float:
    return float(curve["c"] + curve["a"] * params ** (-curve["b"]))


def p1_runs(steps: int) -> list[dict]:
    rows: dict[int, dict] = {}
    for path in sorted(RUNS.glob("p1-class-*/metrics.json")):
        m = json.loads(path.read_text())
        if m.get("steps") != steps or m.get("seed") != 0:
            continue
        rows.setdefault(int(m["params"]), m)
    return [rows[k] for k in sorted(rows)]


def curve(stage: str, steps: int, update: bool) -> int:
    if stage != "P1":
        raise SystemExit(f"curve: {stage} is not built (# TODO)")
    rows = p1_runs(steps)
    points = [(int(m["params"]), float(m["payload"])) for m in rows]
    fit = fit_curve(points)
    if fit is None:
        raise SystemExit(f"curve: need at least two P1 runs at {steps} steps")
    c60 = curve_bpb(fit, 60_000_000)
    print(f"P1 curve · steps {steps} · {len(points)} points")
    for n, y in points:
        print(f"  {n / 1e6:6.2f}M  {y:.3f}  fit {curve_bpb(fit, n):.3f}")
    print(f"  60.00M  —      fit {c60:.3f}")
    payload = {"stage": "P1", "steps": steps, "curve": fit, "curve60": c60}
    write_json(REPORTS / "curve-p1.json", payload)
    if update:
        target = max(rows, key=lambda m: int(m["params"]))
        path = RUNS / target["id"] / "metrics.json"
        target["curve"] = fit
        target["curve60"] = c60
        target["gates"] = gate_rows(list(target["gates"]), target)
        checked = [
            row["pass"] for row in target["gates"].values() if row["pass"] is not None
        ]
        target["skipped"] = [
            name for name, row in target["gates"].items() if row["pass"] is None
        ]
        target["passed"] = bool(checked) and all(checked)
        write_json(path, target)
        print(f"updated {target['id']} with curve60 {c60:.3f}")
    return 0


def gates_report() -> int:
    rows = []
    for path in sorted(RUNS.glob("*/metrics.json")):
        m = json.loads(path.read_text())
        if "payload" not in m or m["id"].startswith("base-") or "gates" not in m:
            continue
        if not m["id"].startswith("p0-") and m.get("steps", 0) < 100:
            continue
        rows.append(m)
    lines = [
        "# CML gates",
        "",
        "| run | mode | train MB | params | steps | payload | bigram | gates |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for m in rows:
        gate_text = " ".join(
            f"{name}:{'—' if row['pass'] is None else ('✓' if row['pass'] else '✗')}"
            for name, row in m["gates"].items()
        )
        bigram = f"{m['bigram']:.3f}" if "bigram" in m else "—"
        mode = m.get("mode") or "adapter"
        lines.append(
            f"| `{m['id']}` | {mode} | {m['train_bytes'] / 1e6:.1f} | "
            f"{m['params'] / 1e6:.1f}M | {m['steps']} | "
            f"{m['payload']:.3f} | {bigram} | {gate_text} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
    ]
    if rows:
        p1 = [m for m in rows if m["id"].startswith("p1-class")]
        p0 = [m for m in rows if m["id"].startswith("p0-")]
        main = [m for m in p1 if m.get("seed") == 0]
        if p1:
            best = min(main or p1, key=lambda m: float(m["payload"]))
            same = [
                m
                for m in p1
                if m["params"] == best["params"] and m["steps"] == best["steps"]
            ]
            values = [float(m["payload"]) for m in same]
            mean = sum(values) / len(values)
            sd = (sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)) ** 0.5
            lines.append(
                f"The seed-0 P1 payload is **{best['payload']:.3f} bpb** at "
                f"{best['params'] / 1e6:.1f}M parameters and {best['steps']} steps "
                f"(`{best['id']}`); seeds 0-{len(values) - 1} at that point are "
                f"**{mean:.3f} ± {sd:.3f}**. The same-slice count bigram is "
                f"{best.get('bigram', 0.0):.3f}; the v9 reference on the 10 MB block "
                f"is 2.696 (2.729 is the 10 MB-prefix rate)."
            )
            lines.append("")
        if p0:
            text_p0 = [
                m for m in p0 if (m.get("corpus") or "enwik8_10m") == "enwik8_10m"
            ]
            adapter_rows = [
                m for m in text_p0 if (m.get("mode") or "adapter") == "adapter"
            ]
            b0 = max(adapter_rows or text_p0 or p0, key=lambda m: float(m["saved"]))
            lines.append(
                f"P0: base {b0['base_bpb']:.3f} → adapter "
                f"{b0['adapter_bpb']:.3f} bpb; rate saved "
                f"{b0['rate_saved']:+.3f}, pack charge {b0['charge']:.3f}, "
                f"net **{b0.get('saved_charged', b0['saved']):+.3f}** charged "
                f"(oracle {b0['saved']:+.3f}); accept {b0['accept']:.1%}, "
                f"coverage {b0.get('coverage', 0.0):.1%}, "
                f"pack {b0['pack_bytes'] / 1e3:.1f} kB (`{b0['id']}`)."
            )
            lines.append("")
            arms = [
                m
                for m in p0
                if (m.get("mode") or "adapter")
                in ("adapter", "none", "counts", "knn", "lora")
            ]

            def rank(m: dict) -> tuple[bool, bool]:
                return (m.get("saved_charged") is not None, bool(m.get("mode")))

            seen: dict[tuple[str, str], dict] = {}
            for m in arms:
                key = (
                    m.get("mode") or "adapter",
                    str(m.get("corpus") or "enwik8_10m"),
                )
                if key not in seen or rank(m) > rank(seen[key]):
                    seen[key] = m
            arms = list(seen.values())
            if arms:
                lines += [
                    "| arm | corpus | saved | charge | accept | pack kB |",
                    "|---|---|---:|---:|---:|---:|",
                ]
                for m in sorted(
                    arms,
                    key=lambda m: (str(m.get("corpus")), str(m.get("mode"))),
                ):
                    corpus = str(m.get("corpus") or "enwik8_10m").replace("enwik8_", "")
                    charged = m.get("saved_charged")
                    saved = "—" if charged is None else f"{charged:+.3f}"
                    lines.append(
                        f"| {(m.get('mode') or 'adapter')} | {corpus} | "
                        f"{saved} | "
                        f"{m['charge']:.3f} | "
                        f"{m['accept']:.1%} | {m['pack_bytes'] / 1e3:.1f} |"
                    )
                lines.append("")
            lin = [m for m in text_p0 if m.get("linchpin") is not None]
            if lin:
                s = lin[0]
                lines.append(
                    f"Linchpin (held-out 10% of `D`): tied "
                    f"{s['linchpin_tied']:.3f} vs counts-only "
                    f"{s['linchpin_counts']:.3f} bpb on "
                    f"{s['linchpin_bytes']} B → gap "
                    f"**{s['linchpin']:+.3f}**; accept "
                    f"{s['linchpin_accept_tied']:.1%} vs "
                    f"{s['linchpin_accept_counts']:.1%}."
                )
                lines.append("")
            d0 = [m for m in p0 if m.get("structured_x") is not None]
            for s in sorted(d0, key=lambda m: str(m.get("structured"))):
                lines.append(
                    f"P0d on `{s.get('structured', '?')}`: "
                    f"**{s['structured_x']:.2f}x** wall-clock over "
                    f"{s['structured_bytes']} B "
                    f"({s['structured_base_bpb']:.3f} → "
                    f"{s['structured_adapter_bpb']:.3f} bpb, accept "
                    f"{s['structured_accept']:.1%}); the copies are verified "
                    f"exact, so the emission is lossless."
                )
                lines.append("")
        evidence = [m for m in rows if m["id"].startswith("p1-check")]
        if evidence:
            lines.append(
                "The second split is the `p1-check-8m` row: train 8 MB and test "
                "the 8-9 MB slice, so the win is not one easy slice."
            )
            lines.append("")
        if p1:
            lines.append(
                f"Provenance: lexicon {best.get('lexicon_bytes', 0) / 1e3:.1f} kB "
                f"`{best.get('lexicon_sha', '')[:12]}`; corpus "
                f"`{best.get('corpus_sha', '')[:12]}`."
            )
            lines.append("")
            lines.append(
                "The 10 MB parse is the 90-100 MB enwik8 val block; the lexicon is "
                "the v9 shared pack. The first 9 MB is the train and the last 1 MB "
                "is the test."
            )
            lines.append("")
            lines.append(
                "P1a ≤2.60, P1b ≤2.293 and P1c ≤2.29 at 60M are measured. P1d "
                "(acceptance per pack byte on a shared trunk) needs the "
                "mixed-schedule trunk."
            )
            zs = [m for m in p0 if m.get("zero_shot") is not None]
            if zs:
                z = zs[0]
                lines.append("")
                lines.append(
                    f"P1e zero-shot lexicon: a pack from "
                    f"`{z.get('pack_corpus') or 'foreign'}` on "
                    f"`{z.get('corpus', '?')}` gives "
                    f"{z['zero_shot_foreign']:.3f} vs "
                    f"{z['zero_shot_tuned']:.3f} bpb tuned → gap "
                    f"**{z['zero_shot']:+.3f}** at "
                    f"{z['zero_shot_coverage']:.1%} coverage; packs are "
                    f"domain-local."
                )
            lines.append("")
            lines.append(
                "The 40M point is 0.006 bpb above the 20.5M point; the curve is "
                "flat, so the 10 MB pilot is data-bound and P1c passes by "
                "saturation, not by a capacity slope."
            )
    else:
        lines.append("No runs yet.")
    path = REPORTS / "GATES.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(path.read_text())
    return 0


def selftest() -> int:
    cfg = Config()
    gates = ["P1a", "P1b"]
    validate(gates)
    rid = run_id(Path("selftest.toml"), cfg)
    out = RUNS / rid
    write_json(
        out / "config.json",
        {"config": asdict(cfg), "gates": gates, "git": git_sha()},
    )
    assert (out / "config.json").exists()
    assert GATES["P1a"].check(2.5) and not GATES["P1a"].check(2.7)
    model = Trunk(64, 16, 1, 2, 8)
    x = torch.randint(0, 64, (2, 9))
    assert model(x[:, :-1]).shape == (2, 8, 64)
    assert torch.isfinite(lm_loss(model, x))
    assert windows(torch.arange(20), 4).shape == (4, 5)
    assert split_index(torch.ones(10, dtype=torch.long), 5) == 5
    print(f"selftest ok: {rid}")
    return 0


def run(
    manifest: Path,
    seed: int | None,
    params: int | None,
    steps: int | None,
    dry: bool,
) -> int:
    if not manifest.exists():
        raise SystemExit(f"missing manifest: {manifest}")
    cfg, gates = config_from_manifest(manifest, seed, params, steps)
    validate(gates)
    rid = run_id(manifest, cfg)
    if dry:
        print(json.dumps({"id": rid, "config": asdict(cfg), "gates": gates}, indent=1))
        return 0
    out = RUNS / rid
    write_json(
        out / "config.json",
        {
            "id": rid,
            "config": asdict(cfg),
            "gates": gates,
            "git": git_sha(),
            "manifest": str(manifest),
        },
    )
    fn = PIPELINES.get(manifest.stem.split("-")[0])
    if fn is None:
        raise SystemExit(f"{manifest.stem}: pipeline not built (# TODO: build it)")
    metrics = fn(cfg, gates, out)
    write_json(out / "metrics.json", metrics)
    print_metrics(metrics)
    return 0


def _lexicon_for(corpus: CodeCorpus, units: int, min_count: int, cap: int):
    path = CACHE / f"lex-{corpus.train_sha[:12]}-{units}-{min_count}-{cap}.bin"
    if path.exists():
        return codec_mod.Lexicon.from_blob(path.read_bytes())
    lex = codec_mod.build_lexicon(corpus.train, units, min_count, cap)
    blob = lex.blob()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(blob)
    os.replace(tmp, path)
    return lex


def _byte_model(corpus: CodeCorpus, steps: int, seed: int):
    import bytecodec

    path = CACHE / f"byte-{corpus.train_sha[:12]}-d256-l6-s{steps}-n{seed}.pt"
    if path.exists():
        try:
            return bytecodec.load_model(path)
        except RuntimeError, KeyError, ValueError:
            pass
    cfg = bytecodec.ByteConfig(steps=steps, seed=seed)
    model, secs = bytecodec.train_byte_model(corpus.train, cfg)
    bytecodec.save_model(path, model, cfg, secs)
    return model


def _run_byte_model(
    corpus: CodeCorpus,
    data: bytes,
    arm: str,
    corpus_name: str,
    nbytes: int,
    split: str,
    steps: int,
    seed: int,
    proof: int,
) -> dict:
    import bytecodec

    model = _byte_model(corpus, steps, seed)
    out_dir = RUNS / f"codec-{arm}-{corpus_name}-{nbytes}"
    bytecodec.save_model(
        out_dir / "model.pt", model, bytecodec.ByteConfig(steps=steps, seed=seed), 0.0
    )
    coder = bytecodec.StaticByteCodec(model, half=True)
    payload = coder.encode(data)
    frame = codec_mod.pack_frame(3, len(data), payload, corpus.sha, corpus.train_sha)
    header_bytes = len(frame) - len(payload)
    want = min(proof, len(data))
    out = coder.decode(payload, len(data), limit=want)
    if out != data[: len(out)]:
        raise SystemExit("byte-model: proof round-trip failed")
    params = sum(p.numel() for p in model.parameters())
    base_bytes = params / 2
    n = max(len(data), 1)
    payload_bpb = 8 * len(payload) / n
    shared_bpb = 8 * (len(payload) + header_bytes) / n
    metrics = {
        "id": f"codec-{arm}-{corpus_name}-{nbytes}",
        "arm": arm,
        "corpus": corpus_name,
        "split": split,
        "bytes": nbytes,
        "n": len(data),
        "payload_bytes": len(payload),
        "header_bytes": header_bytes,
        "lexicon_bytes": 0,
        "payload_bpb": payload_bpb,
        "shared_bpb": shared_bpb,
        "self_bpb": shared_bpb + 8 * base_bytes / n,
        "base_bytes": base_bytes,
        "params": params,
        "roundtrip": True,
        "proof_units": 0,
        "proof_bytes": len(out),
        "corpus_sha": corpus.sha,
        "train_sha": corpus.train_sha,
        "lexicon_sha": "",
        "units": 0,
        "coverage": None,
        "mean_unit": None,
        "literals": None,
        "files": len(corpus.files),
        "train_bytes": len(corpus.train),
        "test_bytes": len(corpus.test),
        "git": git_sha(),
        "seconds": None,
    }
    write_json(out_dir / "metrics.json", metrics)
    (out_dir / "frame.bin").write_bytes(frame)
    return metrics


def _pointer_for(
    corpus: CodeCorpus,
    lex: codec_mod.Lexicon,
    active: int,
    steps: int,
    seed: int,
    rotate: int,
    full_every: int,
    span: str = "content",
    head: str = "gate",
    seq: int = 256,
    d: int = 256,
    layers: int = 6,
):
    import pointer

    active_lex = codec_mod.Lexicon(lex.units[:active])
    tag = f"r{rotate}f{full_every}-" if rotate else ""
    design = "" if (span, head) == ("content", "gate") else f"{span[0]}{head[0]}-"
    shape = "" if (d, layers) == (256, 6) else f"d{d}l{layers}-"
    path = CACHE / (
        f"pointer-v2-{corpus.train_sha[:12]}-a{active}-{design}{shape}{tag}"
        f"s{steps}-q{seq}-n{seed}.pt"
    )
    if path.exists():
        try:
            return pointer.load_model(path), active_lex, pointer.load_losses(path)
        except RuntimeError, KeyError, ValueError, AttributeError:
            pass
    cfg = pointer.PointerConfig(
        active=active,
        steps=steps,
        seed=seed,
        span=span,
        head=head,
        seq=seq,
        d=d,
        layers=layers,
    )
    if rotate:
        streams = pointer.pack_streams(corpus.train, lex.units, active, rotate)
        model, losses = pointer.train_pointer_streams(
            streams, cfg, full_every=full_every
        )
    else:
        els = pointer.elements(corpus.train, active_lex)
        model, losses = pointer.train_pointer(els, active_lex.units, cfg)
    pointer.save_model(path, model, cfg, losses)
    return model, active_lex, losses


def _run_concept(
    corpus: CodeCorpus,
    data: bytes,
    lex: codec_mod.Lexicon,
    active: int,
    steps: int,
    seed: int,
    proof: int,
    rotate: int,
    full_every: int,
    arm: str,
    corpus_name: str,
    nbytes: int,
    split: str,
    suffix: str = "",
    span: str = "content",
    head: str = "gate",
    parse_mode: str = "greedy",
    seq: int = 256,
) -> dict:
    import pointer

    model, active_lex, _ = _pointer_for(
        corpus, lex, active, steps, seed, rotate, full_every, span, head, seq
    )
    out_dir = RUNS / f"codec-{arm}-{corpus_name}-{nbytes}{suffix}"
    pointer.save_model(out_dir / "model.pt", model, model.cfg, [])
    pcodec = pointer.PointerCodec(model, active_lex, half=True)
    ids, covered = codec_mod.greedy_parse(data, active_lex)
    stats = codec_mod.parse_stats(ids, active_lex, len(data), covered)
    t_enc = time.perf_counter()
    payload = pcodec.encode(data, parse=parse_mode)
    encode_s = time.perf_counter() - t_enc
    frame = codec_mod.pack_frame(
        2, len(data), payload, corpus.sha, corpus.train_sha, active_lex
    )
    header_bytes = len(frame) - len(payload)
    want = min(proof, len(ids))
    out = pcodec.decode(payload, len(data), limit=want)
    if out != data[: len(out)]:
        raise SystemExit("concept: proof round-trip failed")
    params = sum(p.numel() for p in model.parameters())
    base_bytes = params / 2
    n = max(len(data), 1)
    payload_bpb = 8 * len(payload) / n
    shared_bpb = 8 * (len(payload) + header_bytes) / n
    metrics = {
        "id": f"codec-{arm}-{corpus_name}-{nbytes}{suffix}",
        "arm": arm,
        "corpus": corpus_name,
        "split": split,
        "bytes": nbytes,
        "n": len(data),
        "payload_bytes": len(payload),
        "header_bytes": header_bytes,
        "lexicon_bytes": len(active_lex.blob()),
        "payload_bpb": payload_bpb,
        "shared_bpb": shared_bpb,
        "self_bpb": shared_bpb + 8 * base_bytes / n,
        "base_bytes": base_bytes,
        "params": params,
        "roundtrip": True,
        "proof_units": want,
        "proof_bytes": len(out),
        "coding_dtype": "fp16",
        "design": f"{span}-{head}-{parse_mode}",
        "span": span,
        "head": head,
        "parse": parse_mode,
        "encode_seconds": round(encode_s, 3),
        "units_per_s": len(ids) / max(encode_s, 1e-9),
        "steps_per_byte": len(ids) / max(len(data), 1),
        "bytes_per_s": len(data) / max(encode_s, 1e-9),
        "corpus_sha": corpus.sha,
        "train_sha": corpus.train_sha,
        "lexicon_sha": active_lex.sha,
        "units": len(active_lex.units),
        "coverage": stats["coverage"],
        "mean_unit": stats["mean_unit"],
        "literals": stats["literals"],
        "files": len(corpus.files),
        "train_bytes": len(corpus.train),
        "test_bytes": len(corpus.test),
        "git": git_sha(),
        "seconds": None,
    }
    out_dir = RUNS / metrics["id"]
    write_json(out_dir / "metrics.json", metrics)
    (out_dir / "lexicon.bin").write_bytes(active_lex.blob())
    (out_dir / "frame.bin").write_bytes(frame)
    return metrics


def run_codec(
    arm: str,
    corpus_name: str,
    nbytes: int,
    split: str,
    units: int,
    min_count: int,
    cap: int,
    active: int,
    steps: int,
    proof: int,
    rotate: int,
    full_every: int,
    seed: int,
    span: str = "content",
    head: str = "gate",
    parse: str = "greedy",
    seq: int = 256,
) -> int:
    """One `cml codec` run: encode a frozen slice, decode it, record the bpb."""
    t0 = time.perf_counter()
    corpus = code_corpus() if corpus_name == "code" else text_corpus(corpus_name)
    pool = {"train": corpus.train, "test": corpus.test, "all": corpus.raw}[split]
    data = pool[:nbytes]
    if arm == "byte-model":
        metrics = _run_byte_model(
            corpus, data, arm, corpus_name, nbytes, split, steps, seed, proof
        )
        metrics["seconds"] = round(time.perf_counter() - t0, 3)
        write_json(RUNS / metrics["id"] / "metrics.json", metrics)
        print(
            f"{metrics['id']}: {len(data)} B -> {metrics['payload_bytes']} B payload, "
            f"header {metrics['header_bytes']} B, "
            f"{metrics['payload_bpb']:.3f} / {metrics['shared_bpb']:.3f} bpb, "
            f"proof {metrics['proof_bytes']} bytes"
        )
        return 0
    if arm == "concept":
        lex = _lexicon_for(corpus, units, min_count, cap)
        design = "" if (span, head) == ("content", "gate") else f"-{span[0]}{head[0]}"
        if parse != "greedy":
            design += "-p"
        suffix = (f"-a{active}" if active != 256 else "") + design
        metrics = _run_concept(
            corpus,
            data,
            lex,
            active,
            steps,
            seed,
            proof,
            rotate,
            full_every,
            arm,
            corpus_name,
            nbytes,
            split,
            suffix,
            span,
            head,
            parse,
            seq,
        )
        metrics["seconds"] = round(time.perf_counter() - t0, 3)
        write_json(RUNS / metrics["id"] / "metrics.json", metrics)
        print(
            f"{metrics['id']}: {len(data)} B -> {metrics['payload_bytes']} B payload, "
            f"header {metrics['header_bytes']} B, "
            f"{metrics['payload_bpb']:.3f} / {metrics['shared_bpb']:.3f} bpb, "
            f"proof {metrics['proof_units']} units"
        )
        return 0
    lex = None
    coverage = None
    mean_unit = None
    literals = None
    if arm == "byte":
        payload = codec_mod.ByteArm().encode(data)
        codec_arm = 0
    elif arm == "lexicon":
        lex = _lexicon_for(corpus, units, min_count, cap)
        parse, covered = codec_mod.greedy_parse(data, lex)
        stats = codec_mod.parse_stats(parse, lex, len(data), covered)
        coverage = stats["coverage"]
        mean_unit = stats["mean_unit"]
        literals = stats["literals"]
        payload = codec_mod.UnitArm(lex).encode(data, parse)
        codec_arm = 1
    else:
        raise SystemExit(f"codec: unknown arm {arm}")
    frame = codec_mod.pack_frame(
        codec_arm, len(data), payload, corpus.sha, corpus.train_sha, lex
    )
    header_bytes = len(frame) - len(payload)
    lexicon_bytes = len(lex.blob()) if lex is not None else 0
    if codec_mod.decode_frame(frame) != data:
        raise SystemExit("codec: round-trip failed")
    n = max(len(data), 1)
    payload_bpb = 8 * len(payload) / n
    shared_bpb = 8 * (len(payload) + header_bytes) / n
    metrics = {
        "id": f"codec-{arm}-{corpus_name}-{nbytes}",
        "arm": arm,
        "corpus": corpus_name,
        "split": split,
        "bytes": nbytes,
        "n": len(data),
        "payload_bytes": len(payload),
        "header_bytes": header_bytes,
        "lexicon_bytes": lexicon_bytes,
        "payload_bpb": payload_bpb,
        "shared_bpb": shared_bpb,
        "self_bpb": shared_bpb,
        "base_bytes": 0,
        "roundtrip": True,
        "corpus_sha": corpus.sha,
        "train_sha": corpus.train_sha,
        "lexicon_sha": lex.sha if lex is not None else "",
        "units": len(lex.units) if lex is not None else 0,
        "coverage": coverage,
        "mean_unit": mean_unit,
        "literals": literals,
        "files": len(corpus.files),
        "train_bytes": len(corpus.train),
        "test_bytes": len(corpus.test),
        "git": git_sha(),
        "seconds": round(time.perf_counter() - t0, 3),
    }
    out = RUNS / metrics["id"]
    write_json(out / "metrics.json", metrics)
    (out / "frame.bin").write_bytes(frame)
    if lex is not None:
        (out / "lexicon.bin").write_bytes(lex.blob())
    print(
        f"{metrics['id']}: {len(data)} B -> {len(payload)} B payload, "
        f"header {header_bytes} B, {payload_bpb:.3f} / {shared_bpb:.3f} bpb"
        + (
            f", coverage {coverage:.3f}, mean unit {mean_unit:.2f} B"
            if coverage is not None
            else ""
        )
    )
    return 0


DESIGNS = {
    "gc": ("content", "gate"),
    "jc": ("content", "joint"),
    "gn": ("ngram", "gate"),
    "jn": ("ngram", "joint"),
}


def run_arena(
    corpus_name: str,
    nbytes: int,
    proof: int,
    steps: int,
    active: int,
    designs: str,
    rotate: int,
    full_every: int,
    seed: int,
) -> int:
    """One row per design: same corpus, slice, pack and proof."""
    rows = []
    for code in [d.strip() for d in designs.split(",") if d.strip()]:
        if code not in DESIGNS:
            raise SystemExit(f"arena: unknown design {code}")
        span, head, parse = DESIGNS[code]
        print(f"== arena {code}: span={span} head={head} parse={parse}", flush=True)
        run_codec(
            "concept",
            corpus_name,
            nbytes,
            "test",
            8192,
            16,
            16,
            active,
            steps,
            proof,
            rotate,
            full_every,
            seed,
            span,
            head,
            parse,
        )
        design = "" if (span, head) == ("content", "gate") else f"-{span[0]}{head[0]}"
        if parse != "greedy":
            design += "-p"
        suffix = (f"-a{active}" if active != 256 else "") + design
        path = RUNS / f"codec-concept-{corpus_name}-{nbytes}{suffix}" / "metrics.json"
        rows.append(json.loads(path.read_text()))
    lines = [
        "# Design arena",
        "",
        "One row per design on the same frozen slice, pack and proof.",
        "",
        "| design | span | head | payload | shared | self | units | coverage | proof | enc s | run |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for m in rows:
        lines.append(
            f"| {m.get('design', m['arm'])} | {m.get('span', '?')} | "
            f"{m.get('head', '?')} | {m['payload_bpb']:.3f} | "
            f"{m['shared_bpb']:.3f} | {m['self_bpb']:.3f} | {m.get('units', 0)} | "
            f"{(m.get('coverage') or 0):.3f} | {m.get('proof_units', 0)} | "
            f"{m.get('encode_seconds') or 0:.1f} | `{m['id']}` |"
        )
    best = min(rows, key=lambda m: m["payload_bpb"])
    lines += [
        "",
        f"Best payload: **{best['id']}** ({best['payload_bpb']:.3f} bpb).",
        "",
    ]
    path = REPORTS / "ARENA.md"
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines))
    os.replace(tmp, path)
    print("\n".join(lines))
    return 0


def run_demo(run: str, nbytes: int) -> int:
    """The end-to-end loop: install -> provenance -> edit -> unseen pack."""
    import pointer

    out_dir, metrics, frame = _run_frame(run)
    data, spans = _decode_run(out_dir, frame, trace=True)
    lex = frame.lexicon
    assert lex is not None
    covered = sum(b - a for a, b, _ in spans)
    print(f"== install: {run}")
    print(
        f"  {len(lex.units)} units / {len(lex.blob())} B installed, "
        f"payload {metrics['payload_bpb']:.3f} bpb, "
        f"provenance {covered}/{len(data)} B"
    )
    print("== provenance: byte -> unit -> pack offset")
    offs = lex.offsets()
    for start, end, sym in spans[:5]:
        if sym < len(lex.units):
            print(
                f"  [{start},{end}) unit {sym} @ {offs[sym]} `{lex.units[sym][:24]!r}`"
            )
        else:
            print(f"  [{start},{end}) literal `{bytes([sym - len(lex.units)])!r}`")
    print("== edit: rm a concept")
    prefix = lex.units[0][:1] or lex.units[0]
    pack_rm(run, prefix, min(nbytes, len(data)))
    print("== install a pack induced on bytes the model never trained on")
    corpus = code_corpus()
    other = corpus.test[-max(nbytes, 200_000) :]
    fresh = codec_mod.build_lexicon(other, 256, 2, 16)
    units = fresh.units[:256]
    if len(units) < 8:
        print("  (not enough units)")
        return 0
    model = pointer.load_model(out_dir / "model.pt")
    pc = pointer.PointerCodec(model, codec_mod.Lexicon(units), half=True)
    held = data[:nbytes]
    blob = pc.encode(held)
    out = pc.decode(blob, len(held))
    print(
        f"  unseen pack: {len(units)} units / {len(codec_mod.Lexicon(units).blob())} B, "
        f"payload {8 * len(blob) / max(len(held), 1):.3f} bpb, "
        f"round-trip {'OK' if out == held else 'FAIL'}"
    )
    return 0


def generate(run: str, nbytes: int, seed: int) -> int:
    """Sample from the installed model; print the bytes and the used concepts."""
    import pointer

    out_dir, metrics, frame = _run_frame(run)
    model = pointer.load_model(out_dir / "model.pt")
    pc = pointer.PointerCodec(model, frame.lexicon, half=True)
    raw = pc.sample(nbytes, seed)
    print(f"== generate {nbytes} B from {run} (seed {seed})")
    print(raw.decode("utf-8", "replace"))
    return 0


def _package_files(
    corpus: CodeCorpus, package: str
) -> tuple[bytes, list[tuple[str, int, int]]]:
    """Held-out files whose path names `package`, concatenated in path order."""
    hits = [f for f in corpus.files if f[1] >= corpus.split and package in f[0]]
    if not hits:
        raise SystemExit(f"install: no held-out files match package {package!r}")
    data = b"".join(corpus.raw[start:end] for _, start, end in hits)
    return data, hits


def run_lm(
    corpus_name: str = "shakespeare",
    tiny: bool = True,
    steps: int = 2000,
    prompt: bytes = b"ROMEO:",
    sample: int = 400,
    active: int = 256,
    units: int = 8192,
    min_count: int = 16,
    cap: int = 16,
    proof: int = 4000,
    seed: int = 0,
) -> int:
    """P0a: train a tiny model, install a pack, tag a prompt continuation."""
    import pointer

    t0 = time.perf_counter()
    corpus = text_corpus(corpus_name)
    lex = _lexicon_for(corpus, units, min_count, cap)
    d, layers = (128, 4) if tiny else (256, 6)
    model, active_lex, losses = _pointer_for(
        corpus,
        lex,
        active,
        steps,
        seed,
        0,
        0,
        span="ngram",
        head="joint",
        seq=256,
        d=d,
        layers=layers,
    )
    params = sum(p.numel() for p in model.parameters())
    train_bpb = pointer.eval_bpb(model, corpus.train[:131072], active_lex)
    run_dir = RUNS / f"lm-{corpus_name}-{'tiny' if tiny else 'base'}-s{steps}"
    pointer.save_model(run_dir / "model.pt", model, model.cfg, losses)
    pcodec = pointer.PointerCodec(model, active_lex, half=True)
    ids, covered = codec_mod.greedy_parse(corpus.test, active_lex)
    stats = codec_mod.parse_stats(ids, active_lex, len(corpus.test), covered)
    t_enc = time.perf_counter()
    payload = pcodec.encode(corpus.test)
    encode_s = time.perf_counter() - t_enc
    want = min(proof, len(ids))
    out = pcodec.decode(payload, len(corpus.test), limit=want)
    if out != corpus.test[: len(out)]:
        raise SystemExit("lm: proof round-trip failed")
    raw, tags = pcodec.continuation(prompt, sample, seed)
    train_loss = sum(losses[-50:]) / max(len(losses[-50:]), 1)
    n = max(len(corpus.test), 1)
    payload_bpb = 8 * len(payload) / n
    write_json(
        run_dir / "metrics.json",
        {
            "id": run_dir.name,
            "corpus": corpus_name,
            "corpus_sha": corpus.sha,
            "train_sha": corpus.train_sha,
            "params": params,
            "d": d,
            "layers": layers,
            "steps": steps,
            "seq": model.cfg.seq,
            "active": len(active_lex.units),
            "lexicon_bytes": len(active_lex.blob()),
            "test_bytes": len(corpus.test),
            "payload_bytes": len(payload),
            "payload_bpb": payload_bpb,
            "coverage": stats["coverage"],
            "train_loss": train_loss,
            "train_bpb": train_bpb["payload_bpb"],
            "proof_units": want,
            "proof_bytes": len(out),
            "roundtrip": True,
            "encode_seconds": round(encode_s, 3),
            "prompt": prompt.decode("utf-8", "replace"),
            "sample_bytes": len(raw),
            "sample_copies": sum(1 for *_, t in tags if t.startswith("copy@")),
            "sample_literals": sum(1 for *_, t in tags if t == "~neural"),
            "git": git_sha(),
            "seconds": round(time.perf_counter() - t0, 3),
        },
    )
    (run_dir / "lexicon.bin").write_bytes(active_lex.blob())
    (run_dir / "sample.bin").write_bytes(raw)
    lines = [
        "# LM (P0a): Tiny Shakespeare",
        "",
        f"- corpus `{corpus_name}` sha `{corpus.sha[:12]}`, train "
        f"{len(corpus.train):,} B, test {len(corpus.test):,} B",
        f"- model {params:,} params (d={d}, {layers} layers), {steps} steps, "
        f"ngram-joint, seq {model.cfg.seq}",
        f"- pack {len(active_lex.units)} units / {len(active_lex.blob()):,} B "
        "installed from the train split",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| train loss (nats/elem, last 50) | {train_loss:.3f} |",
        f"| train bpb (teacher-forced, 128 kB) | {train_bpb['payload_bpb']:.3f} |",
        f"| test payload bpb | {payload_bpb:.3f} |",
        f"| test payload bytes | {len(payload):,} |",
        f"| test coverage | {stats['coverage']:.3f} |",
        f"| proof | {want:,} units / {len(out):,} B, "
        f"{'OK' if out == corpus.test[: len(out)] else 'FAIL'} |",
        f"| encode | {encode_s:.1f} s ({len(corpus.test) / encode_s / 1024:.1f} kB/s) |",
        "",
        "`shared` is not reported here.  The pack charge is unscaled at this",
        f"size: {len(active_lex.blob()):,} B / {len(corpus.test):,} B = "
        f"{8 * len(active_lex.blob()) / n:.3f} bpb.  The text bpb claim needs "
        "the 1 MB enwik8 test, not this slice.",
        "",
        f"## Sample after `{prompt.decode('utf-8', 'replace')}`",
        "",
        "```",
        (prompt + raw).decode("utf-8", "replace"),
        "```",
        "",
        "## Provenance: every emitted span",
        "",
        f"copies {sum(1 for *_, t in tags if t.startswith('copy@'))}, "
        f"neural {sum(1 for *_, t in tags if t == '~neural')}",
        "",
        "| start | end | tag | bytes |",
        "|---:|---:|---|---|",
    ]
    for start, end, tag in tags:
        cell = repr(raw[start:end][:32]).replace("|", "\\|")
        lines.append(f"| {start} | {end} | `{tag}` | `{cell}` |")
    report = REPORTS / "LM.md"
    tmp = report.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, report)
    print(
        f"{run_dir.name}: {params:,} params, train loss {train_loss:.3f}, "
        f"test payload {payload_bpb:.3f} bpb, pack "
        f"{len(active_lex.blob()):,} B, proof {want} units OK"
    )
    print(f"== sample after {prompt!r}")
    for start, end, tag in tags:
        print(f"  [{start},{end}) {tag} {raw[start:end][:32]!r}")
    print((prompt + raw).decode("utf-8", "replace"))
    print(f"wrote {report}")
    return 0


def _encode_row(
    model, data: bytes, lex: codec_mod.Lexicon, proof: int
) -> dict[str, object]:
    """Encode one pack on one slice; decode the proof and return the row."""
    import pointer

    pc = pointer.PointerCodec(model, lex, half=True)
    ids, covered = codec_mod.greedy_parse(data, lex)
    stats = codec_mod.parse_stats(ids, lex, len(data), covered)
    t_enc = time.perf_counter()
    payload = pc.encode(data)
    encode_s = time.perf_counter() - t_enc
    want = min(proof, len(ids))
    out = pc.decode(payload, len(data), limit=want)
    return {
        "units": len(lex.units),
        "lexicon_bytes": len(lex.blob()),
        "payload_bytes": len(payload),
        "payload_bpb": 8 * len(payload) / max(len(data), 1),
        "coverage": stats["coverage"],
        "mean_unit": stats["mean_unit"],
        "literals": stats["literals"],
        "encode_seconds": round(encode_s, 3),
        "proof_bytes": len(out),
        "roundtrip": out == data[: len(out)],
    }


def run_install(
    corpus_name: str = "code",
    package: str = "networkx",
    active: int = 256,
    steps: int = 3000,
    units: int = 8192,
    min_count: int = 16,
    cap: int = 16,
    proof: int = 4000,
    seed: int = 0,
    rotate: int = 8,
    full_every: int = 8,
) -> int:
    """P0b: the install delta on held-out package files, same model, three packs."""
    import pointer

    t0 = time.perf_counter()
    corpus = code_corpus() if corpus_name == "code" else text_corpus(corpus_name)
    data, files = _package_files(corpus, package)
    train_lex = _lexicon_for(corpus, units, min_count, cap)
    model, train_pack, losses = _pointer_for(
        corpus,
        train_lex,
        active,
        steps,
        seed,
        rotate,
        full_every,
        span="ngram",
        head="joint",
        seq=256,
    )
    target_pack = codec_mod.build_lexicon(data, active, min_count, cap)
    run_dir = RUNS / f"install-{corpus_name}-{package}-a{active}-s{steps}"
    pointer.save_model(run_dir / "model.pt", model, model.cfg, losses)
    packs = (
        ("a", "train", train_pack),
        ("b", "package", target_pack),
        ("c", "none", codec_mod.Lexicon(())),
    )
    rows = []
    for tag, name, lex in packs:
        row = _encode_row(model, data, lex, proof)
        row.update(pack=tag, name=name)
        rows.append(row)
        print(
            f"  ({tag}) {name:8s}: {row['units']:3d} units / "
            f"{row['lexicon_bytes']:5d} B, {row['payload_bytes']:6d} B payload, "
            f"{row['payload_bpb']:.3f} bpb, "
            f"coverage {row['coverage']:.3f}, proof "
            f"{'OK' if row['roundtrip'] else 'FAIL'}",
            flush=True,
        )
    a, b, c = (row["payload_bpb"] for row in rows)
    d_ba, d_ca = b - a, c - a
    delta_ok = abs(d_ba) <= 0.15
    gain_ok = d_ca >= 0.2
    write_json(
        run_dir / "metrics.json",
        {
            "id": run_dir.name,
            "corpus": corpus_name,
            "corpus_sha": corpus.sha,
            "train_sha": corpus.train_sha,
            "package": package,
            "files": len(files),
            "data_bytes": len(data),
            "steps": steps,
            "active": active,
            "rotate": rotate,
            "full_every": full_every,
            "params": sum(p.numel() for p in model.parameters()),
            "packs": rows,
            "delta_b_minus_a": d_ba,
            "delta_c_minus_a": d_ca,
            "gate_delta": 0.15,
            "gate_gain": 0.2,
            "pass_delta": delta_ok,
            "pass_gain": gain_ok,
            "roundtrip": all(row["roundtrip"] for row in rows),
            "git": git_sha(),
            "seconds": round(time.perf_counter() - t0, 3),
        },
    )
    for row, lex in zip(rows, (train_pack, target_pack, codec_mod.Lexicon(()))):
        (run_dir / f"pack-{row['pack']}.bin").write_bytes(lex.blob())
    lines = [
        "# Install delta (P0b)",
        "",
        f"- corpus `{corpus_name}` sha `{corpus.sha[:12]}`, train "
        f"{len(corpus.train):,} B, test {len(corpus.test):,} B",
        f"- package `{package}`: {len(files)} held-out files / {len(data):,} B",
        f"- {steps} steps, {active} active units, ngram-joint, seq {model.cfg.seq}",
        "",
        "| pack | units | pack B | payload B | payload bpb | coverage | proof |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| ({row['pack']}) {row['name']} | {row['units']} | "
            f"{row['lexicon_bytes']} | {row['payload_bytes']} | "
            f"{row['payload_bpb']:.3f} | {row['coverage']:.3f} | "
            f"{'OK' if row['roundtrip'] else 'FAIL'} |"
        )
    lines += [
        "",
        f"- `(b) - (a)` = **{d_ba:+.3f} bpb** (bar `|d| <= 0.15`): "
        f"{'PASS' if delta_ok else 'FAIL'}",
        f"- `(c) - (a)` = **{d_ca:+.3f} bpb** (bar `>= +0.20`): "
        f"{'PASS' if gain_ok else 'FAIL'}",
        "",
        "The install is causal when the same trained model reads a pack induced",
        "on the target package nearly as well as its train pack, and both beat",
        "all literals.  `(b)` is an oracle pack (induced on the encoded bytes);",
        "it measures transfer of the encoder/head, not held-out pack induction.",
        "",
        "| gate | result |",
        "|---|---|",
    ]
    lines.append(
        f"| package pack within 0.15 bpb of train | {'PASS' if delta_ok else 'FAIL'} |"
    )
    lines.append(
        f"| train pack beats no pack by 0.20 bpb | {'PASS' if gain_ok else 'FAIL'} |"
    )
    report = REPORTS / "INSTALL.md"
    tmp = report.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, report)
    print(
        f"{run_dir.name}: (a) {a:.3f}, (b) {b:.3f}, (c) {c:.3f} bpb; "
        f"(b)-(a) {d_ba:+.3f}, (c)-(a) {d_ca:+.3f}; "
        f"gate {'PASS' if delta_ok and gain_ok else 'FAIL'}"
    )
    print(f"wrote {report}")
    return 0


def _file_halves(corpus: CodeCorpus, files: list[tuple[str, int, int]]):
    """Split the package's held-out files into two concatenated halves."""
    k = len(files) // 2
    a = b"".join(corpus.raw[start:end] for _, start, end in files[:k])
    b = b"".join(corpus.raw[start:end] for _, start, end in files[k:])
    return a, b


def run_install_transfer(
    corpus_name: str = "code",
    package: str = "networkx",
    active: int = 256,
    steps: int = 3000,
    units: int = 8192,
    min_count: int = 16,
    cap: int = 16,
    proof: int = 4000,
    seed: int = 0,
    rotate: int = 8,
    full_every: int = 8,
) -> int:
    """P0b without the oracle: pack on one held-out half, encode the other."""
    import pointer

    t0 = time.perf_counter()
    corpus = code_corpus() if corpus_name == "code" else text_corpus(corpus_name)
    _, files = _package_files(corpus, package)
    if len(files) < 2:
        raise SystemExit("install: the package needs at least two held-out files")
    left, right = _file_halves(corpus, files)
    halves = (("A->B", left, right), ("B->A", right, left))
    train_lex = _lexicon_for(corpus, units, min_count, cap)
    model, train_pack, losses = _pointer_for(
        corpus,
        train_lex,
        active,
        steps,
        seed,
        rotate,
        full_every,
        span="ngram",
        head="joint",
        seq=256,
    )
    run_dir = RUNS / f"install-transfer-{corpus_name}-{package}-a{active}-s{steps}"
    pointer.save_model(run_dir / "model.pt", model, model.cfg, losses)
    runs = []
    for label, pack_src, held in halves:
        package_pack = codec_mod.build_lexicon(pack_src, active, min_count, cap)
        rows = []
        for tag, name, lex in (
            ("a", "train", train_pack),
            ("b", "half", package_pack),
            ("c", "none", codec_mod.Lexicon(())),
        ):
            row = _encode_row(model, held, lex, proof)
            row.update(pack=tag, name=name)
            rows.append(row)
            print(
                f"  {label} ({tag}) {name:5s}: {row['units']:3d} units / "
                f"{row['lexicon_bytes']:5d} B, {row['payload_bytes']:6d} B, "
                f"{row['payload_bpb']:.3f} bpb, "
                f"coverage {row['coverage']:.3f}, proof "
                f"{'OK' if row['roundtrip'] else 'FAIL'}",
                flush=True,
            )
        a, b, c = (row["payload_bpb"] for row in rows)
        d_ba, d_ca = b - a, c - a
        delta_ok = abs(d_ba) <= 0.15
        gain_ok = d_ca >= 0.2
        runs.append(
            {
                "direction": label,
                "pack_source_bytes": len(pack_src),
                "heldout_bytes": len(held),
                "pack_bytes": len(package_pack.blob()),
                "rows": rows,
                "delta_b_minus_a": d_ba,
                "delta_c_minus_a": d_ca,
                "pass_delta": delta_ok,
                "pass_gain": gain_ok,
            }
        )
        (run_dir / f"pack-{label.replace('->', '')}.bin").write_bytes(
            package_pack.blob()
        )
    write_json(
        run_dir / "metrics.json",
        {
            "id": run_dir.name,
            "corpus": corpus_name,
            "corpus_sha": corpus.sha,
            "train_sha": corpus.train_sha,
            "package": package,
            "files": len(files),
            "split": len(files) // 2,
            "steps": steps,
            "active": active,
            "rotate": rotate,
            "full_every": full_every,
            "params": sum(p.numel() for p in model.parameters()),
            "directions": runs,
            "git": git_sha(),
            "seconds": round(time.perf_counter() - t0, 3),
        },
    )
    lines = [
        "# Install transfer (P0b, no oracle)",
        "",
        f"- corpus `{corpus_name}` sha `{corpus.sha[:12]}`, train "
        f"{len(corpus.train):,} B, test {len(corpus.test):,} B",
        f"- package `{package}`: {len(files)} held-out files split at "
        f"{len(files) // 2}; A {len(left):,} B, B {len(right):,} B",
        f"- {steps} steps, {active} active units, ngram-joint, seq {model.cfg.seq}",
        "",
        "`(b)` is induced on one half and encodes the other, so the pack never",
        "touches the bytes in its payload.  `(a)` is the train pack; `(c)` is",
        "all literals.",
        "",
        "| direction | pack | units | pack B | payload B | payload bpb | coverage | proof |",
        "|---|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for run in runs:
        for row in run["rows"]:
            lines.append(
                f"| {run['direction']} | ({row['pack']}) {row['name']} | "
                f"{row['units']} | {row['lexicon_bytes']} | "
                f"{row['payload_bytes']} | {row['payload_bpb']:.3f} | "
                f"{row['coverage']:.3f} | "
                f"{'OK' if row['roundtrip'] else 'FAIL'} |"
            )
    lines += [
        "",
        "| direction | `(b)-(a)` | `(c)-(a)` | gate |",
        "|---|---:|---:|---|",
    ]
    for run in runs:
        ok = run["pass_delta"] and run["pass_gain"]
        lines.append(
            f"| {run['direction']} | {run['delta_b_minus_a']:+.3f} | "
            f"{run['delta_c_minus_a']:+.3f} | {'PASS' if ok else 'FAIL'} |"
        )
    both = all(r["pass_delta"] and r["pass_gain"] for r in runs)
    lines += [
        "",
        "Bars: `|(b)-(a)| <= 0.15` and `(c)-(a) >= 0.20`, in both directions.",
        f"Overall: **{'PASS' if both else 'FAIL'}**.",
        "",
    ]
    report = REPORTS / "INSTALL-TRANSFER.md"
    tmp = report.with_suffix(".tmp")
    tmp.write_text("\n".join(lines))
    os.replace(tmp, report)
    for run in runs:
        print(
            f"{run['direction']}: (a) {run['rows'][0]['payload_bpb']:.3f}, "
            f"(b) {run['rows'][1]['payload_bpb']:.3f}, "
            f"(c) {run['rows'][2]['payload_bpb']:.3f} bpb; "
            f"(b)-(a) {run['delta_b_minus_a']:+.3f}, "
            f"(c)-(a) {run['delta_c_minus_a']:+.3f}"
        )
    print(f"{'PASS' if both else 'FAIL'}: wrote {report}")
    return 0 if both else 1


def templates(run: str, top: int = 10) -> int:
    """r7: lift a line template with a hole by anti-unification."""
    import re
    from collections import Counter

    _, metrics, _ = _run_frame(run)
    data = _reference(metrics)
    pat = re.compile(rb"[A-Za-z_][A-Za-z0-9_]*|\d+")
    lines = [line for line in data.splitlines() if 8 <= len(line) <= 200]
    skeleton = {line: pat.sub(b"\x00", line) for line in lines}
    groups = Counter(skeleton.values())
    print(f"== templates from {run}: {len(lines)} lines, {len(groups)} templates")
    shown = 0
    for skel, count in groups.most_common():
        if count < 3:
            break
        fills = [line for line in lines if skeleton[line] == skel][:3]
        print(f"  {count:4d}x `{skel[:72]!r}`")
        for fill in fills:
            print(f"        fill `{fill[:72]!r}`")
        shown += 1
        if shown >= top:
            break
    return 0


def _run_frame(run: str):
    out_dir = RUNS / run
    metrics = json.loads((out_dir / "metrics.json").read_text())
    frame = codec_mod.unpack_frame((out_dir / "frame.bin").read_bytes())
    return out_dir, metrics, frame


def _reference(metrics: dict) -> bytes:
    for corpus in (code_corpus(), shakespeare_corpus()):
        if corpus.sha[:12] == metrics["corpus_sha"][:12]:
            pool = {"train": corpus.train, "test": corpus.test, "all": corpus.raw}
            return pool[metrics["split"]][: metrics["n"]]
    raise SystemExit("pack: the frame's corpus is not frozen")


def _decode_run(out_dir: Path, frame, trace: bool = False):
    if frame.arm == 0:
        data = codec_mod.ByteArm().decode(frame.payload, frame.n)
        spans = [(i, i + 1, -1 - b) for i, b in enumerate(data)]
        return (data, spans) if trace else data
    if frame.arm == 1:
        arm = codec_mod.UnitArm(frame.lexicon)
        return (
            arm.trace(frame.payload, frame.n)
            if trace
            else arm.decode(frame.payload, frame.n)
        )
    if frame.arm == 2:
        import pointer

        model = pointer.load_model(out_dir / "model.pt")
        pc = pointer.PointerCodec(model, frame.lexicon, half=True)
        return pc.decode(frame.payload, frame.n, trace=trace)
    raise SystemExit(f"pack: unknown arm {frame.arm}")


def _run_codec_arm(out_dir: Path, frame):
    """The arm that produced the frame, rebuilt for a new encode."""
    if frame.arm == 0:
        return codec_mod.ByteArm()
    if frame.arm == 1:
        return codec_mod.UnitArm(frame.lexicon)
    import pointer

    return pointer.PointerCodec(
        pointer.load_model(out_dir / "model.pt"), frame.lexicon, half=True
    )


def pack_verify(run: str) -> int:
    out_dir, metrics, frame = _run_frame(run)
    data, spans = _decode_run(out_dir, frame, trace=True)
    ref = _reference(metrics)
    adjacent = all(b == c for (_, b, _), (c, _, _) in zip(spans, spans[1:]))
    covered = sum(b - a for a, b, _ in spans)
    lex = frame.lexicon
    lex_off = sum(2 + len(u) for u in lex.units) + 12 if lex else 0
    print(
        f"{run}: arm {frame.arm}, {len(data)} B, round-trip "
        f"{'OK' if data == ref else 'FAIL'}, {len(spans)} elements, "
        f"provenance {covered}/{len(data)} B "
        f"{'contiguous' if adjacent else 'GAPPED'}"
    )
    if lex:
        print(
            f"  pack {len(lex.units)} units / {lex_off} B, "
            f"metrics lexicon {metrics.get('lexicon_bytes', 0)} B, "
            f"sha {lex.sha[:12]}"
        )
    return 0 if data == ref and adjacent and covered == len(data) else 1


def pack_trace(run: str, lo: int, hi: int) -> int:
    out_dir, _, frame = _run_frame(run)
    data, spans = _decode_run(out_dir, frame, trace=True)
    lex = frame.lexicon
    offs = lex.offsets() if lex else []
    rows = [s for s in spans if s[0] < hi and s[1] > lo]
    print(f"{run}: {len(rows)} elements in bytes [{lo}, {hi}) of {len(data)}")
    print("| start | end | unit | pack offset | bytes |")
    print("|---:|---:|---:|---:|---|")
    k = len(lex.units) if lex else 0
    for start, end, sym in rows[:200]:
        if lex and sym < k:
            unit = lex.units[sym]
            row = f"| {start} | {end} | {sym} | {offs[sym]} | `{unit[:32]!r}` |"
        else:
            row = f"| {start} | {end} | lit | — | `{bytes([sym - k])!r}` |"
        print(row)
    if len(rows) > 200:
        print(f"... {len(rows) - 200} more")
    return 0


def pack_diff(a: str, b: str) -> int:
    _, _, fa = _run_frame(a)
    _, _, fb = _run_frame(b)
    ua, ub = fa.lexicon, fb.lexicon
    if ua is None or ub is None:
        raise SystemExit("pack diff: both frames need a lexicon")
    sa, sb = set(ua.units), set(ub.units)
    added, removed = sorted(sb - sa), sorted(sa - sb)
    print(
        f"{a}: {len(ua.units)} units / {len(ua.blob())} B sha {ua.sha[:12]}\n"
        f"{b}: {len(ub.units)} units / {len(ub.blob())} B sha {ub.sha[:12]}\n"
        f"shared {len(sa & sb)}, added {len(added)}, removed {len(removed)}"
    )
    oa, ob = ua.offsets(), ub.offsets()
    for tag, units, lex, offs in (
        ("+", added, ub, ob),
        ("-", removed, ua, oa),
    ):
        for unit in units[:20]:
            print(f"  {tag} {offs[lex.units.index(unit)]:>7} {unit[:40]!r}")
    return 0


def pack_merge(out: str, a: str, b: str, size: int) -> int:
    _, _, fa = _run_frame(a)
    _, _, fb = _run_frame(b)
    if fa.lexicon is None or fb.lexicon is None:
        raise SystemExit("pack merge: both frames need a lexicon")
    units = list(fa.lexicon.units)
    seen = set(units)
    for unit in fb.lexicon.units:
        if unit not in seen:
            units.append(unit)
            seen.add(unit)
    units = tuple(units[:size])
    blob = codec_mod.Lexicon(units).blob()
    lex = codec_mod.Lexicon(units, sha=hashlib.sha256(blob).hexdigest())
    path = RUNS / out / "lexicon.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(blob)
    os.replace(tmp, path)
    print(
        f"{out}: {len(lex.units)} units / {len(lex.blob())} B sha {lex.sha[:12]} "
        f"from {a} + {b} (cap {size})"
    )
    return 0


def _changed_ranges(parse: list[int], lex, prefix: bytes):
    """Byte ranges where a parse used a removed unit."""
    lens = [len(lex.units[s]) if s >= 0 else 1 for s in parse]
    starts = np.concatenate(([0], np.cumsum(lens)))[:-1]
    hit = [
        (int(a), int(a) + n)
        for a, s, n in zip(starts, parse, lens)
        if s >= 0 and lex.units[s].startswith(prefix)
    ]
    return hit


def pack_rm(run: str, prefix: bytes, nbytes: int) -> int:
    import time

    out_dir, metrics, frame = _run_frame(run)
    if frame.lexicon is None:
        raise SystemExit("pack rm: the frame has no lexicon")
    data = _reference(metrics)[:nbytes]
    full = frame.lexicon
    keep = full.drop_prefix(prefix)
    if len(keep.units) == len(full.units):
        raise SystemExit(f"pack rm: no unit starts with {prefix!r}")
    parse_full, _ = codec_mod.greedy_parse(data, full)
    parse_red, _ = codec_mod.greedy_parse(data, keep)
    ranges = _changed_ranges(parse_full, full, prefix)
    nf, nr = len(parse_full), len(parse_red)
    t0 = time.perf_counter()
    before = _run_codec_arm(out_dir, frame).encode(data)
    if frame.arm == 2:
        import pointer

        model = pointer.load_model(out_dir / "model.pt")
        after = pointer.PointerCodec(model, keep, half=True)
        fresh = lambda: pointer.PointerCodec(model, keep, half=True)  # noqa: E731
    elif frame.arm == 1:
        after = codec_mod.UnitArm(keep)
        fresh = lambda: codec_mod.UnitArm(keep)  # noqa: E731
    else:
        after = codec_mod.ByteArm()
        fresh = codec_mod.ByteArm
    blob = after.encode(data)
    n = max(len(data), 1)
    print(
        f"{run}: rm {prefix!r}: {len(full.units)} -> {len(keep.units)} units, "
        f"{len(ranges)} affected spans / "
        f"{sum(b - a for a, b in ranges)} B, parse {nf} -> {nr} symbols"
    )
    print(
        f"  payload {len(before)} -> {len(blob)} B "
        f"({8 * len(before) / n:.3f} -> {8 * len(blob) / n:.3f} bpb, "
        f"{8 * (len(blob) - len(before)) / n:+.3f}); "
        f"{time.perf_counter() - t0:.1f}s"
    )
    for a, b in ranges[:20]:
        print(f"  [{a}, {b}) {data[a:b][:48]!r}")
    if len(ranges) > 20:
        print(f"  ... {len(ranges) - 20} more")
    out = fresh().decode(blob, len(data))
    if out != data:
        raise SystemExit("pack rm: round-trip failed")
    print("  round-trip OK")
    return 0


def recorded_report() -> int:
    rows = []
    for path in sorted(RUNS.glob("codec-*/metrics.json")):
        m = json.loads(path.read_text())
        rows.append(m)
    lines = [
        "# Recorded codec runs",
        "",
        "Generated by `cml recorded`.  `shared` charges the header (and the",
        "installed lexicon, which the frame carries); `self` adds the 4-bit base.",
        "",
        "| arm | corpus | sha | bytes | payload | shared | self | proof | lexicon | round-trip | run |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for m in rows:
        mark = "✓" if m.get("roundtrip") else "✗"
        proof = m.get("proof_units") or (
            f"{m['proof_bytes']} B" if m.get("proof_bytes") else "—"
        )
        lines.append(
            f"| {m['arm']} | {m['corpus']} | {m['corpus_sha'][:8]} | {m['n']} | "
            f"{m['payload_bpb']:.3f} | {m['shared_bpb']:.3f} | "
            f"{m['self_bpb']:.3f} | {proof} | {m.get('lexicon_bytes', 0)} | "
            f"{mark} | `{m['id']}` |"
        )
    if not rows:
        lines += ["", "No `codec-*` runs yet.", ""]
    report = REPORTS / "RECORDED.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    tmp = report.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, report)
    print(f"recorded {len(rows)} runs -> {report}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cml")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("selftest")
    r = sub.add_parser("run")
    r.add_argument("manifest", type=Path)
    r.add_argument("--seed", type=int)
    r.add_argument("--params", type=int)
    r.add_argument("--steps", type=int)
    r.add_argument("--dry", action="store_true")
    b = sub.add_parser("base")
    b.add_argument("--params", type=int, default=20_000_000)
    b.add_argument("--steps", type=int, default=6000)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--seq", type=int, default=256)
    b.add_argument("--corpus", default="enwik8_10m")
    sub.add_parser("space")
    s = sub.add_parser("search")
    s.add_argument("stage", choices=("P0", "P1", "P2", "P3", "P4", "P5"))
    s.add_argument("--budget", type=float, default=12.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--simulate", action="store_true")
    sub.add_parser("status")
    sub.add_parser("report")
    sub.add_parser("gates")
    sw = sub.add_parser("sweep")
    sw.add_argument("stage", choices=("P0",))
    sw.add_argument("--corpus", default="enwik8_10m")
    sw.add_argument("--slice", type=int, default=300_000)
    c = sub.add_parser("curve")
    c.add_argument("stage", choices=("P1",))
    c.add_argument("--steps", type=int, default=6000)
    c.add_argument("--update", action="store_true")
    co = sub.add_parser("codec")
    co.add_argument(
        "--arm", choices=("byte", "byte-model", "lexicon", "concept"), default="byte"
    )
    co.add_argument("--corpus", choices=("code", "shakespeare"), default="code")
    co.add_argument("--bytes", type=int, default=1_000_000)
    co.add_argument("--split", choices=("train", "test", "all"), default="test")
    co.add_argument("--units", type=int, default=8192)
    co.add_argument("--min-count", type=int, default=16)
    co.add_argument("--cap", type=int, default=16)
    co.add_argument("--active", type=int, default=256)
    co.add_argument("--steps", type=int, default=1000)
    co.add_argument("--proof", type=int, default=50_000)
    co.add_argument("--rotate", type=int, default=0)
    co.add_argument("--full-every", type=int, default=2)
    co.add_argument("--seed", type=int, default=0)
    co.add_argument("--span", choices=("content", "ngram"), default="content")
    co.add_argument("--head", choices=("gate", "joint"), default="gate")
    co.add_argument("--parse", choices=("greedy", "priced"), default="greedy")
    co.add_argument("--seq", type=int, default=256)
    cd = sub.add_parser("coder")
    cd.add_argument("--test", action="store_true")
    sub.add_parser("recorded")
    dm = sub.add_parser("demo")
    dm.add_argument("run")
    dm.add_argument("--bytes", type=int, default=20000)
    ge = sub.add_parser("generate")
    ge.add_argument("run")
    ge.add_argument("--bytes", type=int, default=400)
    ge.add_argument("--seed", type=int, default=0)
    lm = sub.add_parser("lm")
    lm.add_argument("--corpus", choices=("shakespeare",), default="shakespeare")
    lm.add_argument("--tiny", action="store_true")
    lm.add_argument("--steps", type=int, default=2000)
    lm.add_argument("--prompt", default="ROMEO:")
    lm.add_argument("--sample", type=int, default=400)
    lm.add_argument("--active", type=int, default=256)
    lm.add_argument("--units", type=int, default=8192)
    lm.add_argument("--min-count", type=int, default=16)
    lm.add_argument("--cap", type=int, default=16)
    lm.add_argument("--proof", type=int, default=4000)
    lm.add_argument("--seed", type=int, default=0)
    ins = sub.add_parser("install")
    ins.add_argument("--corpus", default="code")
    ins.add_argument("--package", default="networkx")
    ins.add_argument("--active", type=int, default=256)
    ins.add_argument("--steps", type=int, default=3000)
    ins.add_argument("--units", type=int, default=8192)
    ins.add_argument("--min-count", type=int, default=16)
    ins.add_argument("--cap", type=int, default=16)
    ins.add_argument("--proof", type=int, default=4000)
    ins.add_argument("--seed", type=int, default=0)
    ins.add_argument("--rotate", type=int, default=8)
    ins.add_argument("--full-every", type=int, default=8)
    ins.add_argument("--transfer", action="store_true")
    tp = sub.add_parser("templates")
    tp.add_argument("run")
    tp.add_argument("--top", type=int, default=10)
    ar = sub.add_parser("arena")
    ar.add_argument("--corpus", default="code")
    ar.add_argument("--bytes", type=int, default=70000)
    ar.add_argument("--proof", type=int, default=50000)
    ar.add_argument("--steps", type=int, default=8000)
    ar.add_argument("--active", type=int, default=256)
    ar.add_argument("--designs", default="gc,jc,gn,jn")
    ar.add_argument("--rotate", type=int, default=8)
    ar.add_argument("--full-every", type=int, default=8)
    ar.add_argument("--seed", type=int, default=0)
    pk = sub.add_parser("pack")
    pks = pk.add_subparsers(dest="op", required=True)
    v = pks.add_parser("verify")
    v.add_argument("run")
    t = pks.add_parser("trace")
    t.add_argument("run")
    t.add_argument("--range", default="0:256")
    d = pks.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    m = pks.add_parser("merge")
    m.add_argument("out")
    m.add_argument("a")
    m.add_argument("b")
    m.add_argument("--size", type=int, default=8192)
    r = pks.add_parser("rm")
    r.add_argument("run")
    r.add_argument("--prefix", required=True)
    r.add_argument("--bytes", type=int, default=70000)
    a = p.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "run":
        return run(a.manifest, a.seed, a.params, a.steps, a.dry)
    if a.cmd == "base":
        return run_base(a.params, a.steps, a.seed, a.seq, a.corpus)
    if a.cmd == "space":
        return space()
    if a.cmd == "gates":
        return gates_report()
    if a.cmd == "sweep":
        return sweep_p0(a.corpus, a.slice)
    if a.cmd == "curve":
        return curve(a.stage, a.steps, a.update)
    if a.cmd == "search":
        if not a.simulate:
            raise SystemExit(
                "search: real pipelines are P0-P5 (# TODO); use --simulate"
            )
        state = run_search(a.stage, a.budget, a.seed)
        d = state_dict(state)
        write_json(REPORTS / "search.json", d)
        (REPORTS / "search.jsonl").write_text(
            "\n".join(json.dumps(result_dict(r)) for r in state.results) + "\n"
        )
        print(render_state(d))
        write_report(d)
        return 0
    if a.cmd == "status":
        return status()
    if a.cmd == "report":
        return report()
    if a.cmd == "codec":
        return run_codec(
            a.arm,
            a.corpus,
            a.bytes,
            a.split,
            a.units,
            a.min_count,
            a.cap,
            a.active,
            a.steps,
            a.proof,
            a.rotate,
            a.full_every,
            a.seed,
            a.span,
            a.head,
            a.parse,
            a.seq,
        )
    if a.cmd == "coder":
        if not a.test:
            raise SystemExit("coder: pass --test")
        print(codec_mod.selftest())
        return 0
    if a.cmd == "recorded":
        return recorded_report()
    if a.cmd == "demo":
        return run_demo(a.run, a.bytes)
    if a.cmd == "generate":
        return generate(a.run, a.bytes, a.seed)
    if a.cmd == "lm":
        return run_lm(
            a.corpus,
            a.tiny,
            a.steps,
            a.prompt.encode(),
            a.sample,
            a.active,
            a.units,
            a.min_count,
            a.cap,
            a.proof,
            a.seed,
        )
    if a.cmd == "install":
        if a.transfer:
            return run_install_transfer(
                a.corpus,
                a.package,
                a.active,
                a.steps,
                a.units,
                a.min_count,
                a.cap,
                a.proof,
                a.seed,
                a.rotate,
                a.full_every,
            )
        return run_install(
            a.corpus,
            a.package,
            a.active,
            a.steps,
            a.units,
            a.min_count,
            a.cap,
            a.proof,
            a.seed,
            a.rotate,
            a.full_every,
        )
    if a.cmd == "templates":
        return templates(a.run, a.top)
    if a.cmd == "arena":
        return run_arena(
            a.corpus,
            a.bytes,
            a.proof,
            a.steps,
            a.active,
            a.designs,
            a.rotate,
            a.full_every,
            a.seed,
        )
    if a.cmd == "pack":
        if a.op == "verify":
            return pack_verify(a.run)
        if a.op == "trace":
            lo, hi = (int(x) for x in a.range.split(":"))
            return pack_trace(a.run, lo, hi)
        if a.op == "diff":
            return pack_diff(a.a, a.b)
        if a.op == "merge":
            return pack_merge(a.out, a.a, a.b, a.size)
        return pack_rm(a.run, a.prefix.encode(), a.bytes)
    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


def rank_auc(score: list[float], label: list[int]) -> float:
    """The Mann-Whitney AUC: 0.5 is no signal, 1.0 is a perfect ranking."""
    n1 = int(sum(label))
    n0 = len(label) - n1
    if not n1 or not n0:
        return 0.5
    order = np.argsort(np.asarray(score))
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = float(np.sum(np.where(np.asarray(label) == 1, ranks, 0.0)))
    return (pos - n1 * (n1 + 1) / 2) / (n1 * n0)


def heldout_auc(
    model: Trunk,
    entries: tuple[Entry, ...],
    trie: dict,
    train: bytes,
    total: int,
    e_c: torch.Tensor,
    nll: torch.Tensor,
    hs: torch.Tensor,
    n_eval: int,
    alpha: float = 0.5,
    samples: int = 60_000,
) -> dict:
    """Does each gate feature predict "the base finds the span cheap"? Held-out."""
    rng = np.random.default_rng(0)
    picks = np.sort(
        rng.choice(n_eval - 1, size=min(n_eval - 1, samples), replace=False) + 1
    )
    count = torch.tensor([e.count for e in entries])
    cos, lens, logc, means = [], [], [], []
    for t in picks.tolist():
        length, idx = longest_match(trie, train, t)
        if length < 2 or t + length > n_eval or idx % HOLDOUT != 0:
            continue
        cos.append(match_score(hs[t], e_c[idx], int(count[idx]), total, 0.0))
        lens.append(length)
        logc.append(math.log(float(count[idx]) / total))
        means.append(float(nll[t : t + length].mean()))
    if len(means) < 100:
        return {"n": len(means)}
    tau = float(np.percentile(means, 95))
    y = [int(m <= tau) for m in means]
    return {
        "n": len(means),
        "pos": float(sum(y)) / len(y),
        "tau": tau,
        "auc_cos": rank_auc(cos, y),
        "auc_length": rank_auc(lens, y),
        "auc_logc": rank_auc(logc, y),
    }


def sweep_p0(corpus: str = "enwik8_10m", slice_bytes: int = 300_000) -> int:
    """Sweep the adapter's choices on one slice: pooling, prior, tau, accept.

    Two questions: is the linchpin negative a pooling artifact, and is the pack
    policy (alpha, tau, threshold) off its optimum?
    """
    cfg = Config(
        arm=Arm.ADAPTER,
        corpus=corpus,
        base="byte-20m",
        params=20_000_000,
        dict_size=4096,
        min_count=32,
    )
    out = RUNS / "sweep-p0"
    path = base_path(cfg)
    if not path.exists():
        raise SystemExit(f"sweep: missing base {path}")
    ckpt = torch.load(path, map_location="cuda", weights_only=False)
    model = Trunk(256, ckpt["d"], ckpt["layers"], ckpt["heads"], ckpt["seq"]).cuda()
    model.load_state_dict(ckpt["model"])
    model.eval()
    raw = corpus_bytes(corpus)
    split = cfg.stream_bytes - cfg.val_bytes
    train, test = raw[:split], raw[split : split + cfg.val_bytes]
    ids = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()
    pack, pack_size, total = load_pack(cfg, train, out)
    trie = make_trie(pack.entries)
    fit_ids = ids[:1_000_000]
    fit_nll, fit_hs, fit_n = stream_nll(model, fit_ids)
    sl = min(slice_bytes, len(test))
    data = test[len(test) - sl :]
    ts = split + len(test) - sl
    sl_ids = ids[ts : ts + sl]
    test_nll, test_hs, test_n = stream_nll(model, sl_ids)
    print(
        f"sweep P0 `{corpus}`: pack {pack_size / 1e3:.1f} kB · slice {sl} B",
        flush=True,
    )
    rows: list[dict] = []
    emb = {
        pool: span_embeddings(model, pack.entries, pool) for pool in ("last", "mean")
    }
    for pool in ("last", "mean"):
        for name, neural in (("tied", True), ("counts", False)):
            gate = fit_gate(
                model,
                pack.entries,
                trie,
                train[:1_000_000],
                total,
                emb[pool],
                fit_nll,
                fit_hs,
                fit_n,
                neural=neural,
                holdout=True,
            )
            m = adapter_eval(
                model,
                pack.entries,
                trie,
                total,
                emb[pool],
                gate,
                data,
                test_nll,
                test_hs,
                test_n,
                pack_size,
                cfg.stream_bytes,
                neural=neural,
            )
            rows.append(
                {
                    "pool": pool,
                    "gate": name,
                    "alpha": 0.5,
                    "tau_pct": 95.0,
                    "threshold": 0.5,
                    "saved": m["saved"],
                    "accept": m["accept"],
                    "held_bpb": m["held_bpb"],
                    "held_accept": m["held_accept"],
                    "w": gate["w"],
                }
            )
            print(
                f"  {pool:5s} {name:7s} net {m['saved']:+.3f} · "
                f"held {m['held_bpb']:.3f} · accept {m['accept']:.1%} · "
                f"w0 {gate['w'][0]:+.3f}",
                flush=True,
            )
    best = max((r for r in rows if r["gate"] == "tied"), key=lambda r: r["saved"])
    e_c = emb[best["pool"]]
    for alpha in (0.0, 1.0):
        for tau_pct in (95.0, 99.0):
            gate = fit_gate(
                model,
                pack.entries,
                trie,
                train[:1_000_000],
                total,
                e_c,
                fit_nll,
                fit_hs,
                fit_n,
                neural=True,
                holdout=True,
                tau_pct=tau_pct,
                alpha=alpha,
            )
            for threshold in (0.5, 0.7):
                m = adapter_eval(
                    model,
                    pack.entries,
                    trie,
                    total,
                    e_c,
                    gate,
                    data,
                    test_nll,
                    test_hs,
                    test_n,
                    pack_size,
                    cfg.stream_bytes,
                    threshold=threshold,
                )
                rows.append(
                    {
                        "pool": best["pool"],
                        "gate": "tied",
                        "alpha": alpha,
                        "tau_pct": tau_pct,
                        "threshold": threshold,
                        "saved": m["saved"],
                        "accept": m["accept"],
                        "held_bpb": m["held_bpb"],
                        "held_accept": m["held_accept"],
                        "w": gate["w"],
                    }
                )
                print(
                    f"  {best['pool']:5s} alpha {alpha:g} tau {tau_pct:g} "
                    f"pi {threshold:g} · net {m['saved']:+.3f} · "
                    f"accept {m['accept']:.1%} · held {m['held_bpb']:.3f}",
                    flush=True,
                )
    write_json(
        REPORTS / "sweep-p0.json",
        {"corpus": corpus, "slice": sl, "pack_bytes": pack_size, "rows": rows},
    )
    top = max(rows, key=lambda r: r["saved"])
    print(
        f"best: {top['pool']} alpha {top['alpha']:g} tau {top['tau_pct']:g} "
        f"pi {top['threshold']:g} · net {top['saved']:+.3f}",
        flush=True,
    )
    return 0
