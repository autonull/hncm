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
import random
import tomllib
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

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
    base: str = ""
    corpus: str = "enwik8_10m"
    lexicon: str = ""


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
    "P0a": Gate("saved", Convention.SHARED, ">=", 0.15),
    "P0b": Gate("accept", Convention.RATE, ">=", 0.50),
    "P0c": Gate("delta", Convention.RATE, "<=", 0.02),
    "P0d": Gate("structured_x", Convention.RATE, ">=", 1.5),
    "P0e": Gate("text_delta", Convention.RATE, "<=", 0.01),
    "P0f": Gate("pack_mb", Convention.RATE, "<=", 1.0),
    "P0g": Gate("provenance", Convention.RATE, ">=", 1.0),
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
    "P0": ("P0a", "P0b", "P0c", "P0d", "P0e", "P0f", "P0g"),
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
    path: Path, seed: int | None, params: int | None = None
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
        steps=optim.get("steps", 2000),
        batch=optim.get("batch", 32),
        lr=optim.get("lr", 3e-4),
        mix=optim.get("mix", 0.5),
        base=model.get("base", ""),
        corpus=data.get("corpus", "enwik8_10m"),
        lexicon=data.get("lexicon", ""),
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


def selftest() -> int:
    cfg = Config()
    gates = ["P1a", "P1b"]
    validate(gates)
    rid = run_id(Path("selftest.toml"), cfg)
    out = RUNS / rid
    write_json(out / "config.json", {"config": asdict(cfg), "gates": gates})
    assert (out / "config.json").exists()
    assert GATES["P1a"].check(2.5) and not GATES["P1a"].check(2.7)
    print(f"selftest ok: {rid}")
    return 0


def run(manifest: Path, seed: int | None, params: int | None, dry: bool) -> int:
    if not manifest.exists():
        raise SystemExit(f"missing manifest: {manifest}")
    cfg, gates = config_from_manifest(manifest, seed, params)
    validate(gates)
    rid = run_id(manifest, cfg)
    if dry:
        print(json.dumps({"id": rid, "config": asdict(cfg), "gates": gates}, indent=1))
        return 0
    out = RUNS / rid
    write_json(out / "config.json", {"id": rid, "config": asdict(cfg), "gates": gates})
    raise SystemExit(f"{manifest.stem}: pipeline not built (# TODO: build it)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cml")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("selftest")
    r = sub.add_parser("run")
    r.add_argument("manifest", type=Path)
    r.add_argument("--seed", type=int)
    r.add_argument("--params", type=int)
    r.add_argument("--dry", action="store_true")
    sub.add_parser("space")
    s = sub.add_parser("search")
    s.add_argument("stage", choices=("P0", "P1", "P2", "P3", "P4", "P5"))
    s.add_argument("--budget", type=float, default=12.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--simulate", action="store_true")
    sub.add_parser("status")
    sub.add_parser("report")
    a = p.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "run":
        return run(a.manifest, a.seed, a.params, a.dry)
    if a.cmd == "space":
        return space()
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
    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
