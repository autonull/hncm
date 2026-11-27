# cml

The concept memory LM: the v11 codebase for the plan in `PLAN.md`.
The active plan is `../TODO8.md` (multi-byte concepts + RLFP); `../TODO7.md`
is the codec record and the auditable-codec product.

## The 50% target

The target is concrete: **50% fewer bits per byte than an equal-compute
byte LM on the pinned 756,752 B code test.** Against the current
`1.693` byte reference, that is `0.847` shared bpb. It is not a cosmetic
gain: it is `9.4x` compression on code, or `1.8x` the current codec
ratio. That would be remarkable, and it is the right ambition.

| milestone | shared bpb | vs 1.693 byte | compression | state |
|---|---:|---:|---:|---|
| 20M, 48 MB, pack 256 | 1.518 | -10.3% | 5.3x | **measured** |
| 20% | 1.354 | -20% | 5.9x | next |
| 30% | 1.185 | -30% | 6.8x | path |
| 40% | 1.016 | -40% | 7.9x | path |
| **50%** | **0.847** | **-50%** | **9.4x** | north star |

The denominator is the current 4.86M byte model (`1.693 / 1.694`). A real
50% claim needs a 20M byte control at the same train/test/steps; that
control is the first experiment, not an afterthought.

Yes, 50% would be remarkable. At `0.847` shared bpb the codec would be
`9.4x` compression, `1.8x` the current ratio, and far enough ahead of a
byte LM that it stops being a rounding error. That is the result worth
building toward. The current `1.518` is the first 10%; it is evidence
the direction is real, not proof the destination is reached.

## Why 50% is plausible

The byte LM pays for every repeated byte. The codec does not:

- The 256-unit pack covers 45% of test bytes. A larger pack or a copy
  arm can cover more, and the pack charge is only 0.03 bpb at 756 KB.
- Code is highly repetitive. Exact span copy at near-zero bits is the
  lever a byte LM does not have.
- The 20M model is still first-grid: 48 MB train, 256 units, 8k steps.
  None of the three codec-specific levers is exhausted.

The path is not "more of the same scale." The next gains come from the
pack, copy, and literals, not another seed.

| path | lever | expected | risk | next test |
|---|---|---|---|---|
| **A pack** | 256 -> 1024/2048 at 20M | -0.15 to -0.30 bpb | pack charge | active sweep at 48 MB |
| **B copy** | copy prior exact spans | -0.10 to -0.25 bpb | leakage | T2 copy arm |
| **C literals** | better byte head | -0.05 to -0.15 bpb | small | literal ablation |
| **D capacity** | 40M/100M | -0.05 to -0.15 bpb | baseline scales | equal-compute 40M |
| **E RLFP** | preference optimization | product win, not bpb | preference data | 2k-pair pilot |

Those are hypotheses, not measured results. They are cheap enough to
test, and each one attacks a different part of the 50% gap.

## The next three experiments

1. **Equal-compute 20M byte control.** Same 48 MB train, same test, same
   steps. Without it, the 20M codec number is a capacity result.
2. **20M pack sweep at 48 MB.** `active` 256, 1024, 2048; add `active`
   to the run id first. The pack charge grows, but so does coverage.
3. **T2 copy arm.** Copy a prior exact span; never search for the target
   span. This is the codec's unique lever and the cheapest route to a
   large gain.

If the pack/copy arms clear 20-30%, the 50% target is a real program.
If they do not, the codec still has a measured compression result, a
strong T1 span capability, and a clear next lever in capacity or RLFP.
That is a foundation, not a verdict.

## Build before the next run

Three small changes make the next experiments honest:

- `cml scale`: include `active` in the run id and metrics. Today a pack
  sweep would overwrite the 256-unit run.
- `cml codec --arm byte-model`: accept `--params/--d/--layers`. The
  current byte control is fixed at 4.86M; a 20M control is required.
- `cml scale`: add a `--copy` arm or a T2 copy fallback. The copy lever
  is the codec's unique advantage and is not yet in T2.

These are small, testable changes. They are the difference between
another seed and a real attempt at 50%.

## What works

These are measured wins and opportunities, not a final verdict. Do not
regress them silently.

| win | measurement | evidence |
|---|---|---|
| **Concept-only capability** | **0.650** exact held-out spans, **1.720** bits/span, **1.00** decisions/span | `reports/SCREEN.md` |
| Concept vs trunk | 0.650 vs **0.480** exact; 1.72 vs 2.59 bits/span | `reports/SCREEN.md` |
| Concept vs byte | **3.1x** exact; **3.0x** fewer decisions; **7.4x** fewer bits/span | `reports/SCREEN.md` |
| Compression | **1.600** bpb teacher, **1.692** payload, **1.722** shared | `reports/CONCEPT.md` |
| T2 review | promoted codec: **1.692** payload / **1.722** shared / **1.600** teacher; blend excluded | `reports/PORTFOLIO.md` |
| T2 scale | 24 MB: 10M **1.538** / 1.568 (seeds 0-4 spread 0.024); 48 MB: 20M **1.488** / 1.518 | `reports/SCALE.md` |
| Equal-params pointer vs byte | 5.21M pointer `1.583 / 1.613` vs 4.86M byte `1.693 / 1.694`; **6.5% payload**, 4.8% shared | `reports/SCALE.md`, `reports/CONCEPT.md` |
| Equal-data scale | 20M vs 10M at 48 MB: **-0.081 payload**; capacity, not data alone | `reports/SCALE.md` |
| Exact round-trip | 50,000 units / 81,531 B | `reports/CONCEPT.md` |
| Local deletion | 35 targets removed by construction; 165 survivors ±0.006; **+0.146 bpb** full encode | `reports/EDIT-CURVE.md` |
| Merge | 256 -> 320 units; **+0.108 bpb**; exact round-trip | `reports/SCREEN.md` |
| Concept draft | **0.717** one-step acceptance; **1.560** accepted concepts per trunk call at K=8 | `reports/EDIT-CURVE.md`, `reports/BREADTH.md` |
| Long-key copy | key 6: **0.650** exact, 0.475 hit, 6.193 bits/span; matches concept exact but too expensive to replace it | `reports/BREADTH.md` |
| Edit locality | `self` **-0.000** bpb, `return` +0.026, indent +0.028, space +0.146, newline +0.228; all round-trip | `reports/BREADTH.md` |

Commands that produce them: `cml train --units codec`, `cml screen`,
`cml editcurve`, `cml breadth`, `cml packs`, `cml portfolio`, `cml scale`.
Scope: 5.21M-20.02M pointer; T1 uses 70k held-out bytes and 200 spans;
T2 uses 24-48 MB train and the pinned 756,752 B test. The 50% target
needs an equal-compute 20M byte control plus the pack/copy arms; those
are the next experiments, not blockers.

## What this covers

The v11 design is four bets from seven sketches (`../hncm.r1-7.md`). `cml`
turns that design into one typed pipeline, one CLI, one run store, and one
gate table. The plan holds the design; this page maps the ideas to files.

| original idea | bet | experiment | where in `cml` | status |
|---|---|---|---|---|
| r1 adapter; equal-charged-bytes; install | B1 | P0 | `Arm.ADAPTER`, `Entry`, `Pack`, `GATES` P0a-h | oracle +0.758; charged −2.224 → P0a fails; the name is the cost |
| r2 C2LM; hybrid output; copy bypass; charge in loss | B3 | P3 | `Arm.LEDGER`, `GATES` P3a-d | manifest; P3 |
| r3 PhraseLM; lexicon-agnostic; three heads; safe mode | B1/B2 | P0, P1 | `Arm.ADAPTER`/`CONCEPT`, `GATES` | manifests; P0/P1 |
| r4 Concept-LM; neural prices; joint EM; model-class test | B2 | P1, P2 | `Arm.CONCEPT`, `GATES` P1a-e/P2a-d | P1 measured; T2 codec 1.488 / 1.518; P2 text-class next |
| r5 LedgerLM; persistence; market; alignment; dual pathway; learning curve; editing | B3 | P3, P4 | `Arm.LEDGER`, `GATES` P3a-d/P4a-f | manifests; P3/P4 |
| r6 CMT; tied embeddings; pack; provenance; constrained; telemetry; two clocks | B3 | P0, P1, P2, P4 | `Pack`, `GATES` P0/P1/P2/P4 | P0/P1 measured; T2 codec 1.488; P2/P4 next |
| r7 CoLib; holes; librarian; composition; planning; auditable thought | B4 | P5 | `Arm.LIBRARY`, `GATES` L-G0…L-G5, P5e-g | manifest; P5 |

The four bets, and where they live:

| bet | experiments | plan | status |
|---|---|---|---|
| B1 the adapter | P0 | §4 B1 | P0 measured; the span-name charge kills the flat copy codec |
| B2 the vocabulary | P1, P2 | §4 B2 | P1 measured (1.920 bpb); T2 codec 1.488 / 1.518; P2 text-class next |
| B3 the ledger | P3, P4 | §4 B3 | manifests |
| B4 the library | P5 | §4 B4 | manifest |

### r4 coverage

| r4 idea | where in the plan | status |
|---|---|---|
| neural prices replace counts; the model-class inversion | §4 B2; P1 | plan; P1 |
| the charge as anti-collapse regularizer; code earns bytes | §1; §4 B2 | plan |
| separable, swappable, auditable lexicon; LM + codec | §4 B2/B3; `Pack` | plan; P0/P4 |
| concept + p(escape)/resume; T_b 10-50M | §4 B2 | plan; P1 |
| e(c)=enc(bytes)+r_c; shared encoder; residual for frequent | §4 B1/B3 | plan; P0 |
| trie/hierarchical/MIPS head, 16k-1M | §4 B2 | plan; P1 |
| Viterbi under neural prices; no parse at generation | §4 B2 | plan; P1 |
| EM: sample parse; gradients; add/prune | §4 B2 | plan; P3 |
| grounded mode; hallucinated atoms impossible | §4 B3 | plan; P4 |
| verify by reconstruction | §4 B3 | plan; P4 |
| mid-span abort; entropy cap | §4 B1 | plan; P0 |
| zero-shot 0.1 bpb; residual-only | §6 P1e | plan; P1 |
| speculative 7B verifier ≥1.3x REST | §6 P4f | plan; P4 |
| target domains; not chat SOTA | §10 | plan |
| lineage: SentencePiece/Brent-Goldsmith/LCT | §4 B2; §2 | plan; README |

### r5 coverage

| r5 idea | where in the plan | status |
|---|---|---|
| persistent versioned ledger; 64-256 MB | §4 B3; §3 | plan; P4 |
| market utility/rent; discrete admit/evict | §4 B3 | plan; P4 |
| n10 scan candidates; cached | §4 B3 | plan; P4 |
| dual pathway; concept+byte; abstain gate; atomic | §4 B3 | plan; P3 |
| L_align; L_ground; L_multi | §4 B2/B3 | plan; P3 |
| BLT pays per occurrence; ledger once; −20% latency | §4 B3 | plan; P4 |
| FLOPs/byte learning curve; 3-6x; KV ~4x | §4 B3 | plan; P3 |
| adaptation without weights; frozen trunk | §4 B3 | plan; P4 |
| editing/unlearning/privacy/citations | §4 B3 | plan; P4 |
| standard prompt; open backbone; grep ledger | §4 B1/B3 | plan; P0 |
| S0/S1/S2/S3; equal FLOPs vs byte/BPE/BLT; four ablations | §6 P3/P4 | plan; P3/P4 |
| rent λ regurgitation knob; verbatim-copy | §4 B3; P4 | plan; P4 |
| dynamic-vocabulary neighbor; PPL 8.03 caution | §2; §4 B3 | plan; README |
| auditor framing; lexical not semantic | §10 | plan |
| open 1B distill; editing/unlearning/citations | §6 P4 | plan; P4 |

### r6 coverage

| r6 idea | where in the plan | status |
|---|---|---|
| counts as a prior; neural score over tied embeddings; backbone frozen; pack | §4 B1/B3 | plan; P0/P4 |
| charge as a training term when D grows; exact spans unchanged | §4 B2/B3 | plan; P1/P3 |
| DP at fit; lazy gate at decode | §4 B1/B2 | plan; P0/P1 |
| linchpin `e_c = Enc(bytes_c)`; precompute once; 1-3M head; gate | §4 B1 | plan; P0 |
| score + gate + mixture; 20-50M backbone; 4-64k D | §4 B1/pilot | plan; P0/P1 |
| teacher-force MDL-DP; NLL + β·pack; optional LoRA | §4 B2 | plan; P1 |
| v1 parallel pass; v2 input tokens; two clocks | §4 B2; P3d | plan; P1/P3 |
| quality floor (NEST); compact drafter | §4 B1 | plan; P0 |
| provenance; pack ops; hard-exact; telemetry | §4 B3 | plan; P4 |
| speed on synseg/JSON/log | §4 B3; P3 | plan; P3 |
| novelty map (CoG/NEST/kNN/memory layers/BLT/REST) | §2 neighbors | plan; README |
| v0/v1/v2; REST/NEST baseline; gate miscalibration; bottom line | §6 P0/P1; §8; §10 | plan; P0 |

### r7 coverage

| r7 idea | where in the plan | status |
|---|---|---|
| hierarchy L0/L1; middle layer = library | §4 B4 | plan; P5 |
| L0 parse = neurally-priced DP; decoder-synchronous | §4 B4 | plan; P5 |
| event = (concept_id, fills); recursive/fallback; cross-attend | §4 B4 | plan; P5 |
| concept examples (code/JSON/genitive/header/API) | §4 B4 | plan; P5 |
| e(c) = template + hole signature; zero-shot; memcpy; hallucination selection | §4 B4 | plan; P5 |
| factorization; 1.327 order-2 | §4 B4; §1 | plan; P5 |
| librarian: scan/anti-unify/held-out DL; DreamCoder; untried | §4 B4 | plan; P5 |
| hard-EM; objective; anti-collapse; held-out gating; inference | §4 B4 | plan; P5 |
| capabilities: compositionality/planning/auditable/knowledge/context | §4 B4 | plan; P5 |
| performance 3-10x FLOPs/byte; named/persistent/stationary; segmentation rigidity | §4 B4 | plan; P5 |
| novelty table (LCM/BLT/H-Net/DreamCoder/COCONUT) | §2 neighbors | plan; README |
| L-G0..L-G5; P5e-g; E0/E1/E2; transfer matrix; API demo; 100M-1B scaling | §4 B4; §6 P5 | plan; P5 |
| verdict: a bet; components separate; G2/G4 decide | §4 B4; §10 | plan |

## What is built now

- `cml.py`: `Config`, `Entry`, `Pack`, `Run`, `Arm`, `Convention`, `Gate`,
  `GATES`, manifest parsing, run IDs, atomic writes, `selftest`, CLI.
- `Space`, `ArmSpec`, `STAGE_AXES`: the valid experiment space per stage.
- `SearchState`, `run_search`: successive halving + UCB + recombination.
- `render_state`, `write_report`: the dashboard and `reports/PROGRESS.md`.
- `Lexicon`, `read_lexicon`, `read_parse`: the v9 pack reader (the strings and
  the cached 10 MB parse); no old imports.
- `Trunk`, `Block`, `lm_loss`, `eval_bits`, `run_p1`: the causal unit trunk,
  the 9 MB train / 1 MB test split, the CE-to-bpb judge, the checkpoint.
- `fit_curve`, `curve`, `gates_report`: the scaling curve, `P1c`, and
  `reports/GATES.md`.
- `run_base`, `cml base`: the frozen byte base (21.9M, 6000 steps, 1.827 bpb).
- `substring_counts`, `induce_pack`, `make_trie`, `longest_match`: `D` and the
  exact-span pack (4096 spans, 90.1 kB, provenance offsets).
- `span_embeddings`, `stream_nll`, `fit_gate`, `adapter_eval`, `run_p0`: the
  frozen encoder state, the 3-parameter gate, verification, and the copy codec.
- `cml train`, `cml sample`, `cml stats`: raw-stream training, prompt
  continuation with temperature/top-k, and the run-store dashboard.
- `cml train --units codec --model pointer|trunk`: the 5.21M concept LM on the
  256-unit pack, its 4.93M fixed-vocab control, decision accounting, and
  `reports/CONCEPT.md` (1.600 / 1.516 bpb, exact round-trip).
- `cml screen`: capability screen on 200 held-out code spans; byte, copy,
  order-1 counts, order-2 library, concept pointer, fixed-vocab trunk, plus
  gradient-free pack removal and merge; `reports/SCREEN.md`.
- `cml editcurve`: causal pack-edit curve, concept-draft acceptance, a
  4-concept speculative block, and the concept+blend T3/T4 arms;
  `reports/EDIT-CURVE.md`. Concept-only exact is 0.650 at 1.72 bits/span.
  The blend edit is local (−0.012 on survivors). The blend draft is
  calibration-limited, not trunk-limited: τ1 K8 is 1.480; τ4 K16 is
  4.650 (pointer 4.450).
- `cml breadth`: small hyperparameter grids for copy, template/hole, library,
  edit prefixes, and speculative draft length; `reports/BREADTH.md`. Copy
  keys 1 -> 12; the concept+copy blend is 0.705 at 2.362 bits (target) and
  0.680 at 1.698 (dominates the concept point). Template radius 2 is 0.340.
- `cml packs`: one pointer per active pack size (128/256/512/1024),
  fixed byte spans, end-to-end T1, parse decisions, coverage, and pack
  bytes; `reports/PACKS.md`. 256 wins all axes (0.415 / 3.133 / 1.00);
  128, 512, and 1024 are dominated.
- `cml portfolio --round N --families ... --task ... --budget ...`:
  aggregates every measured arm into a T1 Pareto frontier, family
  best/median/worst, task views, a T2 compression review, combinations,
  and a resumable 50/30/20 (round 1) or 70/20/10 (later) allocation;
  `reports/PORTFOLIO.md`. It parks families without killing them.
- `cml scale --sizes 5m,10m`: keeps the pinned T2 test and the 256-unit
  pack, adds unique train code up to 24 MB, and trains the requested
  pointers; `reports/SCALE.md`. 5M/10M are targets (5.21M/9.17M actual).
  A nonzero `--seed` gets its own run id; seed 1 repeats the 5M point at
  +0.019 payload. `--scale-cap 48000000` extends unique code to 48 MB.
  Pointer training checkpoints every 500 steps and resumes after a crash.
- `overnight.py`: one resumable queue task per systemd timer invocation;
  a stale `running` task is marked interrupted and retried from the
  checkpoint. It writes `reports/OVERNIGHT.md` and per-task logs.
- `experiments/p0..p5.toml`: one manifest per experiment, no per-experiment
  Python.
- `tests/test_invariants.py`: gate ops, manifest, space, search, render,
  windows/split, the trunk, the curve, the parse round-trip, the pack
  round-trip, and the stream judge.
- `pyproject.toml` + `uv.lock`; `ruff` clean.

P1 is measured: 1.920 bpb at 20.5M params/6000 steps on the held-out 1 MB
slice (1.912 ± 0.008 over seeds 0-2); 1.962 at 8.9M, 1.926 at 39.9M; the 60M
fit is 1.919. The same-slice count bigram is 2.667, so the neural margin is
0.75 bpb. P1a/P1b/P1c pass0.75 bpb. P1a/P1b/P1c pass. The second split (8-9 MB) gives 1.992, so the win
is not one slice.

## Fully trained text LM

`cml train` tokenizes a raw stream with the v9 lexicon, trains the unit
`Trunk`, evaluates a held-out slice, and samples. `cml sample` prompts any
saved run.

- 20,461,440 params (d384, 8 layers), 90 MB train / 10 MB held-out, 30k
  steps, 759 s.
- test payload **1.517 bpb** (unigram 3.131, bigram 2.508). The 10 MB P1
  model (1.920) overfits 1.516/1.920; the 90 MB run is the usable one.
- `uv run cml sample --run text-enwik8_100m-d384l8-s30000`
- `reports/LM-100M.md`, `reports/SAMPLE.md`, `reports/STATS.md`.

P0 core is measured: base 1.827 → adapter 0.997 bpb (net +0.758 after the
0.072 pack charge), acceptance 79.4%, coverage 80%, provenance 100%.
P0a/b/c/e/f/g pass; P0d, kNN/LoRA/no-D at equal charge, P1d and P1e are next.

Not built yet: P0d structured latency, the equal-charge kNN/LoRA/no-D
baselines; the 100 MB verdict (P2); the ledger/gate (P3); the ops (P4); the
library (P5). `cml run` dispatches P0/P1 and raises the TODO for the later
stages. The search `--simulate` is a scheduler test, not evidence.

## The experiment system

- `cml space` prints the valid arms per stage and the axes.
- `cml search P1 --simulate --budget 20` runs the scheduler on a synthetic
  reward; it is a scheduler test, not evidence.
- `cml status` prints the live dashboard; `cml report` writes
  `reports/PROGRESS.md`.
- The scheduler is successive halving + UCB + greedy recombination. A hard
  gate fails a branch; the budget is a soft cap.
- The dashboard shows the frontier, gate ladder, leaderboard, budget, tree,
  axes, next and sparkline. The report adds the decision log and the per-axis
  frontier counts.

## Why this shape

The old pattern was one script per idea (`../hncm5.py`, `../codec.py`,
`../concept_lm.py`, `../n10.py`): duplicated loaders, duplicated judges, prose
gates, scattered JSONs. The new pattern is one typed pipeline (`cml.py`), one
manifest per experiment, one run store (`runs/<id>/`), one gate table
(`GATES`). Split `library.py` only at P5; split `model.py` only when a second
model arm appears. No pre-split, no back-compat, no old imports.

## The rules this README does not repeat

- The design rules are `PLAN.md` §1.
- The four bets are `PLAN.md` §2.
- The charge arithmetic is `PLAN.md` §3.
- The architecture is `PLAN.md` §4.
- The codebase pattern is `PLAN.md` §5.
- The gates, risks and time are `PLAN.md` §6/§8/§9.

## The case for continuing

- A working exact codec with round-trip and provenance is already rare.
- The 5.21M pointer beats the 4.86M byte model on both payload and
  shared bpb: `1.583 / 1.613` vs `1.693 / 1.694`.
- The 20M point is the first 10%; the pack and copy levers are still
  untested, and both attack the part of the gap that scale cannot.
- A 20-30% result would already be a strong outcome. 50% is the
  category change, and it is specific enough to test.

The project is not out of ideas. It is at the point where the next
experiment should be chosen for leverage, not for another seed.

## Run

```bash
uv sync
uv run cml selftest
uv run cml space
uv run cml search P1 --simulate --budget 20
uv run cml status
uv run cml report
uv run cml run experiments/p1-class-10m.toml --params 20_000_000 --steps 6000
uv run cml run experiments/p1-class-10m.toml --params 40_000_000 --steps 6000
uv run cml run experiments/p1-check-8m.toml
uv run cml curve P1 --steps 6000 --update
uv run cml gates
uv run cml base --params 20_000_000 --steps 6000 --seq 256
uv run cml run experiments/p0-adapter.toml
uv run cml lm --corpus shakespeare --tiny --steps 2000 --prompt "ROMEO:" --sample 400
uv run cml install --corpus code --package networkx --active 256 --steps 3000
uv run cml train --corpus enwik8_100m --params 20000000 --steps 30000
uv run cml sample --run text-enwik8_100m-d384l8-s30000
uv run cml stats
```

`run-id = <manifest>-<sha8(config)>-s<seed>`; runs write to
`runs/<id>/{config.json,metrics.json,artifacts/}`. The gate table is `GATES`
in `cml.py`; the manifests reference names. The numbers in `PLAN.md` §1 are
the initial `GATES`; `cml.py` is the source after P0. The P4/P5 bars are
placeholders until their baselines are measured.

The old scripts are frozen references; `cml` does not import them. Data
artifacts are read from `../runs/v9/`.
