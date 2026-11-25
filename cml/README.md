# cml

The concept memory LM: the v11 codebase for the plan in `PLAN.md`.

## What this covers

The v11 design is four bets from seven sketches (`../hncm.r1-7.md`). `cml`
turns that design into one typed pipeline, one CLI, one run store, and one
gate table. The plan holds the design; this page maps the ideas to files.

| original idea | bet | experiment | where in `cml` | status |
|---|---|---|---|---|
| r1 adapter; equal-charged-bytes; install | B1 | P0 | `Arm.ADAPTER`, `Entry`, `Pack`, `GATES` P0a-g | manifest + gates; v0 in P0 |
| r2 C2LM; hybrid output; copy bypass; charge in loss | B3 | P3 | `Arm.LEDGER`, `GATES` P3a-d | manifest; P3 |
| r3 PhraseLM; lexicon-agnostic; three heads; safe mode | B1/B2 | P0, P1 | `Arm.ADAPTER`/`CONCEPT`, `GATES` | manifests; P0/P1 |
| r4 Concept-LM; neural prices; joint EM; model-class test | B2 | P1, P2 | `Arm.CONCEPT`, `GATES` P1a-e/P2a-d | manifests; P1/P2 |
| r5 LedgerLM; persistence; market; alignment; dual pathway; learning curve; editing | B3 | P3, P4 | `Arm.LEDGER`, `GATES` P3a-d/P4a-f | manifests; P3/P4 |
| r6 CMT; tied embeddings; pack; provenance; constrained; telemetry; two clocks | B3 | P0, P1, P2, P4 | `Pack`, `GATES` P0/P1/P2/P4 | manifests; P0/P1/P2/P4 |
| r7 CoLib; holes; librarian; composition; planning; auditable thought | B4 | P5 | `Arm.LIBRARY`, `GATES` L-G0…L-G5, P5e-g | manifest; P5 |

The four bets, and where they live:

| bet | experiments | plan | status |
|---|---|---|---|
| B1 the adapter | P0 | §4 B1 | manifest + gates; v0 next |
| B2 the vocabulary | P1, P2 | §4 B2 | manifests |
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
- `experiments/p0..p5.toml`: one manifest per experiment, no per-experiment
  Python.
- `tests/test_invariants.py`: gate ops, manifest, space, search, render.
- `pyproject.toml` + `uv.lock`; `ruff` clean.

Not built yet: the lexicon, parse, encoder, trunk, trainer, judge, pack, trace,
and the library. P0 builds the adapter; P1 the class; P2 the 100 MB verdict;
P3 the gate; P4 the ops; P5 the library. `cml run` writes the run config and
raises the TODO until P0 lands. The search `--simulate` is a scheduler test,
not evidence.

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

## Run

```bash
uv sync
uv run cml selftest
uv run cml space
uv run cml search P1 --simulate --budget 20
uv run cml status
uv run cml report
uv run cml run experiments/p0-adapter.toml --dry
uv run cml run experiments/p0-adapter.toml   # after P0 is built
```

`run-id = <manifest>-<sha8(config)>-s<seed>`; runs write to
`runs/<id>/{config.json,metrics.json,artifacts/}`. The gate table is `GATES`
in `cml.py`; the manifests reference names. The numbers in `PLAN.md` §1 are
the initial `GATES`; `cml.py` is the source after P0. The P4/P5 bars are
placeholders until their baselines are measured.

The old scripts are frozen references; `cml` does not import them. Data
artifacts are read from `../runs/v9/`.
