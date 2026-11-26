# TODO8 — a portfolio system for concept-native LMs

The codec's multi-byte units are not only a compression alphabet. They are a
candidate action space for an exact, editable code LM. This file is a **portfolio
plan**, not a single-bet plan: every relevant family gets a fair grid, a shared
task, and a budget. The system keeps a Pareto frontier and invests more only
where a family earns it. A family is never killed by one configuration.

`TODO7.md` remains the codec record and the auditable-codec product. `cml/PLAN.md`
holds the older design; this file is the active experiment system.

## 1. The system

The unit of work is an **arm**:

```
arm = (family, mechanism, hyperparameters, data, seed, compute)
```

Each arm runs the same task suite and writes one capability vector:

```
exact_span, bits_per_span, decisions_per_span,
copy_hit, target_calls, edit_delta, roundtrip,
payload_bpb, shared_bpb, pack_bytes, seconds
```

The system does five things:

1. **Baseline** every family on a fast shared task.
2. **Grid** each family over its own hyperparameters.
3. **Pareto-rank** the arms; no single metric decides.
4. **Combine** promoted mechanisms (concept + copy, edit + spec, pack + library).
5. **Scale** only combinations that survive the frontier.

The goal is not one metric. The goal is to find the most useful, unique
end-to-end combinations and keep the evidence for every family.

## 2. Shared tasks

| task | input | output | cost | use |
|---|---|---|---|---|
| **T1 span repair** | 70k held-out bytes, 200 matched spans | exact span, bits/span, decisions/span | seconds-minutes | fast capability |
| **T2 compression** | 756,752 B held-out code | payload/shared bpb, round-trip, provenance | minutes | codec claim |
| **T3 edit** | T1 plus a deleted/merged prefix | survivor exact, removed count, bpb delta, round-trip | minutes | causal editability |
| **T4 draft** | T1 contexts | accepted concepts per target call, K | minutes | speculative value |
| **T5 install** | held-out package/half | bpb delta, provenance | minutes | transferable vocabulary |
| **T6 library** | T1 contexts plus train templates | exact, fill accuracy | hours | composition |
| **T7 preference** | 10k repair pairs | pairwise accuracy, win rate, KL | hours-days | RLFP, optional |

T1 is the common screen. T2 is the honest compression check. T3-T6 test the
properties that make the architecture different from a byte LM. T7 is not the
first gate; it runs only when a combination earns it.

## 3. Families and first grids

| family | mechanism | current evidence | first grid | task |
|---|---|---|---|---|
| **r1** adapter/install | charged copy / equal bytes | oracle +0.758; charged −2.224; P0a fails | dict size, charge, no-pack, transfer | T5 |
| **r2** copy/hybrid | copy a prior context | key 1→6: exact 0.135→0.650; key 6 6.193 bits, 0.475 hit | key 2-8, confidence, offset threshold, pointer fallback | T1, T4 |
| **r3** PhraseLM | lexicon-agnostic heads | concept-only 0.650; unconstrained 0.415 | gate/joint, concept-only, temperature, three heads | T1, T2 |
| **r4** concept LM | neural prices | 1.600 teacher / 1.692 payload / 1.722 shared | pack 128-2048, seq 128-256, steps 2k-8k | T1, T2 |
| **r5** LedgerLM/edits | persistence, market, editing | `self` −0.000 bpb; space +0.146; newline +0.228; merge +0.108 | prefix choice, merge size, held-out win rate | T3 |
| **r6** CMT/pack | tied entries, provenance | 2,771 B pack; 100% round-trip | active 128-2048, install/transfer, pack bytes | T2, T5 |
| **r7** CoLib/library | holes, templates, composition | order 1→5: exact 0.030→0.205 | order 5, template/hole, retrieval, anti-unification | T6 |
| **r8** speculative | pointer drafts, trunk verifies | K=8: 1.560 accepted/call; K=4: 1.410 | K 2-16, draft cost, verifier calls | T4 |
| **r9** RLFP/DPO | preference projection | untested; decision gate is span-policy | pairs 2k-10k, SFT, DPO, KL | T7 |
| **r10** scale | larger model/data | 20M overfits at 7.2 MB; host restarts | 5M/10M/20M, more unique code | T2 |

The family list is not a kill list. It is the current search space. A family
with a weak first grid gets another config before it is parked.

## 4. What is established

| result | number | evidence |
|---|---|---|
| concept-only T1 | **0.650 exact / 1.720 bits / 1.00 decision** | `reports/SCREEN.md` |
| trunk T1 | 0.480 exact / 2.590 bits / 1.00 decision | `reports/SCREEN.md` |
| byte T1 | 0.210 exact / 12.711 bits / 3.04 decisions | `reports/SCREEN.md` |
| concept vs byte | 3.1x exact, 3.0x fewer decisions, 7.4x fewer bits | `reports/SCREEN.md` |
| compression | 1.600 teacher / 1.692 payload / 1.722 shared | `reports/CONCEPT.md` |
| round-trip | 50,000 units / 81,531 B | `reports/CONCEPT.md` |
| local deletion | 35 removed; 165 survivors ±0.006; +0.146 bpb | `reports/EDIT-CURVE.md` |
| merge | 256→320; +0.108 bpb; exact | `reports/SCREEN.md` |
| copy key 6 | 0.650 exact, 0.475 hit, 6.193 bits | `reports/BREADTH.md` |
| library order 5 | 0.205 exact, 5.970 bits | `reports/BREADTH.md` |
| edit prefixes | `self` −0.000, `return` +0.026, indent +0.028, space +0.146, newline +0.228 | `reports/BREADTH.md` |
| spec K=8 | 1.560 accepted concepts per trunk call | `reports/BREADTH.md` |

The concept-only action space is the current frontier on T1. The trunk wins
teacher-forced compression but not the concept action space. Neither result
settles the portfolio.

### 4.1 Hypotheses

| id | claim | status |
|---|---|---|
| H1 | the 5M concept LM reads the codec pack | **pass on bpb/round-trip**; span-policy 3.498 B/decision; all-symbol 1.478 is the compression cost |
| H2 | copy, edit, and draft add capability | **open**; key 6 copy 0.650, `self` edit −0.000, K=8 1.560 accepted/call |
| H3 | concept DPO beats byte DPO | **untested**; only after a combination reaches the T1-T4 frontier |
| H4 | pack edits change behavior without gradients | **local**; removed targets are lost, survivors unchanged |
| H5 | the combination scales | **deferred**; 20M on 7.2 MB overfits; needs more code |

## 5. Fair allocation

The system must not spend all energy on the current leader. Use a fixed budget
per round.

### Round 0: baseline grid
- Every family gets **3-6 configs** and a **2 h** cap.
- Fast task T1 first; T2 only for families that compress.
- Record best, median, and worst, not only the winner.
- No family is dropped after Round 0.

### Round 1: frontier + exploration
- **50%** of time to Pareto-frontier arms.
- **30%** to near-frontier arms (within 20% on one axis).
- **20%** exploration reserve for parked families and new combinations.
- Promote combinations, not just single families.

### Round 2: combinations and scale
- **70%** to the best combinations.
- **20%** to near-frontier alternatives.
- **10%** to a deliberate wildcard.
- Scale only combinations that keep round-trip and provenance.

Hard caps:
- 5M parameters and 70k bytes for the fast loop.
- 2 h per arm; no 20k-step run.
- 756k/T2 only for promoted arms.
- Checkpoint every arm; a host restart must not lose a run.

## 6. Promotion and parking

An arm is **promoted** when it is on the Pareto frontier or gives a clear
tradeoff:

- exact +10% with bits +10% or less,
- bits −10% with exact −5% or less,
- decisions −20% with exact −5% or less,
- target calls −20% with exact unchanged,
- a new capability (edit, transfer, library) with exact ≥0.40.

An arm is **parked** when its whole grid is dominated. Parking is not a kill:
record the grid, the best/worst, and the condition that would reopen it. The
exploration reserve can reopen any parked family when a new combination or task
makes it relevant.

Do not:
- call one config a verdict,
- compare across different tasks,
- promote a bpb win that loses round-trip or provenance,
- start RLFP before a combination survives T1-T4,
- train 20M on 7.2 MB.

## 7. Combination matrix

The unique result is likely a combination, not a single family.

| combination | tests | why |
|---|---|---|
| concept + long-key copy | T1, T4 | copy for repeats, concept for novel spans |
| concept + pack size | T1, T2 | more concepts may raise exactness or reduce decisions |
| concept + edit | T3 | delete/merge a concept and measure survivors |
| concept + spec K=8 | T4 | fewer target calls at the same exactness |
| pack + install | T5 | transfer a pack without a gradient |
| library + concept | T1, T6 | template composition for holes |
| all + RLFP | T7 | only after T1-T6 produce a frontier |

Each combination is an arm with its own budget. Do not assume the best single
family is the best combination.

## 8. Metrics and artifacts

Every run records:
- config, seed, git, corpus/pack SHA,
- exact, bits/span, decisions/span,
- copy hit and offset cost,
- target calls and draft length,
- edit delta, removed/survivor counts,
- payload/shared bpb and round-trip,
- seconds and GPU-hours.

Artifacts:
- `reports/SCREEN.md`, `reports/BREADTH.md`, `reports/EDIT-CURVE.md`,
- `reports/PORTFOLIO.md` and `portfolio.json` (new),
- per-arm `runs/<id>/metrics.json`.

`PORTFOLIO.md` must show the frontier, the best config per family, the
exploration reserve, and the next round's allocation.

## 9. Milestones

| M | ships | acceptance | cost |
|---|---|---|---|
| R0 | shared suite + portfolio schema | T1-T5 run; every family has a grid | 0.5 d |
| R1 | Round 1: all families | best/median/worst; frontier | 1 d |
| R2 | combinations | 3-5 combined arms on T1/T4/T3 | 1 d |
| R3 | compression/scale | T2 on promoted combos; 5M/10M | 1 d |
| R4 | RLFP optional | 2k-10k pairs; SFT; concept vs byte DPO | 1-2 d |
| R5 | decision | scale the Pareto winner or record the negative | 0.5 d |

M1 is already printed. R1 is the next build. R4 is not the first gate.

## 10. Interfaces

Existing:
```bash
uv run cml train --corpus code --units codec --model pointer --params 5000000 --steps 8000
uv run cml screen --spans 200
uv run cml breadth --spans 200
uv run cml editcurve --spans 200
```

New portfolio interface:
```bash
uv run cml portfolio --round 1 --families all --task t1 --budget 2h
uv run cml portfolio --round 2 --promote frontier --task t1,t3,t4
uv run cml portfolio --round 3 --combine concept,copy,spec --task t2
```
`cml portfolio` runs the grid, scores the vector, writes `PORTFOLIO.md`, and
returns the next allocation. It must be resumable.

## 11. Risks

- **Portfolio sprawl.** Too many arms; enforce 2 h and 3-6 configs per family.
- **Metric tunnel vision.** Pareto, not one number.
- **Copy leakage.** Never search for the target span in the context; use only
  prior context. The first copy probe was invalid and was replaced.
- **Edit tautology.** Removed targets are lost by construction; report
  survivors separately.
- **DPO overfit.** Held-out prompts, KL, and both decision definitions.
- **Scale illusion.** 20M on 7.2 MB overfits; more unique code first.
- **Host instability.** Checkpoint every arm; no 20k runs.

## 12. Next 72 hours

1. **R1 copy/hybrid.** Key 4-8, confidence, offset threshold, pointer fallback.
   Target: exact ≥0.70 at ≤2.5 bits/span, or a new Pareto point.
2. **R1 pack size.** Active 128/256/512/1024; concept-only T1; record exact,
   bits, decisions, pack bytes.
3. **R1 edits.** Prefixes `self`/`return`/indent/space/newline on T3; test
   whether the free `self` deletion changes held-out behavior.
4. **R1 spec.** K=4/8/12; accepted/call and draft cost; target ≥1.5.
5. **R1 library.** Order 5 and template/hole composition; target exact ≥0.40.
6. **R2 combinations.** concept+copy, concept+edit, concept+spec; pick the
   Pareto winners.
7. **R3/R4.** Only then T2 scale and RLFP.

## References

- `TODO7.md`: the codec, install negative, and auditable product.
- `cml/PLAN.md` §11: name/charge accounting.
- `cml/pointer.py`: content-addressed pack reader.
- `cml/codec.py`: `Lexicon`, `greedy_parse`, `build_lexicon`, arms.
- `cml/cml.py`: `Trunk`, `train_text`, `p1_sample`, `cml train/sample/stats`,
  `cml screen/breadth/editcurve`, pack ops.
