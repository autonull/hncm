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

### 2.1 Round 1-3 progress (2026-11-26)

| item | state | evidence |
|---|---|---|
| shared suite + portfolio schema | **done** | `cml portfolio --round 1 ...` writes `PORTFOLIO.md`, `portfolio.json`, per-arm `runs/portfolio/` |
| copy key 4-12, confidence/cap/pointer | **done** | no cap: 0.720 exact / 4.871 bits; key 7 cap 256 / τ 0.10: 0.665 / 2.146 |
| edits, survivor exact | **done** | `self` is free (40 B, 0.000 exact); newline costs 0.072 exact |
| spec K 4/8/12 | **done** | K=8: 1.560; K=12: 1.505 accepted/call |
| library template/hole | **done, below target** | radius 2: 0.340 exact / 4.937 bits (target 0.40) |
| pack size 128-1024 | **done** | 256 wins all axes: 0.415 / 3.133 / 1.00; 128, 512, 1024 dominated |
| concept+copy on T1 | **done, target met** | blend key 10 α0.90: 0.705 / 2.362; key 6 α0.50: 0.680 / 1.698; no cap 0.720 / 4.782 |
| concept+edit on T3 | **done, local** | survivors 0.733 vs base 0.745 (−0.012); `self` free |
| concept+spec on T4 | **done, calibration was the limit** | τ4 K16 blend 4.650 accepted/call; τ1 1.480; pointer τ4 4.450 |
| R3 T2 review | **done, pass** | 5.21M pointer: 1.692 payload / 1.722 shared / 1.600 teacher; 50k proof / 81,531 B; blend is not a codec |
| R3 scale 5M/10M | **done, promoted** | 24 MB train, same 756,752 B test: 5M 1.583 / 1.613; 10M (9.17M actual) 1.538 / 1.568; −0.154 payload vs base |
| R4 RLFP | **blocked** | needs a T1-T4 survivor |

Round 2 frontier: 6 arms, all `r2+r3/r4`. The blend dominates the
concept-only point (0.650 / 1.720): key 6 α0.50 gives 0.680 at 1.698;
key 10 α0.90 gives 0.705 at 2.362, so the T1 target **is met**. Pack
size is settled at 256. T3 is local (−0.012 on survivors). T4 was a
draft-calibration limit, not a trunk limit: τ1 gives 1.480 accepted/call,
τ4 K16 gives 4.650. R2 is complete. R3 confirms the promoted codec:
T2 passes, and more unique train code improves the 5M point from 1.692
to 1.583 payload; the 10M point reaches **1.538 / 1.568**. The T2 result
is a codec result, not a T1 blend result.

## 3. Families and first grids

| family | mechanism | current evidence | first grid | task |
|---|---|---|---|---|
| **r1** adapter/install | charged copy / equal bytes | oracle +0.758; charged −2.224; P0a fails | dict size, charge, no-pack, transfer | T5 |
| **r2** copy/hybrid | copy a prior context | key 1→12; blend 0.705 / 2.362 (target), 0.680 / 1.698; no cap 0.720 / 4.782 | key 4-12, blend α, margin, stack, concept fallback | T1, T4 |
| **r3** PhraseLM | lexicon-agnostic heads | concept-only 0.650; unconstrained 0.415 | gate/joint, concept-only, temperature, three heads | T1, T2 |
| **r4** concept LM | neural prices | 1.600 teacher / 1.692 payload / 1.722 shared; scale 10M 1.445 / 1.538 / 1.568 | pack 128-2048, seq 128-256, steps 2k-8k | T1, T2 |
| **r5** LedgerLM/edits | persistence, market, editing | `self` −0.000 bpb / 0.000 exact; newline +0.228 / −0.072 exact; blend survivors −0.012 | prefix choice, merge size, held-out win rate | T3 |
| **r6** CMT/pack | tied entries, provenance | 256 wins T1: 0.415 / 3.133 / 1.00; 128, 512, 1024 dominated | active 128-1024, install/transfer, pack bytes | T1, T2, T5 |
| **r7** CoLib/library | holes, templates, composition | order 1→5: exact 0.030→0.205; template radius 2: 0.340 | order 5, template/hole, retrieval, anti-unification | T6 |
| **r8** speculative | pointer drafts, trunk verifies | τ4 K16: 4.650 accepted/call; τ1 K8: 1.480; draft temperature is the lever | K 2-16, draft temp, verifier calls | T4 |
| **r9** RLFP/DPO | preference projection | untested; decision gate is span-policy | pairs 2k-10k, SFT, DPO, KL | T7 |
| **r10** scale | larger model/data | 24 MB train: 5M 1.583 / 1.613, 10M 9.17M 1.538 / 1.568 payload/shared; 10M promotes | 5M/10M/20M, more unique code | T2 |

The family list is not a kill list. It is the current search space. A family
with a weak first grid gets another config before it is parked.

## 4. What is established

| result | number | evidence |
|---|---|---|
| concept-only T1 | **0.650 exact / 1.720 bits / 1.00 decision** (dominated by blend) | `reports/SCREEN.md` |
| trunk T1 | 0.480 exact / 2.590 bits / 1.00 decision | `reports/SCREEN.md` |
| byte T1 | 0.210 exact / 12.711 bits / 3.04 decisions | `reports/SCREEN.md` |
| concept vs byte | 3.1x exact, 3.0x fewer decisions, 7.4x fewer bits | `reports/SCREEN.md` |
| compression | 1.600 teacher / 1.692 payload / 1.722 shared | `reports/CONCEPT.md` |
| round-trip | 50,000 units / 81,531 B | `reports/CONCEPT.md` |
| T2 review (promoted codec) | 1.692 payload / 1.722 shared / 1.600 teacher; round-trip OK; blend excluded | `reports/PORTFOLIO.md` |
| T2 scale 5M, 24 MB train | 1.583 payload / 1.613 shared / 1.490 teacher | `reports/SCALE.md` |
| T2 scale 10M, 24 MB train | **1.538 payload / 1.568 shared / 1.445 teacher**, 9.17M actual | `reports/SCALE.md` |
| local deletion | 35 removed; 165 survivors ±0.006; +0.146 bpb | `reports/EDIT-CURVE.md` |
| merge | 256→320; +0.108 bpb; exact | `reports/SCREEN.md` |
| copy key 6, no cap | **0.720 exact**, 0.340 hit, 4.871 bits | `reports/BREADTH.md` |
| copy key 7, cap 256, τ 0.10 | 0.665 exact, 2.146 bits (new near-concept point) | `reports/BREADTH.md` |
| concept+copy key 6, no cap | 0.720 exact, 4.782 bits | `reports/BREADTH.md` |
| concept+copy dist key 10, min 1 | 0.710 exact, 2.767 bits | `reports/BREADTH.md` |
| concept+copy dist key 6, min 2 | 0.690 exact, 2.404 bits | `reports/BREADTH.md` |
| concept+copy blend key 10, α 0.90 | **0.705 exact, 2.362 bits (target met)** | `reports/BREADTH.md` |
| concept+copy blend key 6, α 0.50 | 0.680 exact, 1.698 bits (dominates concept) | `reports/BREADTH.md` |
| concept+blend edit, `space` delete | survivors 0.733 vs 0.745; −0.012 exact | `reports/EDIT-CURVE.md` |
| concept+blend draft, τ1 K8 | 1.480 accepted/call (overconfident) | `reports/EDIT-CURVE.md` |
| concept+blend draft, τ4 K16 | **4.650 accepted/call**; pointer 4.450 | `reports/EDIT-CURVE.md` |
| pack 256, fixed spans | 0.415 exact, 3.133 bits, 1.00 decision | `reports/PACKS.md` |
| pack 128/512/1024 | 0.375/0.395/0.300 exact; all dominated by 256 | `reports/PACKS.md` |
| template radius 2 | 0.340 exact, 4.937 bits (was library 0.205) | `reports/BREADTH.md` |
| library order 5 | 0.205 exact, 5.970 bits | `reports/BREADTH.md` |
| edit prefixes | `self` −0.000 bpb, 0.000 exact; newline +0.228 bpb, −0.072 exact | `reports/BREADTH.md` |
| spec K=8 | 1.560 accepted concepts per trunk call | `reports/BREADTH.md` |

The concept-only point (0.650 / 1.720) is now dominated by the blend
(0.680 / 1.698). The T1 frontier is six `r2+r3/r4` arms from 0.675 / 1.667
to 0.720 / 4.782, and the target 0.705 / 2.362 is met. Pack size is
settled at 256. The concept head lives inside the blend as the fallback.
The trunk wins teacher-forced compression but not the action space. On T2,
the promoted codec passes and improves with more unique train code:
1.692 → 1.583 (5M) → 1.538 (10M) payload. That is a codec result, not a
T1 blend result.

### 4.1 Hypotheses

| id | claim | status |
|---|---|---|
| H1 | the 5M concept LM reads the codec pack | **pass on bpb/round-trip**; span-policy 3.498 B/decision; the T1 concept point (0.650/1.720) is dominated by the blend |
| H2 | copy, edit, and draft add capability | **pass on copy, local on edit, calibrated on draft**; blend 0.705 at 2.362 bits, 0.680 at 1.698; blend edit −0.012; T4 τ4 K16 4.650 |
| H3 | concept DPO beats byte DPO | **untested**; only after a combination reaches the T1-T4 frontier |
| H4 | pack edits change behavior without gradients | **local**; removed targets are lost, survivors unchanged |
| H5 | the combination scales | **pass on T2**; 24 MB train: 5M 1.583 / 1.613, 10M 1.538 / 1.568 payload/shared; 20M remains untested |

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
- `reports/PACKS.md`,
- `reports/PORTFOLIO.md` and `portfolio.json` (new),
- per-arm `runs/<id>/metrics.json`.

`PORTFOLIO.md` must show the frontier, the best config per family, the
exploration reserve, and the next round's allocation.

## 9. Milestones

| M | ships | acceptance | cost | status |
|---|---|---|---|---|
| R0 | shared suite + portfolio schema | T1-T5 run; every family has a grid | 0.5 d | **done** |
| R1 | Round 1: all families | best/median/worst; frontier | 1 d | **done** |
| R2 | combinations | 3-5 combined arms on T1/T4/T3 | 1 d | **done** |
| R3 | compression/scale | T2 on promoted combos; 5M/10M | 1 d | **done** |
| R4 | RLFP optional | 2k-10k pairs; SFT; concept vs byte DPO | 1-2 d | blocked |
| R5 | decision | scale the Pareto winner or record the negative | 0.5 d | **partial: T2 scale promoted; RLFP blocked** |

M1 is already printed. R0-R3 are printed. T2 passes on the promoted codec
(1.692 → 1.583 → 1.538 payload), and the 10M scale point promotes. R4 is
still not the first gate: no combination has survived T1-T4 yet.

## 10. Interfaces

Existing:
```bash
uv run cml train --corpus code --units codec --model pointer --params 5000000 --steps 8000
uv run cml screen --spans 200
uv run cml breadth --spans 200
uv run cml editcurve --spans 200
uv run cml packs --sizes 128,256,512,1024
```

New interfaces:
```bash
uv run cml portfolio --round 1 --families all --task t1 --budget 2h
uv run cml portfolio --round 2 --families r2,r3/r4 --task t1,t4 --budget 6h
uv run cml portfolio --round 3 --families all --task t2 --budget 6h
uv run cml scale --sizes 5m,10m --steps 8000 --seed 0
```
`cml portfolio` does not train; it aggregates every measured grid, scores the
T1 frontier, writes `PORTFOLIO.md` and `portfolio.json`, and returns the next
allocation. Rounds are stored in `portfolio.json`, so a rerun replaces only
that round and keeps the history.

`cml scale` keeps the pinned 756,752 B T2 test and the 256-unit pack, adds
unique train code up to 24 MB, and trains the requested 5M/10M pointers. It
writes `reports/SCALE.md` and per-arm `runs/codec-scale-*/metrics.json`.
5M and 10M are parameter targets, not exact counts (5.21M and 9.17M).

## 11. Risks

- **Portfolio sprawl.** Too many arms; enforce 2 h and 3-6 configs per family.
- **Metric tunnel vision.** Pareto, not one number.
- **Copy leakage.** Never search for the target span in the context; use only
  prior context. The first copy probe was invalid and was replaced.
- **Edit tautology.** Removed targets are lost by construction; report
  survivors separately.
- **DPO overfit.** Held-out prompts, KL, and both decision definitions.
- **Scale illusion.** 20M is untested; 24 MB is still finite. Extend unique
  code before parameters.
- **Host instability.** Checkpoint every arm; no 20k runs.

## 12. Next round (R5)

Done in R3: T2 review on the promoted codec, the 5M/10M scale, and the
10M promotion. Still open: RLFP, a combined T1-T4 survivor, and 20M.

1. **R5 decision.** Promote the T2 10M codec (1.538 payload / 1.568
   shared) as the codec winner. Keep the T1 blend (0.680 / 1.698) as the
   span-policy winner. Do not merge the two numbers or average them.
2. **R4 RLFP.** Still blocked: no combination survives T1-T4. If one does,
   start with 2k-10k pairs, SFT, and both decision definitions.
3. **20M only with more unique code.** Do not train 20M on 7.2 MB; the
   24 MB scale corpus is the current limit. Extend unique code first.
4. **Seed repeat.** Repeat the 10M T2 point at seed 1 before treating
   the 0.154 payload gain as a stable claim.

The exploration reserve reopens parked `r1`, `r1/r6`, `r2`, `r3/r4`,
`r3/r4/r6`, `r6`, and `r7` only with a new grid or a new combination.
r6 keeps 256 as the pack; r5 and r8 stay active through their T3/T4
capability rows; r10 stays active through its T2 scale row.

## References

- `TODO7.md`: the codec, install negative, and auditable product.
- `cml/PLAN.md` §11: name/charge accounting.
- `cml/pointer.py`: content-addressed pack reader.
- `cml/codec.py`: `Lexicon`, `greedy_parse`, `build_lexicon`, arms.
- `cml/cml.py`: `Trunk`, `train_text`, `p1_sample`, `cml train/sample/stats`,
  `cml screen/breadth/editcurve/packs/portfolio/scale`, pack ops.
- `cml/reports/SCALE.md`: the R3 5M/10M T2 scale grid and verdict.
