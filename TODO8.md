# TODO8 — the 50% concept codec plan

Status: 2026-11-27. R3 is complete. The 20M codec is at `1.488` payload /
`1.518` shared on the pinned 756,752 B test. The 50% target is `0.847`
shared bpb against a 20M byte control. RLFP is untested. The next move is
a cheap, ordered experiment ladder, not another scale seed.

## 0. One claim, one target

**Claim.** An exact, editable codec beats a byte LM at equal compute on
held-out code, with exact round-trip and provenance.

**Target.** 50% fewer bits per byte than a 20M byte LM on the pinned
756,752 B code test:

| milestone | shared bpb | vs 1.693 byte | compression | state |
|---|---:|---:|---:|---|
| 20M, 48 MB, pack 256 | 1.518 | -10.3% | 5.3x | **measured** |
| 20% | 1.354 | -20% | 5.9x | next |
| 30% | 1.185 | -30% | 6.8x | path |
| 40% | 1.016 | -40% | 7.9x | path |
| **50%** | **0.847** | **-50%** | **9.4x** | north star |

The denominator must be a 20M byte control on the same train/test/steps.
The current 4.86M byte model (`1.693 / 1.694`) is a reference, not the
equal-compute control. A 20M-vs-4.86M win is capacity, not architecture.

## 1. Current evidence

| result | number | evidence |
|---|---|---|
| T1 concept | 0.650 exact / 1.720 bits / 1.00 decision | `reports/SCREEN.md` |
| T1 blend | **0.680 exact / 1.698 bits** | `reports/BREADTH.md` |
| T1 trunk | 0.480 / 2.590 / 1.00 | `reports/SCREEN.md` |
| T1 byte | 0.210 / 12.711 / 3.04 | `reports/SCREEN.md` |
| T2 5M, 24 MB | 1.583 / 1.613 / 1.490 teacher | `reports/SCALE.md` |
| T2 10M, 24 MB | 1.538 / 1.568 / 1.445 | `reports/SCALE.md` |
| T2 10M, 48 MB | 1.569 / 1.599 / 1.476; data alone hurts | `reports/SCALE.md` |
| T2 20M, 48 MB | **1.488 / 1.518 / 1.395** | `reports/SCALE.md` |
| T2 5M seed spread | 0.019 over seeds 0-3 | `reports/SCALE.md` |
| T2 10M seed spread | 0.024 over seeds 0-4 | `reports/SCALE.md` |
| Equal-params 5.21M vs 4.86M byte | 1.583 / 1.613 vs 1.693 / 1.694 | `reports/SCALE.md` |
| Equal-data 20M vs 10M, 48 MB | -0.081 payload | `reports/SCALE.md` |
| T3 edit | survivors 0.733 vs 0.745; -0.012 | `reports/EDIT-CURVE.md` |
| T4 spec | 4.650 accepted/call; pointer 4.450 | `reports/EDIT-CURVE.md` |
| Round-trip | 50,000 units / 81,531 B | `reports/CONCEPT.md` |

**Established:** the codec works, round-trips, scales, and beats a
smaller byte model on payload and shared bpb by about 6%.
**Not established:** an equal-compute 20M win, a per-file product win,
an RLFP win, or a T1-T4 combined survivor.

## 2. Progress map

| area | current | next | gate |
|---|---|---|---|
| **Compression** | 1.488 / 1.518 at 20M | pack, copy, literal | 20/30/50% |
| **T1 capability** | 0.680 / 1.698; frontier 0.675-0.720 | hold; do not regress | T1 frontier |
| **T3 edit** | local, -0.012 | delta codec | survivors |
| **T4 spec** | 4.650 accepted/call | hold | no regression |
| **Combination** | T1/T3/T4 separate | one T1-T4 arm | holds each gate |
| **RLFP** | untested | 2k-pair pilot | +5 points |
| **Scale** | 20M / 48 MB | 40M only after pack/copy | equal compute |
| **Evaluation** | 756 KB in-domain | out-of-domain, per-file | honest claim |
| **Product** | exact round-trip | per-file/adaptive pack | shared bpb |
| **Infrastructure** | checkpoint/resume | caches, no timer | no OOM |

## 3. The minimum-resource ladder

Shrink for **search**, not for the final claim. A tiny model can rank
levers; only a 20M point can prove the 50% target.

| stage | model | train/test | pack | steps | cost | purpose |
|---|---|---|---|---|---|---|
| **0 analysis** | none | 756 KB | 256-2048 | 0 | <5 min CPU | coverage, copy, per-file |
| **1 tiny screen** | `d64 l2 h4 seq64`, ~0.2-0.4M | 1 MB / 512 KB | 64/128/256 | 300, batch 8 | 10 min, <1 GB | rank pack/copy/literal |
| **2 5M confirm** | `d256 l6 h8 seq128`, 5.2M | 8 MB / 512 KB | 256 | 1000, batch 16 | 15 min, <2 GB | confirm top two |
| **3 20M verify** | `d448 l8 h8 seq256`, 20.0M | 48 MB / 756 KB | 256/1024 | 8000, batch 32 | 60-90 min, <3 GB | 20/50% claim |
| **4 RLFP pilot** | frozen 5M trunk + head | 2k pairs | 256 | 500-1000 | 2 min, <1 GB | DPO vs SFT |

One full cycle is **about 2 GPU-hours and under 3 GB**. Run one task at
a time. Checkpoint every 500 pointer steps. Never use a timer that can
kill an active task.

**Rules that save the most:**

1. **Freeze the trunk.** Train one 5M trunk; freeze the blocks; sweep only
   the token embedding, span encoder, and head for pack/copy variants.
2. **Nested packs.** Build a 2048-unit pack once; evaluate prefixes
   256/512/1024/2048. Do not build each pack independently.
3. **Cache parses.** Store `(pack_sha, data_sha) -> int32` arrays; cast to
   long per batch. Halve memory versus int64.
4. **Use 512 KB test at tiny scale.** At 64 KB the 2.7 KB pack costs
   `0.338` bpb and hides the lever. At 512 KB it costs `0.043`.
5. **One seed screens; three seeds confirm.** Seeds are for claims, not
   for ranking levers.
6. **No corpus rebuild per arm.** Reuse the 1 MB, 8 MB, and 48 MB slices.

## 4. Stage 0: CPU analysis, before any GPU

This stage is free and can kill a bad lever before training.

1. **Per-file overhead.** For 1/4/10/100 KB files, compute
   `payload + 8 * pack_bytes / n` with the 2.7 KB and 0.7 KB packs.
   If the codec loses at 10 KB, the product needs an adaptive pack.
2. **Pack coverage.** Parse the 756 KB test with 256/512/1024/2048-unit
   packs. Report coverage, literal fraction, mean unit, pack bytes, and
   the theoretical payload floor. If coverage does not rise by >0.10, the
   pack is not the lever.
3. **Copy preprocessor.** Run LZ77 with a 4 KB window over the 756 KB
   test; encode matches as (offset, length), literals to the existing
   codec. Charge offset/length bits. If it saves <5%, copy at this
   granularity is not the lever.
4. **Literal fraction.** Measure the share of escaped bytes and their
   cost. A literal head can only recover that share.

**Decision.** Keep the two levers with >`0.10` bpb potential. If none has
it, stop compression and pivot to delta/adaptive pack, not another seed.

## 5. Stage 1: tiny screen

Six arms, one shared data slice:

| arm | pack | copy | literal |
|---|---|---|---|
| A | 64 | no | no |
| B | 128 | no | no |
| C | 256 | no | no |
| D | 256 | LZ preprocessor | no |
| E | 256 | no | byte head |
| F | 128 | LZ preprocessor | byte head |

Model `d64 l2 h4 seq64`, 1 MB train / 512 KB test, 300 steps, batch 8.
Report payload, shared, coverage, literal fraction, round-trip, seconds.
Promote only arms that beat arm C by >`0.05` payload. Keep at most two.

## 6. Stage 2: 5M confirm

Two arms plus controls:

- the best pack/copy arm from Stage 1;
- the 256-unit baseline;
- a 5.2M byte control on the same 8 MB train / 512 KB test.

Model `d256 l6 h8 seq128`, 1000 steps, batch 16. Freeze the trunk after
the baseline; train only the embedding/encoder/head for the variant.
Promote only a `>0.10` payload gain over the 5M baseline. Keep one arm.

## 7. Stage 3: 20M verify

The only stage that can support the 50% target:

- 20M byte control: 48 MB train, 756 KB test, 8000 steps;
- 20M pack 256: existing `1.488 / 1.518`;
- 20M pack 1024: freeze the 20M blocks; train embedding/encoder/head
  2000 steps, not 8000;
- 20M copy: LZ preprocessor proxy on the existing 20M model; no new
  training; if it saves >5%, build the full copy arm;
- two seeds for the winner only if it needed a full train.

Model `d448 l8 h8 seq256`, 8000 steps, batch 32, 48 MB train /
756 KB test. Report payload, shared, teacher, round-trip, provenance.
**Gate:** the winner must beat the measured 20M byte control by >20%
shared bpb. At the current `1.693` reference that is `1.354`; 50%
(`0.847`) is the stretch.

## 8. Stage 4: RLFP pilot

RLFP is the unique capability lever and does not need a T1-T4 survivor.
It does not directly lower T2 bpb; it improves the span policy.

- **Data:** 2,000 pairs, 80/20 train/held-out.
- **Chosen:** the exact target span from a T1 context.
- **Rejected:** the policy's top wrong span, or a deleted/merged span.
- **Policy:** softmax over frozen candidate logits; no full LM training.
- **Reference:** the SFT policy before DPO.
- **Loss:** DPO on candidate logits.
- **Metrics:** pairwise accuracy, win rate, KL, exact span accuracy,
  accepted/call.
- **Promote:** +5 points pairwise over SFT on held-out pairs.
- **Label:** synthetic-preference pilot, not human RLHF.

Cost: 2 minutes, under 1 GB. Run it in parallel with Stage 0/1; it does
not compete for the GPU.

**Combination check.** Only after T1, T3, and T4 each hold separately.
One arm, 2 h cap, all three tasks. Do not sweep combinations before the
individual levers are measured.

## 9. Levers and expected gains

| lever | expected | cost | risk | test |
|---|---:|---|---|---|
| pack 256 -> 1024/2048 at 20M | -0.15 to -0.30 | 2 runs | pack charge | Stage 3 |
| T2 copy / LZ preprocessor | -0.10 to -0.25 | CPU + 1 run | leakage | Stage 0/3 |
| better literal head | -0.05 to -0.15 | 1 head | small | Stage 1 |
| 40M capacity | -0.05 to -0.15 | 1 run | byte scales | after Stage 3 |
| RLFP | product/capability | 2 min | synthetic | Stage 4 |

These are hypotheses, not measured results. Pack and copy are the big
levers. If they reach `-0.40`, the codec is at ~`1.12`; capacity and
literals can plausibly reach `0.9`. That is the 50% zone.

## 10. Out-of-the-box alternatives, only if stuck

Try one at a time, after a plateau, with a stop rule.

| alternative | idea | cost | when |
|---|---|---|---|
| **adaptive per-file pack** | build a small pack per file/repo; charge once | CPU | per-file test fails above 100 KB |
| **retrieval pack** | retrieve exact spans from an index at encode/decode | CPU index + 5M | static pack plateaus |
| **delta T2** | encode version-to-version diffs; copy the old version | CPU + 1 run | full-file T2 plateaus |
| **hierarchical pack** | 256 phrase + 2K phrase + byte fallback | 1 model | pack coverage plateaus |
| **grammar/AST pack** | identifiers/indent/keywords as units | CPU | substring pack plateaus |
| **learned literal model** | small byte LM for escapes | 1 head | literal fraction high |
| **domain routing** | two packs by language/package | 2 models | transfer fails |
| **speculative span decode** | draft spans; trunk verifies | 1 run | latency, not bpb |
| **RLFP on edits** | original vs mutated code pairs | 2 min | T1 pairs saturate |

A retrieval pack must charge its index or be installed once; otherwise
it is not a compression claim. An adaptive pack only helps if its pack
charge is <`0.2` bpb, which means <`256` B at 10 KB; measure that floor
before building one. **Delta T2 is the strongest product alternative**:
versioned code can copy the old file and pay only the diff. None of these
fixes 1-10 KB files unless the pack charge falls below `0.2` bpb.

## 11. Gates and kill/park rules

| gate | rule |
|---|---|
| G0 | exact round-trip and provenance |
| G1 | equal-compute 20M byte control exists |
| G2 | 20% shared bpb (`1.354`) |
| G3 | 30% (`1.185`) |
| G4 | 50% (`0.847`) |
| G5 | per-file shared bpb beats byte at 10 KB |
| G6 | RLFP +5 points pairwise on held-out |

- If Stage 0 finds no lever >`0.10`, pivot to adaptive/delta; do not train.
- If Stage 1 finds no arm >`0.05`, stop the pack/copy branch.
- If Stage 2 finds no arm >`0.10`, stop before 20M.
- If Stage 3 misses 20%, record the result and keep T1/T3/RLFP.
- If per-file fails, the codec is a large-stream result, not a product.
- If RLFP misses +5 points, record the negative and park RLFP.
- Do not run 40M before Stage 3. Do not run 20M seed sweeps.

## 12. Implementation

Small changes, in this order:

1. `cml coverage`: CPU parse/coverage for 256/512/1024/2048; cache int32.
2. `cml copyprobe`: LZ77 preprocessor; payload and round-trip.
3. `cml perfile`: 1/4/10/100 KB shared bpb table.
4. `cml scale`: `active` in run id and metrics; `--preset tiny|5m|20m`;
   `--freeze-trunk`; `--copy`.
5. `cml codec --arm byte-model --params/--d/--layers`: 20M control.
6. `cml rlfp --pairs 2000 --run <5M>`: DPO vs SFT; `reports/RLFP.md`.
7. Reports: `reports/COVERAGE.md`, `reports/COPY.md`,
   `reports/PER-FILE.md`, `reports/T2.md`, `reports/RLFP.md`.

Commands after the build:

```bash
uv run cml coverage --sizes 256,512,1024,2048 --bytes 756752
uv run cml copyprobe --window 4096 --bytes 756752
uv run cml perfile --sizes 1024,4096,10240,102400
uv run cml scale --sizes 20m --active 1024 --steps 8000 --seed 0
uv run cml codec --arm byte-model --params 20000000 --steps 8000
uv run cml rlfp --pairs 2000 --run concept-pointer-code-p5m-s8000
```

Tests: the `active` run id, byte-model params, copy round-trip, RLFP
held-out split, and exact codec round-trip. No coverage target for the
rest.

## 13. Risks

- **Capacity confound.** A 20M pointer vs a 4.86M byte model is not
  equal compute. G1 is mandatory.
- **Pack charge.** 2.7 KB is 0.338 bpb at 64 KB and 0.029 at 756 KB.
- **Copy leakage.** Copy only prior context; never search for the target.
- **Per-file illusion.** A 756 KB stream hides the pack charge; G5 is
  mandatory for a product claim.
- **Synthetic RLFP.** T1 preference pairs test the mechanism, not human
  alignment. Label it correctly.
- **Host OOM.** Checkpoint; cap memory; one task; no timer that kills.
- **Portfolio sprawl.** Enforce 3-5 arms per stage and a 2 h cap.

## 14. Next actions

0. Do not restart the 5-minute timer. Use one explicit task per stage.
1. Run Stage 0: coverage, copy, per-file. No GPU.
2. Run Stage 1: six tiny arms. 10 minutes.
3. Run the RLFP pilot in parallel. 2 minutes.
4. Promote at most two arms to Stage 2.
5. Add the 20M byte control before Stage 3.
6. Run Stage 3 on one or two arms, two seeds.
7. Decide: 20% claim, 50% program, or park compression.
8. Only then consider 40M or an out-of-the-box alternative.

## References

- `TODO7.md`: codec record and auditable product.
- `cml/README.md`: the 50% target and progress map.
- `cml/PLAN.md`: design and charge arithmetic.
- `reports/SCALE.md`, `reports/PORTFOLIO.md`, `reports/CONCEPT.md`,
  `reports/EDIT-CURVE.md`.
- `cml/cml.py`, `cml/pointer.py`, `cml/codec.py`.
