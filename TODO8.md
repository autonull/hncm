# TODO8 — the 50% concept codec plan

Status: 2026-11-27. R3 is complete. Stage 0 is measured: copy saves
`15.1%` (2.141 vs 2.522 bpb, exact), pack coverage rises `+0.364`, and
the static pack loses per-file. The concept pilot is measured: typed
holes, exact realization, and a `0.230` bpb representation gain on
unseen compositions (a pilot, not reasoning). The cortex pilot is a
negative at this grain: the term model leads on seen (`-0.022` bpb) and
loses the unseen edge (`+0.004`). The depth sweep shows the cortex
winning unseen from depth 5 (`-0.021` at depth 6) while its novel-edge
pay falls to `+0.012` against `+0.003` for bytes. The window check
(§1.5) shows that trend was partly truncation: at `seq 256` the gap
stops closing at `+0.008`, so bytes match the term model once a term
fits the context. The 20M codec is at `1.488` payload /
`1.518` shared on the pinned 756,752 B test. The 50% target is `0.847`
shared bpb against a 20M byte control. RLFP is untested. The next move is
a composition task that a byte context cannot recompute, plus the Stage 3
copy proxy.

**Backup.** `TODO8.compression.md` is the original compression plan,
unchanged: the state before the concept track, including its full Stage 1
ladder. If the concept test does not separate from a matched byte
Transformer, resume that plan; its Stage 0 step is already answered by
§1.1, and its Stage 3 copy proxy is answered by `reports/COPY.md`.

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

## 1.1 Stage 0: measured 2026-11-27

CPU only, no training. `reports/COVERAGE.md`, `reports/COPY.md`,
`reports/PER-FILE.md` and their JSON siblings; commands in §12.

| lever | number | gate | verdict |
|---|---:|---|---|
| pack coverage, 256 -> 2048 units | 0.453 -> 0.817 (+0.364) | +0.10 | **holds** |
| pack charge | 2.8 KB -> 22.1 KB on 756 KB | - | small at this size |
| unit arm (static pack) | 2.773-3.197 vs byte 2.522 | G5 | **loses** |
| LZ copy, 4 KB window | 2.141 vs 2.522 (-15.1%), 86.5% copied | 5% | **holds** |
| per-file pack charge | best +0.148 bpb at 100 KB | G5 | **fails** |

**Read.** The pack is a coverage lever, not a codec on its own: the plain
unit arm loses 0.25-0.68 bpb to the byte arm, while the pointer arm
already pays (`1.518` shared at 20M). Copy is the big lever: 15.1% with
an exact round-trip and no training, so a Stage 3 proxy can price it in
front of the frozen 20M model. The per-file product claim fails for a
static pack at every size (1/4/10/100 KB); the product needs an adaptive
or delta pack, not a bigger static one. Stage 1's ranking job is served
at 756 KB: skip straight to the Stage 3 copy proxy.

## 1.2 Concept representation: measured 2026-11-27

`cml/concepts.py`: templates with typed holes, recursive `Term` composition,
exact `realize`, an MDL library gate, and `cml composition`. Report:
`reports/COMPOSITION.md`. Synthetic arithmetic grammar, depth 3, seed 0,
1,000 terms per split; the held-out edge is `ADD -> MUL`, never in train.

| split | nodes | realized B | concept bpb | shared | byte arm |
|---|---:|---:|---:|---:|---:|
| train | 6,876 | 26,437 | 0.956 | 1.007 | 1.232 |
| seen | 6,950 | 26,769 | 0.952 | 1.004 | 1.226 |
| unseen | 10,255 | 39,732 | 0.931 | 0.965 | 1.161 |
| twin | 10,255 | 39,732 | 0.919 | 0.953 | 1.147 |

**Read.** Realization is exact and the term is the message; the library
charge is 171 B. The novel edge costs `+0.012` bpb in the term code and
`+0.014` in the byte arm at matched shape (the `twin` control rewrites
`ADD -> MUL` to `ADD -> SUB`). Both arms generalize this edge, so at this
grain the term gain is symbolization, not composition: `0.230` bpb, not
the 50% claim. That is a pilot limit, not a gate. The separating test
needs a harder composition task (argument dependence, wider depth) and a
matched byte Transformer; the encoder and decoder both hold the term, so
this prices representation, not reasoning.

**Keep the compression baseline.** The span lexicon stays a compression
dictionary; terms are a different kind: an operation with holes that the
library composes, not a byte string that repeats.

## 1.3 Cortex pilot: measured 2026-11-27

`cml cortex`: a 22-symbol term model against a matched 256-symbol byte
model, both `d128 l4 h4 seq128`, 3,000 steps, 30,000 terms per split,
810,496 params, 108 s on one GPU. Report: `reports/CORTEX.md`.

| split | bytes | cortex bpb | byte bpb | delta |
|---|---:|---:|---:|---:|
| seen | 1,451,231 | 0.958 | 0.980 | -0.022 |
| unseen | 2,035,976 | 0.970 | 0.966 | +0.004 |
| twin | 2,035,976 | 0.938 | 0.959 | -0.021 |

**Read.** The novel edge costs `+0.032` bpb for the term model and
`+0.007` for bytes at matched shape. The term model leads on seen and
loses the unseen edge: the byte arm absorbs the new composition better.
So §1.2's gain is a *coding* gain when the term is given, not a
*prediction* gain. That is a pilot limit: this grammar has random values
and shallow structure, so local byte contexts recompute it. The
separating test needs argument dependence, long-range binding, and a
depth where bytes cannot recover the structure. Until then the concept
track has no evidence over bytes, and `TODO8.compression.md` is the
fallback.

**Pilot defects found and fixed.** At 3,000 terms the model memorized
(train `1.16` bits/token, held-out `5.23`); at 50,000 terms it stopped
below the marginal. The pilot needs both enough terms and a converged
run; 30,000 terms and 3,000 steps is the smallest point where train and
held-out agree.

## 1.4 Cortex depth sweep: measured 2026-11-27

`cml cortex --depths 3,4,5,6`: one matched pair per depth, 20,000 terms
per split, 2,500 steps, batch 32, `d128 l4 h4 seq128`, 810,496 params,
358 s on one GPU. Report: `reports/CORTEX-DEPTH.md`.

| depth | seen delta | unseen delta | cortex edge pay | byte edge pay | gap |
|---:|---:|---:|---:|---:|---:|
| 3 | +0.017 | +0.051 | +0.038 | +0.014 | +0.023 |
| 4 | -0.016 | +0.011 | +0.028 | +0.008 | +0.020 |
| 5 | -0.028 | -0.011 | +0.019 | +0.005 | +0.014 |
| 6 | -0.032 | -0.021 | +0.012 | +0.003 | +0.009 |

**Read.** The cortex wins the total rate from depth 4 and wins unseen
from depth 5 (`-0.021` at depth 6). Its novel-edge pay falls with depth
(`+0.038` to `+0.012`) while the byte arm stays near `+0.003`; the gap
closes about `0.005` bpb per depth. The runs reproduce across seeds
(depth 3-5 moved by at most `0.003`). Depth 6 terms approach the
128-symbol context, so the trend needs a window check before it means
anything: `§1.5` shows it was partly truncation.

## 1.5 Cortex context check: measured 2026-11-27

`cml cortex --depths 6,7 --seq 256 --tag seq256`: the same matched pair
at a 256-symbol context, where a term fits. 20,000 terms per split,
2,500 steps, 810,496 params, 122 s on one GPU. Reports:
`CORTEX-DEPTH-seq128.md` and `CORTEX-DEPTH-seq256.md`.

| window | depth | seen delta | unseen delta | cortex edge pay | byte edge pay | gap |
|---|---:|---:|---:|---:|---:|---:|
| 128 | 6 | -0.032 | -0.021 | +0.012 | +0.003 | +0.009 |
| 256 | 6 | -0.001 | +0.013 | +0.012 | +0.003 | +0.009 |
| 256 | 7 | +0.001 | +0.011 | +0.009 | +0.002 | +0.008 |

**Read.** At the 256-symbol window the gap **stops closing**: `+0.009` to
`+0.008` from depth 6 to 7, and the seen delta is parity. The `seq 128`
trend was partly context truncation, not structure. So at this grain the
concept-prediction hypothesis is negative: a byte model with enough
context matches the term model on the novel composition, and it is
slightly better on the unseen split. The next test is not more depth; it
is argument dependence, where the byte context cannot recognize the type
at all. `TODO8.compression.md` is the live fallback until that runs.

## 1.6 Copy arm in front of the trained trunk: measured 2026-11-27

`cml copycodec --run codec-byte-model-code-756752 --bytes 756752 --window
65536 --min-match 8 --cap 256`; `reports/COPY-MODEL.md`.

| arm | payload B | bpb |
|---|---:|---:|
| recorded byte model | 160,092 | 1.693 |
| LZ copy + trunk | 131,757 | **1.393** |

**Read.** Copy saves **17.7%** on the same 756 KB bytes: the Stage 3 gate
(5%) holds, and the payload also beats the 20M pointer codec's `1.488`. A
CPU preprocessor plus a 4.86M byte model beats a 20M neural codec on
payload: copy is the lever, not capacity. Two defects gave the earlier
losses (copied bytes coded as literals; a rolling position table out of
distribution at 6.94 bpb) and the fix is what pays. The cost is 2,431 s
for one encode pass, so the product path needs a faster literal step
(batched windows or a smaller trunk), not a bigger window. This is
orthogonal to the concept track's negative Cortex result.

## 2. Progress map

| area | current | next | gate |
|---|---|---|---|
| **Compression** | 1.393 copy + 4.86M byte; 1.488 / 1.518 at 20M | faster copy literals | 20/30/50% |
| **Concepts** | typed holes, exact realize; at `seq 256` the edge gap saturates at +0.008 | argument dependence, where bytes cannot infer the type | P5d |
| **T1 capability** | 0.680 / 1.698; frontier 0.675-0.720 | hold; do not regress | T1 frontier |
| **T3 edit** | local, -0.012 | delta codec | survivors |
| **T4 spec** | 4.650 accepted/call | hold | no regression |
| **Combination** | T1/T3/T4 separate | one T1-T4 arm | holds each gate |
| **RLFP** | untested | 2k-pair pilot | +5 points |
| **Scale** | 20M / 48 MB | 40M only after copy | equal compute |
| **Evaluation** | 756 KB in-domain; per-file fails | adaptive/delta pack | honest claim |
| **Product** | exact round-trip; copy exact | per-file/adaptive pack | shared bpb |
| **Infrastructure** | checkpoint/resume; CPU probes | caches, no timer | no OOM |

## 3. The minimum-resource ladder

Shrink for **search**, not for the final claim. A tiny model can rank
levers; only a 20M point can prove the 50% target.

| stage | model | train/test | pack | steps | cost | purpose |
|---|---|---|---|---|---|---|
| **0 analysis** | none | 756 KB | 256-2048 | 0 | **7 min CPU, done** | coverage +0.364, copy -15.1%, per-file fails |
| **1 tiny screen** | `d64 l2 h4 seq64`, ~0.2-0.4M | 1 MB / 512 KB | 64/128/256 | 300, batch 8 | 10 min, <1 GB | **skipped**: Stage 0 answers its pack/copy arms |
| **2 5M confirm** | `d256 l6 h8 seq128`, 5.2M | 8 MB / 512 KB | 256 | 1000, batch 16 | 15 min, <2 GB | the literal head only, if needed |
| **3 20M verify** | `d448 l8 h8 seq256`, 20.0M | 48 MB / 756 KB | 256/1024 | 8000, batch 32 | 60-90 min, <3 GB | 20/50% claim; copy proxy first |
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

**Result (2026-11-27).** Both levers hold: coverage `+0.364`, copy
`-15.1%`. The static unit arm loses to the byte arm, so the pack pays
only through the pointer. Details in §1.1 and the three reports.

## 5. Stage 1: tiny screen

**Skipped (2026-11-27).** Stage 0 measured its pack and copy arms at
756 KB with exact codecs; the only open question is the literal head,
which Stage 2 covers if the copy proxy plateaus. Keep this section for
the record.

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
**This is span selection, not language-model alignment:** SFT precedes
any DPO, and the pilot is labelled synthetic-preference, not human RLHF.

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

Done in `codec.py` (the `CopyArm` and `lz_parse` primitives) and
`stage0.py` (the three analyses and their renderers); the CLI is
`cml.py`. Next changes, in this order:

1. **done** `cml coverage`: parse + measured unit arm per pack;
   `--sizes 256,512,1024,2048`; cached lexicons, no GPU.
2. **done** `cml copyprobe`: LZ77 tokens with a slot/raw/length charge;
   exact round-trip; `--window/--min-match/--cap/--depth`.
3. **done** `cml perfile`: 1/4/10/100 KB table, the pack charged once
   per file, byte baseline on the same chunks.
4. **done** `cml composition`: templates with typed holes, recursive
   `realize`, the MDL library gate, and an unseen-edge split with a
   shape-matched `twin` control (`concepts.py`).
5. **done (negative)** `cml cortex`: a 22-symbol term model against a
   matched byte model; the byte arm wins the novel edge at this grain
   (`reports/CORTEX.md`).
6. **done (direction)** `cml cortex --depths 3,4,5`: the edge gap falls
   with depth, but the byte arm still pays less (`reports/CORTEX-DEPTH.md`).
7. `cml composition --hard`: argument dependence and depth 6 at
   `seq 256`, where a byte context cannot recompute the structure;
   rerun the matched pair.
8. `cml scale --copy`: put the measured LZ tokens in front of the
   frozen 20M model; charge only the residual literals; report the
   proxy saving on `reports/T2.md`. This is the next GPU lever.
9. `cml codec --arm byte-model --params/--d/--layers`: the 20M control
   (G1) must be a manifest, not an ad-hoc run.
10. `cml rlfp --pairs 2000 --run <5M>`: DPO vs SFT; `reports/RLFP.md`.
11. Product pack: an adaptive or delta pack per file; a static pack is
    measured as a loss at 1-100 KB and is not the answer.
12. Reports: `COVERAGE.md` (done), `COPY.md` (done), `PER-FILE.md`
    (done), `COMPOSITION.md` (done), `CORTEX.md` (done),
    `CORTEX-DEPTH.md` (done), `T2.md`, `RLFP.md`.

Commands after the build:

```bash
uv run cml coverage --sizes 256,512,1024,2048 --bytes 756752
uv run cml copyprobe --window 4096 --bytes 756752
uv run cml perfile --sizes 1024,4096,10240,102400
uv run cml composition --pairs 1000 --depth 3 --seed 0
uv run cml cortex --pairs 30000 --depth 4 --steps 3000
uv run cml cortex --depths 3,4,5,6 --pairs 20000 --steps 2500
uv run cml cortex --depths 6,7 --seq 256 --tag seq256
uv run cml copycodec --run codec-byte-model-code-756752 --bytes 756752
uv run cml scale --sizes 20m --copy --steps 8000 --seed 0
uv run cml codec --arm byte-model --params 20000000 --steps 8000
uv run cml rlfp --pairs 2000 --run concept-pointer-code-p5m-s8000
```

Tests: the `active` run id, byte-model params, copy round-trip and
determinism, per-file charge arithmetic, typed-hole arity, the MDL gate,
exact realization and term round-trip, the unseen-edge split, the term
stream round-trip, the matched pilot smoke, RLFP held-out split, and
exact codec round-trip. No coverage target.

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

Two tracks share one baseline; keep both.

**Concept track (architectural milestone).**
1. **Done.** Typed holes, recursive composition, exact realization, the
   MDL library gate, and the unseen-composition pilot (`§1.2`).
2. **Done, negative at depth 4.** The matched Cortex pilot (`§1.3`): the
   term model leads on seen, the byte arm wins the novel edge.
3. **Done, negative.** The depth sweep (`§1.4`) and its window check
   (`§1.5`): the apparent closure was context truncation. At `seq 256`
   the edge gap saturates at `+0.008`, so bytes match the term model
   once a term fits the context.
4. Next: argument dependence, where a byte context cannot infer the type
   of a hole from its bytes. Design the grammar so the type is not
   written, rerun the matched pair, and score exact type recovery.
5. Only then scale the Cortex and compare against a matched byte
   Transformer at 5M and 20M params.

**Compression track.**
6. **Done.** Stage 0: coverage, copy, per-file.
7. **Done.** The copy arm on a 4.86M byte model saves 17.7% (1.393) and
   beats the 20M pointer codec's 1.488 payload. Next: make the literal step
   fast (batched windows or a smaller trunk) and rerun the full round-trip.
8. Run the RLFP span pilot in parallel (CPU, 2 minutes).
9. Add the 20M byte control manifest before any 20% claim.
10. Decide: 20% claim, 50% program, or park compression.
11. Product pack: adaptive or delta, never a bigger static pack (G5).

## References

- `TODO7.md`: codec record and auditable product.
- `cml/README.md`: the 50% target and progress map.
- `cml/PLAN.md`: design and charge arithmetic.
- `reports/SCALE.md`, `reports/PORTFOLIO.md`, `reports/CONCEPT.md`,
  `reports/EDIT-CURVE.md`.
- `cml/cml.py`, `cml/pointer.py`, `cml/codec.py`.
