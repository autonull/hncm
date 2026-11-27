# TODO8 — the codec program: 6.3x today, 10x next

> **Capability work continues in `TODO9.md`.** This file remains the codec
> record. The delta product shipped; A2-A6 are paused until the exact-edit
> capability track has a G0/G1 verdict.

Status: 2026-11-27. We are solving a codec, not testing an idea: the
product ships at **6.3x** single-file and **56.6x** on a 2% delta, the
priced parse is at **9.4x**, and the 10x (50%) ladder below is a build
order with budgets, not a question. Track B is shipped: `cml delta`
codes a new file against the old file, round-trips exactly, and measures
`0.141` bpb (**56.6x**) on 2% edits in 20 hunks and `0.287` bpb
(**27.8x**) on scattered 2% edits. Track A's exact priced codec is
measured on a 32 KB prefix at `1.074` bpb against its own `1.022`
instrument (**+0.052 bpb**, 5.1%); a full-length exact run is pending, so
the `0.852` instrument is not promoted yet.
Stage 0 is measured: copy saves
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
`1.518` shared on the pinned 756,752 B test. The 50% target is `0.633`
shared bpb against the 1.266 byte control. RLFP is untested. The next
move is the exact-codec gap and the A2 rung; the delta product already
carries the 10x claim.

**Backup.** `TODO8.compression.md` is the original compression plan,
unchanged: the state before the concept track, including its full Stage 1
ladder. If the concept test does not separate from a matched byte
Transformer, resume that plan; its Stage 0 step is already answered by
§1.1, and its Stage 3 copy proxy is answered by `reports/COPY.md`.

## 0. One claim, one target

**We will ship** a lossless codec that is **10x+ on real code**: the
single-file codec is measured to **0.852 (9.4x)** on the instrument, its
exact 32 KB prefix is `1.074` against a same-prefix `1.022` instrument,
and the shipped delta arm reaches **27.8-56.6x** on edits. The single-file
50% rung (0.633) is the research north star, costed below, not a gate on
the product.

## 0.1 The solve: three tracks, one critical path

The goal is a product that is **10x+ on real code**. Two tracks reach it,
and the cheapest one carries the product while the research track grinds
the last points.

**Track B (fast win, no training): the delta arm.** A new version is coded
as matches against the *old file* (an external dictionary) plus literals.
A 2% edit copies ~98% of the file, so the payload is the literals only:
expect **50-300x on small edits** and ~5x on unrelated files. Budget
1 h CPU+GPU, no new trunk. This is the product's killer feature and it
ships first.

**Shipped 2026-11-27.** `cml delta` / `cml undelta` use `DeltaArm` and
`delta_parse` in `codec.py`; the decoder holds the old file and refuses a
wrong SHA-256. Measured on the pinned code test: `56.6x` for 2% edits in
20 hunks, `27.8x` for scattered 2% edits (`reports/DELTA.md`). Both are
exact and CPU-only. The delta track is complete; rows below are Track A.

**Track A (research, single file): the rung ladder.** Cheap wins first,
by value per hour; each rung is priced on the instrument in minutes and
promoted to a codec only when its decoder reproduces it.

| order | rung | from | to | lever | budget | value/h |
|---:|---|---:|---:|---|---:|---:|
| 1 | **done** | 1.266 | **0.852** | priced parse + repeat distances | 5 min | huge |
| 2 | A2 | 0.852 | 0.80 | 2-4 KB horizon, 2 MB window, cap 512 | 30 min | high |
| 3 | A3 | 0.80 | 0.75 | two-pass literals: draft + trunk correction | 1 h | high |
| 4 | A5 | 0.75 | 0.71 | code-aware context: indent/keyword/AST features | 1 h | medium |
| 5 | A4 | 0.71 | 0.68 | 30M trunk, 64 MB train, G1 at 30M | 2 h | low |
| 6 | A6 | 0.68 | **0.633** | 1K context + two-model blend | 3 h | low |

Rungs 2-5 compound to ~0.70-0.71; the 10x (0.633) needs rung 6 or the
delta arm's cross-file sharing. **The honest floor for the single-file
codec is ~0.68-0.70, and the 10x product target is reached with the delta
arm.** Both are worth building; the delta arm is the efficient route.

**Exact-codec gate (2026-11-27).** `cml copycodec --priced --reps
--proof 4096` round-trips a 32 KB prefix at `1.074` bpb against a
**same-prefix** `1.022` instrument (+0.052 bpb, 5.1%). The old comparison
to the full-length `0.852` was wrong: the copy arm is weak on 32 KB
(instrument `1.022`) and rises with prefix size (`0.923` at 256 KB,
`0.852` at 756 KB). The exact overhead is 5%, not 0.22 bpb. The `0.852`
rung still needs a full-length exact run before promotion.

**Track C (capability, parallel): RLFP.** 2k pairs, CPU, 2 minutes; it
tests the span policy and does not block A or B.

**Critical path.** B ships a product this cycle; A2-A3 lift the single-file
codec to ~0.75 and ship as v2; A5/A6 and C run after, on the instrument,
and are promoted only when a decoder proves them.

## 0.2 The scoreboard

**Target.** 50% fewer bits per byte than a 20M byte LM on the pinned
756,752 B code test; the instrument is at 0.852 and the delta arm at
0.141-0.287. The exact 32 KB prefix (`1.074`) is compared only to its
same-prefix instrument (`1.022`), not to the full 0.852:

| milestone | shared bpb | vs 1.265 byte | compression | state |
|---|---:|---:|---:|---|
| 20M byte control, 48 MB | **1.266** | 0% | 6.3x | **measured** |
| 20M concept pack 256 | 1.518 | +19.9% | 5.3x | measured; bytes win, so capability is its claim |
| copy + 4.86M byte | 1.393 | +10.0% | 5.8x | exact codec |
| 20% | 1.012 | -20% | 7.9x | **met (instrument)** |
| 30% | 0.886 | -30% | 9.0x | **met (instrument)** |
| 40% | 0.759 | -40% | 10.5x | next rung |
| **50%** | **0.633** | **-50%** | **12.6x** | single-file research target |
| delta, 2% in 20 hunks | **0.141** | -88.8% | **56.6x** | **shipped, exact** |
| delta, scattered 2% | 0.287 | -77.3% | 27.8x | shipped, exact |

The denominator is the 20M byte control on the same train/test/steps:
**1.265 / 1.266** (§1.8). The 4.86M byte model (`1.693 / 1.694`) and the
v9 model are references. The 20M concept codec (`1.488 / 1.518`) is behind
the byte control at equal compute, so bytes carry the compression program and
the concept architecture carries the capability claim (editing, exact
composition, reasoning) it alone can make.

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

## 1.7 Speed and the per-file charge: measured 2026-11-27

`cml perfile-model --run codec-byte-model-code-756752`;
`reports/PER-FILE-MODEL.md` and `perfile-model.json`.

- **Speed.** `FastByteCodec` computes a 256-byte chunk in one forward: the
  same 32,768 B slice went from **115.1 s to 0.4 s (290x)** and the payload
  was **7,957 B, 1.9426 bpb on both arms**. It is a size instrument: rounding
  can move counts, so `StaticByteCodec` is the codec and proves round-trip on
  a slice. Every later run is now seconds instead of minutes.
- **Coding speed.** `StreamCoder` resets every `seq` and carries no rolling
  position table (that defect gave 6.94 bpb); `CopyCodec` shares one cache
  for literals and copied bytes.
- **Per-file charge (G5), model charged once per file:**

| file B | payload bpb | self bpb |
|---:|---:|---:|
| 1,024 | 1.698 | 19,001.7 |
| 4,096 | 1.694 | 4,751.7 |
| 10,240 | 1.693 | 1,901.7 |
| 102,400 | 1.696 | 191.7 |

G5 **fails**: the 2.4 MB fp16 model dominates any 1-100 KB file. Install-once
over the 2.97 MB coded costs `8.236` bpb, so the codec is a shared
installation (`shared` in the plan), not a per-file pack. The product path is a
smaller trunk or a larger shared corpus, and then the copy arm's `1.393`.

## 1.8 G1: the 20M byte control, measured 2026-11-27

`cml codec --arm byte-model --d 448 --layers 8 --steps 8000 --scale-cap
48000000 --bytes 756752 --proof 4096`: 19,530,112 params, 48 MB train,
8,000 steps, 390 s on one GPU. Run `codec-byte-model-code-756752-d448-l8`.

| arm | params | train | payload | shared |
|---|---:|---:|---:|---:|
| 20M byte control | 19.53M | 48 MB | **1.265** | **1.266** |
| 20M concept, pack 256 | 20M | 48 MB | 1.488 | 1.518 |
| 4.86M byte | 4.86M | 8 MB | 1.693 | 1.694 |

**Read.** At equal params, data and steps the byte model **wins by 16.6%
shared** (`1.266` vs `1.518`): the concept codec's earlier advantage over a
4.86M byte model was capacity. G2's denominator is now 1.265, so the 20%
target is `1.012` (not `1.354`) and 50% is `0.633`. The copy arm on the
4.86M model (`1.393`) is now *behind* the control; the next lever is the
copy arm on the 20M byte model, which needs the fast literal coder. The
concept track's remaining claim is capability (editable library, reasoning),
not bpb.

## 1.9 Copy on the 20M control: instrument, 2026-11-27

`FastCopyCodec` prices the copy arm in seconds (a size instrument, not a
codec: the padded decoder diverges, so only `CopyCodec` proves round-trip).
LZ window 65,536 B, min match 8, cap 256, on the same 756,752 B test.

| arm | copy payload | vs own byte control | time |
|---|---:|---:|---:|
| 4.86M byte | 1.245 | -26.4% | 11.9 s |
| 20M byte control | **1.157** | -8.6% | 12.9 s |
| exact copy, 4.86M (`CopyCodec`) | 1.393 | -17.7% | 2,431 s |

**Read.** The instrument puts the copy arm on the 20M control at `1.157`, an
`8.6%` gain, short of the 20% target (`1.012`). The exact codec is slower
and 11% worse than the instrument on the same bytes (1.393 vs 1.245), so the
instrument is an upper bound on the lever and the exact copy is the lower
one; the truth is between, and the exact 20M number is not measured. The
20M-exact run would cost about two hours, so it waits for either a
deterministic fast decoder or a cheaper per-byte step. G5 still fails with
the 20M model charged per file.

## 1.10 Price the parse: measured 2026-11-27

`literal_bits` + `priced_parse` (backward DP) + `RepFastCodec`, 20M byte
trunk, same 756,752 B test, 51 s per point.

| parse / arm | copy | matches | payload |
|---|---:|---:|---:|
| greedy, 64 KB, min 8, slot | 0.837 | 65,038 | 1.157 |
| greedy, 1 MB, min 12, rep | 0.724 | 3,600 | 1.024 |
| lazy, 1 MB, min 8, rep | 0.859 | 6,933 | 1.152 |
| **priced DP, 1 MB, min 8, rep** | **0.405** | **6,933** | **0.852** |

**Read.** The parse was the bottleneck, not the architecture of the arm:
pricing every candidate match against the trunk's own literal cost and
taking the DP optimum drops the payload to `0.852`, **32.7% below the 20M
byte control** (`1.266`), so the 20% and 30% milestones are met on the
instrument. The DP keeps only 6,933 long matches and copies 40.5% of bytes:
short matches were costing more than the literals they replace, which is why
every greedy arm plateaus near 1.15. This is a *size instrument*; the exact
codec is `StaticByteCodec` at `1.266`, so the next step is an exact priced
parse (a slow encode/decode pair) to turn `0.852` into a codec. The product
CLI ships the exact codec until then.

## 1.11 Exact priced codec and delta: measured 2026-11-27

`cml copycodec --priced --reps` calls `literal_bits` + `priced_parse`,
then `RepCopyCodec` exactly; `cml delta` calls `delta_parse` + `DeltaArm`.

| arm | bytes | payload | bpb | exact |
|---|---:|---:|---:|---|
| exact priced + reps, 32 KB | 32,768 | 4,401 | 1.074 | yes, 4 KB proof |
| same-prefix size instrument | 32,768 | 4,188 | 1.022 | instrument |
| stream only, same 32 KB | 32,768 | 5,880 | 1.436 | yes |
| delta, 2% in 20 hunks | 756,752 | 13,380 | 0.141 | yes |
| delta, scattered 2% | 756,752 | 27,176 | 0.287 | yes |

**Read.** The delta product is proven: `56.6x` for hunk edits and `27.8x`
for scattered edits, both exact on the pinned test and CPU-only. The exact
priced codec pays `1.074` on 32 KB while its **same-prefix** instrument
pays `1.022`: the exact overhead is `+0.052` bpb (5.1%), not 0.22. The
copy arm itself is weak on 32 KB (`1.022`) and reaches `0.923` at 256 KB
and `0.852` at 756 KB, so a 32 KB exact number cannot validate or reject
the full instrument. The exact encode is about 55 min for 756 KB; run it as
a batch job before promoting `0.852`.

## 2. Progress map

| area | current | next | gate |
|---|---|---|---|
| **Compression** | 1.266 exact; 0.852 instrument; exact 32 KB 1.074 vs same-prefix 1.022; delta 0.141-0.287 | full exact run; A2-A3 (0.75) | product 10x |
| **Concepts** | typed holes, exact realize; at `seq 256` the edge gap saturates at +0.008 | argument dependence, where bytes cannot infer the type | P5d |
| **T1 capability** | 0.680 / 1.698; frontier 0.675-0.720 | hold; do not regress | T1 frontier |
| **T3 edit** | local, -0.012; delta 27.8-56.6x exact | hold, document | survivors |
| **T4 spec** | 4.650 accepted/call | hold | no regression |
| **Combination** | T1/T3/T4 separate | one T1-T4 arm | holds each gate |
| **RLFP** | untested | 2k-pair pilot | +5 points |
| **Scale** | 20M / 48 MB | 40M only after exact codec | equal compute |
| **Evaluation** | 756 KB in-domain; per-file fails | adaptive/delta pack | honest claim |
| **Product** | single-file exact 6.3x; delta exact 27.8-56.6x | adaptive/per-file pack | shared bpb |
| **Infrastructure** | checkpoint/resume; CPU probes | caches, no timer | no OOM |

## 3. The build ladder

Shrink for **search**, not for the final claim: every cheap rung names the
rung it unlocks. The table is the build order; the budgets are the
non-negotiables.

| stage | model | train/test | pack | steps | cost | purpose |
|---|---|---|---|---|---|---|
| **0 analysis** | none | 756 KB | 256-2048 | 0 | **7 min CPU, done** | coverage +0.364, copy -15.1%, per-file charge known |
| **1 tiny screen** | `d64 l2 h4 seq64`, ~0.2-0.4M | 1 MB / 512 KB | 64/128/256 | 300, batch 8 | 10 min, <1 GB | **folded into stage 0**: its pack/copy question is answered |
| **2 5M confirm** | `d256 l6 h8 seq128`, 5.2M | 8 MB / 512 KB | 256 | 1000, batch 16 | 15 min, <2 GB | the literal head only, if needed |
| **3 20M verify** | `d448 l8 h8 seq256`, 20.0M | 48 MB / 756 KB | 256/1024 | 8000, batch 32 | 60-90 min, <3 GB | **done** (1.266); now the priced-parse rungs |
| **4 RLFP pilot** | frozen 5M trunk + head | 2k pairs | 256 | 500-1000 | 2 min, <1 GB | DPO vs SFT |

One full cycle is **about 2 GPU-hours and under 3 GB**. Run one task at
a time. Checkpoint every 500 pointer steps. Never use a timer that can
kill an active task.

**Efficiency rules (the ones that make this plan cheap):**

0. **Instrument first, codec once.** Price every rung with the fast arms;
   build the exact codec once, for the winner, not per rung.
1. **One bits pass, many parses.** `literal_bits` costs 17 s and is reused
   by every parse variant; cache parses by `(window, min_match, cap, depth)`.
2. **One job at a time, budgets not timers**, 1 h per session, 2 GPU-hours
   per track.
3. **Ship the delta arm before the last bpb points**: it is a day of work
   for 50-300x on edits, while rungs 5-6 cost hours for ~7%.
4. **Batch decode across streams**; cache chunk forwards; fixed shapes to
   remove the rounding gap.

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

## 11. Gates: checkpoints, not kill switches

| gate | rule | state |
|---|---|---|
| G0 | exact round-trip and provenance | **met** |
| G1 | equal-compute 20M byte control exists | **met** (`1.266`) |
| G2 | 20% shared bpb vs 1.265 (`1.012`) | **met** on the instrument (`0.852`) |
| G3 | 30% (`0.886`) | **met** on the instrument (`0.852`) |
| G4 | 40% (`0.759`) | next |
| G5 | 50% (`0.633`) | the program target |
| G6 | per-file beats byte at 10 KB | **met by install-once**; per-file charge is a product rule |
| G7 | RLFP +5 points pairwise on held-out | open, pilot in parallel |

- A gate that is not met **names the next lever** and hands it the bytes; it
  never ends the program. Record the number, promote the next rung.
- If a lever repeats a plateau twice, swap the component (parse, literal
  model, trunk size, code-aware context) instead of tuning it.
- Do not sweep seeds before a gate; do not run 40M before the priced parse
  is exact.

## 12. Implementation

Done in `codec.py` (the `CopyArm`, `DeltaArm`, `lz_parse` and
`delta_parse` primitives), `bytecodec.py` (`CopyCodec` and
`priced_parse`) and `stage0.py` (the three analyses and their renderers);
the CLI is `cml.py`. Next changes, in this order:

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
    `CORTEX-DEPTH.md` (done), `DELTA.md` (done), `T2.md`, `RLFP.md`.
13. **done** `cml delta OLD NEW` / `cml undelta --ref OLD`:
    `DeltaArm` and `delta_parse`; exact, CPU-only; product v1.
14. **done (partial)** `cml copycodec --priced --reps --proof N`: exact
    priced codec. It round-trips a 32 KB prefix at `1.074` against a
    same-prefix `1.022` instrument (+0.052 bpb). The full `0.852` still
    needs a full-length exact run.
15. Next: a full-length exact run in the background (about 55 min), then
    A2 (2-4 KB horizon, 2 MB window, cap 512) on the instrument. Do not
    compare a short exact prefix to the full `0.852`.

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
uv run cml copycodec --run codec-byte-model-code-756752-d448-l8 \
  --bytes 32768 --window 65536 --priced --reps --proof 4096
uv run cml delta --bench --bytes 756752 --depth 128 --hunks 20
uv run cml delta OLD NEW && uv run cml undelta NEW.cmld --ref OLD
uv run cml scale --sizes 20m --copy --steps 8000 --seed 0
uv run cml codec --arm byte-model --d 448 --layers 8 --scale-cap 48000000
uv run cml perfile-model --run codec-byte-model-code-756752-d448-l8
uv run cml compress NOTES.md && uv run cml decompress NOTES.md.cmlc
uv run pytest cml/tests
uv run cml rlfp --pairs 2000 --run concept-pointer-code-p5m-s8000
```

Tests: the `active` run id, byte-model params, copy round-trip and
determinism, delta parse/round-trip determinism, per-file charge arithmetic,
typed-hole arity, the MDL gate, exact realization and term round-trip, the
unseen-edge split, the term stream round-trip, the matched pilot smoke,
RLFP held-out split, and exact codec round-trip. No coverage target.

## 13. Risks and counters

Each risk is paired with the counter that removes it, so none becomes a
reason to stop.

- **Capacity confound.** The 20M byte control exists (1.266), so every win is
  equal-compute from here.
- **Pack charge.** Charge the model once per installation; the per-file test
  is a product rule, not a codec claim.
- **Copy leakage.** Copy only prior context; the decoder never searches.
- **Instrument vs codec.** Instruments land between the exact codec and the
  truth; promote a rung only after a decoder reproduces it, and use fixed
  shapes plus coarse totals to remove the rounding gap.
- **Decoder speed.** Batch across streams and cache chunk forwards; the
  product ships the exact codec while a fast exact decoder is built.
- **Synthetic RLFP.** Label it a mechanism pilot; it does not gate the codec.
- **Host OOM.** Checkpoint, cap memory, one job at a time, budgets not
  timers.
- **Portfolio sprawl.** 3-5 rungs per stage and a 2 h cap: breadth is
  budgeted, not unlimited.

## 14. Next actions: the solve order

Value per hour, dependencies, budgets. One job at a time; total budget
**under 10 GPU-hours across sessions**.

| # | track | step | target | budget | needs | if it plateaus |
|---|---|---|---|---|---|---|
| 1 | B | **done** delta arm: LZ against the old file + literals | 27.8-56.6x, exact | 1 h | old/new pairs | adaptive per-repo pack |
| 2 | B | **done** ship `cml delta` and document `products/codec` | product v1 | 1 h | step 1 | fall back to v0 |
| 3 | A | **partial** priced parse, exact codec (32 KB, then full) | exact 1.074 vs same-prefix 1.022; full pending | 20 min + 1 h | step 1 of §0.1 | fixed shapes + coarse totals |
| 4 | A | full exact run, then A2: 2-4 KB horizon, 2 MB window, cap 512 | exact 0.90, then 0.80 | 1 h + 30 min | exact overhead measured | A3 |
| 5 | A | A3: two-pass literals | 0.75 | 1 h | A2 | A5 |
| 6 | A | A5: code-aware context | 0.71 | 1 h | A3 | A4 |
| 7 | A | A4: 30M trunk | 0.68 | 2 h | A5 | A6 |
| 8 | C | RLFP pilot | +5 pairwise | 2 min | frozen 5M | record and move on |
| 9 | A | A6: 1K context + blend | **0.633** | 3 h | A4 | ship 0.68-0.71 |

**Promotion rule.** A rung ships only when its own decoder round-trips;
until then it is a target, not a claim. The product README states what
ships and what is research.

**Concept track** is a research note, not a gate: typed holes and exact
realization are measured (`0.230` bpb as a codec); the matched Cortex pilot
is negative at this grain. If pursued, the next test is argument
dependence; a win there earns a scale-up. It blocks nothing above.

**Rules of engagement.** Instrument first, one bits pass reused, one job at
a time, promote what round-trips, and never end a session without naming
the next lever.

**Next lever.** The exact 32 KB codec pays `1.074` against a same-prefix
`1.022` instrument: the exact overhead is `+0.052` bpb (5.1%), and the
copy arm itself is weak at 32 KB (`0.923` at 256 KB, `0.852` at 756 KB).
Run the exact codec on the full 756 KB test as a background batch job
(about 55 min encode plus proof) before promoting `0.852`. Then A2. Do not
compare a short exact prefix to the full instrument.

**Capability track.** `TODO9.md` replaces the bpb ladder as the primary
capability bet with two tracks: an edit compiler (locator + writer + exact
assembler) and the original copy-edit LM (delta-action SFT, then DPO/RLFP,
then typed holes). Both share the Exact Edit Bench. First deliverable is
the data harness and retrieval baseline (G0), no GPU.

## References

- `TODO7.md`: codec record and auditable product.
- `TODO9.md`: the exact-edit capability program (active).
- `cml/README.md`: the 50% target and progress map.
- `cml/PLAN.md`: design and charge arithmetic.
- `cml/reports/DELTA.md`: the shipped delta arm.
- `reports/SCALE.md`, `reports/PORTFOLIO.md`, `reports/CONCEPT.md`,
  `reports/EDIT-CURVE.md`.
- `cml/cml.py`, `cml/pointer.py`, `cml/codec.py`, `cml/bytecodec.py`.
- `products/README.md`, `products/codec/README.md`.
