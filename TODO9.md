# TODO9 — the better-LM program: exact memory, one benchmark

Status: 2026-11-27. The goal is a **better language model at equal
compute**: a frozen base plus an exact, MDL-priced, installable span
memory. The delta codec and the Exact Edit Bench are the substrate. E1/C1
are applications of the memory, not the claim.

## Status 2026-11-27: Track A paused on bits; exact codecs shipped

> Continuation and the working plan: `TODO10.md` (2026-11-28). This file
> is the research record; TODO10 has the current shippable products, the
> ByteCopy LM experiment, and the next queues.

| work | state | artifact |
|---|---|---|
| **Track A better-LM** | **active** | A0.0+A0.1 pass; **A0.2 bits fail**; A1 blocked |
| data + benchmark | **done** | `cml/editdata.py`; 5 repos, 12,811 edits, repo split |
| retrieval G0 | **done** | `cml/reports/EDIT-LM.md`; exact `0.000` on 1,648 test commits |
| E1 locator | **paused after GE1 fail** | `cml/reports/EDIT-LOCATOR.md` |
| C1 copy-SFT | **paused after GC1 fail** | `cml/reports/EDIT-COPY.md` |
| delta product | **shipped** | `products/delta/README.md`; **56.6x** hunk edits, exact |
| memory product | **shipped** | `products/memory/README.md`; **40.0x** over 1,823 versions, exact |
| repo-memory product | **shipped** | `products/repo-memory/README.md`; **20.7x** on 80 files, exact `path:line` search |
| agent memory | **shipped** | `products/memory-mcp/README.md`; 5 exact MCP tools |
| exact tools | **shipped** | patch-guard, repo-delta (85.8x), log-delta (27.3x) |
| ByteCopy LM | **prototype** | `products/copy-lm/README.md`; exact copies, joint gate next |

**Banked products.** `cml delta` is exact at `56.6x` on 2% hunk edits and
`27.8x` scattered. `cml chain` is exact at `40.0x` over five real files
(90.1 MB → 2.26 MB), `6.5x` better than gzip. These are evidence and
fallback for the architecture, not the goal. Tests: `55 passed`.

**G0 recorded: pass.** Retrieval on 1,648 held-out `tiangolo/fastapi`
commits: exact `0.000`, valid `0.027`, output `0.991x`. Train `5,672`
(`httpx`, `flask`, `requests`); val `5,491` (`pytest`). The cross-repo
transplant cannot see the query repository's history; that is the point of
G0. Artifacts: manifest `cml/runs/edit-manifest.json` (ignored), blob
cache `~/.cache/hncm/edit` (308 MB). Reuse them.

**A0.0 recorded 2026-11-27: pass.** On the frozen 19.5M `d448-l8`
trunk (`seq=256`) and the 756,752 B held-out code, a 1,024-unit exact-span
pack gives free-id payload **0.684** bpb plus **0.120** bpb pack charge =
**0.804** bpb, against the LZ copy arm `0.852`; 3.19x fewer trunk calls and
100% provenance. 256 units fails (1.052); 4,096 ties (0.845; the 0.482 pack
charge cancels the coverage gain). The margin is an upper bound: ids are
free and the literal bits are a size instrument. Artifact:
`cml/reports/MEMORY-PROBE.md`; `cml memory-probe --oracle`. Next is A0.1.

**A0.1 recorded 2026-11-27: pass.** The same 48M-train code trunk at
`seq=2048` (20.3M params, 8,000 steps) codes the same 756,752 B test at
**0.974** bpb, against `1.265` for the 256-byte trunk: **−0.291 bpb
(−23%)**, proof round-trip OK on 4,096 B. The 2K base also sharpens the
oracle: the 1,024-unit free-id pack totals **0.643** bpb, `0.209` below
the LZ copy arm, at 3.19x fewer calls (`MEMORY-PROBE-2k.md`). A 4,096-unit
pack still pays 0.482 bpb of charge, so pack induction (A0.2) must beat
the 1,024 point on total bits, not on coverage. The run exposed and fixed
a real bug: `StaticByteCodec._feed` built an autograd graph, so the KV
cache leaked at long context. Next is A0.2.

**A0.2 recorded 2026-11-27: bits fail; speed/provenance pass.** On the
`-q2048` base, a 1,024-unit pack from the 48M train (15,429 B, 0.163 bpb
charge) with the fitted 3-parameter gate and a priced id cannot beat the
stream. Free-id gate: **0.912** bpb (only +0.058 over the 0.970 stream,
below the 0.1 gate); count prior **2.768**; fixed index **2.895**; tied
softmax **2.906**. The gate costs ~0.1 bpb and the id ~9-10 bits/copy; the
pack's chosen-unit counts are nearly flat, and the tied softmax is uniform
(+0.004 vs counts): the last-state content embedding does not name the
unit. A0b: count-id shuffle **+0.414** bpb (identity is load-bearing); the
softmax shuffle is +0.030 (that code ignores ids). Speed **1.84x** at 100%
provenance. Artifact `cml/reports/MEMORY-GATE.md`.

**A0.2 decision and revival condition.** The blocker is the id price plus
the per-file pack charge, not the copy decision. Do not start A1 for bits;
record the negative and ship the codecs. The bits claim revives only on
(a) a stream large enough to amortize the pack (≥10 MB with a ≤64 KB pack)
and (b) a trained pointer id at ≤2 bits/copy, or spans long enough that
copies/B falls ~5x. The 756,752 B single-file regime cannot pay for
either. The speed/provenance path stays alive as a product capability.

**Reframe: memory as exact context, not a bitstream.** A0.2 failed because
it asked the model to *code* span ids. `repo-memory` ships the same
substrate as an exact, searchable index: install, `path:line` search,
SHA-verified read, detach. The model does not pay an id to recall a span;
it queries the index and conditions on the exact bytes. The better-LM
question then becomes: at equal compute, does a frozen base plus an exact
queryable repo memory beat a dense base on repo tasks (exact context,
answer, patch), not "does it win bpb". The next product is `memory-mcp`,
which exposes the index to an agent. That is the direct path from the
shipped memory to the architecture claim.

**E1/C1 are paused, not closed.** Their first runs produced no exact file,
but the controls also exposed likely implementation and regime defects:

- E1: zeroed-file baseline loss matched the locator at every log point; a
  50-file/5,000-step control still emitted no spans; line pooling scored
  `0.055`. The next tests are oracle-message (does the pipeline work at
  all?), balanced train/eval priors, and candidate/pointer localization.
- C1: `valid = 0.960` but `exact = 0`; `exact@8 = 0`. The grammar is
  learned, the content is not. The next tests are one-window exact overfit,
  a text/pointer action grammar, and beam-plus-verifier.

Do not run the same formulation longer. Each revival experiment below has
a cheap correctness control, a measurable signal, and a kill rule. A
negative result that passes its control is evidence; one that fails its
control is a bug.

## The claim: a better language model

At equal parameters, FLOPs, and charged bytes, the memory-augmented LM is a
**better LM** than:

- a byte/patch transformer at equal FLOPs: lower bits/byte on code,
  JSON, and logs;
- the same model without memory: same or better quality, faster structured
  decode;
- LoRA and kNN at equal charged bytes: better adaptation per byte;
- a dense model on knowledge operations: exact provenance, exact deletion,
  install/detach/diff, and typed-hole composition.

This is not a chat-SOTA claim. It is a better-LM-at-equal-compute claim,
measured in bits, tokens per second, adaptation bytes, and exact
capabilities. Products are the memory substrate and the fallback, not the
main line.

## Ambition: the memory-augmented LM

`hncm.r1-r7.md` converge on one thesis, stated seven ways:

> A frozen/amortized neural base plus an exact, MDL-priced, transmissible,
> editable memory of spans, with a copy-or-compute gate. Knowledge is data;
> compute routes around known spans; every emitted span is verifiable.

That is a better-LM claim, not a codec claim. It is bigger than exact edit
prediction: the memory is the adapter, the tokenizer, the drafter, and the
audit trail at once. Products are evidence; the architecture is the goal.

| plan | name | core claim | decisive test |
|---|---|---|---|
| r1 | frozen base + charged span memory | adapter = compressor = LoRA competitor = drafter | equal-charged-bytes probe vs kNN and LoRA |
| r2 | C2LM | route each span between zero-compute COPY and neural COMPUTE | G1 quality, G2 speed |
| r3 | PhraseLM | installable lexicon; lexicon-agnostic pretraining | held-out install ≥ 0.15 bpb |
| r4 | Concept-LM | the vocabulary is data; neural prices + MDL | P0 ≤ 1.83 bpb |
| r5 | LedgerLM | persistent ledger with an admission market | S1 vs byte/BPE/BLT at equal FLOPs |
| r6 | CMT | concept pack + tied embeddings + provenance | v0: structured ≥ 1.5x at exact quality |
| r7 | CoLib | exact concepts with typed holes under a cortex | G0 parity, G2 shuffle, G4 composition |

**What we already own.** Exact delta (`56.6x`) and history chains (`40x`);
a trained 20M byte trunk; pointer/concept code; the Exact Edit Bench; the
round-tripping judge. These are the memory substrate and the evaluation.

**What is missing.** Content-addressed span embeddings tied to the
backbone; a copy-or-compute gate; pack operations (attach/detach/diff/rm);
provenance tags; a two-clock concept input; typed holes.

**What we got wrong.** E1/C1 tried to predict an unknown edit from a commit
message. The r-series says the memory should *hold* known spans and route
around them; predicting the edit is a narrow application, not the
architecture. Keep E1/C1 as diagnostics and applications, not as the main
line.

## Track A: memory-augmented LM

A0-P and A0.0 are inference-only and reuse the codecs, the trunk, and the
corpus. A0.1 trains a 2K-context base only if A0.0 passes.

| id | experiment | cost | gate |
|---|---|---|---|
| **A0** | on the 2K base: induce a 4k-64k exact-span pack; tied embeddings from the last-state readout; fit a 3-parameter gate; provenance tags (256 B oracle is A0.0) | **minimal A0.2 failed on bits**; speed 1.84x | text ≤ 0.01 bpb loss; code/JSON ≥ 0.1 bpb win (**fail**); structured ≥ 1.5x decode at exact quality (**pass**); pack ≤ 1 MB (**pass**); provenance round-trip 100% (**pass**); A0b shuffle ≥ 0.2 (**pass**) |
| **A0b** | shuffle control: permute concept ids, keep embeddings | hours | **done, pass**: count-id shuffle costs +0.414 bpb; identity is load-bearing |
| **A1** | equal-charged-bytes: pack vs kNN vs LoRA at 64k/256k/1MB | 1 week | **blocked**: A0.2 bits fail; do not start |
| **A2** | train the head: mixture NLL + charge on DP parses | 1-2 weeks | **gated on a ≥10 MB stream or long spans**; tied head must reach ≤2 bits/copy |
| **A3** | two-clock / CoLib: concept tokens, holes, planning | 1 GPU-month | G0 parity, G2 shuffle, G4 composition ≥ +10 points |

### Efficiency audit: three fixes

1. **Base context.** The frozen 20M trunk had `seq = 256` (run
   `codec-byte-model-code-756752-d448-l8`); it could not condition on a
   4k-64k pack. A0.0 measured the 256-byte oracle (pass) and A0.1 trained
   the `seq=2048` 20M trunk (0.974 bpb, −23%). A0.2 used `-q2048`; the
   remaining blocker is the id price and pack charge, not the base.
2. **Metric.** bpb alone duplicates the LZ copy arm (`0.852`) and the 40x
   chain. The unique claim is install, exact delete, and provenance at equal
   quality. Gate on bits-to-adapt and exact deletion first; bpb is a safety
   check.
3. **Time to signal.** Split A0: a 2-hour oracle, a 1-2 hour base, then
   1-day probes. No 3-5 day run before the oracle.

| id | probe | cost | stop rule |
|---|---|---|---|
| **A0-P** | pack ops on the exact chain: attach/detach/rm/diff plus provenance; exact delete | 1-2 CPU-days | cannot delete exactly → no product |
| **A0.0** | oracle copy inside 256 B: free pack IDs, charged pack bytes; compare to the LZ copy arm | 2 h CPU | **done, pass**: 0.804 vs 0.852 bpb, 3.19x calls |
| **A0.0b** | if bits fail: speed/provenance upper bound on the same 256 B probe | 2 h CPU | not needed; A0.0 passed on bits |
| **A0.1** | 2K-context 20M trunk on code, only if A0.0 passes | 1-2 GPU-h | **done, pass**: 0.974 vs 1.265 bpb (−23%) |
| **A0.2** | tied span embeddings + 3-parameter gate on the 2K base | 1 day | no signal over counts → stop |
| **A0.3** | provenance, copy acceptance, structured decode on the 2K base | 1 day | gates in the A0 table |

A0-P is a supporting product, not the main gate: it needs no base and
no training. Concretely: `cml chain attach/detach/diff/rm --index`, rebuild
the chain after a delete, verify every remaining version, and record the
new ratio and removed provenance. A0.0 decides whether the LM integration
is worth a base. Do not train A0.1 until A0.0 passes.

**Decision rule.** If A0 or A0b fails, record the negative, ship the codecs,
and do not claim an architecture. If A0 passes, A1 decides whether the
memory beats LoRA/kNN at equal bytes. If A1 passes, Track A becomes the
main research line and E1/C1 become copy-path applications.

**Budget.** A0-P/A0.0 are inference-only; A0.1 is a 1-2 GPU-hour base
extension. A1 is one week.
A2/A3 need a separate 1 GPU-month; the current 8 GPU-hour budget does not
cover them. Do not start A2 before A0 and A1 pass.

## 0. One claim, three formulations

**Track A (main).** At equal parameters, FLOPs, and charged bytes, a frozen
base plus an exact span memory is a better LM: lower bits on code/JSON/logs,
faster structured decode, better adaptation per byte than LoRA/kNN, and
exact provenance/deletion.

**Track E (edit compiler, application).** A small model that predicts
*where* a file changed and *what* replaces it beats a larger full-file byte
LM on real commits, at a small fraction of the output bytes.

**Track C (copy-edit LM, application).** A small LM that emits an exact
delta program beats a larger full-file byte LM on real commits, at 5-50x
fewer output tokens.

All are verified by exact decode, exact match, or test pass. Track A is the
better-LM claim; E and C are applications of the same memory.

## 1. Why this is the better bet

- **The memory is exact and cheap.** Delta is 56.6x and chains 40x on real
  code, byte-exact. Exact spans are a safe substrate: a wrong copy is
  caught by a digest, not averaged away.
- **The base is amortized.** Intelligence lives in frozen weights; the
  per-domain pack is kilobytes. Adaptation is install, not training.
- **Copy/compute routes around known spans.** Known bytes cost a copy
  decision, not per-byte FLOPs; novel bytes get the neural path. That is
  where better bits and speed at equal compute come from.
- **The claim is falsifiable at equal compute.** Byte/BPE/patch baselines
  and LoRA/kNN at equal charged bytes are the controls. No vague "feels
  better" claim.
- **E1/C1 are applications.** Predict-where-it-changed and emit-a-delta
  are ways to use the memory, not the memory thesis.

## 2. Benchmarks: better-LM and edit application

**Track A (better LM).** At equal parameters, FLOPs, and charged bytes:

- bits/byte on code, JSON, and logs;
- structured tokens/s at equal quality;
- adaptation per charged byte vs LoRA and kNN;
- provenance round-trip and exact deletion.

**Edit application.** Given `old`, the commit message, and `new`, predict
`new` exactly. This is the **Exact Edit Bench**: real commits, split by
repository, exact-match and test-pass labels. Track E/C use it; it is not
the Track A claim.

## 3. Data

Cache at `~/.cache/hncm/edit`. Record a manifest with repo, parent sha,
commit sha, path, message, file sizes, diff size, and split.

Pinned source set (permissive, test-rich):

- `psf/requests`
- `pallets/flask`
- `pytest-dev/pytest`
- `encode/httpx`
- `tiangolo/fastapi`

Filters:

- non-merge commit, one text file changed;
- file ≤ 64 KB, diff ≤ 20% of the file (changed lines / new lines);
- non-empty message, no generated/vendor files;
- parent blob and commit blob both available.

Split by **repository**, not commit:

| split | repos | commits (mined) |
|---|---|---:|
| train | 3 | 5,672 |
| val | 1 | 5,491 |
| test | 1 | 1,648 |

Store blobs by SHA. Deduplicate near-identical edits by content hash.
Record repo SHAs and mining config; a run without provenance is not
evidence.

## 4. Track E: the edit compiler

```
old file + message
      |
      v
[ edit locator ]  -> changed spans
      |
      v
[ edit writer ]   -> replacement text / typed holes
      |
      v
[ exact assembler ] -> copy unchanged bytes, splice changes
      |
      v
new file, verified byte-for-byte
```

Phase 1 copies from `old` only. Phase 2 may copy from already-emitted
output. The locator is a span-classification model; the writer is a small
autoregressive model over replacement bytes or typed holes.

Why this is efficient:

- the locator is token classification, not autoregressive action decoding;
- the writer generates only changed bytes;
- the assembler is exact and deterministic;
- synthetic edits give controllable pretraining;
- the locator is a standalone product.

## 5. Track C: the original copy-edit LM

The model emits a serialized delta action sequence:

```
COPY(offset, length)   # exact bytes from old
LIT(span)              # new bytes
HOLE(type)             # typed hole, later
END
```

The target is `delta_parse(new, old)`, serialized as actions. Restrict
phase 1 to copies from `old`; output copies are phase 2. Decode with the
exact delta decoder and compare to `new`.

This track is not a fallback. It is the original idea: a single
autoregressive policy over edits, with SFT, DPO/RLFP, and typed holes.

## 6. Arms

### Track A: the better-LM claim

| arm | what it does | role | cost |
|---|---|---|---|
| **A0.0** oracle | free pack IDs inside 256 B; charged pack bytes | decides the LM path | 2 h CPU |
| **A0.1** base | 2K-context 20M byte trunk | minimum base for a pack | 1-2 GPU-h |
| **A0.2** gate | tied span embeddings + copy/compute gate | first real better-LM test | 1 day |
| **A1** equal bytes | pack vs kNN vs LoRA at equal charged bytes | adaptation claim | 1 week |
| **A3** two-clock | concept tokens, holes, planning | capability claim | GPU-month |

### Edit applications (Track E/C)

| arm | what it does | role | cost |
|---|---|---|---:|
| **R** retrieval | nearest old/new pair or patch by message; apply exactly | cheap baseline; kills a fake win | 30 min CPU |
| **F** full-file | same trunk generates the whole new file | compute-matched baseline | 1-2 GPU-h |
| **E1** locator | old + message -> changed spans | application gate | 1 GPU-h |
| **E2** locator + writer | spans + replacement bytes -> exact assembler | application product | 1-2 GPU-h |
| **C1** copy-SFT | SFT on serialized delta actions | application arm | 1-2 GPU-h |
| **D** copy-DPO | SFT + DPO on exact-decode preferences | preference step | 1 GPU-h |
| **G** copy-RLFP | GRPO/REINFORCE if DPO plateaus | fallback | 1 GPU-h |
| **H** holes-SFT | copy + typed holes for new code | compositional application | 1 GPU-h |
| **V** verifier rerank | sample K, decode each, keep exact/test-pass | test-time compute | 2-4x inference |

Track A order: A0.0 -> A0.1 -> A0.2 -> A1 -> A3. Do not start A0.1 until
A0.0 passes. Edit order: R -> F -> E1 -> E2 -> C1 -> D -> G -> H -> V; run
it only after Track A has a signal or as a cheap application.

## 7. Metrics and pre-registered gates

**Track A primary (the better-LM claim):**

- **bits/byte at equal parameters and FLOPs** vs byte/BPE/patch baselines;
- **structured tokens/s at equal quality**;
- **adaptation per charged byte** vs LoRA and kNN;
- **provenance round-trip**, **exact deletion**, and **shuffle cost**
  (A0b must cost ≥0.2 bpb).

**Edit application metrics:**

- **exact match**: assembled file equals `new` byte-for-byte;
- **test pass** on the subset with runnable tests;
- **valid decode**: actions assemble without error;
- **output bytes**: payload bytes divided by new-file bytes;
- **locator span F1** and **location exact** (Track E);
- **latency** and **test-time compute**.

Secondary: edit distance, copy-span F1, instruction counterfactual accuracy
(same old file, different messages).

Gates:

| gate | test | promote if |
|---|---|---|
| **G0** | retrieval on 500 held-out commits | retrieval exact < 20%; else harden |
| **GE1** | locator vs message-only baseline | span F1 +20 points |
| **GE2** | locator + writer vs full-file | +10 exact or +20 test pass, ≤20% output bytes |
| **GC1** | copy-SFT vs full-file | +10 exact or +20 test pass, ≤5% output bytes |
| **G3** | DPO/RLFP on the best SFT arm | +3 exact over SFT; else drop |
| **G4** | holes-SFT vs best arm | +5 points on held-out compositions |
| **G5** | verifier rerank | +2 points at ≤4x test-time compute |
| **G6** | head-to-head E2 vs C1 | pick the winner; record the loser |

G0 recorded 2026-11-27: exact `0.000` on 1,648 test commits, pass. Next is
GE1. Test-pass labels are not mined yet; exact match is the only label in
the manifest. Add test execution only when a gate needs it, and keep the
same split.

GE1 recorded 2026-11-27: byte F1 `0.000` versus the message-only baseline
`0.000` on 500 val files, fail. A 50-file, 5,000-step control also scored
`0.000`; line-level pooling scored `0.055` on 50 files. The locator emits
no usable spans and the loss is unstable; see `cml/reports/EDIT-LOCATOR.md`.
Do not run E2 on this locator.

GC1 recorded 2026-11-27: exact `0.000`, exact@8 `0.000`, valid `0.960`,
output `0.057x` on 50 val files, fail. Exact zero makes a +10-point lift
impossible; see `cml/reports/EDIT-COPY.md`. Do not start DPO or holes from
this arm until C1-R0..R2 pass.

A failed first run with failed controls is a bug. A failed first run with
passed controls is evidence. Neither stops the architecture track; both choose
the next experiment.

## 8. Implementation order

Track A leads. The edit applications come after the memory has a signal.

1. **Data + benchmark** (done): `cml/editdata.py`, repo split, manifest.
2. **Exact memory substrate** (done): delta, chain, 20M byte trunk.
3. **A0.0 oracle** (done): `cml memory-probe --oracle` on the 256 B
   trunk; free pack IDs, charged pack bytes, versus the LZ copy arm.
   Pass at 1,024 units: **0.804** vs **0.852** bpb.
4. **A0.1 2K base** (done): extend the byte trunk to `seq=2048` on
   code; **0.974** bpb vs **1.265** at seq 256 (−23%), proof OK.
5. **A0.2 gate** (done, bits fail): tied span embeddings + a
   three-parameter copy/compute gate on the 2K base. Free-id gate 0.912
   bpb; count-prior ids 2.768; speed 1.84x; A0b +0.414. The id price and
   pack charge are the blocker.
6. **A1 equal bytes** (blocked): do not start on an A0.2 bits fail; a
   large-stream or long-span revival must come first.
7. **A0-P pack ops**: attach/detach/diff/rm and exact delete on the
   chain; repo-wide memory. Supporting product, CPU only.
8. **E1/C1 controls**: oracle-message, matched priors, anchored subset,
   one-window overfit, text/pointer grammar, beam verifier.
9. **E1b/E1c and C1b/C1c**: only after the controls pass; applications
   of the memory.
10. **E2/D/H/V**: only after a capability gate passes.
11. Reports and tests: `reports/REPO-MEMORY.md`,
    `reports/EDIT-LM.md`, `tests/test_editlm.py`, `tests/test_memory.py`.

Commands. Run from `cml/`; new Track A commands are marked *to add*.

```bash
# delivered substrate
uv run cml editdata --repos requests,flask,pytest,httpx,fastapi \
  --cache ~/.cache/hncm/edit --out runs/edit-manifest.json
uv run cml chain --repo ~/.cache/hncm/edit/repos/psf__requests.git \
  --path requests/models.py --out runs/models.cmlh

# Track A, delivered
uv run cml memory-probe --oracle --run codec-byte-model-code-756752-d448-l8
uv run cml memory-probe --oracle --tag 2k \
  --run codec-byte-model-code-756752-d448-l8-q2048
uv run cml codec --arm byte-model --seq 2048 --steps 8000 --d 448 --layers 8 \
  --scale-cap 48000000 --bytes 756752
uv run cml memory-gate --run codec-byte-model-code-756752-d448-l8-q2048 \
  --units 1024 --min-count 16

# applications, only after Track A has a signal
uv run cml editlm --arm locator --manifest runs/edit-manifest.json \
  --split val --limit 50 --steps 500
uv run cml editlm --arm copy --sft --manifest runs/edit-manifest.json \
  --split val --limit 50 --steps 500 --samples 8

uv run pytest tests
```

## 9. Budget

The main GPU budget goes to Track A. Products are CPU-only and must not
displace the architecture gates. Current pool: **8 GPU-hours** and
**4 CPU-days**; A2/A3 need a separate **1 GPU-month** and only start after
A0 and A1 pass.

| bucket | budget | rule |
|---|---:|---|
| **A0.0 oracle copy** | 2 h CPU | **done, pass**; 0.804 vs 0.852 bpb at 1,024 units |
| **A0.1 2K base** | 1-2 GPU-h | **done, pass**: 0.974 bpb at seq 2048 vs 1.265 |
| **A0.2 tied embeddings + gate** | 1 day | **done, bits fail**: 0.912 free-id, 2.768 priced; speed 1.84x |
| **A1 equal-charged-bytes** | 1 week | **blocked** until A0.2 bits revives |
| **A0-P pack ops + repo memory** | 2-3 CPU-days | supporting product; never first in a report |
| exact priced codec proof | 2-3 h | supporting product |
| E1 controls | 45 min GPU | applications of memory |
| C1 controls | 45 min GPU | applications of memory |
| E1b/E1c models | 2 GPU-h | only after controls pass |
| C1b/C1c models | 2 GPU-h | only after controls pass |
| D/H/V | 1 GPU-h | only after a capability gate passes |
| **A2/A3** | 1 GPU-month | only after A0/A1 pass; separate budget |

Use a 5M-20M trunk and 2-4K context. Do not sweep seeds before a control
passes. Do not run 40M. Track A uses the 20M trunk for A0.0; A0.1 is the
only base extension. If a hypothesis has no signal on 50 files, stop it and
preserve the budget for the next hypothesis.

## 10. The main line and the fallback

The **better-LM claim is the main line**: Track A. The exact codecs are the
memory substrate, the fallback, and evidence that exact spans are cheap.
They are not the architecture claim.

| line | what | role |
|---|---|---|
| **Track A** | frozen base + exact pack + copy/compute gate | **main claim** |
| delta / chain | exact version and history memory | substrate and fallback |
| exact priced codec | 9.4x exact | supporting product |
| CI logs / verifier | exact structured memory | supporting product |
| locator / edit compiler | applications of the memory | after Track A |

**Win scorecard.** A better-LM win is lower bits at equal FLOPs, faster
decode at equal quality, better adaptation per charged byte, or a
capability dense models lack. A product ratio is a supporting win, not a
substitute for the architecture claim.

| horizon | win | metric | state |
|---|---|---|---|
| now | **better-LM oracle** | A0.0 0.804 vs 0.852 bpb, 3.19x calls | **pass** |
| now | **A0.2 gate** | fitted gate + priced ids ≥0.1 bpb win | **bits fail**: id cost 1.9 bpb; speed 1.84x, A0b pass |
| next | **equal-byte adapter** | pack beats kNN/LoRA on ≥ 2/3 domains | A1 |
| research | **composition** | G4 +10 points on held-out slots | A3 |
| supporting | delta / memory | 56.6x / 40.0x exact | shipped |
| fallback | exact priced codec | 9.4x exact | proof pending |

**Portfolio rule.** Every job must either move a better-LM gate or ship an
exact capability no baseline has. Do not headline a codec ratio as an
architecture result.

## 11. Risks and counters

- **Retrieval solves it.** Harden with held-out repos, longer edits, and
  counterfactual messages; if retrieval still wins, stop.
- **Exact match is too hard.** Keep exact as the primary label; edit
  distance and recall@k are secondary and pre-registered.
- **Leakage.** Split by repo, deduplicate within a split, never put the
  target file in the prompt.
- **Instruction ignored.** Include counterfactual pairs: same old file,
  different messages, different targets.
- **LZ parse is not the edit.** Track E trains on real changed spans, not
  `delta_parse` offsets. Track C keeps the action space.
- **DPO reward hacking.** The reward is exact decode, not a learned model.
- **Action grammar errors.** Constrained decoding or a valid-action mask;
  invalid parses score zero.
- **Memory is decorative.** A0b must cost ≥ 0.2 bpb when concept ids are
  shuffled; if not, the claim is a patch model, not exact memory.
- **Adapter loses to LoRA/kNN.** A1 compares at equal charged bytes; if the
  pack loses, ship the codecs and record the negative.
- **Over-copying / provenance.** Verification mode bounds quality; the pack
  is inspectable and every emitted span carries `(concept-id, offset)`.
- **Compute creep.** One job at a time; small trunk; budgets, not timers.
- **Bug vs evidence.** A failed control means a bug, not a verdict. A
  failed gate with passed controls is evidence; record it and move on.

## 12. Forward plan: better LM first

The main line is Track A. A0.0 is first because two hours decide whether
the LM integration is worth a base. Products run in parallel CPU time and
are never allowed to displace the architecture gates.

**Execution order by expected value.**

1. **A0.0 oracle copy** (done, pass): free pack IDs inside 256 B,
   charged pack bytes. 1,024 units → 0.804 bpb vs the LZ copy arm
   0.852, 3.19x fewer calls, 100% provenance. 256 units fails and
   4,096 ties; the next pack point is A0.1.
2. **A0.1 2K base** (done, pass): 0.974 bpb at `seq=2048`, −0.291
   against the 256-byte trunk. The base can now read a longer pack.
3. **A0.2 gate + tied embeddings** (done, bits fail): free-id gate
   0.912, count-prior ids 2.768, speed 1.84x, A0b +0.414. The id price
   and per-file pack charge dominate; tied softmax does not beat counts.
   Revival needs a ≥10 MB stream (≤64 KB pack) and ≤2 bits/copy, or
   long spans that cut copies/B ~5x.
4. **A1 equal-charged-bytes** (blocked): do not run on a failed A0.2
   bits gate; the adaptation claim waits for a revived pack.
5. **A0-P pack ops + repo-wide memory** (CPU, parallel): exact delete,
   provenance, repo-level memory ratio. Supporting wins, not the claim.
6. B controls E1-R0, E1-R1, C1-R0: 45 minutes; E1/C1 are applications
   of the memory, not the main line.
7. Exact priced codec proof: supporting product only.
8. A0.3/A2/A3 (provenance, trained head, two-clock) need the separate
   GPU-month and only start after A0.2 and A1 pass.

### Phase A0: architecture probe (the r-series bet)

Freeze the existing 20M byte trunk
(`codec-byte-model-code-756752-d448-l8`). Induce a 4k-64k exact-span pack
with the existing DP/EM. Embed each span from the trunk's causal last-state
readout (not mean pooling). Add a small score head and a three-parameter
copy-or-compute gate; tag every emitted span with `(concept-id, offset)`.

Gates, in order:

1. text ≤ 0.01 bpb loss vs the frozen trunk;
2. code/JSON ≥ 0.1 bpb win;
3. structured ≥ 1.5x decode at exact quality;
4. pack ≤ 1 MB; provenance round-trip 100%;
5. **A0b shuffle control**: permute concept ids, keep embeddings; it must
   cost ≥ 0.2 bpb, or identity is decorative.

Kill rule: if A0 or A0b fails, record the negative, ship the codecs, and
do not claim an architecture. If it passes, A1 compares the pack against
kNN and LoRA at equal charged bytes. Do not train a new base for A0-P/A0.0;
A0.1 is the only allowed extension.

### Phase A: bank exact wins

1. **Repo-wide memory** (done, shipped). `cml repo` indexes all text
   files under HEAD: per-file chains, manifest, exact search. 80
   `fastapi` files, 9.84 MB raw → 787,482 B = **12.5x**, 4.3x smaller
   than per-file gzip, 47 s CPU, every SHA-256 checked. Product:
   `products/repo-memory`. The index is the exact access layer; a single
   global gzip is smaller but has no file/version boundary or
   `path:line` search.
2. **Full exact priced codec.** Run the pending full-length proof on the
   756,752 B test. Promote `0.852` (9.4x) only if the exact decoder
   reproduces every byte; otherwise keep `1.266` (6.3x). Cost: 2-3 h.
3. **CI log codec.** Mine or generate repeated pytest logs; delta each run
   against the previous; exact round-trip. Gate: ≥ 10x on near-identical
   logs and ≥ 2x on changed logs. Cost: 1 day.
4. **Patch verifier.** Apply a proposed diff, check bytes, reject
   non-exact. Ship as a guard, not a compressor. Cost: 1 day.

### Phase B: revival controls (≤ 15 min each, no model scaling)

Do the control before any long run. A control that fails is a bug.
Implement each row as a `--control` flag in `cml/editlm` and write its
result to the track report. Controls may see the answer; they are not
benchmark results and must never be reported as a win.

| id | track | control | pass signal | if it fails |
|---|---|---|---|---|
| **E1-R0** | E1 | oracle message: put the changed line/token in the message for 20 train files | byte F1 ≥ 0.50 | debug `_batch`, labels, window coverage, message padding |
| **E1-R1** | E1 | same balanced windows in train and eval; compare zeroed-file baseline | content delta ≥ 0.10 | content path broken; inspect attention/gradients |
| **E1-R2** | E1 | anchored subset: message token unique in `old` and inside a changed span | top-1 hit ≥ 0.50 | scope product to recall@k, not exact@1 |
| **C1-R0** | C1 | overfit one window to exact | exact = 1, valid = 1 | teacher-forcing/label/grammar bug |
| **C1-R1** | C1 | text grammar (decimal offsets) vs binary varints on 50 windows | valid ≥ 0.95, exact > 0 | pointer/copy head |
| **C1-R2** | C1 | beam K = 16 plus exact verifier, temperature sweep | exact@K > 0 | target/model, not search |

### Phase C: revival models (one GPU job at a time, ≤ 1 GPU-h)

Only run one after its control passes and its control row is recorded.

- **E1b candidate locator.** Enumerate candidate lines/identifier
  occurrences from `old`; score message-candidate pairs with a
  cross-encoder; hard negatives from repeated tokens. Train on anchored +
  unanchored pairs. Gate: candidate line F1 ≥ 0.30; then rerun GE1.
- **E1c pointer locator.** Initialize from the trained 20M byte trunk
  (`d448`); freeze blocks; train a span/copy head. Gate: byte F1 ≥ 0.20.
- **C1b pointer copy-LM.** Replace varint offsets with a pointer head over
  `old` (source offset, length); keep a literal head. Use the 20M trunk
  as initialization. Gate: valid ≥ 0.95 and exact ≥ 0.05 on 50 files.
- **C1c chunked actions.** Apply C1b to all 5,672 files with overlapping
  action windows; exact assembly across windows. Gate: exact ≥ 0.05.

### Phase D: gated capability

- **E2 writer** only after GE1b passes.
- **D/DPO** only after C1b has exact@K > 0 and a recorded SFT baseline.
- **H/V** only after the preceding gate; V is rerank, not a rescue.

## 13. Revival program: hypotheses and kill rules

### Track A: memory-augmented LM (r-series)

| # | hypothesis | cheap test | if true | if false |
|---|---|---|---|---|
| 1 | the pack is an adapter, not a patch index | A1 pack vs kNN/LoRA at equal bytes | architecture claim | ship codecs; no architecture |
| 2 | discrete identity is load-bearing | A0b shuffle ids | exact spans carry the win | it is a patch model; narrow the claim |
| 3 | tied content embeddings transfer | hold out 10% of the pack; predict similar spans | zero-shot install | per-domain training |
| 4 | copy-or-compute gate is safe | verification mode bounds quality | product path | gate redesign |
| 5 | exact concepts + holes compose | r7 G4 slot split | capability claim | efficiency paper only |

Kill rules for Track A: no structured win at A0 → stop; no shuffle cost at
A0b → stop; pack loses to LoRA and kNN at A1 → stop; do not scale A2/A3
on a failed A1.

### E1 locator

| # | hypothesis | cheap test | if true | if false |
|---|---|---|---|---|
| 1 | target/mask/window bug | oracle message | pipeline needs no model | debug the data path |
| 2 | train/eval prior mismatch | identical balanced windows | fix sampling; rerun | content path broken |
| 3 | message ambiguity | anchored subset | scope to anchored/recall@k | candidate ranking |
| 4 | dense byte head is wrong | candidate-line cross-encoder | line F1 ≥ 0.30; E2 writer | pointer/line model |
| 5 | capacity/init | 20M trunk, freeze + head | +0.10 byte F1 | pretrained model required |

### C1 copy-LM

| # | hypothesis | cheap test | if true | if false |
|---|---|---|---|---|
| 1 | labels/teacher forcing | one-window overfit | exact = 1 | fix alignment first |
| 2 | varint offsets are hard | text grammar | valid ≥ 0.95 | pointer head |
| 3 | no copy pointer | pointer head over `old` | exact > 0 on 50 | action model is wrong |
| 4 | too little data/context | all 5,672 files, chunked | exact ≥ 0.05 | narrow the task |
| 5 | search is weak | beam 16 + exact verifier | exact@K > 0 | DPO is not justified |

**Kill rules.**

1. A failed control: debug, do not scale.
2. No signal on 50 files: stop that hypothesis, not the track.
3. One GPU job at a time; ≤ 1 GPU-h per hypothesis.
4. Do not change the label, split, or `max_diff` after seeing a result.
5. A negative result with passed controls is evidence; record it in the
   track report and move to the next hypothesis.
6. A gate that passes names the next product; a gate that fails names the
   next experiment.

## 14. BOLO: additional products to create

Ranked by user value per CPU-day. Each is exact, CPU-only, and reuses
the substrate. None depends on a new model or a bpb win.

| product | what it solves | command | state |
|---|---|---|---|
| **repo-memory** | exact installable repo history + search | `cml repo` | **shipped**, 20.7x |
| **patch-guard** | apply a diff and refuse non-exact bytes | `cml patch` | **shipped** |
| **memory-mcp** | expose the repo index to an agent | `cml mcp` | **shipped**, 5 tools |
| **repo-delta** | exact patch between two commits | `cml repo-delta` | **shipped**, 85.8x example |
| **log-delta** | exact CI logs, delta each run | `cml logdelta` | **shipped**, 27.3x |
| **pack-doctor** | rank spans by net bits at the measured bpb | `cml pack fit` | **shipped** |
| **span-index** | per-file phrasebook; amortize ids locally | `cml span-index` | next; 2 days |
| **exact-priced-codec** | full-length 0.852 proof, promote on exact decode | `cml copycodec --proof 756752` | next; 3 h |
| **concept-cards** | typed holes with exact realization | `cml concepts` | next; 1 week |

The defect the A0.2 run exposed is the guide: a flat span pack scores
spans by bytes saved at 8 bpb, but the base is ~1 bpb and a copy also
pays an id. At 1 bpb a 3-byte span saves ~3 bits and a 9-bit id loses
6; at 16 bytes it saves ~15 and wins. `span-index` (local phrasebook)
and `pack-doctor` (net-bit ranking) attack that directly; `repo-memory`
sidesteps it by giving the model exact context instead of asking it to
code spans.

## 15. Model invention: ByteCopy LM

A genuine model-side invention, not a bitstream patch: a byte-level trunk
with a **copy head** (pointer over the context), a **run head** (span
length), and a **gate**. Copied spans are exact and update the cache in a
chunked forward; novel bytes go through the byte head. No tokenizer, no
flat span id, no pack charge. Training is self-supervised: the label is the
longest prior match in the context (min 8 B, cap 64 B).

The prototype (`cml/copylm.py`, `cml/reports/COPY-LM.md`) trains a 403k
frozen copy head on 2 MB of code. Exact copies work; the gate is not
calibrated on frozen last states: 52.6% copy precision at gate 0.9, 0.96x
bytes/step. The joint repair (R0-R9) added a relative-position bias, a
top-2 pointer margin, an exact-span gate, a run head conditioned on the
chosen source, a multi-positive pointer loss, 2x training volume, and a
2x pointer. At gate 0.99 the span is 98.0% at 98.1% source, but at only
1.2% coverage. The operating point stalls at 66-67% span precision at
~5% coverage, 0.97x. The gate cannot exceed source precision, so the
source pointer is the binding term; the B1 repair loop is a recorded
stop in `TODO10.md`. The last test is B2: point into a retrieved exact
pack.

The product path is `memory-mcp`: the model queries an exact repo index and
conditions on the bytes. ByteCopy LM is the trained-model path; the MCP
server is the installable-memory path.

## References

- `TODO8.md`: codec record; delta product and exact codec.
- `cml/reports/DELTA.md`: 56.6x / 27.8x exact delta.
- `cml/reports/REPO-MEMORY.md`: 40.0x exact history chain.
- `cml/codec.py`: `DeltaArm`, `delta_parse`.
- `cml/memory.py`: `CMLH`, chain encode/decode.
- `cml/bytecodec.py`: `RepCopyCodec`, exact verifier.
- `cml/pointer.py`: pointer/copy head for the action sequence.
- `cml/concepts.py`: typed holes and exact realization.
- `products/codec/README.md`, `products/delta/README.md`,
  `products/memory/README.md`: what ships today.
