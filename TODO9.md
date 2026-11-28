# TODO9 — the exact-edit capability program: two tracks, one benchmark

Status: 2026-11-27. `TODO8.md` shipped the delta codec. This plan changes
the objective from bits per byte to **exact edit capability**. It keeps
the original copy-edit LM and adds a cheaper edit-compiler track. Both are
funded; the benchmark decides which one wins.

## Status 2026-11-27: wins banked; E1/C1 paused, not abandoned

| work | state | artifact |
|---|---|---|
| data + benchmark | **done** | `cml/editdata.py`; 5 repos, 12,811 edits, repo split |
| retrieval G0 | **done** | `cml/reports/EDIT-LM.md`; exact `0.000` on 1,648 test commits |
| E1 locator | **paused after GE1 fail** | `cml/reports/EDIT-LOCATOR.md` |
| C1 copy-SFT | **paused after GC1 fail** | `cml/reports/EDIT-COPY.md` |
| delta product | **shipped** | `products/delta/README.md`; **56.6x** hunk edits, exact |
| memory product | **shipped** | `products/memory/README.md`; **40.0x** over 1,823 versions, exact |

**Banked wins.** `cml delta` is exact at `56.6x` on 2% hunk edits and
`27.8x` scattered. `cml chain` is exact at `40.0x` over five real files
(90.1 MB → 2.26 MB), `6.5x` better than gzip. Both are CPU-only. Tests:
`55 passed`.

**G0 recorded: pass.** Retrieval on 1,648 held-out `tiangolo/fastapi`
commits: exact `0.000`, valid `0.027`, output `0.991x`. Train `5,672`
(`httpx`, `flask`, `requests`); val `5,491` (`pytest`). The cross-repo
transplant cannot see the query repository's history; that is the point of
G0. Artifacts: manifest `cml/runs/edit-manifest.json` (ignored), blob
cache `~/.cache/hncm/edit` (308 MB). Reuse them.

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

## Ambition: the memory-augmented LM

`hncm.r1-r7.md` converge on one thesis, stated seven ways:

> A frozen/amortized neural base plus an exact, MDL-priced, transmissible,
> editable memory of spans, with a copy-or-compute gate. Knowledge is data;
> compute routes around known spans; every emitted span is verifiable.

That is an architecture claim, not a codec claim. It is bigger than exact
edit prediction: the memory is the adapter, the tokenizer, the drafter, and
the audit trail at once.

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

A0 is the decisive, inference-only probe. It reuses the codecs, the trunk,
and the corpus; no new base training.

| id | experiment | cost | gate |
|---|---|---|---|
| **A0** | freeze the 20M byte trunk; induce a 4k-64k exact-span pack; tied embeddings from the trunk's last-state readout; fit a 3-parameter gate; provenance tags | 3-5 days, inference only | text ≤ 0.01 bpb loss; code/JSON ≥ 0.1 bpb win; structured ≥ 1.5x decode at exact quality; pack ≤ 1 MB; provenance round-trip 100% |
| **A0b** | shuffle control: permute concept ids, keep embeddings | hours | must cost ≥ 0.2 bpb, or identity is decorative and the claim narrows to patches |
| **A1** | equal-charged-bytes: pack vs kNN vs LoRA at 64k/256k/1MB | 1 week | pack wins ≥ 2 of 3 structured domains; prose may go to LoRA |
| **A2** | train the head: mixture NLL + charge on DP parses | 1-2 weeks | tied embeddings beat counts on held-out spans; ≥ 20% acceptance per pack byte |
| **A3** | two-clock / CoLib: concept tokens, holes, planning | 1 GPU-month | G0 parity, G2 shuffle, G4 composition ≥ +10 points |

**Decision rule.** If A0 or A0b fails, record the negative, ship the codecs,
and do not claim an architecture. If A0 passes, A1 decides whether the
memory beats LoRA/kNN at equal bytes. If A1 passes, Track A becomes the
main research line and E1/C1 become copy-path applications.

**Budget.** A0 is inference-only on the existing trunk. A1 is one week.
A2/A3 need a separate 1 GPU-month; the current 8 GPU-hour budget does not
cover them. Do not start A2 before A0 and A1 pass.

## 0. One claim, two formulations

**Track E (edit compiler).** A small model that predicts *where* a file
changed and *what* replaces it beats a larger full-file byte LM on real
commits, at a small fraction of the output bytes.

**Track C (original copy-edit LM).** A small LM that emits an exact delta
program beats a larger full-file byte LM on real commits, at 5-50x fewer
output tokens.

Both are verified by exact decode, exact match, or test pass. Neither is a
bpb claim.

## 1. Why this is the better bet

- The delta arm is exact and measured: **56.6x** on 2% edits in 20 hunks,
  **27.8x** on scattered 2% edits (`cml/reports/DELTA.md`).
- The exact decoder is a free verifier. The reward is binary.
- A full-file LM pays for every byte; both edit tracks pay only for
  changes.
- The synthetic delta benchmark is replaced by real old/new commit pairs.
- Track E is cheaper to train; Track C is more general and is the original
  bet. Run both.

## 2. The shared task and benchmark

Given:

- `old`: the parent file bytes;
- `message`: the commit message (and optional failing test name);
- `new`: the target file bytes.

The benchmark is **Exact Edit Bench**: real commits, split by repository,
with exact-match and test-pass labels. It is shared by both tracks and is
itself a product.

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

## 6. Arms: a variety of solutions

| arm | what it does | role | cost |
|---|---|---|---:|
| **R** retrieval | nearest old/new pair or patch by message; apply exactly | cheap baseline; kills a fake win | 30 min CPU |
| **F** full-file | same trunk generates the whole new file | compute-matched baseline | 1-2 GPU-h |
| **E1** locator | old + message -> changed spans | fast Track E gate | 1 GPU-h |
| **E2** locator + writer | spans + replacement bytes -> exact assembler | Track E product | 1-2 GPU-h |
| **C1** copy-SFT | SFT on serialized delta actions | original Track C arm | 1-2 GPU-h |
| **D** copy-DPO | SFT + DPO on exact-decode preferences | RLFP-style preference step | 1 GPU-h |
| **G** copy-RLFP | GRPO/REINFORCE if DPO plateaus | verifiable-reward fallback | 1 GPU-h |
| **H** holes-SFT | copy + typed holes for new code | compositional capability | 1 GPU-h |
| **V** verifier rerank | sample K, decode each, keep exact/test-pass | test-time compute | 2-4x inference |

Order: R -> F -> E1 -> E2 -> C1 -> D -> G -> H -> V. Do not train D before
C1 has a recorded baseline. Do not drop C because E is cheaper: C is the
general arm and the original bet.

## 7. Metrics and pre-registered gates

Primary:

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
passed controls is evidence. Neither stops the product track; both choose
the next experiment.

## 8. Implementation order

Delivered: data, benchmark, retrieval, delta product, memory product.
Paused: E1/C1. The revival controls and models are ordered in §12-13,
not repeated here.

1. `cml/editdata.py`: mining, filters, repo split, manifest, blob cache.
   Tests: deterministic split, no repo leakage, diff-size filter.
2. `cml/editlm.py`: action grammar for both tracks, target builders,
   exact assembler, metrics.
3. `cml editdata --repos ... --out ...`: build manifest and split.
4. `cml editlm --arm retrieval`: G0 baseline; `reports/EDIT-LM.md`.
5. `cml editlm --arm locator`: E1; **paused after GE1 fail**; run the
   E1-R0..R2 controls first (`reports/EDIT-LOCATOR.md`).
6. `cml editlm --arm compiler`: E2; wait for a locator that passes GE1b.
7. `cml editlm --arm copy --sft`: C1; **paused after GC1 fail**; run the
   C1-R0..R2 controls first (`reports/EDIT-COPY.md`).
8. `cml editlm --arm copy --dpo`: D; only after C1b has exact@K > 0.
9. `cml editlm --arm copy --rlfp`: G only if D plateaus.
10. `cml editlm --arm holes --sft`: H; after the preceding gate.
11. `cml editlm --arm compiler --rerank`: V; after E2; rerank only.
12. Reports and tests: `reports/EDIT-LM.md`, `tests/test_editlm.py`.

Commands. Run from `cml/`; `runs/` is ignored.

```bash
# delivered
uv run cml editdata --repos requests,flask,pytest,httpx,fastapi \
  --cache ~/.cache/hncm/edit --out runs/edit-manifest.json
uv run cml editlm --arm retrieval --manifest runs/edit-manifest.json \
  --split test
uv run cml chain --repo ~/.cache/hncm/edit/repos/psf__requests.git \
  --path requests/models.py --out runs/models.cmlh
# paused: controls first (§12), then one model job per hypothesis
uv run cml editlm --arm locator --manifest runs/edit-manifest.json \
  --split val --limit 50 --steps 500
uv run cml editlm --arm copy --sft --manifest runs/edit-manifest.json \
  --split val --limit 50 --steps 500 --samples 8
# E2/D/H/V commands stay out until their gate passes
uv run pytest tests
```

## 9. Budget

Total for products and controls: **8 GPU-hours** and **4 CPU-days**. One
GPU job at a time. Track A A2/A3 needs a separate **1 GPU-month** and is
not covered here; start it only after A0 and A1 pass.

| bucket | budget | rule |
|---|---:|---|
| product track (delta, memory, logs, verifier) | 4 CPU-days | bank first; no GPU |
| **A0/A0b concept-pack probe** | 3-5 days, inference | decisive architecture test; no new base training |
| **A1 equal-charged-bytes** | 1 week | pack vs kNN vs LoRA |
| exact priced codec proof | 2-3 h | one batch; promote only if exact |
| E1 controls | 45 min GPU | ≤ 15 min each |
| C1 controls | 45 min GPU | ≤ 15 min each |
| E1b/E1c models | 2 GPU-h | ≤ 1 GPU-h each |
| C1b/C1c models | 2 GPU-h | ≤ 1 GPU-h each |
| D/H/V | 1 GPU-h | only after a gate passes |
| **A2/A3** | 1 GPU-month | only after A0/A1 pass; separate budget |

Use a 5M-20M trunk and 2-4K context. Do not sweep seeds before a control
passes. Do not run 40M. If a hypothesis has no signal on 50 files, stop it
and preserve the budget for the next hypothesis. Track A uses the 20M trunk
that already exists; do not train a new base for A0.

## 10. Product track: bank exact wins first

The capability tracks are research; the codec products are exact and
measured. Bank them while the revival controls run.

| product | state | next action | value |
|---|---|---|---|
| **`cml delta`** | shipped; **56.6x** hunk, **27.8x** scattered, exact | keep; add CI log packaging | immediate |
| **`cml chain`** (repo memory) | shipped; **40.0x** over 1,823 versions, exact, 6.5x gzip | repo-wide batch: all files, one ratio | high; 1-2 days |
| **exact priced codec** | 32 KB prefix exact **1.074**; full proof pending | run full-length exact proof; promote **9.4x** only if exact | high; 2-3 h |
| **CI log codec** | idea | delta repeated test logs; exact round-trip | medium; 1 day |
| **patch verifier** | idea | apply a proposed diff, reject non-exact; no training | medium; 1 day |
| **locator API** | paused | anchored/recall@k after E1b; not exact@1 | medium |
| **edit compiler** | paused | only after GE1b and GC1b | the capability win |

**Portfolio rule.** Every job must either ship a measured exact artifact or
move a pre-registered gate. A job that does neither is recorded and
stopped. Product work is CPU-only; one GPU job at a time.

**Win scorecard.** A headline win is exact and ≥ 5x on a real corpus, or a
capability gate passing on held-out data. Anything else is a diagnostic.

| horizon | win | metric | state |
|---|---|---|---|
| now | delta product | 56.6x exact | shipped |
| now | repo memory (5 files) | 40.0x exact, 6.5x gzip | shipped |
| next | repo-wide memory | ≥ 20x exact, ≥ 3x gzip | planned |
| next | exact priced codec | 9.4x exact | proof pending |
| next | CI log codec | ≥ 10x near-identical | planned |
| research | **A0 concept pack** | code/JSON ≥ 0.1 bpb, ≥ 1.5x | **next probe** |
| research | E1b locator | byte F1 ≥ 0.20 | control first |
| research | C1b copy-LM | exact ≥ 0.05 on 50 | control first |

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

## 12. Forward plan: bank, probe, control, then revive

Phase A banks exact products. Phase A0 is the architecture probe from the
r-series: no new base training, frozen 20M trunk, exact concept pack.
Phase B is cheap diagnostics. Phase C is one GPU job per hypothesis.
Phase D stays gated.

**Execution order by expected value.**

1. A1 repo-wide memory: the 40x win becomes a repo-level product claim.
2. **A0 concept-pack probe + A0b shuffle control** (3-5 days, inference):
   the decisive test of the memory-as-adapter ambition. If it fails, the
   architecture claim stops here and the codecs ship.
3. B controls E1-R0, E1-R1, C1-R0: 45 minutes, they decide whether the
   next model run is a bug fix or a new hypothesis.
4. A2 exact priced codec proof: promote 9.4x if exact; otherwise keep 6.3x.
5. **A1 equal-charged-bytes** (1 week) only if A0/A0b pass: pack vs kNN
   vs LoRA at equal bytes. This is the architecture's product test.
6. B controls E1-R2, C1-R1, C1-R2.
7. C models E1b/E1c and C1b/C1c, one at a time.
8. A3/A4 CI logs and patch verifier in leftover CPU time; A2/A3
   (train head, two-clock) need the separate GPU-month.

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
kNN and LoRA at equal charged bytes. Do not train a new base for A0.

### Phase A: bank exact wins

1. **Repo-wide memory.** Extend the chain runner to `--paths` (comma
   list) or a repo manifest; process all files in `fastapi` and `pytest`;
   report one raw/chain/gzip total. Gate: chain ≥ 20x and ≥ 3x gzip. Cost: 1-2 days
   CPU. This is the headline product claim.
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
