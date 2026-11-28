# TODO9 — the exact-edit capability program: two tracks, one benchmark

Status: 2026-11-27. `TODO8.md` shipped the delta codec. This plan changes
the objective from bits per byte to **exact edit capability**. It keeps
the original copy-edit LM and adds a cheaper edit-compiler track. Both are
funded; the benchmark decides which one wins.

## Status 2026-11-27: first session complete; G0 passed

| §8 step | state | artifact |
|---:|---|---|
| 1 editdata | **done** | `cml/editdata.py`: git mining, filters, repo split, manifest, blob cache |
| 2 editlm core | **done** | `cml/editlm.py`: copy actions, spans, exact assemblers, metrics |
| 3 CLI | **done** | `cml editdata`, `cml editlm` |
| 4 retrieval G0 | **done** | `cml/reports/EDIT-LM.md`, `cml/reports/editlm.json` |
| 5 E1 locator | **attempted; GE1 fail** | `cml/reports/EDIT-LOCATOR.md` |
| 6 E2 writer | **blocked** | needs a locator that passes GE1 |
| 7 C1 copy-SFT | **attempted; GC1 fail** | `cml/reports/EDIT-COPY.md` |
| 8-11 D/H/V | **not started** | blocked by GC1 |

**G0 recorded: pass.** Retrieval on 1,648 held-out `tiangolo/fastapi`
commits: exact `0.000`, valid decode `0.027`, output `0.991x`, line edit
distance `0.020`. Train `5,672` commits (`httpx`, `flask`, `requests`);
val `5,491` (`pytest`). The cross-repo transplant rarely aligns context,
so the baseline is weak by construction; it cannot see the query
repository's history. That is the point of G0: it is not a fake win.

Artifacts are local: manifest `cml/runs/edit-manifest.json` (ignored),
blob cache `~/.cache/hncm/edit` (308 MB). Reports are tracked. Reuse
them; do not re-mine unless the config changes.

**Next:** stop capability training. Both tracks failed their first gate, so
package the exact delta product (`cml delta`) and keep the benchmark. If
work resumes, the lever is the task/data, not more steps. Do not train D,
H, or V without a gate that passes.

**E1 recorded: GE1 fail.** The 5.2M byte locator (`d256 l6`, 1024 B
window, 1200 steps) scores byte F1 `0.000` against the message-only
baseline's `0.000` on 500 val files; it emits no spans. Synthetic
pretraining fits (`loss 0.34`), but real fine-tuning does not transfer and
collapses to the prior. A 50-file, 5,000-step control also scored `0.000`,
so this is not a budget problem. A line-level prototype (8-line windows,
byte pooling) on the same 50 files scored `0.055` with an unstable loss.
The next lever is the model input/architecture (explicit message-to-file
cross-attention or a different decoder), not more steps on this trunk. E2
must not reuse this locator. C1 is independent and should run first.

**C1 recorded: GC1 fail.** Pilot on files that fit one 1024 B window:
596 train, 50 val, 1,000 steps, 5.2M causal byte LM. Exact `0.000`,
valid program `0.960`, output `0.057x` of the new file. Sampling `K = 8`
programs per file and keeping any exact assembly also gives `exact@8 0.000`:
test-time search does not find the target. A 5,000-step run was worse
(valid `0.300`, output `0.086x`). Constrained decoding keeps the bytes
grammar-valid, but the model does not learn the edit content. Since exact
is zero, a +10-point lift over any baseline is impossible; the full-file
arm is not needed to reject this arm. Do not start DPO or holes from it.

**Program decision.** Both funded tracks now fail their first gate: GE1
`0.000`, GC1 `0.000`. Per §7, stop capability training and ship the delta
product. The next lever is the task/data (retrieval prior, smaller
grammar, or a different edit distribution), not more GPU hours.

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

GC1 recorded 2026-11-27: exact `0.000`, valid `0.370`, output `0.044x` on
200 val files, fail. Exact zero makes a +10-point lift impossible; see
`cml/reports/EDIT-COPY.md`. Do not start DPO or holes from this arm.

If G0 shows retrieval solving the task, harden before training. If GE2 and
GC1 both fail, stop and ship the delta product. A gate that is not met
names the next lever; it does not end the program.

## 8. Implementation order

Add two modules and one CLI command. Keep `TODO8` codec code untouched.

1. `cml/editdata.py`: mining, filters, repo split, manifest, blob cache.
   Tests: deterministic split, no repo leakage, diff-size filter.
2. `cml/editlm.py`: action grammar for both tracks, target builders,
   exact assembler, metrics.
3. `cml editdata --repos ... --out ...`: build manifest and split.
4. `cml editlm --arm retrieval`: G0 baseline; write `reports/EDIT-LM.md`.
5. `cml editlm --arm locator`: E1; synthetic pretrain, then real commits.
   **Attempted; GE1 fail** (`reports/EDIT-LOCATOR.md`).
6. `cml editlm --arm compiler`: E2; writer + exact assembler. Wait for a
   locator that passes GE1.
7. `cml editlm --arm copy --sft`: C1; teacher-forced delta actions.
   **Attempted; GC1 fail** (`reports/EDIT-COPY.md`).
8. `cml editlm --arm copy --dpo`: D; sample K, score exact/test pass,
   build preference pairs, DPO. No learned reward model.
9. `cml editlm --arm copy --rlfp`: G only if D plateaus.
10. `cml editlm --arm holes --sft`: H; typed holes + exact realizer.
11. `cml editlm --arm compiler --rerank`: V; sample K and verify.
12. Reports and tests: `reports/EDIT-LM.md`, `tests/test_editlm.py`.

Commands, after the build. Run from `cml/`; `runs/` is ignored.

```bash
uv run cml editdata --repos requests,flask,pytest,httpx,fastapi \
  --cache ~/.cache/hncm/edit --out runs/edit-manifest.json
uv run cml editlm --arm retrieval --manifest runs/edit-manifest.json \
  --split test
uv run cml editlm --arm locator --manifest runs/edit-manifest.json \
  --split val --limit 500 --pretrain synthetic --steps 1200
# E2 only after GE1 passes: locator + writer
uv run cml editlm --arm compiler --manifest runs/edit-manifest.json \
  --from edit-locator --steps 2000
uv run cml editlm --arm copy --sft --manifest runs/edit-manifest.json \
  --from codec-byte-model-code-756752-d448-l8 --steps 2000
uv run cml editlm --arm copy --sft --manifest runs/edit-manifest.json \
  --split val --limit 200 --steps 1000
# D/H/V only after a gate passes; both funded gates failed.
uv run cml editlm --arm copy --dpo --manifest runs/edit-manifest.json \
  --from edit-copy-sft --pairs 2000
uv run cml editlm --arm holes --sft --manifest runs/edit-manifest.json \
  --from edit-copy-sft
uv run cml editlm --arm compiler --rerank --manifest runs/edit-manifest.json \
  --from edit-compiler --samples 8
uv run pytest tests
```

## 9. Budget

Under **8 GPU-hours total**, one job at a time:

| step | budget |
|---|---:|
| data + retrieval | 2 h CPU |
| full-file baseline | 1-2 GPU-h |
| locator + writer (Track E) | 2-3 GPU-h |
| copy-SFT (Track C) | 1-2 GPU-h |
| DPO / RLFP | 1 GPU-h |
| holes | 1 GPU-h |
| rerank | inference only |

Use a 5M-20M trunk and 2-4K context. Do not sweep seeds before G1. Do not
run 40M.

## 10. Products/wins within reach

| product | what it is | effort | value |
|---|---|---:|---|
| **`cml delta`** | exact version compression; 56.6x on edits | built; 1 day to package | immediate |
| **Exact Edit Bench** | real commits, repo split, exact/test-pass | 1 day CPU | foundation + artifact |
| **Edit locator API** | old file + instruction -> changed spans | 1 week | useful to any coding agent |
| **Edit compiler** | locator + writer + exact assembler | 2-3 weeks | the capability win |
| **Repo memory** | delta chain across versions; exact retrieval | 2-3 days | agent context compression |
| **Patch verifier** | apply a proposed diff, reject non-exact results | 1 day | hallucination guard |
| **CI log codec** | delta repeated test logs | 1 day | small but real |

The fastest conclusive win is the Exact Edit Bench plus the retrieval
baseline. If retrieval solves it, save the GPU budget. If not, the
benchmark is the foundation for both tracks.

## 11. Risks and counters

- **Retrieval solves it.** Harden with held-out repos, longer edits, and
  counterfactual messages; if retrieval still wins, stop.
- **Exact match is too hard.** Pre-register test pass and edit distance as
  secondary; do not move the goal after seeing results.
- **Leakage.** Split by repo, deduplicate near-identical edits, never put
  the target file in the prompt.
- **Instruction ignored.** Include counterfactual pairs: same old file,
  different messages, different targets.
- **LZ parse is not the edit.** Track E trains on real changed spans, not
  `delta_parse` offsets. Track C keeps the original action space; compare
  them on the same benchmark.
- **DPO reward hacking.** The reward is exact decode, not a learned model;
  edit distance is only a tie-break.
- **Action grammar errors.** Use constrained decoding or a valid-action
  mask; invalid parses score zero.
- **Compute creep.** One job at a time; small trunk; budgets, not timers.

## 12. Next-session start

Read this file first, then `cml/reports/EDIT-LM.md`,
`cml/reports/EDIT-LOCATOR.md`, and `cml/reports/EDIT-COPY.md`. G0 passed;
GE1 and GC1 both failed.

**Stop capability training.** Ship `cml delta`: it is exact, CPU-only, and
`56.6x` on hunk edits. Keep the Exact Edit Bench and both reports as the
foundation. Do not train DPO, holes, or rerank from a zero-exact arm.

If the capability work resumes, change the task/data, not the schedule:

1. retrieval prior or a smaller action grammar (C1 family);
2. explicit message-to-file cross-attention (E1 family);
3. a different edit distribution with longer, less ambiguous messages.

Do not raise `max_diff` to make exact match easier; a gate that fails names
the next lever, not a new goal.

## 13. What would make the old ideas work

Three cheaper revivals were tested after GE1/GC1 failed:

| idea | result |
|---|---|
| **V verifier rerank** | C1 `exact@8 = 0.000`; sampling never found the target |
| **candidate-line ranking** | top-1 changed-line hit `0.109`, byte F1 `0.030` on 46 val files |
| **anchored edits** | only `15/300` val and `21/300` test files have a message token that occurs once in `old` and inside the changed span |

The blocker is the task, not the decoder. A 5M scratch LM on a commit
subject cannot recover an edit that the message does not determine. The
old ideas become viable only with a different input or model:

1. **Richer context**: a failing test name, stack trace, issue text, or
   agent conversation; the message alone is not an instruction.
2. **Pretrained code model**: 5M scratch bytes are too small; use an
   existing code LM with constrained decoding and the exact verifier.
3. **Narrower product**: an anchored locator API scored by recall@k, not
   exact@1; or repo memory that retrieves a known past patch.
4. **Different granularity**: predict a typed edit class (rename, import,
   version bump) and synthesize the patch, then verify exactly.

Do not spend more GPU hours on the same formulation. The delta product is
exact today; the Exact Edit Bench is the asset for the next attempt.



- `TODO8.md`: codec record; delta product and exact codec.
- `cml/reports/DELTA.md`: 56.6x / 27.8x exact delta.
- `cml/codec.py`: `DeltaArm`, `delta_parse`.
- `cml/bytecodec.py`: `RepCopyCodec`, exact verifier.
- `cml/pointer.py`: pointer/copy head if the action sequence is long.
- `cml/concepts.py`: typed holes and exact realization.
- `products/codec/README.md`: what ships today.
