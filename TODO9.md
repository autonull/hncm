# TODO9 — the exact-edit capability program

Status: 2026-11-27. `TODO8.md` shipped the delta codec. This plan changes
the objective from bits per byte to **exact edit capability**. It is the
active capability track; codec rungs A2-A6 are paused, not deleted.

## 0. One claim

**A small LM that emits an exact edit program beats a larger full-file
byte LM on real code edits, at 5-50x fewer output tokens.** The decoder
verifies every result. Success is exact match or test pass, not bpb.

The claim is falsifiable. A retrieval/diff baseline and a full-file LM
are measured on the same held-out repositories. If retrieval already wins,
the capability is not real and the track stops.

## 1. Why this is the better bet

- The delta arm is exact and measured: **56.6x** on 2% edits in 20 hunks,
  **27.8x** on scattered 2% edits (`cml/reports/DELTA.md`).
- `delta_parse(new, old)` is a free teacher. No human labels.
- The exact decoder is a free verifier. The reward is binary.
- A full-file LM pays for every byte. A copy-edit LM pays only for
  changes: copy spans, literals, and typed holes.
- TODO8 A2-A6 optimize a proxy and cap near `0.68-0.71` bpb. This track
  produces a capability result: instruction-conditioned, exact editing.
- The synthetic delta benchmark is replaced by real old/new commit pairs.

## 2. The task

Given:

- `old`: the parent file bytes;
- `message`: the commit message (and optional failing test name);
- `new`: the target file bytes.

The model emits an action sequence:

```
COPY(offset, length)   # exact bytes from old
LIT(span)              # new bytes
HOLE(type)             # phase 3 only
END
```

The target sequence is the `delta_parse(new, old)` result, serialized as
actions. Restrict phase 1 to copies from `old`; copies from already-emitted
output are phase 2. Decode the actions with the exact delta decoder and
compare to `new`.

Start with serialized delta tokens. A pointer/copy head over old-file
positions is the fallback if the action sequence is too long.

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
- file ≤ 64 KB, diff ≤ 20% of the file;
- non-empty message, no generated/vendor files;
- parent blob and commit blob both available.

Split by **repository**, not commit, so held-out code cannot leak:

| split | repos | commits |
|---|---|---:|
| train | 3 | ~5,000 |
| val | 1 | ~500 |
| test | 1 | ~500 |

Store blobs by SHA. Deduplicate near-identical edits by content hash.
Record the exact repo SHAs and the mining config; a run without provenance
is not evidence.

## 4. Arms: a variety of solutions

| arm | what it does | role | cost |
|---|---|---|---:|
| **R** retrieval | nearest old/new pair or patch by message similarity; apply it | cheap baseline; kills a fake win | 30 min CPU |
| **F** full-file | same trunk, generate the whole new file | compute-matched baseline | 1-2 GPU-h |
| **S** copy-SFT | SFT on serialized delta actions | first real arm | 1-2 GPU-h |
| **D** copy-DPO | SFT + DPO on exact-decode preferences | RLFP-style preference step | 1 GPU-h |
| **G** copy-RLFP | GRPO/REINFORCE if DPO plateaus | last resort, verifiable reward | 1 GPU-h |
| **H** holes-SFT | copy + typed holes for new code | compositional capability | 1 GPU-h |
| **V** verifier rerank | sample K parses, decode each, keep the exact/test-pass one | test-time compute | 2-4x inference |

The order is R → F → S → D → G → H → V. Do not train D before S has a
recorded baseline. Do not start H before D or G has a verdict.

## 5. Metrics and pre-registered gates

Primary:

- **exact match**: decoded file equals `new` byte-for-byte;
- **test pass** on the subset with runnable tests;
- **valid decode**: action sequence decodes without error;
- **output bytes**: payload bytes divided by new-file bytes;
- **latency** and **test-time compute**.

Secondary: edit distance, copy-span F1, instruction counterfactual accuracy
(same old file, different messages).

Gates:

| gate | test | promote if |
|---|---|---|
| **G0** | retrieval on 500 held-out commits | retrieval exact < 20%; else harden the task |
| **G1** | copy-SFT vs full-file | +10 points exact or +20 test pass, at ≤5% output bytes |
| **G2** | copy-DPO vs copy-SFT | +3 points exact; else drop DPO |
| **G3** | holes-SFT vs copy-SFT | +5 points on held-out compositions; else drop holes |
| **G4** | verifier rerank | +2 points at ≤4x test-time compute; else drop |

If G1 fails and retrieval wins, stop the track and ship the delta product.
A gate that is not met names the next lever; it does not end the program.

## 6. Implementation order

Add two modules and one CLI command. Keep `TODO8` codec code untouched.

1. `cml/editdata.py`: mining, filters, repo split, manifest, blob cache.
   Tests: deterministic split, no repo leakage, diff-size filter.
2. `cml/editlm.py`: action grammar, target builder from `delta_parse`,
   exact decoder check, metrics.
3. `cml editdata --repos ... --out ...`: build the manifest and split.
4. `cml editlm --arm retrieval`: G0 baseline; write `reports/EDIT-LM.md`.
5. `cml editlm --arm full --train`: F baseline on the same trunk.
6. `cml editlm --arm copy --sft`: S arm; teacher-forced delta actions.
7. `cml editlm --arm copy --dpo`: D arm; sample K, score exact/test pass,
   build preference pairs, DPO. No learned reward model.
8. `cml editlm --arm copy --rlfp`: G arm only if D plateaus.
9. `cml editlm --arm holes --sft`: H arm; typed holes + exact realizer.
10. `cml editlm --arm copy --rerank`: V arm; sample K and verify.
11. Reports and tests: `reports/EDIT-LM.md`, `tests/test_editlm.py`.

Commands, after the build:

```bash
uv run cml editdata --repos requests,flask,pytest,httpx,fastapi \
  --cache ~/.cache/hncm/edit --out cml/runs/edit-manifest.json
uv run cml editlm --arm retrieval --manifest cml/runs/edit-manifest.json
uv run cml editlm --arm full --manifest cml/runs/edit-manifest.json \
  --run codec-byte-model-code-756752-d448-l8 --steps 2000
uv run cml editlm --arm copy --sft --manifest cml/runs/edit-manifest.json \
  --run codec-byte-model-code-756752-d448-l8 --steps 2000
uv run cml editlm --arm copy --dpo --manifest cml/runs/edit-manifest.json \
  --from edit-copy-sft --pairs 2000
uv run cml editlm --arm holes --sft --manifest cml/runs/edit-manifest.json \
  --from edit-copy-sft
uv run cml editlm --arm copy --rerank --manifest cml/runs/edit-manifest.json \
  --from edit-copy-sft --samples 8
uv run pytest cml/tests
```

## 7. Budget

Under **6 GPU-hours total**, one job at a time:

| step | budget |
|---|---:|
| data + retrieval | 2 h CPU |
| full-file baseline | 1-2 GPU-h |
| copy-SFT | 1-2 GPU-h |
| DPO | 1 GPU-h |
| holes | 1 GPU-h |
| rerank | inference only |

Use a 5M-20M trunk and 2-4K context. Do not sweep seeds before G1. Do not
run 40M.

## 8. Risks and counters

- **Retrieval solves it.** Harden with held-out repos, longer edits, and
  counterfactual messages; if retrieval still wins, stop.
- **Exact match is too hard.** Pre-register test pass and edit distance as
  secondary; do not move the goal after seeing results.
- **Leakage.** Split by repo, deduplicate near-identical edits, and never
  put the target file in the prompt.
- **Instruction ignored.** Include counterfactual pairs: same old file,
  different messages, different targets.
- **DPO reward hacking.** The reward is exact decode, not a learned model;
  edit distance is only a tie-break.
- **Action grammar errors.** Use constrained decoding or a valid-action
  mask; invalid parses score zero.
- **Compute creep.** One job at a time; small trunk; budgets, not timers.

## 9. Fresh-session start

Read this file first, then `TODO8.md` §1.10 and `cml/reports/DELTA.md`.

First session deliverable, no GPU:

1. `cml/editdata.py` + deterministic repo split + manifest.
2. `cml editlm --arm retrieval` + G0 on 500 held-out commits.
3. `cml/reports/EDIT-LM.md` with the G0 number and the task difficulty.

Do not start A2-A6. Do not train a model before G0. The first session is
done when G0 is recorded, not when a model is trained.

## References

- `TODO8.md`: codec record; delta product and exact codec.
- `cml/reports/DELTA.md`: 56.6x / 27.8x exact delta.
- `cml/codec.py`: `DeltaArm`, `delta_parse`.
- `cml/bytecodec.py`: `RepCopyCodec`, exact verifier.
- `cml/pointer.py`: the pointer/copy head if the action sequence is long.
- `cml/concepts.py`: typed holes and exact realization.
- `products/codec/README.md`: what ships today.
