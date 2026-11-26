# TODO8 — concepts as the post-training action space

The codec's multi-byte units are not only a compression alphabet. They are the
concept model's vocabulary and RLFP's action space. A model that predicts and
acts on these units makes fewer, longer decisions; a preference signal can be
projected onto exact concepts; and `rm`/`merge` edit the policy without a
gradient step. The gate is preference reward per compute, not bpb.

`TODO7.md` remains the codec record and the auditable-codec product. This file
is the synergy plan.

## Progress (2026-11-26)
| M | status | evidence |
|---|---|---|
| M0 | **done** | §0 pins DPO, span repair, verifiable bytes, and the byte/word/concept controls |
| M1 | **printed; H1 partial** | 5.21M pointer: **1.600 bpb**, 1.692 payload, 1.722 shared, 50k-unit round-trip; 4.93M trunk control: **1.516 bpb**; `reports/CONCEPT.md` |
| M2 | **blocked** | H1's decision gate fails under the printed all-symbol definition; resolve §11.1 first |
| M3-M5 | not started | |
### M1 measurements
| model | params | steps | epochs | bits/byte | payload | shared | B/dec all | B/dec span |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pointer | 5.21M | 8,000 | 13.4 | 1.600 | 1.692 | 1.722 | 1.478 | 3.498 |
| trunk | 4.93M | 8,000 | 13.4 | 1.516 | - | - | 1.478 | 3.498 |
- 8,000 steps draws 65.5M elements: 13.4 epoch-equivalents over 4,897,323
  train elements. With `rotate=8`, only 1/8 of the steps read the full pack;
  the rest cycle four chunk packs. The first 20,000-step trial (no rotation,
  seq 128) was 33.5 full-pack epochs and overfit: train loss 0.709
  nats/decision, test 1.769 bpb (2.615 nats/decision). It is an overfit
  control, not a larger-data test.
- Both models read the same 256-unit greedy parse: pack 2,771 B, coverage
  0.453, literal fraction 0.809.
- The trunk is the fixed-vocabulary control: slightly better at this scale,
  but a new concept needs a new row and a retrain.
- H1: bits/byte **PASS** (1.600 ≤ 1.75), round-trip **PASS** (50,000 units /
  81,531 B), all-symbol B/decision **FAIL** (1.478 < 2.5), span-only
  B/decision **PASS** (3.498 ≥ 2.5).
### 11.1 The decision-gate inconsistency
The 256-unit pack covers 45.3% of test bytes. The parse has 511,933 decisions
for 756,752 B: **1.478 B/decision** over all symbols, or **3.498 B/decision**
over matched spans. H1 passes the 2.5 bar only if a literal escape is not a
policy decision. H3 asks for 2x fewer decisions: concept uses 0.676
decisions/byte, only **1.48x fewer** than byte decisions; span-only is
0.130 span decisions/byte, **7.7x fewer**.
Choose before M2:
1. **All symbols (strict).** H1 fails and H3's decision clause fails. The
   action-space claim at 256 cannot pass; a larger pack or a new metric is
   required.
2. **Span policy.** A policy decision is a matched concept; the byte escape is
   charged separately. H1 passes (3.50), H3 has 7.7x fewer decisions. Report
   the 0.809 literal fraction and the escape bytes; do not call this 2x fewer
   all-symbol decisions.
Recommendation: use span-policy decisions for the RLFP action claim, keep
all-symbol decisions as the honest compression cost, and gate H3 on both.
Do not switch definitions silently.
### Capability screen (2026-11-26)
`cml screen` removes 200 matched spans from held-out code and measures exact
reconstruction, bits/span, and decisions/span on the same task:

| mechanism | family | exact span | top-1 | bits/span | decisions/span |
|---|---|---:|---:|---:|---:|
| trunk | r3/r4 control | 0.480 | 0.480 | 2.590 | 1.00 |
| pointer | r3/r4/r6 | 0.415 | 0.415 | 3.133 | 1.00 |
| byte | r1 baseline | 0.210 | 0.610 | 12.711 | 3.04 |
| counts | r1/r6 copy | 0.040 | 0.040 | 2.815 | 1.00 |

- The concept action space reconstructs **2.0x** more exact spans than byte
  (0.415 vs 0.210) with **3.0x** fewer decisions (1.00 vs 3.04). This is the
  first direct capability win for the multi-byte action space; it is not a
  preference win.
- The trunk still leads (0.480). At 5M the fixed-vocab control is stronger;
  the pointer's advantage must come from editability or scale, not accuracy.
- `counts` is weak (0.040): a bigram copy table is not enough. Do not use it
  as the install baseline.
- The screen tests only one capability. `r5` edit and `r7` library need their
  own probes on the same task; `r2` copy bypass needs a charged hybrid. Do not
  call this full r1-r7 coverage.
### 11.2 Remaining work
- **M2 pairs.** Build 10k span-repair pairs from the held-out code split;
  preferred = exact bytes, dispreferred = failed mutation. Hold out prompts.
- **M2 SFT.** Train the concept policy on the same pack; verify the random
  (~50%) and SFT (>55%) pairwise baselines.
- **M3.** Run concept/byte/word DPO at equal pairs and compute; report H3 with
  both decision definitions and held-out win rate.
- **Scale test.** 20M/20k is deferred: it overfits at this corpus size and the
  host restarted during the long runs. Re-run only on a stable host with
  checkpointing; the 5.21M pointer is the H1 feasibility point.
## 0. RLFP definition

**RLFP = RL from Preferences.** It covers RLHF (reward model + PPO), DPO,
IPO/KTO/ORPO/SimPO, and preference-based GRPO. The plan does not treat these as
one algorithm; it pins one first and keeps the others as controls.

- **Default: DPO.** Offline, stable, no reward model, cheap on pre-tokenized
  pairs. It is the fastest test of the action-space claim.
- **Second: PPO/GRPO** with a reward model or a verifiable reward, only if DPO
  leaves headroom or online exploration is required.
- **Preference source:** verifiable code repair. Preferred = the patch passes
  tests or reconstructs the held-out bytes; dispreferred = it fails. This is
  self-supervised and auditable. Model-generated pairs are the fallback when
  the task is too easy.
- **Metrics:** pairwise accuracy, win rate against the SFT policy, reward
  margin, KL to the reference, decisions per byte, and GPU-hours.

## 1. What is measured (do not re-derive)

| fact | number | where |
|---|---|---|
| codec, 756,752 B, 256 units | **1.692 payload / 1.722 shared** | `runs/codec-concept-code-756752-nj/` |
| byte model, same slice | 1.693 / 1.694 | `runs/codec-byte-model-code-756752/` |
| oracle install | +0.088 bpb, circular | `runs/install-code-networkx-a256-s3000/` |
| no-oracle install | +0.209 / +0.280 bpb, 8k | `runs/install-transfer-code-networkx-a256-s8000/` |
| text LM, 90 MB / 10 MB | 20.46M, **1.517 bpb**, promptable | `runs/text-enwik8_100m-d384l8-s30000/` |
| 10 MB P1 model | 1.920 bpb; train 1.516 → overfit | `runs/p1-class-10m-3084fd95-s0/` |
| codec ops | provenance 100%, `trace`, `rm`, `merge` | `reports/INSTALL*.md` |
| compute | 20M/90 MB/30k steps = 759 s; batch 64 +28-38% at 20M | profile in `TODO7.md` |

The install result is the key correction: an unseen pack costs 0.21-0.28 bpb.
That is a compression cost, not a policy verdict. RLFP can amortize it if a
concept action replaces several byte actions.

## 2. The missing link

Today the codec has the pack and the text LM has the v9 word lexicon. They are
separate systems. The pointer reads packs, but it is a codec, not a policy.
The next build joins them:

1. **Concept LM.** Tokenize the code corpus with the codec pack. Train the
   concept model on those unit ids (pointer or Trunk). The action vocabulary is
   the pack; every action maps to exact bytes.
2. **Preference policy.** Treat the concept model's next-concept distribution
   as the policy. For DPO, the policy is the concept LM and the reference is a
   frozen copy. For PPO/GRPO, add a value head.
3. **Concept-level preference projection.** Tokenize preferred and
   dispreferred responses with the same pack; compute the preference loss on
   concept tokens. One action covers multiple bytes, so a response has fewer
   decisions and less credit-assignment variance.
4. **Editable policy.** `rm`/`merge`/install change the action set. The
   content-addressed pointer recomputes embeddings without a gradient step; a
   fixed-vocab Trunk is the control because it needs a new row.

Architectural choice:

| model | install/rm cost | use |
|---|---|---|
| pointer, content-addressed | zero weights; recompute `e_c` | preferred for editability |
| Trunk, tied unit embeddings | new row + retrain | fixed-pack baseline |

The 5.2M codec pointer is the proof and the built M1 model. The 20M pointer is
a deferred scale test: at 7.2 MB it overfits before scale helps, and long runs
restarted the host. Do not train another fixed-vocab LM and call it the concept
model.

## 3. Hypotheses and gates

| id | hypothesis | gate |
|---|---|---|
| H1 | a 5M concept LM reads the codec pack | test bits/byte **≤ 1.75** on 756,752 B; round-trip exact; ≥2.5 B/decision; 20M is deferred scale |
| H2 | the preference task is learnable | ≥10k verifiable pairs; SFT pairwise accuracy > 55%; random ≈ 50% |
| H3 | concept actions give RLFP leverage | concept DPO beats byte DPO by **≥5 points** pairwise accuracy at equal compute, or matches it with **≥2x fewer decisions** |
| H4 | pack edits change the policy without gradients | `rm`/`merge` moves win rate in the predicted direction; provenance 100% |
| H5 | the synergy scales | if H3/H4 pass, scale the concept model and pairs; if not, record the negative |

H3 is the claim. H1 is the feasibility check. H2 is the data check. H4 is the
auditable-policy claim. Do not gate on bpb; report it as context.

## 4. Milestones

| M | ships | acceptance | cost |
|---|---|---|---|
| M0 | pin RLFP: method, task, reward, metrics | DPO chosen; repair task and pair generator specified; byte/word/concept controls named | 0.5 d |
| M1 | **printed; H1 partial** | `cml train --corpus code --units codec --model pointer\|trunk`; 5.21M pointer at 8k steps; H1 bpb/round-trip pass; all-symbol decision gate fails; `reports/CONCEPT.md` | done |
| M2 | **blocked** | resolve §11.1, then ≥10k pairs; SFT concept policy; byte and word controls; `reports/RLFP-DATA.md` | 1 d |
| M3 | DPO loop | concept vs byte vs word at equal compute; H3; `reports/RLFP.md` | 1-2 d |
| M4 | pack edits | `rm`/`merge` after DPO; H4; `reports/RLFP-EDIT.md` | 1 d |
| M5 | decision | scale if H3/H4 pass; otherwise record the negative and return to the auditable codec | 1 d |

M1 is printed. Resolve the §11.1 decision definition before M2. M2-M3 are the
bet. M4 is the unique part; no byte-level post-training can delete a concept
and measure the policy response.

## 5. Target interfaces

The M1 concept commands exist. The `cml rl` commands are still the interface
contract; do not build them before M2 pairs.

```bash
# concept LM on the codec vocabulary (M1)
uv run cml train --corpus code --units codec --model pointer --params 5000000 --steps 8000
uv run cml train --corpus code --units codec --model trunk --params 5000000 --steps 8000
# 8k steps = 13.4 epochs over 4.9M train elements; 20M/20k is the deferred
# overfit control.

# preference pairs and SFT baseline (M2)
uv run cml rl --task repair --make-pairs --pairs 10000 --pack <lexicon>
uv run cml rl --task repair --method sft --actions concept --pairs 10000

# DPO action-space comparison (M3)
uv run cml rl --task repair --method dpo --actions concept --pairs 10000
uv run cml rl --task repair --method dpo --actions byte --pairs 10000
uv run cml rl --task repair --method dpo --actions word --pairs 10000

# editable policy (M4)
uv run cml pack rm <run> --prefix <concept>
uv run cml rl --task repair --eval-only --run <dpo-run>
```

Keep the existing `cml train`, `cml sample`, `cml pack`, and `cml stats`.
Add `cml rl` only when M1 prints a concept LM; do not build the RL loop first.

## 6. Preference data

The first task is **span repair**:

- Take a held-out function from the code test split.
- Remove one span (a line, an expression, or a multi-byte unit).
- Ask the policy to reconstruct the span.
- Preferred = exact bytes, or the function's tests pass.
- Dispreferred = a mutation that fails, or a model sample that fails.

This gives pairs without human labels and without a reward model. Keep a
held-out prompt set so DPO cannot memorize pairs. For each prompt, report the
pair construction rule, the reward, and the pack SHA.

If the task is too easy (both sides pass), increase the removal span or use a
bug-injection mutation. If it is too hard (both sides fail), use a shorter span
or give the policy the function body.

## 7. Metrics and artifacts

Every run records:

- `pairwise_accuracy`: preferred chosen over dispreferred.
- `win_rate`: against the SFT policy on held-out prompts.
- `reward_margin`: mean preferred minus dispreferred log-prob.
- `kl`: policy to reference.
- `decisions_per_byte` and `bytes_per_decision`.
- `bits_per_byte` (context, not the gate).
- `gpu_seconds` and `pairs`.
- `pack_sha`, `corpus_sha`, `policy_sha`, `git`.

Artifacts: `reports/CONCEPT.md`, `reports/RLFP-DATA.md`, `reports/RLFP.md`,
`reports/RLFP-EDIT.md`, plus the usual `runs/<id>/metrics.json` and model.

## 8. Risks and kill criteria

- **Reward hacking.** The repair reward is exact bytes or tests; keep a
  held-out test set and never train on it.
- **Vocabulary/task misalignment.** If the concept boundaries do not match the
  repair span, the policy may need a byte escape. Charge it; report the mix.
- **DPO overfit.** Use held-out prompts and a KL term; report both.
- **Edit instability.** `rm`/`merge` can create out-of-distribution concepts;
  measure calibration and the win-rate delta, not just the mean.
- **Fixed-vocab control.** A Trunk needs a new row for a new concept, so it is
  the control, not the target.
- **Kill H1:** concept LM > 1.9 bits/byte at 5M/8k on 756,752 B → the vocabulary
  is not carrying structure; stop.
- **Kill H3:** concept DPO does not beat byte DPO at equal compute → the
  synergy claim fails; ship the auditable codec.
- **Kill H4:** pack edits have unpredictable reward effects → the
  auditable-policy claim fails; do not market it.

## 9. Do not

- Do not gate on bpb. The install no-oracle result already fails that gate.
- Do not use an oracle pack induced on the encoded bytes.
- Do not sweep pack size; 256 is optimal for the codec.
- Do not build PPO/GRPO before DPO shows a signal.
- Do not train a fixed-vocab concept model and claim install/editability.
- Do not claim a preference win on training pairs; report held-out prompts.
- Do not add dependencies; use the existing torch/numpy stack.

## 10. First week, in order

1. **Pin M0.** Write the repair task, the pair rule, and the DPO loss. Name the
   byte and word controls.
2. **M1.** Add `--units codec` and `--model pointer|trunk` to `cml train`;
   tokenize the 7.24 MB train split with the 256-unit pack; train a 5M
   pointer at 8k steps; print bits/byte, B/decision, and round-trip. H1
   bpb/round-trip pass; resolve the decision gate in §11.1 before M2.
3. **M2.** Build 10k repair pairs from the code test split; train the SFT
   concept policy; verify the random and SFT baselines.
4. **M3.** Run concept/byte/word DPO at equal pairs and compute; report H3.
5. **M4.** `rm`/`merge` the post-DPO pack; measure the held-out win-rate
   delta and provenance without a gradient step.

M1 is the only build that starts before the preference data exists. Everything
else waits for the concept LM to print.

## References

- `TODO7.md`: the codec, the install negative, and the auditable product.
- `cml/PLAN.md` §11: the name/charge accounting.
- `cml/pointer.py`: the content-addressed pack reader.
- `cml/codec.py`: `Lexicon`, `greedy_parse`, `build_lexicon`, the arms.
- `cml/cml.py`: `Trunk`, `train_text`, `p1_sample`, `cml train/sample/stats`,
  and the pack ops.
