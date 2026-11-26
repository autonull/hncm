# TODO8 — concepts as the post-training action space

The codec's multi-byte units are not only a compression alphabet. They are the
concept model's vocabulary and RLFP's action space. A model that predicts and
acts on these units makes fewer, longer decisions; a preference signal can be
projected onto exact concepts; and `rm`/`merge` edit the policy without a
gradient step. The gate is preference reward per compute, not bpb.

`TODO7.md` remains the codec record and the auditable-codec product. This file
is the synergy plan.

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

The 5.2M codec pointer is the proof; a 20M pointer is the scale test. Do not
train another fixed-vocab LM and call it the concept model.

## 3. Hypotheses and gates

| id | hypothesis | gate |
|---|---|---|
| H1 | a 20M concept LM reads the codec pack | test bits/byte **≤ 1.75** on 756,752 B; round-trip exact; ≥2.5 B/decision |
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
| M1 | concept LM | `cml train --corpus code --units codec --model pointer\|trunk`; 20M pointer; H1 on 756,752 B; `reports/CONCEPT.md` | 2-3 h |
| M2 | preference data + SFT baseline | ≥10k pairs; SFT concept policy; byte and word controls; `reports/RLFP-DATA.md` | 1 d |
| M3 | DPO loop | concept vs byte vs word at equal compute; H3; `reports/RLFP.md` | 1-2 d |
| M4 | pack edits | `rm`/`merge` after DPO; H4; `reports/RLFP-EDIT.md` | 1 d |
| M5 | decision | scale if H3/H4 pass; otherwise record the negative and return to the auditable codec | 1 d |

M1 starts now. M2-M3 are the bet. M4 is the unique part; no byte-level
post-training can delete a concept and measure the policy response.

## 5. Target interfaces

These commands do not exist yet; they are the interface contract.

```bash
# concept LM on the codec vocabulary (M1)
uv run cml train --corpus code --units codec --model pointer --params 20000000 --steps 20000
uv run cml train --corpus code --units codec --model trunk --params 20000000 --steps 20000

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
- **Kill H1:** concept LM > 1.9 bits/byte at 20M on 756,752 B → the vocabulary
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
   tokenize the 756 kB train split with the 256-unit pack; train a 20M
   pointer; print bits/byte, B/decision, and round-trip. If H1 fails, stop.
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
