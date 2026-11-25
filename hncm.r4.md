Yes — but only if you invert three of the repo's own measured negatives instead of fighting them. The concepts survive; almost everything around them changes. Here is a concrete, buildable design.

# Concept-LM (C-LM): a neural LM whose vocabulary is data

**One line:** a decoder-only transformer that autoregressively emits *indices into a persistent, MDL-charged lexicon of exact byte spans*, with a priced neural byte-escape — where the lexicon is a separable, swappable, auditable adaptation artifact, and the same model is also a codec.

## The three inversions (each one is a repo measurement turned into a design rule)

1. **"The model class was the deficit" (0.67 of the 0.83 gap).** → The concept model is no longer a count table. It is a contextual neural transformer over *compositional* concept embeddings. Counts survive only as bookkeeping (usage, charge, priors), never as the predictor. This kills the sparsity wall (7.7 obs/context) by parameter sharing.
2. **"Free dense codes collapse" (v6).** → The code is discrete, exact (an index, zero residual), and **charged in the loss**. The charge is not accounting anymore — it is the anti-collapse regularizer. v6 died because the code was free; C-LM's code must earn its bytes.
3. **"The archive/shared-context economics win" (v10).** → The shared, transmissible unit is no longer a compressed archive; it is the **lexicon**. Weights are amortized globally and frozen; domain knowledge lives in a kB–MB lexicon you fit in minutes and swap without a gradient step.

## Architecture

```
text ──► trie lattice of lexicon spans ──► Viterbi under NEURAL prices ──► concept sequence
                                                                              │
                                              ┌───────────────────────────────┘
                                              ▼
                                   Concept Transformer T_c  (decoder-only, 50–300M)
                                     predicts next concept + p(escape)
                                              │
                          ┌───────────────────┴───────────────────┐
                          ▼ concept id                              ▼ escape
                   lexicon[id] → exact bytes                Byte channel T_b (10–50M)
                   (zero residual, one call)                emits bytes until "resume"
```

- **Lexicon L:** versioned trie of exact spans (1–256 B). Each entry: bytes, usage count, embedding `e(c) = ByteEnc_ψ(bytes(c)) + r_c`. `ByteEnc_ψ` is a small shared byte encoder; `r_c` is a learned residual only for frequent concepts. **New/rare spans are carried by the shared encoder** — this is what makes zero-shot lexicon swap work and what lets statistical strength flow between related spans.
- **T_c:** contextual, long-range, replaces the order-1 table. Output head over |L| (16k–1M) via trie-factorized / hierarchical softmax or a MIPS head, so the big vocabulary doesn't dominate FLOPs.
- **T_b + switch:** a neural byte fallback (not the SVD cold start) with a learned, MDL-priced escape/resume. Uncovered text is priced by a model that generalizes, which is exactly what "coverage 0.912 → rate 4.283" said was missing.
- **Parser:** their DP, unchanged in spirit, but prices come from T_c/T_b. At *generation* time there is no parsing — T_c emits concepts and the lexicon writes bytes. Parsing is only for encoding/conditioning/training. That asymmetry is the repo's best idea, kept.

## Objective (two-part MDL, their judge becomes the loss)

```
J(θ, φ, ψ, L) = Σ_seq [ Σ_t −log₂ p_θ(c_t | c_<t)
                      + Σ_escape −log₂ p_φ(b | ·) + switch bits ]
              + λ · bytes(L)          ← the charge; λ chosen on held-out
parse π*(x)  = argmin_π code_{θ,φ}(π(x))   (Viterbi on the lexicon lattice)
```

Training is their EM loop, upgraded: **E** = parse under current neural prices (sample from the lattice posterior, not just argmin, for regularization — the subword-regularization trick); **M** = gradients on θ, φ, ψ; **L-step** = add spans where usage × surprise-reduction exceeds charge, prune low-usage entries (their usage-cost EM, now neural). This is the SentencePiece unigram-LM EM/Viterbi loop [[35]] with the unigram replaced by a contextual neural model and a charge term — that specific upgrade is the research core.

## Why it's useful / usable / performant (the honest Pareto)

- **Adaptation without training.** Fit a domain lexicon with frozen weights (minutes), ship kB–MB. vs LoRA (training, MBs), RAG (no exactness), and vocabulary-adaptation work like AdaptiVocab [[14]]: here the artifact is *exact byte spans, charged, model-in-the-loop parsed*, and it is the knowledge itself, not token merges.
- **Exact grounded atoms.** Generated identifiers, API names, JSON keys, boilerplate are byte-exact copies of known-good spans. In trie-constrained "grounded mode" over a repo, hallucinated *atoms* are impossible (composition can still be wrong — honest).
- **Span-at-a-time generation.** Sequence length n/ℓ̄; on structured data ℓ̄ = 20–200 B → large KV/attention/FLOP win; on prose ℓ̄ ≈ BPE → parity with token LMs, better than byte LMs. This is the BLT/H-Net dividend [[23]], but with *persistent, named, priced* concepts instead of ephemeral per-sequence patches.
- **Auditable, deletable knowledge.** What the model knows about your domain is an inspectable, prunable list of spans with prices. Unique.
- **One artifact = LM + codec.** The same model compresses (transmit the parse, their bitstream) and generates → "verify by reconstruction" for data integrity.

**Target domains (where it wins):** repo-scale code assistants, JSON/log/structured generation, domain boilerplate, on-device personalization, auditable/deletable domain models. **Not** open-domain chat SOTA per parameter. Say that plainly.

## Novelty, positioned honestly

The components have lineage: MDL segmentation goes back to Brent/Goldsmith [[4]]; unigram-LM does EM+Viterbi with a unigram [[35]]; BLT does entropy patches [[23]]; recent MDL tokenizer induction (e.g. LCT) builds reusable units [[1]]; REST does retrieval drafting. **None of them is this synthesis:** a *persistent, charged, separable lexicon* that is simultaneously (a) the vocabulary of a neural LM, (b) a swappable adaptation artifact, (c) an auditable knowledge store, and (d) a codec — with segmentation defined as the argmin of the *neural* model's own code length and the charge acting as the training regularizer. The unclaimed part is the combination and the charge-as-regularizer inversion, not any single ingredient.

## Risks and cheap falsifiers (your culture)

| risk | check | gate / stop |
|---|---|---|
| neural prices don't beat counts at pilot scale | **P0 below** | ≤1.83 bpb go; >2.2 stop |
| joint EM oscillates / lexicon churn | parse stability, churn <10%/round | held-out J must drop each round; else soft-lattice relaxation |
| new-span cold start breaks swap | zero-shot new-lexicon eval | within 0.1 bpb of tuned; else residual-only tuning |
| span commitment hurts quality | mid-span abort token; entropy-capped span length | equal verified quality at equal FLOPs vs byte baseline |
| structured win doesn't generalize | your N2 real suite | beat PPMd + zstd-dict + small byte transformer |

## The one decisive first experiment (≈1 GPU-day, reuses everything)

**P0 — the model-class test.** Freeze the cached v9 DP parses on the 90 MB enwik8 train. Train a 20–60M concept transformer on those token sequences (your `fit-seed` path, no new scan). Evaluate the 10 MB val with the *same* parse, neural prices replacing the order-1 table.
- **Gate:** ≤1.83 bpb (beats PPMd) → the concept channel is viable; proceed to joint EM (P1), lexicon-as-adapter (P2), generation (P3), speculative mode with an open 7B verifier (P4, ≥1.3× wall-clock vs REST).
- **Gray 1.83–2.2:** the class fix works but needs the joint parse loop; do P1.
- **Stop >2.2:** a neural concept model doesn't pay at this scale; record the negative and keep the archive product.

P0 isolates exactly the variable your decomposition blamed, costs a day, and cannot be confounded by coverage or parsing. If it passes, you have a novel architecture whose every design choice is the fixed point of a measurement you already made — and whose first artifact is a lexicon you can ship, swap, audit, and compress with.

