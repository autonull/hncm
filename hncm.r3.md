# PhraseLM — generate the novel, copy the known

**The invention in one sentence:** a mixed-granularity neural LM whose unit of computation is either a byte or an *exact span from an installable, MDL-induced phrasebook* — trained to read **any** phrasebook, so domain adaptation becomes installing a kilobyte-scale data artifact (no gradients), which simultaneously improves quality (exact domain strings) and speed (one global step per span instead of per byte).

This keeps everything H-NCM proved valuable — exact spans, the DP/EM induction, the charge, the verification discipline — and replaces the one thing it proved fatal: the count-table predictor becomes a neural backbone, and the phrasebook demotes from "the model" to "an installable lexicon the model reads." The core scientific bet is new and cheaply falsifiable: **lexicon-agnostic pretraining transfers to unseen phrasebooks at test time.**

## 1. Architecture

**Units.** At each step the model emits one unit `u ∈ {256 bytes} ∪ D`, where `D` is the installed phrasebook (10–500 KB of exact byte spans + counts, induced by your existing DP+EM machinery). The stream is self-delimiting — a gate bit, then a byte or a span index — so the decoder never reruns the DP. Your v9 decoder-synchrony result carries over directly.

**Three components:**
- **Local encoder** (small causal conv/transformer over bytes): embeds any span *from its bytes*, order-preserving — the v6 lesson (a mean pool cannot carry byte order) is why this is a sequence net, not a pool. Runs at byte rate, ~15–20% of FLOPs, MegaByte-style [[31]].
- **Global transformer** over *unit* embeddings — the expensive part runs at unit rate, not byte rate. This is the BLT/MegaByte compute argument [[36]], but the units are exact reusable spans with identity, not entropy patches.
- **Three heads:** a pointer `q(s|h)` = softmax over phrasebook embeddings (dot-product, same cost class as a vocab head; embeddings computed once at install by the local encoder), a byte head `p(b|h)`, and a gate `λ(h) = P(copy)`. Emission is the mixture — a pointer-sentinel architecture [[11]], generalized from "copy a word from recent context" to "copy a span from an installed lexicon."

**Why transfer is plausible (the mechanistic argument):** a new span gets its embedding *from its bytes through the same local encoder*, so it lands in the model's representation space with zero training. The pointer learns context→span-*content* matching, not context→arbitrary-ID. This is exactly what vocabulary expansion cannot do — new token IDs need learned embeddings and continued pretraining [[45]], and even domain n-gram vocab replacement requires embedding adaptation [[43]]. Here, install = encode 32k spans once (milliseconds), done.

**Safe mode (the quality guarantee).** After a copy, the byte head scores the span's bytes in one parallel local pass; if mean surprisal > τ, roll back and generate byte-wise. τ is calibrated on held-out data to bound the perplexity delta. This is REST's verification insight moved *inside* the model: the phrasebook drafts, the model's own byte path verifies, and output quality is bounded by construction. Fast mode skips rollback with a calibrated conservative gate.

**D = ∅ mode:** a plain byte/patch transformer. No regression, always usable, and the training anchor.

## 2. Training

1. **Induce** per-domain phrasebooks with the repo's existing DP + EM usage-cost loop, budget chosen by charged MDL on held-out bytes — your stop-rule becomes the objective.
2. **Hard-EM:** parse each training sequence with current model costs (cached by corpus×lexicon SHA — your existing cache), then train the mixture NLL on that parse: `L = −Σ [y·log λq(s*) + (1−y)·log(1−λ)p(b*)]`. The MDL parse *is* the copy supervision — no labels, and the parse guarantees the target span matches ground truth exactly.
3. **Lexicon-agnostic protocol (the novel bet):** every batch samples a phrasebook from a pool — same-domain, cross-domain, noisy, and ∅ — with copy-path dropout. The model must remain a strong byte LM under *any* lexicon and exploit a good one. This forces the pointer to generalize by content, which is what makes installation work zero-shot.

## 3. Why it satisfies every adjective

| Requirement | Argument |
|---|---|
| **Viable** | Every component has a working precedent: local/global hierarchy [[31]], patch compute [[36]], gated copy [[11]], span-copy generation [[1]], internal verification (REST). The only unproven part is lexicon transfer — and the pilot fits your 3080 (20–40M params, 100 MB–1 GB). |
| **Useful** | "LoRA you can read": a 10–500 KB auditable, diffable, mergeable file that adapts an LLM instantly. Kills BPE pathologies on identifiers, numbers, morphology. Personalization/edge without fine-tuning. The same artifact doubles as a drafter for big models (last turn's E1). |
| **Usable** | Standard `generate()` + `install(D)` / `uninstall()`; streaming bytes; no-lexicon fallback; the phrasebook is the H-NCM prior artifact you already ship. No DP at inference. |
| **Unique** | The adapter *class* is new: discrete, exact, transmissible, charged — vs LoRA (dense, trained), RAG (fuzzy passages), vocab expansion (needs training [[45]]), AdaptiVocab (needs adaptation [[43]]). |
| **Performant** | Units/byte `u = (1−c) + c/L̄`. From your measurements: enwik8 at 32k dict (c=0.825, ~3.06M units/10 MB) → **~3.3× fewer global steps**, ~10× cheaper long-context attention, ~3× smaller KV cache; shakespeare (37,168 calls/111 KB) → ~3×; synseg (485 calls/1.2 MB) → ~1000×. Projections, labelled — but from your own committed coverage numbers. |
| **Novel** | Honest delta: pointer-sentinel copies words from context, no installable lexicon [[11]]; Copy-Is-All-You-Need copies spans from an external collection, not a charged adapter or compute unit [[1]]; prompt-lookup/REST are inference-only drafting, not a trained architecture [[21]]; BLT/H-Net segmentation lives in weights, not installable data [[36]]. The synthesis — **lexicon-agnostic training + transmissible MDL lexicon + spans as atomic compute units + internal verification** — is unoccupied, and the transfer claim is a real scientific question, not a rebrand. |

## 4. Kill-gates (your culture)

- **P0 — the core bet (3–5 days, 3080):** 20–40M params, train on two domains with rotating lexicons; evaluate on a *held-out* domain with its phrasebook installed vs D=∅. **Gate:** ≥0.15 bpb improvement from installation alone, and safe-mode copy acceptance ≥50% at ≤0.02 bpb quality delta. **Stop:** no cross-domain transfer → fall back to per-domain continued pretraining (still useful, less novel) or ship as a drafter only.
- **P1 — speed (2 days):** wall-clock vs the same-size byte transformer and vs prompt-lookup on the same model [[21]]. **Gate:** ≥1.3× on text, ≥3× on structured at equal quality. **Stop:** gate overhead eats the savings at low coverage → restrict to structured domains (where your measured win already lives).
- **P2 — scale (only if P0+P1 pass):** 100–300M params, 5–10B tokens; gate: the transfer effect survives scale and beats an equal-FLOP BPE transformer on domain suites.

## 5. Honest risks

Copy miscalibration (safe mode bounds it, at a speed cost); lexicon overfitting (the rotating-lexicon protocol is the test, not the guarantee); DP parse cost at train time (cache, amortize); the phrasebook is literally your data — auditable but a memorization/privacy surface; and the novelty is a synthesis, not a new primitive. None of these is fatal; all are measurable before scale.

**Bottom line:** the viable invention is not "H-NCM as an LLM" — it's H-NCM's phrasebook, charge, and verification demoted into a *transmissible adapter* for a neural backbone that is trained to read any lexicon. One model plus kilobyte-scale installable lexicons: better and faster domain language modeling with zero training at deployment. Every piece is precedented except the one claim that matters, and P0 falsifies it in a week on hardware you already own.

