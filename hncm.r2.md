Yes — but the move is not to make H-NCM predict better. It is to **invert what each part is for**: let the network *generalize*, let the phrasebook *remember exactly*, and let a description-length budget *decide which knowledge lives where*. That partition is the invention. Everything else is proven parts.

# C2LM — the Copy-or-Compute LM

**One line:** a byte-patch transformer whose domain knowledge is an exact, MDL-charged, transmissible phrasebook, and whose inference routes each span between a zero-compute COPY path and a neural COMPUTE path, trained under one loss that charges the memory.

This keeps every asset your repo already owns (the DP exact-cover tiling, the EM usage-cost model, the charge accounting, the structured win, the judge) and discards the one thing the measurements killed: the bigram as *predictor*. The phrasebook stops predicting and starts *remembering*; the transformer does the predicting.

## The novel core: a charged knowledge partition

Train with

```
L = CE(patch sequence)  +  λ · |phrasebook|_bytes  +  μ · E[deep-stack FLOPs]
```

The charge term `λ·|P|` is your v7 rule promoted into the loss. It creates a pressure no current LLM has: for every regularity in the data, the optimizer must choose whether to encode it in **weights** (free at inference, expensive to train, opaque, not transmissible) or in the **phrasebook** (charged, exact, inspectable, deletable, shippable). General/compositional structure migrates to weights; repetitive, domain-specific, exact structure migrates to data. Memorization becomes *explicit and auditable* — you can read, prune, delete, and ship what the model "knows" about your domain. That externalization-under-MDL is the new bit.

Be honest about what is *not* new: MDL tokenization is crowded in 2026 (MDL-calibrated BPE stopping [[19]], the Latent Core Tokenizer [[25]]); memory-in-the-LM exists (Memory Layers add trainable KV parameters without FLOPs [[1]]); compute routing exists (Mixture-of-Depths skips layers per token [[10]], GateSkip [[37]], Mixture-of-Recursions [[17]]); and the copy-vs-recall choice is already studied [[29]]. The novelty is the **intersection**: an *exact span* memory that is a *charged, transmissible, deletable output path*, partitioned from weights by one MDL objective. None of the above stores exact text spans you can delete and ship, and none routes to *zero-compute emission*.

## The architecture

```
bytes ──► MDL tiling (your DP/EM) ──► patches: [phrase spans | residual byte runs]
                                            │
              ┌─────────────────────────────┴─────────────────────────────┐
        local byte encoder                                          trie over phrasebook
        (phrase embedding =                                         + tiny confidence scorer
         compose(span bytes))                                             │
              │                                                    P(phrase|ctx) > τ ?
        global transformer  ◄── COMPUTE ──┐                               │ yes
              │                           └───────────────────────────────┤
        hybrid head: next patch = phrase index | byte residual            │ COPY
              │                                                           │
        phrase index ──► exact bytes (lookup)  ◄──────────────────────────┘
```

- **Units.** Patches come from your existing MDL tiling, so the memory and the network operate on the same spans. Phrase embeddings are **composed from the span's bytes** by the local encoder (plus a small free residual). This is the fix to the v9 wall: similar phrases share byte structure, so the transformer generalizes across them; exactness lives only in the *output* (index → bytes), not the representation.
- **Two output modes.** (a) *Span emission*: predict a whole phrase index in one forward pass → many bytes per pass (the multi-token/BLT/H-Net benefit, always on, safe). (b) *COPY bypass*: when a cheap trie retrieval over the phrasebook plus a tiny scorer is confident, emit the span with **no global-stack FLOPs**. Mode (a) stands even if (b) never calibrates — that de-risks the whole thing.
- **Memory compiler.** An offline pass (your DP/EM) re-induces the phrasebook from the domain corpus, prunes entries whose `usage × bits-saved < charge`, recomputes embeddings. Offline, not online gradient churn — stable.
- **Self-verification (optional).** After a COPY, the local model scores the span; if surprise exceeds a threshold, recompute. Internalized speculative decoding with no second model — the drafter is the charged trie, the verifier is the model's own local head.

## Why it is viable, useful, usable, performant

- **Viable:** every component is proven at scale — byte-patch transformers (BLT/H-Net), trainable sparse memory [[1]], compute routing [[10]], copy mechanisms [[32]], retrieval drafting (REST). The only new research risks are router calibration and the charge term's usefulness, both testable at 30–100M params.
- **Useful:** domain adaptation by *shipping a ≤MB phrasebook* to a frozen base model — no fine-tuning; exact boilerplate reproduction (code, JSON schemas, logs); **machine unlearning becomes trivial** for externalized knowledge (delete the entry, the exact span is provably gone); a privacy-auditable memory (it is literally your data's spans, so you can inspect and filter it).
- **Usable:** standard autoregressive decoding; deployment is `base weights + phrasebook file`; no verifier model, no retrieval pipeline at serve time.
- **Performant:** compute is *content-adaptive*. On repetitive structured text (your measured win) most patches COPY → target ≥2× tokens/s at equal quality; on novel prose it degrades gracefully to the base transformer (no regression). Like memory layers it adds knowledge without FLOPs [[1]], but the knowledge is exact and transmissible.
- **Unique:** the product is a base model plus a marketplace of tiny, auditable, deletable, transferable domain phrasebooks — the archive economics ("pay once, reuse across streams") applied to LLM knowledge.

## Gates and cheap decisive tests (reuse the repo)

- **G1 quality:** within 0.02 bpb of an equal-FLOP byte-patch transformer on text (no regression from memory/router).
- **G2 speed:** ≥2× tokens/s on JSON/log/code at identical outputs.
- **G3 adaptation:** a shipped phrasebook beats a LoRA of equal bytes on domain perplexity.
- **G4 deletion:** removing an entry provably removes its spans from generation.
- **G5 charge honesty:** the MDL loss correlates with held-out compression; total = amortized weights + phrasebook bytes + emitted bits.

Experiments, in your stop-rule culture:
- **E0 (days):** tile enwik8 + your structured corpora with the existing DP/EM; train a 30–100M byte-patch transformer over the hybrid vocabulary with byte-composed phrase embeddings; baselines BPE and BLT patches at equal FLOPs. *Gate:* match BPE on text, beat it on structured. Tests the tiling+composition fix **without** the router.
- **E1 (week):** add span emission + threshold trie-COPY. *Gate:* G2; stop if quality drops on prose.
- **E2 (week):** memory compiler + charge pruning; ship the phrasebook across two checkpoints; deletion test. *Gates:* G3/G4.
- **E3:** learned router + full MDL loss; **ablate λ** — does MDL pruning beat plain frequency pruning? If not, the charge is decorative and you say so.

## What would kill it (honest risks)

1. **Router miscalibration** → wrong copies. Mitigate with threshold + self-verify + mode-(a) fallback; the safe variant carries the paper even if bypass fails.
2. **GPU batching irregularity** from variable spans — the win is largest in single-stream/edge/structured deployment, not big-batch novel prose. State the scope.
3. **Byte-composed embeddings** may lag free embeddings at scale — use the hybrid.
4. **The charge term reduces to frequency pruning** — E3's ablation is the falsifier; pre-register it.
5. **Privacy:** the phrasebook *is* memorized user spans — a leakage hazard and the deletability feature at once; gate induction with privacy filters.

## Bottom line

Don't build a better phrasebook predictor — your data already buried that. Build **an LM that compiles its own tokenizer and its own exact memory from experience under a description-length budget, and routes around the memory when copying is cheaper than thinking.** The phrasebook's weakness (no generalization) becomes irrelevant because the network generalizes; its strengths (exact, compact, charged, transmissible, deletable) become the product. That is a viable, useful, usable, unique, performant, and — at the intersection, not the parts — novel architecture. And E0 reuses your DP, EM, charge, corpora, and judge verbatim, so you can falsify the core in a week rather than a quarter.

