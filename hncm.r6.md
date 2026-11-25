Yes — but the invention requires exactly one structural inversion, and your own measurements dictate it. Here is a concrete architecture, built from H-NCM's organs, that is trainable on your 3080, deployable as a CLI, and has capabilities no current LLM has.

# CMT: the Concept Memory Transformer (H-NCM v11)

**One sentence:** a small byte transformer whose *working vocabulary* is an MDL-induced, content-addressed memory of exact spans ("concepts") — induced per archive, tied to the backbone's geometry, priced by counts, emitted through a learned copy-or-compute gate — so the phrasebook becomes the vocabulary, the archive becomes the adapter, and the charge becomes the regularizer.

## The five mutations, each forced by a measurement

| H-NCM had | CMT changes it to | Forced by |
|---|---|---|
| Counts *are* the model | Counts are a **prior**; a neural score over **backbone-tied concept embeddings** is the model | The 0.67 model-class gap; 7.7 obs/context sparsity wall |
| Model transmitted per archive | Backbone frozen and global; only the **concept pack** (dictionary + counts + tiny head) is transmitted | LLM economics — but the charge discipline survives exactly where it's a feature |
| Compression objective | Generation objective **with the charge as a training term** | Same MDL, now the judge of adaptation cost |
| DP at encode, never decode | DP at *fit* (induction), lazy gate at *decode* | Your TODO5 N4 lazy-parse plan, promoted to the core |
| Exact spans | **Unchanged** | This is the source of every unique capability — do not touch it |

The linchpin is the tied embedding: `e_c = Enc_backbone(bytes_c)`, precomputed once at induction. Similar spans get similar vectors, so statistical strength is shared — the exact thing your count table could not do — while a concept's *identity* remains discrete and byte-exact. Generalization comes from the backbone; exactness, auditability, and cheap adaptation come from the dictionary. Your v9 negative is the design spec.

## The architecture

Components: (1) a 20–50M byte transformer backbone (BLT-style patches optional), trained once on general text+code+JSON; (2) the concept memory: dictionary `D` (4k–64k exact spans, induced by your existing DP/EM usage-cost machinery), tied embeddings, count features; (3) a tiny scoring head (~1–3M trainable params); (4) a gate.

At position `t`, backbone state `h_t`:

```
score(c) = (W_q h_t)·e_c / √d  +  λ·log p_count(c | c_prev)     c ∈ D
π_t      = σ(w·[h_t, max score, coverage features])            # copy-or-compute gate
p(unit)  = π_t · softmax(score) ⊗ exact_bytes(c)  +  (1−π_t) · p_byte(·|h_t)
```

Emit a concept → write its exact bytes, one decision. Emit fallback → the backbone is just a byte LM. **Training:** teacher-force the MDL-DP parse of the corpus; loss = mixture NLL + β·|pack| (the charge); re-induce `D` on an EM schedule. No RL, no exotic optimization — it's pointer-generator-style training on a derived tokenization. Trainable surface: projections + gate + optional LoRA.

**Performance, in two honest stages.** v1: when a concept fires, its bytes are *known*, so the backbone advances over the whole span in one parallel pass instead of `|span|` sequential sampling steps — speculative decoding's latency win, with the memory as the drafter and no second model. At your measured structured coverage (~0.9, spans of 8–20 B) that is a 5–10× cut in sequential depth; on enwik8 (0.815 coverage) ~3×. v2: concept tokens enter the backbone's *input* sequence (embedding + length), cutting FLOPs too — this is where CMT becomes a two-clock architecture, the exact-span cousin of BLT/H-Net patching. **Quality floor:** a verification mode (backbone checks emitted span bytes, rejects the tail below threshold) guarantees output quality ≥ backbone, always — the NEST trick, but the drafter is trained and compact [[11]].

## The unique capabilities (the reason to build it)

1. **Provenance-typed generation.** Every output byte is tagged `(concept-id, source-offset)` or `NEURAL`. The model shows its work at byte granularity: which words are copied evidence, which are invention. NEST does post-hoc token-level attribution [[12]]; here attribution is *by construction*, exact, and free. For RAG, agents, compliance, science — a hallucination *label* no current LLM can emit.
2. **Knowledge as a pack — git for model knowledge.** `cmt fit repo/ → repo.pack` (~200 KB, SHA-verified); `attach`, `detach`, `diff`, `merge`, `rm <concept>`. Adaptation is literal readable spans + counts: you can inspect, edit, and *exactly delete* what was learned. No catastrophic forgetting (backbone frozen), no gradients on user data at serve time, on-device personalization you can audit. LoRA is opaque weights; memory layers are parameters [[20]]; RAG pays context cost forever. Your append/dedup/verify/shipped-prior machinery maps 1:1 onto pack operations. This is a genuinely new *operations* story for LLMs, and it's your home turf.
3. **Hard-exact constrained generation.** Restrict `D` to a schema/protocol/DSL's validated phrases → every concept emission is guaranteed-valid by construction, stronger than token-level grammar masking.
4. **Speed where your method already wins:** the synseg/JSON/log structured advantage becomes generation latency, not just compression.
5. **Calibrated novelty telemetry.** Gate abstention rate + pack charge = a runtime "I'm out of my depth" signal for drift detection and abstention — the MDL accounting turned into epistemics.

## Honest novelty map

COG already generates by copying spans from a static collection [[1]]; NEST retrieves and verifies tokens for attribution [[11]]; kNN-LM interpolates single-token distributions over a massive datastore [[30]]; memory layers add trainable key-value parameters [[20]]; BLT/H-Net learn patches neurally; REST drafts from suffix arrays. **None makes the vocabulary itself an MDL-charged, transmissible, editable exact-span memory that is trained into the model and compresses its compute.** The claim is that synthesis plus the charge discipline plus the pack-ops capability — and the tied-embedding mechanism as the specific cure for your measured sparsity wall. That is a defensible paper-shaped contribution, not a repackaging.

## Build plan (3080, your gate culture)

- **v0, week 1 — inference-only, zero training.** Freeze a small byte backbone (train 20–40M on enwik8+code+JSON in ~1–2 days, or start from an open byte model). Induce `D` with the existing DP/EM; tied embeddings; cosine+count score; a 3-parameter gate fit on held-out; verification mode on. *Gates:* structured ≥1.5× latency at backbone-exact quality; general text ≤0.01 bpb loss; pack ≤1 MB; provenance round-trip 100% (your judge).
- **v1, weeks 2–3 — train the head.** Mixture NLL + charge on DP parses. *Gate:* ≥20% acceptance-per-pack-byte over v0. **Linchpin falsifier:** hold out 10% of `D`, test prediction of unseen-but-similar spans — tied embeddings must beat counts-only, or the core hypothesis is dead and you record the negative.
- **v2, weeks 4–6 — two-clock backbone.** Concept tokens in the input sequence, fine-tune. *Gate:* ≥2× throughput on structured at equal quality.
- **Stop rules:** gate can't beat a tuned REST/NEST baseline at equal datastore bytes → ship v0 as an inference technique, not an architecture; verification acceptance collapses on free generation vs teacher-forcing → exposure-bias problem, fix the gate before scaling.

## The risk that kills it

Gate miscalibration under distribution shift: the parse you trained on is not the text you generate. The cheap check is v0's verification mode — it bounds the damage to zero quality loss and measures the acceptance gap directly, in week one, before any training investment.

**Bottom line:** keep the phrasebook, replace the accountant with a neural one, freeze the world model, and turn the charge into an adapter. The result is not "H-NCM as an LLM" — it's an LLM whose memory is your archive format: exact, priced, transmissible, editable, and self-attributing. Every unique capability traces back to the one property you refused to give up in v6→v7: the code is an exact span, and the decoder writes the true bytes.

