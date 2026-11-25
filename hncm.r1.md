Yes — but only if you change two load-bearing things and keep everything else. The measured evidence tells you exactly which: **the model class must become neural, and the charge boundary must move.** What survives is the part that is actually yours: the exact-span phrasebook, the DP analysis-by-synthesis tokenization, the archive economics, and the judge. Reassembled around a frozen neural base, that is a defensible novel architecture — not "a transformer with extra steps," but **an LM whose adaptation layer is a lossless compressor of your data.**

## What to keep, what to change (each change pinned to a measured failure)

| Keep / change | Why (your numbers) |
|---|---|
| **Change:** concept bigram counts → frozen neural sequence model over span embeddings | The 0.67 bpb model-class gap; 7.7 obs/context; online negative 2.39–2.45. Counts cannot share statistical strength; this is the wall, not a tuning problem |
| **Change:** "whole model must transmit" → split: **amortized frozen base (global) + charged adapter (per archive)** | The anti-LLM economics. LLMs are the best predictors and worst transmitters; keep the charge where it's a feature (domain memory), drop it where it's a cage (general intelligence) |
| **Change:** coverage requirement → memory is *optional*; a competent neural byte/token path always runs | "Coverage 0.912, rate 4.283": unseen transitions dominated because the fallback was weak. The dictionary becomes an accelerator, not a liability |
| **Keep:** exact spans, dictionary-as-data, charged and inspectable | This is the novel organ. Exactness buys auditability, privacy, and a free decode path |
| **Keep:** DP + EM tokenization, now priced by the neural model | Analysis-by-synthesis segmentation under an MDL objective — genuinely different from BPE, entropy patching (BLT), or learned routing (H-Net) |
| **Keep:** the judge — held-out bits *including the adapter charge* | No LLM paper compares adapters at equal transmitted bytes. This discipline is the scientific contribution regardless of outcome |
| **Keep:** few-calls decoder | Your hncm7 measurement: synseg 0.996 coverage / **485 calls per 1.2 MB**, syngrid **1 call**. With a strong base supplying quality, that dividend becomes real |

One caution carried forward: the v6 negative (pooled code can't carry byte order) forbids *permutation-invariant* span embeddings, not neural ones. Pool over a **causal** encoder's states, and keep exact bytes in the dictionary for the copy path.

## The architecture

**Identity:** a frozen general LM + a per-archive, MDL-charged, exact-span memory that (a) chooses the tokenization, (b) biases prediction, and (c) can *emit verified exact spans in one step*. The archive ships its own adapter — `adapter.hncm` is simultaneously a compressor, a LoRA competitor, and a speculative drafter.

**v1 (minimal, testable in days — no new base training):**
- Frozen open LLM (1–7B, or a small byte/patch model — BLT weights and H-Net code are public [[14]][[23]]).
- Phrasebook memory fitted per corpus with your existing scan + EM + DP, but prices come from the frozen LM: an entry is worth its bytes iff it saves held-out bits. Charge-aware knapsack pruning (this is CREST's datastore-compaction problem [[3]] solved with an MDL objective instead of heuristics).
- **Copy/generate gate:** at each step, score dictionary continuations (longest-suffix retrieval, concept-prior ranking — your bigram survives as the *ranking prior*, where weak models are fine). Copy if expected bits saved > charge price; otherwise the base generates. Copied spans are verified by the base exactly as in speculative decoding, so output quality is *identical to the base by construction* — REST's insight [[2]], now inside the architecture rather than bolted on.
- Fit = scan → EM/DP → prune → pack → record charge. Minutes on your hardware, no gradient farm.

**v2 (the full novel architecture, weeks):** add the hierarchy — causal byte frontend → DP segmentation over the sparse Aho-Corasick match lattice (O(matches), not O(L×dict) — this also fixes your 0.02 MB/s DP lesson) → span embeddings (causal pool + charged per-entry residual) → latent transformer stepping **per span, not per byte** → byte decoder only for novel spans. The memory *is* the tokenizer, per domain: the charge decides how much vocabulary is worth buying. That is the claim BLT/H-Net don't make — their segmentation is fixed or learned globally; yours is MDL-optimal *per archive and transmissible*.

**Prompting, usability:** normal — the base handles conditioning; out-of-domain the matches vanish, the gate closes, and you recover the base exactly (no forgetting, testable). Product properties no LoRA has: the adapter is kB–MB, inspectable byte-for-byte, diffable, deletable (privacy claim you can *verify*), and ships inside the archive next to the data it models. CLI continuity: `hncm lm fit <dir>` / `hncm lm run --adapter`.

## Why this is novel (honest positioning)

- vs **kNN-LM / RETRO**: uncharged GB-scale datastores, token-level, no pruning objective, no exactness guarantee, no transmission story. Yours: charged, MDL-pruned, exact-span, archive-shipped.
- vs **LoRA/adapters**: continuous, opaque, not exact, not a compressor. The headline experiment — **bits saved per adapter byte, phrasebook vs LoRA at equal charged size** — is one nobody runs, and it's a paper even if the answer is mixed.
- vs **BLT / H-Net / MegaByte**: they fix *where compute goes*; they have no per-domain memory and no charge. Your segmentation is driven by a transmissible MDL memory.
- vs **REST / multi-token prediction**: drafting is an inference trick; here the same memory is the adaptation mechanism, and the charge is the training objective.

Novel combination, not novel physics — every piece is de-risked by an existing result, which is exactly what you want.

## The decisive experiment (your culture: gates, stops, pre-registration)

**E1 — equal-charged-bytes probe (2–3 days, frozen base, no training):** on your structured corpora + an enwik8 slice, fit phrasebook adapters at 64 kB / 256 kB / 1 MB; compare held-out bpb against (a) base alone, (b) kNN datastore at equal bytes, (c) LoRA at equal bytes.
- **Gate:** phrasebook beats all three on ≥2 of 3 structured domains at ≤1 MB, with zero quality regression elsewhere (verification guarantees it) and ≥1.5× decode speedup from the copy path.
- **Pre-registered expectation:** LoRA wins on prose; the phrasebook wins on repetitive/structured data where exact spans recur (your synseg 1.389 and 485-calls results are the evidence). If it loses everywhere → clean negative, publish the charged-comparison table, fall back to the drafter product from last turn.

**E2 (1 week):** gate + residuals + charged fit CLI (`adapter.hncm`), round-trip judge on the copy path. **E3 (weeks):** the v2 span-latent hierarchy, only if E1 passes.

## Failure modes to watch

1. **Gate over-copying** (memorization/PII): the charge + held-out selection is the control; the dictionary is inspectable and scrubbable — make that a tested feature, not a promise.
2. **Base ceiling:** small byte LMs trail token LMs at equal size; v1 sidesteps this by using a token base, v2 inherits BLT's scaling evidence [[15]].
3. **The niche risk:** if the win is only structured data, the claim narrows to "domain-adaptive LM with transmissible exact memory" — logs, code repos, legal/medical corpora, on-device personalization. That is still useful, usable, and novel; it is not a general-chat LLM, and you should not want it to be — that race is lost on arrival.

**Bottom line:** stop asking the phrasebook to *be* the intelligence. Let a frozen base supply the intelligence, and make the phrasebook the thing no current LLM has: a per-archive, exact, auditable, MDL-priced memory that tokenizes, adapts, and decodes in one mechanism — judged, as always, in recorded bits. E1 tells you in three days whether the core claim is real.

