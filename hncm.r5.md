# LedgerLM — H-NCM v11: the tokenizer becomes a priced, editable memory

**Thesis:** keep everything that is *soul* in H-NCM — discrete exact-span concepts, a model that pays for itself in bits, the DP parse, coverage→calls — and replace everything that measurably failed: counts → a neural trunk, first-order → attention, transmit-charge → a **budgeted, growable, auditable concept ledger**, SVD tail → a real byte expert. The result is an LLM whose *primary symbol inventory is a persistent learned memory*, not a fixed tokenizer and not an ephemeral retrieval set.

**The structural insight that makes it work (the v6 fix):** in H-NCM the code had to *carry* the bytes, so a permutation-invariant representation collapsed. Invert it: **the code only *names* the bytes.** The bytes live verbatim in the ledger (exactness is structural, residual zero); the embedding is a *free parameter* that can drift to mean anything — idiom, entity, code pattern — compositionally unreachable from the surface. Exactness and generalization stop competing.

## Architecture

**1. The Ledger `L`** — persistent, versioned, cross-document. Each concept:
`c = ⟨bytes b_c (verbatim), free embedding e_c ∈ R^d, usage n_c, bits-saved S_c, storage K_c, provenance ptrs, timestamps⟩`.
`e_c` warm-starts from the byte expert's *last-state* readout of `b_c` (never mean-pool — v6), then trains freely. Total storage `Σ K_c ≤ B` (64–256 MB). The ledger is a human-readable, greppable, editable file.

**2. Induction engine (a market, not a fit).** Background mining of recurring spans (your `n10.py` GPU scan: 2.7M candidates at count≥32 on 90 MB — reuse it). Price each candidate with the *current neural model*:
`utility(c) = Σ_occ [log p_concept(c|ctx) − log p_bytes(b_c|ctx)] − λ·8·K_c`.
**Admit iff utility > 0; evict when realized utility < rent.** Your EM rule "usage becomes the cost, unused strings drop" becomes an admission economics. The MDL discipline survives — relocated from a per-archive transmission tax (anti-LLM) to a lexicon budget (pro-LLM).

**3. Dual-pathway trunk.** A decoder-only transformer over the *mixed* sequence (concept IDs + raw bytes), plus a small **byte expert** for novel spans (your measured 5.6 bpb uncovered tail says: give the tail a real model). One softmax over `L ∪ {256 bytes} ∪ specials` with a learned abstain gate. Emitting a concept writes its exact bytes **atomically — one forward pass per phrase**; novel text is spelled byte-by-byte. Quality is neural everywhere; verbatim commitment is a first-class output symbol, not a retrieval hack.

**4. Parser.** Training: your DP exact-cover with neural costs, plus parse *sampling* (subword-regularization style) to marginalize segmentation ambiguity. Inference: lazy decoder-synchronous longest-match over an Aho-Corasick automaton of the ledger (your TODO5 N4 "adaptive lazy parse" idea, promoted).

**5. Objectives.** `L_LM` (next-symbol CE — a real, promptable LM) + `L_align` (KL between phrase-level and byte-level distributions at aligned positions — adopted from the dynamic-vocabulary paper, which suffered a PPL regression without robust alignment [[2]]) + `L_ground` (auxiliary reconstruct-`b_c`-from-`e_c`; regularizer only, exactness never depends on it) + `L_multi` (next-k-concepts prediction). Induction is discrete; no gradients through the market.

## What changed, and why each change is forced by your own measurements

| H-NCM | LedgerLM | Forced by |
|---|---|---|
| count bigram over indices | attention trunk over free `e_c` | 0.67 model-class gap; 7.7 obs/context sparsity wall |
| code carries bytes | code names bytes; bytes stored | v6 collapse |
| SVD/count tail | neural byte expert | uncovered tail = the whole deficit |
| fit-once dictionary | growable/evictable ledger | coverage 0.912 → 4.283: static lexicons can't track a domain |
| charge per archive | MDL rent on ledger storage | LLM weights amortize globally; the budget belongs on memory |
| DP at encode time | DP for induction, lazy parse at inference | 0.02 MB/s is not a generation speed |
| "fewer calls" compression claim | atomic phrase emission | calls are a *latency* asset, not a rate asset |

## Novelty, honestly positioned

| Neighbor | What it does | The delta |
|---|---|---|
| Dynamic Vocabulary (EMNLP'24) | spans as generation bricks, but **ephemeral**: retrieved per query, encoded on the fly by a phrase encoder; training phrases dropped after learning [[2]] | persistence + free identity + admission market + byte expert + adaptation-by-growth |
| CoG / REST | copy/retrieve spans from retrieved docs, verify | concepts are *native output symbols* of one model, no separate verifier |
| BLT / H-Net / MegaByte | anonymous dynamic patches, re-encoded every occurrence | named, cached, priced concepts; encode once at admission, reuse forever |
| SuperBPE / MWT | static multi-word vocab (top-k frequency) [[8]] | learned, budgeted, evictable, provenance-carrying; economics not frequency |
| kNN-LM / continuous cache / RETRO | unbounded or heuristic activation/datastore memory [[40]][[45]] | exact-span entries with MDL admission, editable, auditable |

Each piece has a neighbor; the synthesis — **persistence × free embeddings × MDL market × dual pathway × lexicon-growth-as-adaptation** — I could not find published. That is the claim, and S1 below is designed to falsify it cheaply. Note the cautionary datum: the dynamic-vocabulary paper's PPL *worsened* (8.03 vs 3.60) even as MAUVE/latency improved [[2]] — phrase emission without alignment and a byte fallback is a known failure mode; `L_align` + the expert + parse sampling are the explicit fixes.

## Why it is performant (your numbers, reused)

- Coverage 0.8–0.9 → 3–6× shorter sequences → attention FLOPs down ~quadratically on covered mass; **KV cache shrinks with coverage** → ~4× context for free.
- Atomic emission → fewer decode steps; the ephemeral-phrase version already measured −20% latency [[2]]; persistent free embeddings additionally skip re-encoding every recurrence (BLT pays patch encoding per occurrence; LedgerLM pays once, at admission).
- **The unique curve: FLOPs/byte *decreases as the ledger learns the domain*.** Efficiency has a learning curve. No fixed tokenizer and no anonymous-patch model has this.

## Why it is useful and usable

- **Domain adaptation without weight updates:** deploy on logs/JSON/code → induction admits domain concepts → sequences shorten, quality rises, trunk frozen. Your synseg win is the evidence that domains have learnable phrasebooks. Auditable, reversible, cheap.
- **Surgical editing/unlearning:** a wrong fact stored as a concept is one ledger row — edit or delete it, exactly. Diffuse weights cannot do this.
- **Privacy:** memorization is *explicit and searchable* — verbatim bytes with provenance. You can find and remove training text; no neural-weight LLM offers that.
- **Citations/provenance for free** (every concept points at sources; ephemeral phrases already improved citation reliability [[2]] — persistent provenance is stronger).
- **Usable interface:** standard prompt→text; plug-and-play on an open backbone (the dynamic-vocab paper did exactly this with GPT-2/TinyLlama [[2]]); the ledger is a file you can grep.

## Risks, gates, staged falsification (your culture)

- **S0 (2–3 days):** freeze spec; reuse scan/DP/EM/corpora; ledger + automaton parser + mixed dataloader.
- **S1 (1–2 weeks, one 3080):** 30–60M dual-pathway vs byte-transformer vs BPE vs BLT-style patches, equal FLOPs, enwik8 + structured suite. **Gate:** ≥0.1 bpb over byte baseline *or* ≥1.5× tokens/s at equal bpb, **and concepts ≥ anonymous patches** — if patches match, concept identity is decorative: stop, publish the negative, pivot to the drafting niche.
- **S2 (1 week), four ablations isolating the novel claims:** free vs composed embeddings (the v6 fix); MDL admission vs top-k (the market); phrase vs byte emission (the speed); ledger-growth vs LoRA on a new domain (the adaptation). **Gate:** growth within 20% of LoRA quality at ≤1% of its compute, with editability LoRA lacks.
- **S3:** scale; distill from an open 1B backbone; evaluate editing, unlearning, citations, verbatim-copy rate (rent `λ` is the regurgitation knob).
- Pre-registered negatives: PPL regression beyond 5% at equal FLOPs → alignment failure, fix or stop; ledger non-stationarity → versioned ledger, slow admission, EMA stats.

**Honest scope:** these are *lexical/phrasal* concepts — idioms, entities, boilerplate, code patterns — not semantic abstraction. Clustering `e_c` into senses is v2; don't claim it now.

## The part I like most

The compressor doesn't die — **it becomes the auditor.** The ledger is literally a charged model; your round-tripping recorded-size judge now measures the ledger's MDL health while the LM loss measures its intelligence. Two judges, one artifact, and the discipline that made v9's negative trustworthy becomes the thing that makes v11's positive trustworthy.

Bottom line: this is not H-NCM patched into an LLM — it is H-NCM's *accounting soul* transplanted into the one place a symbolic memory genuinely beats diffuse weights: a persistent, priced, editable concept inventory inside a neural trunk. Every failed component is replaced for a measured reason; every kept component is kept because it measured well; and the whole thing is falsifiable on your GPU in under three weeks.
