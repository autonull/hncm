# CoLib — the Concept Library Model

**One sentence:** a hierarchical LLM whose middle layer thinks in a *persistent, MDL-priced library of exact concepts with typed holes* — discrete identity at the interface, distributed neural pricing underneath — grown by a wake–sleep induction loop, so that compositionality, planning, and knowledge-as-data become architectural properties rather than emergent hopes.

This is a design proposal, not a result. But it is built so that your two measured negatives become the load-bearing design rules, and so it can be killed in a GPU-month.

## 1. The lessons that force the design

- **v6 (collapse):** a pooled continuous span code is permutation-invariant and the optimizer zeroes it. → Concepts must be **exact and discrete at the interface**: identity is an index into stored bytes, not a vector.
- **v9 (sparsity wall):** independent count parameters per context cannot share statistical strength; unseen transitions dominated at 0.912 coverage. → Pricing must be **distributed**: a concept's embedding is *computed from its content*, so a never-seen concept is a new point in a shared space, not a cold parameter row.
- **The economics inversion (last turn):** weights are amortized globally and free at inference; the *library* is per-domain, cheap, charged. MDL moves from transmission constraint to **training objective**.

The synthesis: **discrete identity, distributed price, exact expansion, holes for abstraction, charge for growth.** No prior system has all five.

## 2. Architecture

```
bytes ──► L0 Realizer (tiny, ~10M): entropy signal, fallback coding, exact expansion
   │  parse = neurally-priced DP (your Viterbi), decoder-synchronous
   ▼
L1 Cortex (the scaled transformer): autoregressive over CONCEPT EVENTS
   event = (concept_id, slot fills…)   fills are recursive events or fallback patches
   cross-attends into the Library (long-term memory)
   ▲
Library: c = (exact template with typed holes, e(c), usage stats, provenance, DL cost)
   ▲
Librarian (wake–sleep): propose → abstract (anti-unification) → price on held-out → accept/prune
```

**The concept.** `def <name>(<args>): <body>`; `"<key>": <value>,`; `the <X> of the <Y>`; an Apache header; an API call shape. Exact bytes where fixed, **typed holes** where variable. `e(c)` comes from a small *causal* encoder over the template (last-state, not mean — order-sensitive, the v6 fix) plus the hole-type signature. Because `e(c)` is a function of content, new concepts are usable zero-shot; because identity is discrete, emission is a memcpy — **zero neural FLOPs and zero hallucination *within* a concept**. Hallucination moves to concept *selection*, which is auditable.

**The cortex.** All the scale lives here, but it runs on events ~5–20 bytes long. Factorization: `p(event) = p(id | history) × Π p(fill | id, history)`. This is where your oracle gap goes: the 1.327 bpb in-sample order-2 structure is real mutual information between concepts; a deep model with shared representations is the estimator that can actually collect it.

**The librarian.** Your existing machinery, repurposed: the GPU candidate scan proposes from fallback hotspots and frequent event n-grams; **anti-unification lifts repeated substructure into holes** (Stitch-style top-down, corpus-guided compression [[39]]); a candidate is accepted iff *held-out* description length drops — your charge rule, now an inductive bias. This is DreamCoder's wake–sleep library learning [[11]], which has never been run with a scalable neural cortex over open text.

**Training.** Hard-EM: DP parses under current neural prices are targets; train cortex + realizer on bits/byte (the judge stays bpb); sleep phase grows/prunes the library; re-parse with versioned replay (your fingerprint/resume discipline). Objective: `L(library) + bits(data | library, weights)`. **Anti-collapse is structural:** the fallback path is priced per byte and the concept channel is the only discounted path — v6 collapsed because the residual was free; here the residual is the expensive route. Library growth is held-out-gated, so training-loss gaming can't inflate it.

**Inference.** The cortex emits events; exact spans are written verbatim; holes and novelty go through fills/fallback. No encoder-only information: the parse *is* the generation (your "decoder never reruns the DP" rule, generalized).

## 3. Why this buys higher-order capabilities — mechanism by mechanism

| Capability | Mechanism | Falsifier |
|---|---|---|
| Systematic compositionality | Holes make concepts *functions over fills*; the combination rule is shared across fills, not memorized per string | Held-out slot-combination split (train partial A×B, test the rest) |
| Affordable planning | Concept-level horizon is 5–20× shorter; beam/MCTS over library moves with DL cost-to-go; COCONUT shows latent-space planning beats token CoT on backtracking tasks [[19]] | Search win at equal FLOPs vs token-level CoT |
| Auditable, verifiable thought | Thoughts are discrete library programs — readable, cacheable, re-expandable, type-checkable; COCONUT's vectors are none of these [[22]] | Re-expansion + type checks catch ≥X% of structured-output errors by construction |
| Knowledge as data | Install/patch/delete a library entry = adaptation and surgical editing *without gradients*; provenance gives exact quotation | New-API install beats LoRA on bits-to-adapt × wall-clock |
| Archive-scale context | 100k events ≈ megabytes of text at fixed attention budget — "the archive is the unit" becomes native | Long-range retrieval/consistency at equal context FLOPs |

The planning point is the deep one: reasoning in a space where each step is a **reusable, typed, exact unit** makes search cheap *and* verification possible. That is the higher-order bet.

## 4. Performance and novelty

**Performance:** one cortex call per 5–20 bytes; exact emission is free; only fills and novelty cost per-byte compute. Target: 3–10× fewer big-model FLOPs/byte at equal bpb. Against BLT/H-Net — which segment bytes into *anonymous, ephemeral, content-defined* patches [[25]] — CoLib's units are **named, persistent, and reused across documents**, so the cortex's input distribution is stationary; that should improve sample efficiency, and it is a measurable claim, not a promise. H-Net's own ablations show the chunking machinery is load-bearing [[30]]; CoLib's risk is the mirror image — segmentation rigidity — absorbed by fallback + induction.

**Novelty delta** (each neighbor has some of the five properties; none has all):

| System | Unit | Persistent identity | Exact | Holes/abstraction | MDL growth | Neural scale |
|---|---|---|---|---|---|---|
| LCM (Meta) | SONAR sentence embedding [[46]] | no | no | no | no | yes |
| BLT / H-Net | ephemeral patches [[25]] | no | no | no | no | yes |
| DreamCoder / LILO | symbolic library [[11]][[34]] | yes | yes | yes | yes | **no cortex** |
| COCONUT | continuous thoughts [[17]] | no | no | no | no | yes |
| **CoLib** | exact span + holes | **yes** | **yes** | **yes** | **yes** | **yes** |

("Learned concept library" exists in symbolic regression [[43]]; the phrase isn't the novelty — the text-LM architecture is.)

## 5. Risk register and kill gates

- **G0 parity (week 1):** 30M cortex + frozen BPE+suffix library vs same-FLOP byte/patch baseline: within 0.03 bpb on text, ≥0.1 win on code/JSON. Miss → fix implementation before tuning (your rule).
- **G1 usage:** concept-covered **bit** fraction ≥50% on target domains. Below → the channel is decorative; narrow the claim to structured/code (the synseg lesson).
- **G2 shuffle control (the v6 ghost):** permute concept ids, keep embeddings — must cost ≥0.2 bpb, or identity isn't load-bearing and you've rebuilt BLT with extra steps.
- **G3 induction:** held-out DL strictly decreases per sleep cycle; report the library *transfer matrix* across domains.
- **G4 capability:** ≥+10 pts on held-out slot combinations vs flat baseline at equal FLOPs. Miss → it's an efficiency paper, not a capability paper. Say so.
- **G5 adaptation:** library install (CPU minutes) beats LoRA (GPU) on bits-to-adapt × time.
- Known hard risks: parse non-stationarity (freeze windows + replay), exposure bias between DP targets and self-generated parses (scheduled sampling over parses), induction cost (cap candidates via your scan).

## 6. Minimal decisive program (~1 GPU-month, your 3080 starts it)

1. **E0 (week 1):** G0/G1/G2 on enwik8 slice + a code/JSON corpus, 30M cortex, frozen library.
2. **E1 (weeks 2–3):** the induction loop on a code corpus; G3 + the "learn a new API in minutes" demo (G5).
3. **E2 (weeks 3–4):** combination splits; G4.
4. **E3:** 100M–1B equal-FLOP scaling check vs an H-Net-style baseline before any big claim.

**What transfers from the repo:** the scan/DP/EM/charge code *is* the librarian; the round-tripping bpb judge *is* the objective; the `.hncm` prior format *is* the library file; primed blocks *are* library-primed context; fingerprints/resume *are* versioned wake–sleep. The project's identity survives the pivot: **"one model, many streams" becomes "one cortex, many libraries."**

## Verdict

Honest status: a bet, with every component demonstrated *separately* — library learning [[11]], hierarchical byte modeling [[25]], concept-level autoregression [[44]], latent planning [[17]] — and the combination (exact, hole-bearing, MDL-grown concepts under a scaled cortex) untried. The two failures that killed H-NCM as an LLM are precisely the two constraints this design satisfies: exactness defeats collapse, distributed pricing defeats sparsity, and holes defeat the memorization of combinations. If G2 and G4 pass, you have something genuinely new: a model whose knowledge grows as *data*, whose thoughts are *programs of exact fragments*, and whose planning horizon is measured in concepts rather than tokens. If they fail, you'll know in a month, cheaply — which is the only kind of hope worth having.

