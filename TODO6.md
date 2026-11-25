# TODO6 — H-NCM v11: the concept memory LM (four bets, one spine, seven stages)

Status: v9's text negative is measured (2.664 at 100 MB; PPMd 1.834; PPMd over
the same parse 1.992). The seven sketches (`hncm.r1-7.md`) are one family, not
seven architectures: an exact span memory, a neural trunk, an MDL price, and a
copy/compute gate. This plan selects **four bets**, builds **one spine**, stages
the architecture in **seven versions**, and orders **six experiments** by
information per GPU-hour. The load-bearing question: does exact-span memory,
priced by MDL and read by a neural trunk, beat the count model at equal parse —
and can it adapt a frozen model by installing kilobytes?

The later sketches are not alternatives; they are the stages. r5's ledger is
the persistent, market-priced memory; r6's pack is its deployment and
provenance format; r7's holes are its abstraction and planning layer. The plan
builds them in that order, each on the same spine.

New here: **the charge arithmetic**. A neural trunk is 10-100x the v9 model,
so every result is reported three ways: `rate` (payload), `shared` (payload +
memory, base amortized), `self` (payload + base + memory). The 100 MB `self`
text win is a measured question, not an assumption; the archive plan (TODO5)
remains the product floor.

## 0. Start here

```bash
python neural.py selftest
python neural.py lex --corpus enwik8_100m --dict 32768 --min-count 32 \
  --out runs/v11/lex100.npz
python neural.py parse --corpus enwik8_100m --lex runs/v11/lex100.npz \
  --out runs/v11/parse100.npz
python neural.py train --arm native --parse runs/v11/parse100.npz \
  --params 60m --steps 20000 --out runs/v11/ckpt-60m.pt
python neural.py eval --ckpt runs/v11/ckpt-60m.pt --val 10000000 \
  --baselines ppmd,n10 --out runs/v11/e1.json
python neural.py distill --ckpt runs/v11/ckpt-60m.pt --bits 4 \
  --out runs/v11/ckpt-4m.pt
python neural.py install --ckpt runs/v11/ckpt-lexagnostic.pt \
  --lex runs/v11/lex-synseg.npz --out runs/v11/adapter-synseg.hncm
python neural.py pack runs/v11/ckpt-lexagnostic.pt \
  --lex runs/v11/lex-synseg.npz --out runs/v11/pack-synseg.hncm
python neural.py attach runs/v11/pack-synseg.hncm --ckpt runs/v11/ckpt.pt
python neural.py trace --ckpt runs/v11/ckpt.pt --lex runs/v11/lex-synseg.npz \
  --prompt "..." --out runs/v11/trace.jsonl
python neural.py bench --ckpt runs/v11/ckpt-lexagnostic.pt --regime auto \
  --out runs/v11/speed.json
# controls (cached; rerun only if cold)
python concept_lm.py run --corpora enwik8_10m --vocab union \
  --dicts 16384 --prunes 16384 --out runs/v11/control-concept.json
python n10.py proof --val 10000000 --out runs/v11/control-n10.json
python codec.py run --corpora shakespeare,synseg,syngrid --try 3
```

Artifacts: `runs/v11/` — lexicons, parses, checkpoints, `e1.json`, `e2.json`,
`speed.json`, `ops.json`, `trace.jsonl`, `library-*.json`, `pack-*.hncm`,
`GATES.md`, `PROOF.md`. Every artifact is keyed by corpus SHA and lexicon SHA;
a rerun is seconds. The controls are the v9 tables, not new numbers.

## 1. What is fixed (the design rules)

| measurement | rule |
|---|---|
| v9's 0.83 gap is 0.67 model class + 0.16 dictionary; PPMd over the parse is 1.992 | the predictor is a neural trunk; counts are features, priors and prices — never the predictor |
| 7.7 observations/context; the online interpolation is a negative (2.39-2.45) | price a span from its **content** (a causal encoder's state), not from a per-ID row; share strength across similar spans |
| v6: a pooled span code conveys ~1 bpb while it costs 3.1; the optimizer collapses it | identity is a **discrete index**; the bytes are stored verbatim; the residual is zero |
| coverage 0.912 → rate 4.283; the uncovered 24% at 5.6 bpb is the deficit | memory is optional; the fallback is a **real byte expert**, not the SVD cold start |
| weights amortize globally; v9's 300 kB model is 0.024 bpb over 100 MB | the base is shared; the per-archive charge is the **lexicon**; report `rate`/`shared`/`self` |
| the decoder never reruns the DP (0.0013 bpb) | no DP at inference; lazy longest-match over an Aho-Corasick automaton; span emission is one decision |
| in-sample oracle 1.327 (order-2) vs 2.072 (order-1) | the headroom is real mutual information; a deep shared estimator is the only way to collect it |
| `calls` in v9 = 1 + single-byte tokens, not total units (253,853 vs 3.73M at 10 MB) | judge speed in **wall-clock at equal recorded bits**; never sell a call-count ratio as a speedup |

**The later rules** (each is a design constraint the later sketches add, and
each is pinned to a measured failure or a known regression):

| rule | from | why |
|---|---|---|
| the memory is **persistent and versioned** | r5 | a static lexicon cannot track a growing domain; coverage 0.912 still leaves 24% unseen |
| the price is **content-addressed and tied** | r5, r6 | the 7.7 obs/context wall and the v6 collapse; similar spans must share strength |
| emission is **atomic** | r5, r6 | one decision per span; zero residual; decoder-synchronous |
| the loss **aligns** phrase and byte distributions | r5 | the dynamic-vocabulary paper regressed PPL (8.03 vs 3.60) without robust alignment |
| a concept is a **template with typed holes** | r7 | composition is the shared rule; memorizing every fill is the failure |
| every output byte is **provenanced** | r6 | audit, citation, deletion; no post-hoc attribution |
| the model is **two clocks** | r3, r6, r7 | the byte expert runs at byte rate; the trunk at unit rate |
| the memory is a **pack** | r6 | install, diff, merge, delete without gradients; "LoRA you can read" |

**The scorecard (what "best" means).**

| claim | metric | bar |
|---|---|---|
| the model class is fixed | payload | ≤1.98 (PPMd-over-parse 1.978) |
| the text win | shared | ≤1.834 (PPMd) |
| self-contained | self | ≤2.664 (beat v9); ≤1.99 (stretch) |
| the adapter | shared, ≤1 MB | ≥0.15 bpb saved vs D=∅ |
| structured | shared/rate | ≥0.1 bpb over byte, or ≥1.5x tokens/s |
| the gate | acceptance/quality | ≥50%, delta ≤0.02 bpb |
| speed | wall-clock | ≥1.3x text, ≥3x structured |
| the ledger | ops | installed subset ≤1 MB, install ≤1 min, delete, provenance 100% |
| the library | held-out slots | +10 points; shuffle ≥0.2 bpb |

## 2. The design space: sketches, bets, stages

All seven are the same spine with four knobs: where the memory sits, how it is
priced, how it is charged, and what it emits.

| sketch | memory sits | price | charge | emits |
|---|---|---|---|---|
| r1 adapter | outside the base | counts + base | adapter bytes | spans |
| r2 C2LM | inside the output | neural + counts | ledger rent | span or byte |
| r3 PhraseLM | installed lexicon | neural | adapter bytes | span or byte |
| r4 Concept-LM | the vocabulary | neural | lexicon bytes | concept |
| r5 LedgerLM | persistent ledger | neural market | ledger rent | concept or byte |
| r6 CMT | tied memory | neural | pack bytes | concept |
| r7 CoLib | library with holes | neural | library bytes | event |

**What each sketch adds** — this is the map from idea to module to experiment:

| sketch | unique addition | module | experiment |
|---|---|---|---|
| r1 adapter | the equal-charged-bytes probe; install/uninstall | `Install`, `Score` | E2 |
| r2 C2LM | the hybrid output; the copy bypass; the charge in the loss | `Gate`, `Market` | E3 |
| r3 PhraseLM | lexicon-agnostic training; three heads; safe mode | `Train`, `Gate` | E1, E2 |
| r4 Concept-LM | neural prices; joint EM; the P0 model-class test | `Trunk`, `EM` | E1, E3 |
| r5 LedgerLM | persistence; the admission market; alignment; the dual pathway; the learning curve; editing | `Ledger`, `Market`, `Align` | E3, E4 |
| r6 CMT | tied embeddings; pack ops; provenance; constrained generation; novelty telemetry; the two-clock backbone | `Pack`, `Trace`, `Gate` | E2, E4 |
| r7 CoLib | holes; the librarian; compositionality; planning; auditable thought | `Library`, `Librarian`, `Holes` | E5 |

**N = 4.** The bets, in build order:

| bet | sketches | claim | decisive test | product |
|---|---|---|---|---|
| **B2 the vocabulary** | r4 | a neural trunk over exact spans beats the count table at equal parse; the text rate returns | E1 | the text/structured rate |
| **B1 the adapter** | r1, r3, r6 | installing a kB lexicon on a frozen trunk beats D=∅, kNN and LoRA at equal charged bytes | E2 | "LoRA you can read" |
| **B3 the ledger** | r2, r5, r6 | native span emission + a market + a copy gate beats a byte LM on structured data and ships edit/delete/provenance | E3, E4 | the auditable memory |
| **B4 the library** | r7 | holes buy systematic composition and cheap planning | E5 | the capability |

The order is information, not ambition: B2 decides whether the pivot is real;
B1 is the product that survives if it is not; B3 is the architecture; B4 is the
capability bet, and it starts only after B2/B3 pass.

**Why these four.** Each is a capability no current LM has, and each is
falsifiable on one GPU:

- **B1** is an adapter you can read, diff, delete and ship: adaptation without
  gradients, at a charge the judge can see; the same artifact ships inside the
  archive and compresses it.
- **B2** is the only route to the text rate that v9 measured as the deficit.
- **B3** is provenance at byte granularity, surgical unlearning, hard-exact
  constrained generation, and speed where the data is structured. Its unique
  curve: FLOPs/byte falls as the ledger learns the domain.
- **B4** is composition, planning and type-checked thought over exact units —
  the higher-order bet.

**Neighbors and the delta.** Each neighbor has one piece; none has the union.

| neighbor | it does | the delta |
|---|---|---|
| dynamic vocabulary | ephemeral phrases, re-encoded per query | persistent entries, free identity, admission, byte expert |
| CoG / REST | copy/retrieve spans, verify | concepts are native output symbols; no separate verifier |
| BLT / H-Net | anonymous patches, re-encoded every occurrence | named, cached, priced, reused across documents |
| kNN-LM / RETRO | an uncharged datastore, token-level | MDL admission, exact spans, editable, provenance |
| DreamCoder / Stitch | a symbolic library, no cortex | exact holes under a scaled neural cortex |
| LoRA | opaque dense weights | exact, auditable, transmissible pack |
| SuperBPE / MWT | a static frequency vocabulary | learned, budgeted, evictable, provenance |
| COCONUT | continuous thoughts | discrete, re-expandable, type-checkable |

**The seven stages.** Each stage is shippable and gated; no stage starts before
the previous gate passes.

| stage | version | adds | experiment |
|---|---|---|---|
| v11a | the spine | lexicon, parse, encoder, trunk, heads, judge | E0 |
| v11b | the native LM | the concept trunk on the fixed parse; distillation | E1 |
| v11c | the adapter | the frozen base + install; safe mode | E2 |
| v11d | the ledger | the market, the dual pathway, the gate | E3 |
| v11e | the ops | pack, provenance, constrained generation, telemetry | E4 |
| v11f | the library | holes, the librarian, planning | E5 |
| v11g | the scale | 100-300M, 1-2B units | E6 |

## 3. The charge arithmetic (the constraint the sketches dodge)

A 60M-parameter trunk is 30 MB at 4-bit, 60 MB at 8-bit, 120 MB at fp16. At
100 MB the 4-bit base alone is 2.40 bpb; no payload recovers that. The budget:

| stream | base | base bpb | payload needed to beat PPMd 1.834 |
|---|---:|---:|---:|
| 100 MB | v9 300 kB | 0.024 | 1.810 (v9 payload 2.640: negative) |
| 100 MB | 4 MB | 0.320 | 1.514 |
| 100 MB | 30 MB | 2.400 | impossible |
| 1 GB | 30 MB | 0.240 | 1.594 |
| 10 GB | 30 MB | 0.024 | 1.810 |
| 1.2 MB (synseg) | 3 MB | 20.0 | impossible |

**The rule:** the neural pivot is a deployment-scale bet, not a per-file bet.
The 100 MB `self` claim is negative unless the trunk is ≤4 MB; the 100 MB
`shared` claim is valid for the adapter; the text-rate claim needs ≥1 GB or a
distilled trunk. The `self` row is not a failure to hide; it is a measured
crossover, and reporting it is the contribution.

**The ledger's charge.** The per-archive charge is the ledger, not the base:
`charge = 8·Σ K_c / n`, summed over the installed subset, where `K_c` is the
entry's packed bytes. The ledger budget `B` (64-256 MB) is a *deployment*
budget, not a per-file one; the per-file charge is the installed subset. This
is the r5 inversion: the MDL tax moves from "transmit the model" to "rent the
memory".

**The crossover.** A 30 MB base at 4-bit amortizes at `240 MB / (1.834 -
payload)` bytes of stream. At payload 1.6 bpb the crossover is ~1.03 GB; at
1.5 bpb it is ~0.72 GB. The `shared` gate at 100 MB is conditional on that
deployment; the `self` gate is the 4 MB distillation. Report both.

## 4. The architecture

### 4.1 The spine

```
bytes ──► lexicon + DP parse ──► units (span ids | byte runs)
                                      │
                    ┌─────────────────┴─────────────────┐
              span content                          unit sequence
              causal encoder                        trunk (Stack)
              e_c = Enc(bytes_c)                    h_t
                    │                                   │
                    └────────► concept head ◄───────────┤
                               byte head ◄───────────────┤
                               gate ◄────────────────────┘
                                      │
                         copy (exact bytes) | compute (byte)
```

- **Lexicon.** Exact spans, ids, counts, rent, provenance. Built by the `n10`
  scan and the `concept_lm` DP/EM; serialized as a bytes blob + JSON.
- **Units.** A byte stream plus the parse → a sequence of ids and byte runs.
  A single-byte run is the fallback; a span id is a copy.
- **Encoder.** A small causal byte encoder (~10M) over span content; `e_c` is
  the last state, never a mean (the v6 rule). New spans land in the same space
  with no training.
- **Trunk.** `hncm5.Stack` over unit embeddings (~30-60M pilot); the expensive
  part runs at unit rate, the encoder at byte rate.
- **Heads.** Concept logits over |D|, byte logits over 256, a gate. The mixture
  is one loss: `CE_concept + CE_byte + μ·KL(phrase || byte)`; the charge
  `λ·8·|D|/n` enters the **induction objective**, not the fixed-D gradient (at
  fixed D it is a constant, so the λ ablation is against frequency pruning).
- **Gate.** Copy if the concept score beats a calibrated threshold; verify the
  span's bytes with the byte head; roll back if surprise exceeds τ. Quality is
  bounded by the base; speed is the dividend.
- **Judge.** Recorded bits = payload + packed memory + packed base at the same
  bytes; `rate`, `shared`, `self`; round-trip every artifact. Reuse `codec.py`.
- **Codec.** The neural CDF replaces the count CDF; `codec.py`'s arithmetic
  coder and decoder are unchanged. The decoder runs the same trunk, so the
  bitstream is still the judge (rule 1).

**Pilot spec (one 3080).**

| module | pilot | scale | note |
|---|---|---|---|
| encoder | 4×256, 4 heads, ~3M | 6×384 | causal, last state |
| trunk | 6×512, 8 heads, ~35M | 12×768, ~110M | `Stack`, unit positions |
| embeddings | tied to the encoder; a free residual for count ≥32 | | the v6 fix |
| schedule | batch 32×1024 units, AdamW 3e-4, warmup 500, 20k steps, clip 1.0 | | fp16/bf16 |
| total | ~40M | 100-300M | 30/60/120 MB at 4/8/16-bit |

### 4.2 B1 — the adapter (r1, r3, r6)

**What it is.** A frozen base plus an installed lexicon `D`. The base generates
as it always does; the lexicon *biases* prediction and the copy path *emits*
exact spans. There are no native concept symbols: the output is bytes, and the
copy is a verified shortcut.

**Install.** `e_c = Enc(bytes_c)` is computed once per entry by the frozen
encoder; `D` is a pack (bytes + embeddings + counts + provenance). Installing
is milliseconds for 32k entries; removing is deleting the pack. No gradients,
no weight update, no continued pretraining.

**Score and gate.**

```
score(c) = (W_q h_t)·e_c/√d + α·log p_count(c | c_prev)
π_t      = σ(w·[h_t, max score, coverage features])
p(unit)  = π_t·softmax(score) + (1-π_t)·p_byte(·|h_t)
```

The count term is a **ranking prior**, not the predictor; the trunk supplies
the context, the count table breaks ties. This is the r1/r6 split.

**Equal-charged-bytes.** Compare base alone, base + `D`, base + kNN, and base
+ LoRA at the same `8·|adapter|/n`: `score = (CE_bits + 8·|adapter|)/nbytes`.
The base weights are excluded (amortized). This is the one comparison no LLM
adapter paper runs; it is the scientific contribution of B1.

**v0 (4 h).** Freeze an open byte base (or the E1 trunk); induce `D` with the
existing DP/EM; fit a 3-parameter gate on the held-out 10%; verification on.
This is the zero-training product; E2 is the trained version.

**Safe mode.** After a copy, the byte head scores the span's bytes in one
parallel pass; if mean surprisal > τ, roll back and generate byte-wise. τ is
calibrated on held-out data to bound the quality delta (≤0.02 bpb). Fast mode
skips the rollback with a conservative gate. Quality is bounded by the base by
construction.

**Product.** `adapter.hncm` — a 10 kB-1 MB pack that is a compressor, an
adapter, and a drafter at once. It ships inside the archive next to the data
it models. "LoRA you can read."

**Gates.** G2a installation ≥0.15 bpb at ≤1 MB; G2b acceptance ≥50% at ≤0.02
bpb delta; G2c no regression on the base's own domain. **Stop:** no transfer →
per-domain adapter only.

### 4.3 B2 — the native LM (r4)

**What it is.** The trunk's output vocabulary *is* the lexicon: at each step it
emits a concept id or a byte. The parse is fixed for the compression eval (the
v9 DP path); the model predicts the next unit; the exact bytes are a lookup.

**Two clocks.** The span encoder runs at byte rate on the current span; the
trunk runs at unit rate over the unit history. The expensive attention is over
units (~2.7 B/unit on enwik8; 5-20 B/unit on structured), so the sequence is
2-20x shorter than bytes. This is the BLT/H-Net dividend, but the units are
named, persistent and exact.

**Joint EM.** E-step: parse under the current neural prices (Viterbi over the
lexicon lattice, chunked); M-step: train the trunk and encoder on the parse;
L-step: add spans whose `usage × surprise-reduction` exceeds the rent, prune
the rest. Parse *sampling* (not just argmin) regularizes the segmentation.

**Distillation.** The 60M teacher is distilled to a 4 MB 4-bit student (prune,
quantize, distill on the teacher's distribution). The student is the
self-contained 100 MB ladder (G1c/G1d); the teacher is the model-class test
(G1a/G1b).

**Gates.** G1a payload ≤1.98; G1b shared ≤1.834; G1c 4 MB self ≤2.664; G1d
4 MB self ≤1.99. **Stop:** payload >2.2.

### 4.4 B3 — the ledger (r2, r5, r6)

**What it is.** A persistent, growable, auditable memory of exact spans that
the trunk reads natively. The ledger is the vocabulary, the adaptation
artifact, the knowledge store and the codec, in one file.

**The record.**

```
c = < bytes b_c (verbatim), id, e_c (tied to the encoder + free residual),
      n_c (usage), S_c (bits saved), K_c (packed bytes), rent,
      provenance[], timestamp, holes? >
```

`e_c` is computed from the span's content by the causal encoder; the free
residual is only for frequent entries. The identity is the id; exactness is
structural (bytes stored, residual zero).

**The market.** The admission price is the ledger's MDL rule:

```
utility(c) = Σ_occ [log p_concept(c|ctx) - log p_bytes(b_c|ctx)] - λ·8·K_c
```

Admit iff `utility > 0`; evict when the realized utility falls below the rent.
The market is discrete: no gradients through admission. The λ ablation is the
falsifier: if λ=0 and λ>0 give the same ledger, the
market is frequency pruning with extra steps, and the charge is decorative.

**The dual pathway.** One softmax over `D ∪ {256 bytes} ∪ specials`, plus an
abstain gate. A concept emission writes its exact bytes **atomically** — one
forward pass per phrase, zero residual. A byte emission is the fallback. The
trunk's loss is the mixture NLL; the gate learns when to copy.

**Alignment.** `L_align = KL(p_phrase || p_byte)` at aligned positions. The
dynamic-vocabulary paper's PPL regression (8.03 vs 3.60) is the warning: phrase
emission without alignment and a byte fallback regresses quality. `L_ground`
(reconstruct `b_c` from `e_c`) is a regularizer only; exactness never depends
on it. `L_multi` (next-k concepts) is optional.

**The learning curve.** `u(c) = (1-c) + c/L̄` units per byte. At coverage
0.8-0.9 and `L̄ = 5-20 B`, the sequence is 3-6x shorter; the KV cache shrinks
with the sequence, so the context is ~4x longer for free. The unique curve:
**FLOPs/byte falls as the ledger learns the domain**. No fixed tokenizer and
no anonymous-patch model has this.

**The ops.** `pack` (a `.hncm` file: header, bytes, counts, rents, embeddings,
provenance, SHA), `attach`, `detach`, `diff`, `merge`, `rm`, `trace`,
`verify`. The ledger is a file you can grep, diff, ship and delete. This is r6's
"git for model knowledge".

**Provenance-typed generation.** Every output byte is tagged
`(concept-id, source-offset)` or `NEURAL`. The model shows its work at byte
granularity: which words are copied evidence, which are invention. `trace`
writes the tags; the tag is exact by construction, not post-hoc.

**Hard-exact constrained generation.** Restrict `D` to a schema/DSL's validated
phrases; every concept emission is guaranteed-valid by construction. This is
stronger than token-level grammar masking.

**Novelty telemetry.** Gate abstention rate + pack charge = a runtime "I'm out
of my depth" signal, for drift detection and abstention. The MDL accounting
becomes epistemics.

**Failure modes.** Gate miscalibration under distribution shift (the parse you
trained on is not the text you generate); ledger non-stationarity (version the
ledger, slow admission, EMA stats); exposure bias between DP targets and
self-generated parses (scheduled sampling over parses); pack bloat (the rent).

**Gates.** G3a structured ≥0.1 bpb or ≥1.5x; G3b concepts vs patches ≥0.05 or
≥1.2x; G3c gate acceptance/delta; G4a adapter pack ≤1 MB, install ≤1 min; G4b bits per
adapter byte beats LoRA/kNN; G4c deletion/provenance.

### 4.5 B4 — the library (r7)

**What it is.** A hierarchical model whose middle layer thinks in a persistent
library of exact concepts **with typed holes** — discrete identity at the
interface, distributed neural pricing underneath, grown by wake-sleep.

```
bytes ──► L0 Realizer (~10M): entropy signal, fallback coding, exact expansion
   │  parse = neurally-priced DP, decoder-synchronous
   ▼
L1 Cortex (the trunk): autoregressive over CONCEPT EVENTS
   event = (concept_id, slot fills…)   fills are recursive events or bytes
   cross-attends into the Library (long-term memory)
   ▲
Library: c = (exact template with typed holes, e(c), usage, provenance, DL)
   ▲
Librarian (wake-sleep): propose → abstract (anti-unification) → price → accept
```

**The concept.** `def <name>(<args>): <body>`; `"<key>": <value>,`;
`the <X> of the <Y>`; an Apache header; an API call shape. Exact bytes where
fixed, **typed holes** where variable. `e(c)` is the causal encoder's last
state over the template plus the hole-type signature. New concepts are usable
zero-shot; identity is discrete, so emission is a memcpy — zero neural FLOPs
and zero hallucination *within* a concept. Hallucination moves to concept
*selection*, which is auditable.

**The cortex.** All the scale lives here, but it runs on events ~5-20 bytes
long. Factorize `p(event) = p(id|history) × Π p(fill|id, history)`. The
order-2 in-sample headroom (1.327 vs 2.072) is real mutual information between
concepts; a deep model with shared representations is the estimator that can
collect it.

**The librarian.** The repo's scan/DP/EM/charge, repurposed: the GPU candidate
scan proposes from fallback hotspots and frequent event n-grams;
**anti-unification** (top-down, corpus-guided) lifts repeated substructure
into holes; a candidate is accepted iff the *held-out* description length
drops. This is DreamCoder's wake-sleep library learning, never run with a
scalable neural cortex over open text.

**Training.** Hard-EM: parse under current neural prices; train cortex +
realizer on bits/byte; sleep grows/prunes the library; re-parse with versioned
replay. Objective: `L(library) + bits(data | library, weights)`. Anti-collapse
is structural: the fallback is priced per byte; the concept channel is the only
discounted path; growth is held-out-gated, so training-loss gaming cannot
inflate it.

**Inference.** The cortex emits events; exact spans are written verbatim; holes
and novelty go through fills/fallback. No encoder-only information: the parse
*is* the generation (the decoder never reruns the DP, generalized).

**Capabilities and falsifiers.**

| capability | mechanism | falsifier |
|---|---|---|
| systematic compositionality | holes make concepts functions over fills; the rule is shared across fills | held-out slot-combination split (train A×B, test the rest) |
| affordable planning | event horizon is 5-20x shorter; beam/MCTS with DL cost-to-go | search win at equal FLOPs vs token CoT |
| auditable thought | thoughts are discrete library programs; re-expand, type-check | re-expansion + type checks catch structured-output errors |
| knowledge as data | install/patch/delete without gradients; provenance | new-API install beats LoRA on bits-to-adapt × time |
| archive-scale context | 100k events ≈ megabytes at a fixed attention budget | long-range consistency at equal context FLOPs |

**Gates.** L-G0 parity (within 0.03 bpb text, ≥0.1 win code/JSON); L-G1 usage
(concept-covered bit fraction ≥50%); L-G2 shuffle (permuted ids cost ≥0.2 bpb);
L-G3 induction (held-out DL falls per sleep cycle); L-G4 capability (+10 points
on held-out slot combinations); L-G5 adaptation (library install beats LoRA on
bits-to-adapt × time). **Stop:** if only efficiency moves, it is an efficiency
paper, not a capability paper.

### 4.6 The pack and the CLI

The pack is the deployment artifact: `pack-<name>.hncm` with a header, the
lexicon bytes, the counts/rents, the tied embeddings, the provenance table and
a SHA-256. The CLI is the product surface:

```
neural.py fit <dir>            # induce the lexicon + parse
neural.py train --arm native   # the trunk
neural.py install <pack>       # mount a lexicon on a frozen base
neural.py pack <ckpt> --lex …  # write the artifact
neural.py attach/detach <pack>
neural.py diff/merge <pack>…   # compare two lexicons
neural.py rm <pack> <concept>  # surgical deletion
neural.py trace --prompt …     # provenance tags per byte
neural.py verify <pack>        # SHA + round-trip
neural.py bench …              # rate/shared/self; wall-clock
```

The archive format is the same one v10 ships; the pack is a lexicon, not a
model. `attach` on a frozen base is the adapter; `attach` on the trunk is the
ledger. The same file is a compressor, an adapter and a drafter.

### 4.7 The judge

Every claim is a recorded size: `payload + memory + base` at the same bytes,
three conventions (`rate`, `shared`, `self`), a round-trip, and the codec's
entropy-vs-recorded reconciliation within 0.005 bpb. The base is charged in
`self`; the memory is charged at its packed bytes; the adapter comparison is
equal-charged-bytes. No perplexity-only claim; no call-count speed claim.

## 5. The harness and the six experiments

### E0 — the harness (4 h)

Build `neural.py`; round-trip 1 MB; check `recorded = entropy + charge` within
0.005 bpb in both conventions; the parser is decoder-synchronous; the cache is
SHA-keyed; the pack round-trips (write, attach, detach, verify); the trace tags
every byte. **Gate:** every artifact round-trips; the unit sequence reproduces
the byte stream exactly; a 1 MB smoke train/eval runs. **Stop:** a broken
round-trip stops everything (rule 1).

### E1 — the model class (B2, 1 GPU-day)

*Question:* does a neural trunk over the frozen v9 parse beat the count table
at the same parse? *How:* parse the 90 MB train once (cached; the 10 MB pickle
is the smoke test); train 60M on the 90 MB parse; eval the 10 MB val; arms:
fixed-parse vs rotating-lexicon; controls: byte transformer at equal FLOPs,
the v9 count model, PPMd. Report `rate` and `self`.

| gate | arm | bar | meaning |
|---|---|---|---|
| G1a | 60M payload | ≤1.98 | the model class is fixed (PPMd-over-parse 1.978) |
| G1b | 60M shared | ≤1.834 | the text win (lexicon included) |
| G1c | 4 MB distilled, self | ≤2.664 | beats v9 self-contained |
| G1d | 4 MB distilled, self | ≤1.99 | the self-contained text win |

G1b is the **deployment** claim: the 60M base is amortized over ≥1 GB (the
charge table). At 100 MB alone the base is 2.40 bpb, so G1b is not a
self-contained win; G1c/G1d are the self-contained ladder.

*Stop:* payload >2.2 → the neural concept model does not pay at this scale;
record the negative, keep the archive and structured products. Gray 1.98-2.2 →
run one joint-EM round (E3) before deciding.

### E2 — the adapter (B1, 2-3 days)

*Question:* does installing a held-out lexicon on a frozen trunk beat D=∅ and
equal-byte baselines? *How:* first the v0 probe (4 h: an open frozen base,
induced `D`, a 3-parameter gate, verification); then train with rotating
lexicons (same-domain, cross-domain, noisy, ∅; copy-path dropout); freeze;
install 64k/256k/1M lexicons from held-out domains; baselines: base alone, kNN
at equal bytes, LoRA at equal bytes; `score = (CE_bits + 8·|adapter|)/nbytes`.

| gate | test | bar |
|---|---|---|
| G2a | installation at ≤1 MB | ≥0.15 bpb saved |
| G2b | safe mode | ≥50% acceptance at ≤0.02 bpb delta |
| G2c | the base's own domain | no regression |

The readout is the **charged table**: domain × {∅, 64k, 256k, 1M} × {base, D,
kNN, LoRA}. The **linchpin falsifier:** hold out 10% of `D`; test prediction
of unseen-but-similar spans. The tied embeddings must beat counts-only, or the
content-addressing hypothesis is false. *Stop:* no transfer → the adapter is
per-domain only (still a product), and "lexicon-agnostic" is a clean negative.

### E3 — native emission and the gate (B3, 3-4 days)

*Question:* does span emission + a calibrated gate beat a byte LM on structured
data at equal quality, and does joint EM beat the frozen parse? *How:* 2 EM
rounds (parse under neural prices, retrain); add the dual pathway and the gate;
train on synseg/syngrid/shakespeare; measure wall-clock tokens/s and bpb.
E1 is the fixed-parse **compression** number; E3 is the model's own parse and
the **generation** gate.

| gate | test | bar |
|---|---|---|
| G3a | structured | ≥0.1 bpb over the byte baseline, or ≥1.5x tokens/s at equal bpb |
| G3b | concepts vs BLT-style patches | ≥0.05 bpb or ≥1.2x |
| G3c | gate | acceptance and quality delta within the pre-registered bound |

The **learning curve** is the fifth readout: `speed.json` records tokens/s at
batch 1 and 32, bpb, acceptance, rollback rate, and units/FLOPs per byte vs
ledger size. *Stop:* if concepts = patches, stop B3 and ship B1/B2 as the
result.

### E4 — the ledger ops (B3, 1 week)

*Question:* do the operations no LoRA has work, at a price? *How:* the
admission market (`utility > rent`), eviction, `pack/attach/detach/diff/merge/rm`,
provenance round-trip, deletion test, install time; the λ ablation; constrained
generation on a schema; telemetry.

| gate | test | bar |
|---|---|---|
| G4a | adapter pack, installed subset | ≤1 MB, ≤1 CPU-minute |
| G4b | bits saved per adapter byte | beats LoRA and kNN at equal bytes |
| G4c | delete, provenance | deleted spans provably absent; provenance 100% |
| G4d | constrained generation | 100% valid atoms at ≤0.05 bpb cost |
| G4e | telemetry | abstention correlates with held-out error |

Readouts in `ops.json`: install time, pack bytes, bits saved per adapter byte,
delete verification, trace coverage, telemetry AUC. *Stop:* if the market is
frequency pruning with extra steps (the λ ablation is flat), say so; the charge
is decorative.

### E5 — the library (B4, 2 weeks)

*Question:* do holes buy systematic composition, and does the librarian find
them? *How:* anti-unification over the ledger on a code corpus; typed holes;
slot-combination splits; a planning toy with the DL cost-to-go; the held-out DL
per sleep cycle.

| gate | test | bar |
|---|---|---|
| G5a | held-out slot combinations | +10 points at equal FLOPs |
| G5b | shuffle control | permuted ids cost ≥0.2 bpb |
| G5c | wake-sleep | held-out DL falls per cycle |
| G5d | adaptation | install beats LoRA on bits-to-adapt × time |
| G5e | planning | search win at equal FLOPs vs token CoT |

Readouts in `library-*.json`: the induction curve, held-out DL, slot accuracy,
the shuffle delta, the planning win, and install-vs-LoRA bits/time. The
**capability controls** are the point: L-G0 parity, L-G1 usage, L-G2 shuffle,
L-G3 induction, L-G4 combination, L-G5 adaptation. *Stop:* if only efficiency
moves, it is an efficiency paper, not a capability paper; say so.

### E6 — scale (only if E1-E3 pass, 1-2 weeks)

100-300M trunk, 1-2B units, equal FLOPs vs BPE and H-Net-style patches. *Gate:*
no regression; a win on the domain suite at equal charged bytes; the `shared`
crossover moves below 1 GB.

## 6. The decision tree and the minimal decisive run

```
E0 ─► E1 ─┬─ ≤1.834 ─► E2 ─► E3 ─► E4 ─► E5
          ├─ 1.834-1.98 ─► E2 + one EM round
          ├─ 1.98-2.2 ─► gray: E3 only; else archive
          └─ >2.2 ─► stop the text branch; B1/B3 structured
```

**Minimal decisive run: E0 + E1 + E2 (4-5 days).** It answers the two
load-bearing questions — is the model-class fix real, and does installation
transfer? — on one 3080, with cached parses. E3/E4 are the architecture; E5 is
the capability; E6 is the scale.

## 7. Risks and cheap checks

| risk | check | cost |
|---|---|---|
| the trunk cannot beat PPMd at 60M | E1 payload vs 1.834/1.98 | 1 GPU-day |
| the 100 MB self claim is impossible | the charge table; the 4 MB distillation | free |
| installation does nothing | E2 D=∅ vs installed | 2-3 days |
| the gate over-copies | acceptance, quality delta, verification | 1 day |
| concepts are anonymous patches | the patch control; the shuffle test | 3 days |
| joint EM oscillates | churn <10%/round; parse sampling | 1 day |
| the charge is decorative | λ=0 vs λ>0 ablation | 1 day |
| alignment regresses PPL | KL on/off; the dynamic-vocabulary warning | 1 day |
| ledger non-stationarity | versioned ledger; slow admission; EMA | 1 day |
| exposure bias | scheduled sampling over parses | 1 day |
| the induction cost is high | cap candidates via the scan; cache | 1 day |
| the pack bloats | pack ≤1 MB; rent | 1 day |
| the memory is a privacy surface | provenance, inspect, delete | 1 week |
| the two clocks batch poorly | wall-clock at batch 1 and 32 | 1 day |

### What not to do

- Do not train a from-scratch chat LM. The claim is memory, not chat.
- Do not make counts the predictor; that negative is measured.
- Do not pool span bytes; that negative is measured (v6).
- Do not charge the base per archive; a 30 MB base at 100 MB is 2.40 bpb.
- Do not run seven architectures; one spine, four knobs, seven stages.
- Do not judge by perplexity; the recorded bits are the judge.
- Do not sell the `calls` column as a speedup; measure wall-clock.
- Do not put the DP on the inference path.
- Do not claim phrase emission without alignment and a byte fallback.
- Do not claim composition without the slot split and the shuffle control.

## 8. Stop rules

1. The recorded bits are the judge: `payload + memory + base` at the same
   bytes; a rate estimate is not. The round-trip is the artifact.
2. Every budget, table and lexicon is chosen on the held-out 10%; never the val.
3. Report `rate`, `shared` and `self`; the base is amortized only when it is
   actually shared; the crossover is a result.
4. B2 dies at payload >2.2; B1 dies if installation does not pay; B3 dies if
   concepts = patches; B4 dies if the slot gain is <10 points.
5. No speed claim without wall-clock at equal recorded bits; no call-count
   ratio.
6. The archive plan and the structured win are the floor; the pivot cannot
   regress them.
7. If the trunk does not beat the count model at equal parse, the model class
   is not fixed; record the negative and stop the text branch.
8. If only the adapter works, ship it: the architecture claim is the adapter,
   not the LM.
9. If phrase emission regresses PPL >5% at equal FLOPs, fix alignment or stop.
10. If the ledger churns >10% per round, slow admission; do not scale a
    non-stationary ledger.
11. If the adapter pack >1 MB, the per-file install >1 min, or the ledger's
    installed subset >1 MB, the per-file ops claim fails.
12. If holes buy only efficiency, B4 is not a capability result; say so.

## 9. Time

| step | cost | result |
|---|---|---|
| E0 spine | 4 h | the harness, the round-trip, the pack |
| E1 model class | 1 GPU-day | payload vs 1.834/1.98; the distillation |
| E2 adapter | 2-3 days | the install table vs ∅/kNN/LoRA |
| E3 native + gate | 3-4 days | the structured speed/quality |
| E4 ledger ops | 1 week | pack/edit/delete/provenance/constrained |
| E5 library | 2 weeks | the composition/planning gates |
| E6 scale | 1-2 weeks | the deployment crossover |

Minimal: E0+E1+E2, 4-5 days. The full program is ~6 weeks on one 3080, with
every artifact cached and every gate pre-registered.

## 10. The big picture

The seven sketches agree on the mechanism and disagree on the stage. The
mechanism is one model, many streams: an exact span memory (the phrasebook), a
neural trunk (the predictor), an MDL price (the admission), and a copy/compute
gate (the quality bound). The stage is where the memory sits: outside the base
(the adapter), inside the vocabulary (the LM), beneath the compute (the ledger),
above the sequence (the library).

The charge decides the order. A 60M trunk cannot pay at 100 MB; a 300 kB
adapter can. So the vocabulary is the first experiment and the deployment bet;
the adapter is the first shippable product; the ledger is the architecture;
the library is the capability. The best result in the design space is not "a
bigger phrasebook"; it is a small, exact, priced memory that a frozen base can
read — and a text rate that pays at the deployment scale, not the file scale.

The later sketches add the capabilities that make it worth doing:

- **Provenance-typed generation** (r6): every output byte is tagged; the model
  shows its work at byte granularity. No current LLM can emit that label.
- **Knowledge as a pack** (r5, r6): `fit`, `attach`, `detach`, `diff`, `merge`,
  `rm`; adaptation and surgical unlearning without gradients; a memory you can
  grep, ship and delete.
- **Hard-exact constrained generation** (r6): a schema/DSL's validated phrases
  are exact by construction; stronger than token-level grammar masking.
- **The learning curve** (r5): FLOPs/byte falls as the ledger learns the
  domain; efficiency has a slope, not a constant.
- **Composition and planning** (r7): holes make concepts functions; the event
  horizon is 5-20x shorter; thoughts are discrete, cacheable, type-checkable
  programs.
- **Auditable thought** (r7): a thought is a library program; re-expand it,
  type-check it, cache it; the verification is structural, not post-hoc.
- **Archive-scale context** (r7): 100k events ≈ megabytes of text at a fixed
  attention budget; "the archive is the unit" becomes native.

If E1 passes at ≤1.834, the v9 negative is explained and reversed: the model
class was the gap, and a neural trunk over exact spans closes it. If E1 only
reaches 1.98, the class is fixed but the text win is not; the adapter and the
structured win are the result. If E1 stops at >2.2, the neural pivot is a
negative, and the archive, the structured win and the adapter (if E2 passes)
are the product. All three are publishable; the gates make the difference
measured, not hoped.

Provenance: v9's rows are `runs/v9/{proof-100m-seed-10m.json,union.json,
CODEC.md}`; the PPMd-over-parse 1.992 and the 0.67/0.16 decomposition are
TODO5's measured accounting; the 10 MB parse is
`runs/v9/fit-seed-16384-16384-10000000.pkl` (`toks`, 3.73M units, 2.68 B/unit);
the 100 MB scan is `runs/v9/scan100.npz`; the seed is 0. The commands in §0
reproduce the artifacts; every table is keyed by corpus SHA and lexicon SHA.
