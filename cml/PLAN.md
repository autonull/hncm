# TODO6 — H-NCM v11: the concept memory LM (the efficient path)

Status: v9's text negative is measured (2.664 at 100 MB; PPMd 1.834; PPMd over
the parse 1.992). The seven sketches (`hncm.r1-7.md`) are one family: exact
span memory + neural trunk + MDL price + copy/compute gate. This revision
changes two things: **the order of the evidence** (the 4-hour adapter and the
10 MB pilot before the 100 MB run) and **the codebase** (a new `cml/` directory,
one typed pipeline, manifests, a run store). The minimal verdict is 2 days, not
5.

**Status (2026-11-25).** P1 is built and measured before P0. The cached 10 MB
parse makes P1 hours, and its trunk is the frozen base P0 re-runs on (§7 lever
1). The 20.5M unit trunk reaches **1.920 bpb** at seed 0 (1.912 ± 0.008
over seeds 0-2) on the held-out 1 MB slice (9 MB train), and the 40M decision
point reaches 1.926; the 60M curve fit is 1.919. P1a (≤2.60), P1b (≤2.293) and
P1c (≤2.29 at 60M) pass. The same-slice count bigram is 2.667; the neural
trunk is 0.75 bpb lower. The v9 reference on the 10 MB block is 2.696 (the
2.729 is the 10 MB-prefix rate); PPMd over the parse is 1.978. P0, P1d and
P1e are next; the P1 record is
`reports/GATES.md` and `runs/<id>/metrics.json`. The 40M point is 0.006 above
the 20.5M point, so the 10 MB pilot is data-bound; P1c passes by saturation,
not by a capacity slope.

## 0. Start here

```bash
cd cml
uv sync
uv run cml selftest
uv run cml space
uv run cml search P1 --simulate --budget 20
uv run cml status
uv run cml report
uv run cml run experiments/p0-adapter.toml
uv run cml run experiments/p1-class-10m.toml
uv run cml run experiments/p1-class-10m.toml --params 40_000_000
uv run cml curve P1 --steps 6000 --update
uv run cml gates
uv run cml run experiments/p2-class-100m.toml   # only if p1 crosses
uv run cml run experiments/p3-gate.toml
uv run cml run experiments/p4-ops.toml
uv run cml run experiments/p5-library.toml
```

The run store: `cml/runs/<run-id>/{config.json,metrics.json,artifacts/}`.
`run-id = <manifest>-<sha8(config)>-s<seed>`. Atomic writes; resume from the
checkpoint; the summary is `cml/reports/GATES.md`.

## 1. The design rules

| measurement | rule |
|---|---|
| v9's 0.83 gap is 0.67 model class + 0.16 dictionary; PPMd over the parse is 1.992 | the predictor is a neural trunk; counts are features, priors and prices — never the predictor |
| 7.7 observations/context; the online interpolation is a negative (2.39-2.45) | price a span from its **content** (a causal encoder's state), not from a per-ID row |
| v6: a pooled code conveys ~1 bpb while it costs 3.1; the optimizer collapses it | identity is a **discrete index**; the bytes are verbatim; the residual is zero |
| coverage 0.912 → rate 4.283; the uncovered 24% at 5.6 bpb is the deficit | memory is optional; the fallback is a **real byte expert** |
| v9's 300 kB model is 0.024 bpb over 100 MB | the base is shared; the per-archive charge is the **lexicon**; report `rate`/`shared`/`self` |
| the decoder never reruns the DP (0.0013 bpb) | no DP at inference; lazy longest-match; span emission is one decision |
| the in-sample oracle is 1.327 (order-2) vs 2.072 (order-1) | the headroom is real mutual information; a deep shared estimator is the only way to collect it |
| v9's `calls` = 1 + single-byte tokens, not total units (253,853 vs 3.73M at 10 MB) | judge speed in **wall-clock at equal recorded bits**; never sell a call-count ratio |

**The scorecard.**

| claim | metric | bar |
|---|---|---|
| the model class is fixed | payload, 10 MB | ≤2.60 (beat v9 2.729); ≤2.293 (beat PPMd-over-parse) |
| the text win | payload, 100 MB | ≤1.98 (beat PPMd-over-parse 1.978) |
| the text win | shared, 100 MB | ≤1.834 (PPMd) |
| self-contained | self, 100 MB | ≤2.664 (beat v9); ≤1.99 (stretch, 4 MB) |
| the adapter | shared, ≤1 MB | ≥0.15 bpb saved vs D=∅ |
| the adapter v0 | structured/text/pack/provenance | ≥1.5x, ≤0.01 bpb, ≤1 MB, 100% |
| structured | shared/rate | ≥0.1 bpb over byte, or ≥1.5x tokens/s |
| the gate | acceptance/quality | ≥50%, delta ≤0.02 bpb |
| speed | wall-clock | ≥1.3x text, ≥3x structured |
| the ledger | ops | installed subset ≤1 MB, install ≤1 min, delete, provenance 100% |
| the library parity | text/code, usage | within 0.03 bpb, ≥0.1 win, usage ≥50% |
| the library | held-out slots | +10 points; shuffle ≥0.2 bpb; planning win |
| the library capability | auditable/archive | re-expansion + type checks; long-range at equal FLOPs |

## 2. The four bets

| bet | sketches | claim | first test | product |
|---|---|---|---|---|
| **B1 the adapter** | r1, r3, r6 | a kB lexicon on a frozen base beats D=∅, kNN and LoRA at equal charged bytes | P0 | "LoRA you can read" |
| **B2 the vocabulary** | r4 | a neural trunk over exact spans beats the count table at equal parse | P1 | the text/structured rate |
| **B3 the ledger** | r2, r5, r6 | native span emission + a market + a copy gate; pack/provenance/constrained | P3, P4 | the auditable memory |
| **B4 the library** | r7 | holes buy systematic composition and cheap planning | P5 | the capability |

**Why this order.** P0 is 4 hours and tests the product claim with no
training. P1 is 4-6 hours and tests the scientific pivot on the cheap 10 MB
corpus. P2 is the 100 MB verdict only if P1 crosses. The architecture (P3/P4)
and the capability (P5) are later. This is information per GPU-hour, not
ambition.

**What each sketch adds.** r1: equal-charged-bytes. r2: hybrid output + copy
bypass. r3: lexicon-agnostic training + three heads + safe mode. r4: neural
prices + joint EM. r5: persistence + market + alignment + dual pathway +
learning curve + editing. r6: tied embeddings + pack + provenance +
constrained generation + telemetry + two clocks. r7: holes + librarian +
composition + planning + auditable thought.

**Neighbors and the delta.** Each neighbor has one piece; none has the union.

| neighbor | it does | the delta |
|---|---|---|
| CoG / REST | copy/retrieve spans, verify | concepts are native output symbols; no separate verifier |
| NEST | post-hoc token attribution | provenance is by construction, free, byte-granular |
| kNN-LM / continuous cache / RETRO | an uncharged datastore, token-level | MDL admission, exact spans, editable, provenance |
| memory layers | trainable KV parameters | an exact, transmissible, deletable pack |
| BLT / H-Net / MegaByte | anonymous ephemeral patches | named, persistent, priced, reused units |
| DreamCoder / LILO | a symbolic library, no cortex | exact holes under a scaled neural cortex |
| COCONUT | continuous thoughts | discrete, re-expandable, type-checkable |
| LCM | sentence embeddings | exact spans with typed holes |
| LoRA | opaque dense weights | exact, auditable, transmissible pack |
| SuperBPE / MWT | a static frequency vocabulary | learned, budgeted, evictable, provenance |
| dynamic vocabulary | ephemeral phrases, re-encoded per query | persistent entries, free identity, admission, byte expert |
| SentencePiece / MDL segmentation | a static unigram, no charge | neural prices, exact spans, a charge, editable |
| LCT | static reusable units | MDL-charged, editable, transmissible |

## 3. The charge arithmetic

A 60M trunk is 30 MB at 4-bit, 60 MB at 8-bit, 120 MB at fp16. At 100 MB the
4-bit base alone is 2.40 bpb; no payload recovers that.

| stream | base | base bpb | payload needed to beat PPMd 1.834 |
|---|---:|---:|---:|
| 100 MB | v9 300 kB | 0.024 | 1.810 (v9 payload 2.640: negative) |
| 100 MB | 4 MB | 0.320 | 1.514 |
| 100 MB | 30 MB | 2.400 | impossible |
| 1 GB | 30 MB | 0.240 | 1.594 |
| 10 GB | 30 MB | 0.024 | 1.810 |
| 1.2 MB (synseg) | 3 MB | 20.0 | impossible |

The rule: the neural pivot is a deployment-scale bet, not a per-file bet. The
100 MB `self` claim is negative unless the trunk is ≤4 MB; the 100 MB `shared`
claim is valid for the adapter; the text rate needs ≥1 GB or a distilled
trunk. The `self` row is a measured crossover, not a failure to hide.

The ledger's charge is `8·Σ K_c / n` over the installed subset. The ledger
budget `B` (64-256 MB) is a deployment budget; the per-file charge is the
installed subset.

The crossover: a 30 MB base amortizes at `240 MB / (1.834 - payload)`. At
payload 1.6 bpb it is ~1.03 GB; at 1.5 bpb ~0.72 GB. `shared` at 100 MB is
conditional on that deployment; `self` is the 4 MB distillation.

## 4. The architecture

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

**B1 the adapter.** Frozen base + installed `D`; `e_c = Enc(bytes_c)` computed
once at install; a 3-parameter gate first (a logistic on `[h_t, max score,
coverage]`), an MLP only if the gate fails. The pilot backbone is 20-50M;
`D` is 4k-64k exact spans; the scoring head is 1-3M; the gate is 3
parameters (a 2-layer MLP only if the 3-parameter gate fails). The score is
`(W_q h_t)·e_c/√d + α·log p_count(c|c_prev)`; the mixture is
`p(unit) = π_t·softmax(score) + (1-π_t)·p_byte(·|h_t)`; the count term is a
ranking prior, not the predictor. Install is milliseconds; remove is deleting
the pack. Equal-charged-bytes compares base, base+D, base+kNN and base+LoRA at
the same `8·|adapter|/n`. Safe mode: verify the copied span's bytes with the
byte head; roll back if surprise > τ; the quality delta is bounded (≤0.02 bpb).
Span commitment is bounded by a mid-span abort token and an entropy-capped
span length.
**v0 (4 h):** freeze an open byte base; induce `D`; fit the 3-parameter
gate; verification on. Re-run v0 on the P1 trunk after P1. This is the
zero-training product.

**v0 readout.** Structured latency ≥1.5x at backbone-exact quality; text
delta ≤0.01 bpb; pack ≤1 MB; provenance 100%. The baseline includes tuned
REST/NEST at equal datastore bytes; if the gate cannot beat them, v0 is an
inference technique, not an architecture. The quality floor is the base by
construction (NEST's insight); the drafter is the trained, compact ledger.

**B2 the native LM.** The trunk's output vocabulary is the lexicon: a
concept id or a byte; the full `T_c` is 50-300M, the pilot 10-60M. The parse
is fixed for the compression eval; the model predicts the next unit. Two
clocks: the encoder at byte rate, the trunk at unit rate (~2.7 B/unit on
enwik8; 5-200 B/unit on structured, depending on the phrasebook), so the
sequence is 2-20x shorter. Joint EM: parse under neural prices; train;
add/prune by `usage × surprise-reduction`; parse sampling regularizes.
Distillation: only after the 60M teacher passes; a 4 MB 4-bit student is the
self-contained ladder.

**Switch and head.** An escape emits a byte run; the byte channel (`T_b`,
10-50M) emits until a resume token; the switch is learned and MDL-priced.
The concept head is trie-factorized/hierarchical softmax or MIPS for
`|L|` 16k-1M; the pilot's 16k is a flat softmax. Structured spans are
20-200 B where the phrasebook fits; enwik8 is ~2.7 B/unit. The charge is the
anti-collapse regularizer: the code must earn its bytes. The lineage is the
SentencePiece unigram-LM EM/Viterbi loop (the unigram replaced by the
contextual trunk and a charge term), MDL segmentation (Brent/Goldsmith) and
LCT. Optional next-k concept prediction (`L_multi`).

**Training.** Teacher-force the MDL-DP parse; loss = mixture NLL + β·|pack|
when `D` grows (at fixed `D` the charge is the induction objective, not a
gradient); train the surface projections + gate; optional LoRA on the trunk;
re-induce `D` on the EM schedule; pointer-generator-style.

**v1/v2.** v1 advances over a whole span in one parallel pass (target 5-10x
structured, ~3x text); v2 puts concept tokens in the input sequence and cuts
FLOPs; the two-clock target is ≥2x structured at equal quality.

**B3 the ledger.** A persistent, growable, auditable memory of exact spans:
`<bytes, id, e_c, n_c, S_c, K_c, rent, provenance, timestamp, holes?>`. The
market: `utility(c) = Σ_occ [log p_c(c|ctx) - log p_b(b_c|ctx)] - λ·8·K_c`;
admit iff >0; evict below rent; the market is discrete (no gradients through
admission). The dual pathway: one softmax over `D ∪ {256 bytes} ∪ specials`,
an abstain gate; a concept emission writes its exact bytes atomically. The
alignment `KL(p_phrase || p_byte)` is mandatory: the dynamic-vocabulary
regression (8.03 vs 3.60) is the warning; `L_ground` is a regularizer only.
The learning curve `u(c) = (1-c) + c/L̄`: FLOPs/byte falls as the ledger
learns. The ops: pack/attach/detach/diff/merge/rm/trace/verify; provenance
tags every byte `(concept-id, source-offset)` or `NEURAL`; hard-exact
constrained generation restricts `D` to schema/DSL phrases; novelty telemetry
is gate abstention + charge.

**Pack ops.** `fit repo/ → repo.pack` (~200 KB for a repo; SHA-verified),
`attach`, `detach`, `diff`, `merge`, `rm <concept>`, `trace`, `verify`; the
pack is readable spans + counts; inspect, edit, exactly delete; no
catastrophic forgetting (base frozen), no serve gradients; on-device
personalization; LoRA is opaque weights, memory layers are parameters, RAG
pays context forever; append/dedup/verify/shipped-prior map 1:1.

**Grounding and integrity.** Grounded mode restricts generation to a
repo/schema trie; hallucinated atoms are impossible (composition can still
be wrong). The same model compresses and generates, so verify by
reconstruction (encode, decode, compare) is a data-integrity check. The
market's candidates come from the existing `n10` GPU scan (2.7M at count
≥32 on 90 MB; cached). BLT pays patch encoding per occurrence; the ledger
pays once at admission; the ephemeral-phrase version already measured −20%
latency. The rent `λ` is the regurgitation knob; measure the verbatim-copy
rate.

**B4 the library.** L0 realizer (~10M: entropy signal, fallback, exact
expansion; its parse is the neurally-priced DP (Viterbi), decoder-synchronous)
+ L1 cortex (the trunk) over events `(concept_id, fills...)`; a fill is a
recursive event or a fallback patch; the cortex cross-attends into the
library. `p(event) = p(id|history) × Π p(fill|id, history)`; the library
stores exact templates with typed holes; `e(c)` is the causal encoder's last
state over the template plus the hole-type signature. The concepts are a code
definition, a JSON member, a genitive phrase, a license header, an API call
shape; new concepts are zero-shot; identity is discrete, so emission is a
memcpy — zero FLOPs and zero hallucination within a concept; hallucination
moves to concept selection, which is auditable. The librarian is the repo's
scan/DP/EM/charge: propose from fallback hotspots and event n-grams;
anti-unification (top-down, corpus-guided) lifts substructure into holes;
accept iff the held-out DL drops. This is DreamCoder's wake-sleep library
learning, never run with a scalable neural cortex over open text. Hard-EM
training; versioned replay; anti-collapse is structural (fallback priced,
the concept path discounted); growth is held-out-gated. Objective: `L(library) + bits(data | library,
weights)`. Inference: the cortex emits events; exact spans are written
verbatim; holes and novelty go through fills/fallback; no encoder-only
information — the parse is the generation. Capabilities: compositionality
(slot splits),
planning (event horizon 5-20x shorter; beam/MCTS with DL cost-to-go),
auditable thought (re-expand, type-check, cache), knowledge as data
(install/patch/delete), archive-scale context (100k events ≈ MBs).

**B4 performance.** One cortex call per 5-20 bytes; exact emission is free;
only fills and novelty cost per-byte compute; the target is 3-10x fewer
big-model FLOPs/byte at equal bpb. The units are named, persistent and
reused, so the cortex input is stationary; BLT/H-Net patches are anonymous
and ephemeral. H-Net's own ablations show the chunking is load-bearing;
segmentation rigidity is the mirror risk, absorbed by the fallback and the
induction loop. The combination (exact holes under a scaled
neural cortex) is the untried part; every piece is demonstrated separately.

**B4 program (~1 GPU-month).** E0 week 1: parity/usage/shuffle on an enwik8
slice + code/JSON, 30M cortex, frozen library. E1 weeks 2-3: induction on a
code corpus; the "learn a new API in minutes" demo. E2 weeks 3-4: slot
splits. E3: a 100M-1B equal-FLOP scaling check vs an H-Net-style baseline
before the big claim. Report the library transfer matrix across domains. The
repo's scan/DP/EM/charge is the librarian; the bpb judge is the objective;
the `.hncm` prior is the library file; primed blocks are library-primed
context; fingerprints/resume are versioned wake-sleep. "One model, many
streams" becomes "one cortex, many libraries."

**The pack.** At the pilot: a `.npz` (bytes, counts, embeddings) + JSON
(header, provenance, SHA). No custom binary format, no C kernel, no archive
CLI until P4. The pack is the adapter, the ledger and the drafter in one file.

## 5. The new codebase: `cml/`

**The old pattern (retire).** One script per idea (`hncm5.py`, `codec.py`,
`concept_lm.py`, `n10.py`), duplicated corpus loaders, duplicated judges,
prose gates, scattered JSONs. A fix in one does not reach the others. Do not
modify the old files; they are frozen references and data sources.

**The new pattern.** One typed pipeline, one CLI, manifests, a run store, a
gate registry. No one-off experiment scripts.

```
cml/
  pyproject.toml
  uv.lock
  cml.py                 # one file until P4: config, data, lexicon, parse,
                         # model, judge, pack, gates, CLI
  experiments/
    p0-adapter.toml
    p1-class-10m.toml
    p2-class-100m.toml
    p3-gate.toml
    p4-ops.toml
    p5-library.toml
  tests/
    test_invariants.py
  reports/
    GATES.md
  runs/                  # generated; .gitignore
  README.md
```

**Split rule.** `cml.py` stays one file until P4. Split `library.py` at P5
(the second model). Split `model.py` only when a second model arm appears. No
pre-split, no empty modules, no ceremony.

**The typed core.**

```python
class Arm(StrEnum):
    BYTE = "byte"; CONCEPT = "concept"; ADAPTER = "adapter"
    LEDGER = "ledger"; LIBRARY = "library"

class Convention(StrEnum):
    RATE = "rate"; SHARED = "shared"; SELF = "self"

@dataclass(frozen=True, slots=True)
class Config:
    arm: Arm; convention: Convention; seed: int
    dict_size: int; min_count: int
    stream_bytes: int; val_bytes: int
    params: int; steps: int; batch: int; lr: float; mix: float

@dataclass(frozen=True, slots=True)
class Entry:
    data: bytes; count: int; rent: int
    provenance: tuple[str, ...] = ()

@dataclass(frozen=True, slots=True)
class Pack:
    entries: tuple[Entry, ...]; config: Config; sha: str

@dataclass(frozen=True, slots=True)
class Run:
    id: str; config: Config; metrics: dict[str, float]
```

`CDF` and `Model` are `Protocol`s; the coder takes a `CDF`, the trainer takes
a `Model`. `Corpus` is a `Protocol` with `train()`/`val()`.

**The CLI.**

```
cml selftest
cml run experiments/p1-class-10m.toml [--seed 0] [--params 20m] [--dry]
cml pack <file> --out runs/<id>/pack.npz
cml trace <run-id> --prompt ...
```

`run` loads the manifest, validates the gate names against `GATES`, builds the
run id `f"{stem}-{sha8(config)}-s{seed}"`, writes `config.json` atomically,
dispatches through `PIPELINES`, writes `metrics.json` + `artifacts/`, and stops
at the first failed gate. `--dry` prints the config and the gate table.

**The manifest.** One TOML per experiment; no per-experiment Python.

```toml
[run]
name = "p1-class-10m"
seed = 0
gates = ["P1a", "P1b", "P1c"]

[data]
corpus = "enwik8_10m"
val_bytes = 1_000_000
lexicon = "../runs/v9/fit-seed-16384-16384-10000000.pkl"

[model]
arm = "concept"
params = 10_000_000
base = ""

[optim]
steps = 2000
batch = 32
lr = 3e-4
mix = 0.5
```

**Gates.** One `GATES` dict in `cml.py`:

```python
@dataclass(frozen=True, slots=True)
class Gate:
    metric: str; convention: Convention; op: str; bar: float
    def check(self, value: float) -> bool: ...

GATES = {
    "P1a": Gate("payload", Convention.RATE, "<=", 2.60),
    "P1b": Gate("payload", Convention.RATE, "<=", 2.293),
    "P1c": Gate("curve60", Convention.RATE, "<=", 2.29),
}
```

The manifests reference names; the scorecard is generated, not duplicated.
The numbers in §1 are the initial `GATES`; `cml.py` is the source after P0.
The P4/P5 bars are placeholders until their baselines are measured; set them
from the E4/E5 readouts, not from a guess.
`PIPELINES` is one dict (`p0`...`p5`), each a function; no per-experiment
branches outside it.

**Tests.** Only invariants that have broken: round-trip, charge reconciliation
(±0.005 bpb), resume from checkpoint, pack round-trip, shared init, manifest
validation. No coverage target.

**Discipline.** `uv` only (one lock, no `requirements.txt`); ruff format
always, `ruff check --fix` on changed files; no import-time work; atomic
writes; `config.json` includes the git SHA, seed, corpus SHA, lexicon SHA and
gate names; no back-compat; no old imports; data artifacts read directly from
`../runs/v9/`.

**The experiment space.** Every valid arm is a typed point; the space is not
the full Cartesian product. `STAGE_AXES` in `cml.py` is the one source of
truth; `cml space` prints the valid count per stage and the live axes.

| axis | values |
|---|---|
| stage | P0-P5 |
| arm | byte, concept, adapter, ledger, library |
| price | counts, neural, neural+counts |
| charge | none, archive, lexicon, rent, library |
| emission | byte, span, event |
| parse | fixed, joint-em, lazy |
| gate | none, logistic, mlp, verify |
| embedding | random, composed, tied, tied+residual |
| schedule | fixed, rotating, mixed |
| distill | none, 4m |
| data | enwik8_10m, enwik8_100m, synseg, syngrid, shakespeare, src, json, csv |
| scale | 10m, 20m, 40m, 60m, 100m, 300m |
| adapt | null, 64k, 256k, 1m |
| seed | 0, 1, 2 |

The valid points per stage are small (hundreds); the full product is not.
The axis values are the *possibilities*; `STAGE_AXES` is the *constraint*.

**The adaptive policy.** Successive halving with UCB, gated:

1. At the cheapest rung, run every arm in the stage's explore set (one per
   axis value) once.
2. Score = the gate margin (normalized), minus measured cost, plus a UCB
   bonus `sqrt(2 ln N / n)`.
3. Hard-prune a failed gate; prune an arm whose lower confidence bound is
   below the incumbent.
4. Promote the top 1/3 to the next rung (10m → 20m → 40m → 60m; 1k → 5k →
   20k steps); recombine the best value per axis into a candidate, plus
   one-axis mutations, and evaluate the untested ones.
5. Exploit 80% of the budget on the frontier; explore 20% on the
   least-tested axis value (ε = 0.2).
6. Stop when the frontier's gate margin is positive at the top rung, the
   budget is spent, or no arm can beat the incumbent's lower bound.
7. Record every decision (arm, rung, score, UCB, reason, parent, cost) in
   `reports/search.jsonl`; every run in `runs/<id>/`.

The reward is the gate margin, not a proxy. A failed gate is a hard prune;
a marginal gate is a wait.

**The feedback.** `cml status` is the live dashboard; `cml report` writes
`reports/PROGRESS.md`. Both show:

- **Frontier:** best arm per stage with rate/shared/self, gate margins,
  cost, seeds.
- **Gate ladder:** each gate's bar, value, margin, pass/fail.
- **Leaderboard:** top 10 by UCB, with n, mean ± std, cost.
- **Budget:** spent/total, with a bar and burn per stage.
- **Tree:** tested, pruned, promoted, waiting; the prune reason.
- **Axes:** exhausted, live, unexplored values.
- **Next:** the next arm and why (max UCB, explore).
- **Evidence:** config, seed, corpus/lexicon SHA, artifact paths.
- **Sparkline:** the best metric per rung.

**The simulation.** `cml search --simulate` replaces the pipeline with a
synthetic reward to test the scheduler and the report; it is not evidence.
The real search calls `PIPELINES`; the state/report schema is identical, so
the dashboard does not change when the pipelines land. The simulated metric
stands in for the full metric dictionary; the evidence fields are the same.

The report is generated from the run store, not hand-written. It is the
engaging surface: a table, a tree, a budget bar and a one-line "next".


## 6. The efficient ladder

### P0 — the adapter v0 (4 h)

Frozen base (open byte model; re-run on the P1 trunk after P1); induce `D`;
3-parameter gate; verification on; no training. Compare base, +D, +kNN, +LoRA,
and tuned REST/NEST at equal datastore bytes. Hold out 10% of `D`: the tied
embeddings must beat counts-only on unseen-but-similar spans (the linchpin).
**Gates:** P0a ≥0.15 bpb at ≤1 MB; P0b ≥50% acceptance at ≤0.02 bpb delta;
P0c no regression on the base's own domain; P0d structured latency ≥1.5x at
backbone-exact quality; P0e text delta ≤0.01 bpb; P0f pack ≤1 MB; P0g
provenance 100%. **Readouts:** `metrics.json` (the charged table), the
acceptance gap and verification pass rate, `trace.jsonl`. **Stop:** no gain →
per-domain adapter only; if the gate cannot beat tuned REST/NEST at equal
bytes, v0 is an inference technique, not an architecture.

### P1 — the class on 10 MB (4-6 h)

The cached 10 MB parse (`toks`, 3.73M units). Train 10M → 20M → 40M on the
same parse (`--params`); fit `bpb = a·N^-b + c`; the 40M run is the decision
point. The
first run uses the **mixed schedule** (50% fixed parse, 50% rotating lexicon)
so P1 and P0 share the trunk; a fixed-only control only if the fixed readout
is >0.03 bpb worse. The judge is CE-to-bpb for the search; the 1 MB round-trip
is the record. **Gates:** P1a ≤2.60 (beat v9 2.729); P1b ≤2.293 (beat
PPMd-over-parse); P1c the 40M curve extrapolates ≤2.29 at 60M; P1d
acceptance-per-pack-byte ≥20% over P0; P1e zero-shot new-lexicon within
0.1 bpb of tuned (else residual-only tuning). **Stop:** >2.80 or a flat
curve → the class is not fixed at 10 MB; record the negative.

**Measured (fixed-only, 2026-11-25).** Train the first 9 MB of the 10 MB
parse, test the last 1 MB; the parse is the 90-100 MB enwik8 val block and
the lexicon is the v9 shared pack. The bits-per-byte payloads at 6000 steps are
8.9M: 1.962, 20.5M: 1.920 (1.912 ± 0.008 over seeds 0-2), 39.9M: 1.926;
the fitted 60M point is 1.919.
The same-slice count bigram is 2.667, so the neural margin is 0.75 bpb.
The 40M decision point passes P1a (2.60), P1b (2.293) and P1c (2.29 at
60M). A second split (train 8 MB, test the 8-9 MB slice) gives 1.992, so
the win is not one lucky slice. The mixed schedule is not built yet: this
is the fixed-only control, and it does not yet test the one-trunk P1/P0
claim. P1d and P1e need P0. The record is `reports/GATES.md`.

### P2 — the class on 100 MB (1 GPU-day)

Only if P1 crosses. Parse the 90 MB train once; train 60M; eval the 10 MB
val. **Gates:** P2a payload ≤1.98; P2b shared ≤1.834; P2c 4 MB distilled
self ≤2.664; P2d 4 MB self ≤1.99. **Stop:** >2.2 → keep the archive and the
structured win; publish the negative.

### P3 — the gate and the structured suite (2-3 days)

Joint EM (2 rounds), dual pathway, gate, verification; synseg/syngrid/
shakespeare. **Gates:** P3a ≥0.1 bpb over byte, or ≥1.5x tokens/s; P3b
concepts ≥ BLT-style patches; P3c acceptance/delta; P3d ≥2x structured at
equal quality (the two-clock gate; the scorecard's ≥3x is the stretch).
**Readouts:** `speed.json` (tokens/s at batch 1/32, bpb, acceptance, rollback,
learning curve). **Stop:** concepts = patches → stop B3.

### P4 — the ops (1 week)

Market, pack/attach/detach/diff/merge/rm, provenance, deletion, constrained
generation, telemetry. **Gates:** P4a adapter pack/installed subset ≤1 MB,
install ≤1 min; P4b bits per adapter byte beats LoRA/kNN; P4c deletion and
provenance; P4d constrained generation 100% valid at ≤0.05 bpb; P4e telemetry
AUC; P4f speculative mode with an open 7B verifier ≥1.3x wall-clock vs REST.
**Ablations:** free vs composed embeddings; MDL admission vs top-k; phrase
vs byte emission; ledger growth vs LoRA. **Readouts:** verbatim-copy and
regurgitation vs rent `λ`; citations; unlearning; an open 1B distill at S3.
**Stop:** the λ ablation flat → the charge is decorative.

### P5 — the library (3-4 weeks, the r7 program)

E0 week 1: parity/usage/shuffle on an enwik8 slice + code/JSON, 30M cortex,
frozen library. E1 weeks 2-3: induction on a code corpus; the "learn a new
API in minutes" demo. E2 weeks 3-4: slot splits. E3: the 100M-1B equal-FLOP
scaling check vs an H-Net-style baseline before the big claim. Report the
library transfer matrix across domains.

**Gates:** L-G0 parity (within 0.03 bpb text, ≥0.1 win code/JSON); L-G1 usage
(concept-covered bit fraction ≥50%); L-G2 shuffle (permuted ids ≥0.2 bpb);
L-G3 induction (held-out DL falls per sleep cycle); L-G4 capability (+10
points on held-out slots); L-G5 adaptation (install beats LoRA on
bits-to-adapt × time); P5e planning (search win at equal FLOPs vs token CoT);
P5f auditable thought (re-expansion + type checks catch ≥X% structured
errors); P5g archive context (long-range consistency at equal context
FLOPs). **Stop:** only efficiency → not a capability result; if G2 or G4
fails, record the negative; the library is a bet, not a promise.

## 7. Efficiency levers

1. **One run, two readouts.** The mixed schedule trains one trunk for P1 and
   P0; the 0.03 bpb parity check protects the fixed readout.
2. **Scaling ladder.** 10M → 20M → 40M; fit the curve; only run 60M/100 MB if
   the 40M extrapolates below 2.29.
3. **10 MB first.** The 10 MB pilot (3.73M units) is 10x cheaper than the
   100 MB parse; the 100 MB is the verdict, not the first test.
4. **CE judge for search; round-trip for the record.** The arithmetic coder is
   0.02 MB/s; the pilot uses the CE in bpb; the 1 MB round-trip is the proof.
5. **3-parameter gate first.** The MLP only if the logistic fails.
6. **Warm start.** The concept head starts from the count table's logits; the
   embeddings from the encoder; no random init for frequent entries.
7. **Distill only after P2.** The 4 MB student is a second stage, not a
   parallel run.
8. **No C kernel, no binary pack, no custom CLI until P4.** The pilot uses
   `.npz` + JSON, Python, CPU judge.
9. **Early stop.** The CLI stops at the first failed gate; no wasted epochs.
10. **One seed for gates; 3 seeds only for the final table.** The spread is
    reported, not optimized.

## 8. Risks and stop rules

| risk | check | cost |
|---|---|---|
| the class is not fixed at 10 MB | P1a/P1b/P1c | 4-6 h |
| the 100 MB self claim is impossible | the charge table; P2c/P2d | free |
| the adapter does nothing | P0a vs D=∅ | 4 h |
| the gate over-copies | acceptance, delta, verification | 1 day |
| concepts are anonymous patches | P3b; the shuffle test | 3 days |
| joint EM oscillates | churn <10%/round; parse sampling; soft-lattice relaxation | 1 day |
| alignment regresses PPL | KL on/off; the 8.03 warning | 1 day |
| the ledger is non-stationary | versioned ledger; slow admission; EMA | 1 day |
| span commitment hurts quality | mid-span abort; entropy-capped span length | 1 day |
| new-span cold start | zero-shot eval within 0.1 bpb of tuned; else residual-only | 1 day |
| the pack bloats | installed subset ≤1 MB; rent | 1 day |
| the memory is a privacy surface | provenance, inspect, delete | 1 week |

**Stop rules.**

1. The recorded bits are the judge: `payload + memory + base`; the round-trip
   is the artifact.
2. Every budget, table and lexicon is chosen on the held-out 10%; never the val.
3. Report `rate`, `shared` and `self`; the base is amortized only when shared;
   the crossover is a result.
4. B1 dies if P0a fails; B2 dies if P1 is >2.80 or flat; B3 dies if concepts
   = patches; B4 dies if the slot gain is <10 points.
5. No speed claim without wall-clock at equal recorded bits.
6. The archive plan and the structured win are the floor; the pivot cannot
   regress them.
7. If phrase emission regresses PPL >5%, fix alignment or stop.
8. If the ledger churns >10%/round, slow admission; do not scale it.
9. If the adapter works and the LM does not, ship the adapter: the
   architecture claim is the adapter.
10. If the 40M curve does not cross 2.29, do not run 60M/100 MB.
11. If safe-mode acceptance collapses from teacher-forcing to free generation
    (exposure bias), fix the gate before scaling.
12. If the pack >1 MB or the install >1 min, the ops claim fails.
13. If holes buy only efficiency, B4 is not a capability result; say so.

## 9. Time

| step | cost | result |
|---|---|---|
| P0 adapter | 4 h | the install table; the product |
| P1 class 10 MB | 4-6 h | the class gap; the scaling curve |
| P2 class 100 MB | 1 GPU-day | the text verdict; the 4 MB student |
| P3 gate + structured | 2-3 days | the structured speed/quality |
| P4 ops | 1 week | pack/provenance/constrained/telemetry |
| P5 library | 3-4 weeks | composition/planning; the r7 program |

Minimal verdict: **P0 + P1, 1.5 days** (the product and the class gap). Add P2
for the 100 MB text verdict (+1 day). The full program is ~5-6 weeks on one
3080; B4 alone is the r7 1 GPU-month.

## 10. The big picture

The seven sketches agree on the mechanism and disagree on the stage. The
mechanism is one model, many streams: an exact span memory, a neural trunk, an
MDL price, a copy/compute gate. The stage is where the memory sits: outside
the base (the adapter), inside the vocabulary (the LM), beneath the compute
(the ledger), above the sequence (the library).

The charge decides the order. A 60M trunk cannot pay at 100 MB; a 300 kB
adapter can. So the adapter is the first shippable product; the vocabulary is
the scientific pivot; the ledger is the architecture; the library is the
capability. The best result in the design space is a small, exact, priced
memory that a frozen base can read — and a text rate that pays at the
deployment scale, not the file scale.

The later sketches add the capabilities that make it worth doing:
provenance-typed generation (every byte tagged), knowledge as a pack
(fit/attach/detach/diff/merge/rm), hard-exact constrained generation, the
learning curve (FLOPs/byte falls as the ledger learns), composition and
planning over holes, auditable thought (re-expand, type-check), and
archive-scale context (100k events ≈ MBs).

Scope: lexical/phrasal concepts (idioms, entities, boilerplate, code
patterns), not semantic abstraction; clustering `e_c` into senses is v2.
Target domains: repo-scale code assistants, JSON/log/structured generation,
domain boilerplate, on-device personalization, auditable/deletable models.
Exact spans kill BPE pathologies on identifiers, numbers and morphology.
Not open-domain chat SOTA per parameter.

The compressor becomes the auditor: the ledger is a charged model; the
round-trip recorded-size judge measures its MDL health while the LM loss
measures its intelligence.

If P1 passes at ≤2.293, the v9 negative is explained and reversed at 10 MB;
P2 decides 100 MB. If P1 only reaches 2.60, the class is fixed but the text
win is not; the adapter and the structured win are the result. If P1 stops at
>2.80, the neural pivot is a negative, and the archive, the structured win and
the adapter are the product. All three are publishable; the gates make the
difference measured, not hoped.

Provenance: v9's rows are `../runs/v9/{proof-100m-seed-10m.json,union.json,
CODEC.md}`; the PPMd-over-parse 1.992 and the 0.67/0.16 decomposition are
TODO5's measured accounting; the 10 MB parse is
`../runs/v9/fit-seed-16384-16384-10000000.pkl` (`toks`, 3.73M units); the
100 MB scan is `../runs/v9/scan100.npz`; the seed is 0.
