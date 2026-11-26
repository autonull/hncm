# TODO7 — the code codec: capacity in the lexicon, not the weights

Thesis: at equal total bytes, a **small** unit model plus an installed code
lexicon beats a bigger byte model on source code. The model does not learn the
vocabulary; it reads it. Measure that with a real codec, then push it.

This file supersedes the analysis phase. Read `cml/PLAN.md` §11 for the
corrected accounting (the P0 oracle −2.224) and `TODO6.md` for the frozen
state.

## 0a. Progress after the first build (M0-M3)

The corpus is pinned in `~/.cache/hncm/code-current.json`: SHA `1ccd9ffc7f68`,
555 files, 8.0 MB (train 7.24 MB, test 0.757 MB). `cml recorded` owns
`reports/RECORDED.md`; every row below carries that SHA.

| M | state | evidence |
|---|---|---|
| M0 | **done** | `cml/codec.py`: `ArithEncoder/Decoder`, `Counts`, `quant`; `cml coder --test` passes 10k fixed + 10k adaptive symbols and a 200 kB byte round-trip |
| M1 | **done** | byte arm on 756,752 B of test: payload/shared **2.522/2.523 bpb**, exact round-trip |
| M2 | **done** | 8,192-entry lexicon (89,713 B); 200k test **coverage 0.919**, mean unit **3.23 B**; payload **2.321**, shared **5.911** (the 90 kB header dominates at 200k) |
| M3 | **done** (local copy pending) | 4.97M pointer, 8 rotating packs, `full_every=8`, 8,000 steps: 70k test payload **2.386**, shared **2.709** vs adaptive byte **2.807**; proof **50,000 units** exact; unseen-pack gap **−0.121 bpb** (Q1 bar 0.15 passes) |
| M4 | **negative at 1 MB** | 4.86M trained byte trunk, static coder: 70k **1.916/1.922 bpb**. The 4.97M pointer at **756,752 B** records **2.207/2.237**; the byte model at the same n is pending. Caveat: the active pack is still 2.8 kB (256 units) |
| M5 | **ops done; demos pending** | `cml pack verify/trace/diff/merge/rm`. Q2: 756,752 B decode, **511,933 elements**, provenance contiguous 100%, `byte → unit id → pack offset` printed. Q3: `rm self` on a 70k unit-arm slice deletes 27 units, hits 4 spans / 18 B, delta +2 B; `rm s` on a 20k pointer slice hits 222 spans; both round-trip. Adapter control and rent sweep remain |
| M6-M7 | **not started** | The P1 text control and the librarian remain |

Concrete remaining work, in order:

1. **M4 diagnostics.** The trained byte model wins at 70k and the pointer's 1 MB
   row is recorded. The exact batched encode is done: windows are coded as
   per-window segments in groups of 32, so encode and decode share one batched
   step and a cross-process round-trip holds. Remaining: a larger active pack
   (`--active 2048/8192`, retrained, in flight) and local copy symbols. If the
   0.2 bpb bar still fails, accept the negative and write it down.
2. **M5.** The ops are built. Run the adapter arm as the 30-min negative control
   and the rent λ sweep; then Q4's class split.
3. **M6-M7.** The P1 text control, then the librarian.

Two build notes. The pointer codes 228k units in ~80 s at 756 kB (batch 32),
and a 50k-unit proof at that length decodes in seconds; per-window framing
costs ~3 B/window (charged in `header`). `code_corpus(refreeze=True)` takes a
new snapshot; normal runs never read the moving working tree.


## 0. What is measured (start from this, do not re-derive)

| fact | number | where |
|---|---|---|
| P1 unit trunk, 10 MB enwik8 parse | **1.920 bpb**, 1.912 ± 0.008, vocab 16219, d384, 8 layers, seq 128, 20.46M | `runs/p1-class-10m-3084fd95-s0/` |
| P1 test slice | 366,020 units / MB | `metrics.json` |
| P0 adapter charged (name paid) | 3.97 bpb, **−2.224 net** → P0a fails | `runs/p0-adapter-4e989907-s0/` |
| P0 code oracle / charged | 2.739 / **−0.403** | `runs/p0-code-79525d79-s0/` |
| the gate | length is the predictor; tied = counts (0.510/0.510), cos AUC 0.427 | `runs/p0-linchpin-5fb03736-s0/` |
| measured speed (20.46M, batch 32×128) | encode **286k units/s**; `win_step` **6.9 ms/unit** | this session |
| the only decodable P0 row | LoRA −0.033 | `runs/p0-lora-4ab0bed1-s0/` |

Three invariants to carry (they cost this session to learn):

1. **Charge the message, not the decision.** A copied span must name its entry.
2. **Features live in the decoder's state.** Length and match are future info.
3. **CE is a scheduler metric.** The claim is `8·(blob + header + pack + base)/n`.

## 1. Why code first, text second

Code is where the lexicon pays: identifiers, boilerplate, imports and literals
repeat across files, so a unit model sees ~4-6 B per symbol and spends weights
on structure while the pack carries the vocabulary. On 10 MB of text the pack
charge and the trunk charge already lose to PPMd by arithmetic. Code has no such
baseline to beat; it has the 20M byte model at 4.6 bpb (out of domain) and a
real customer: a repo assistant.

The first claim to earn: **a 5M unit model + a 90 kB lexicon ≤ a 5.5M byte
model on held-out Python at equal total bytes**, recorded end-to-end.

The first probe of that claim fails: at 70 kB, a 4.86M trained byte model
records **1.922 bpb**, while the 4.97M pointer plus a 2.8 kB active pack
records **2.709**. The pointer also wins against the *adaptive* byte arm
(2.807), so M3's bar and M4's equal-bytes bar separate cleanly.

## 1a. The budget that forces the pointer head

A vocab softmax does not fit: at d=256, 6 layers, an 8k unit vocabulary costs
2×2.11M for the embedding and the head, and 4.72M for the blocks — 8.9M, not
5M. And a trained row per entry means a new lexicon needs a gradient step, which
kills the install.

The sketches' answer (r3 "pointer", r6 "tied embeddings", r1 "content, not ID")
is a **pointer head**: each installed span gets `e_c = Enc(bytes_c)` from a tiny
local encoder, and the head scores `(W_q h_t)·e_c`. Then:

| part | params |
|---|---:|
| blocks, 6 × 12d² at d=256 | 4.72M |
| local span encoder (2 layers, d=64) | 0.05M |
| `W_q` (d×d) + byte head (256×d) + gate | 0.14M |
| **total** | **≈4.9M** |

A new entry costs zero parameters: install = encode its bytes once. This is the
one sketch idea the plan must adopt *before* M3, because it changes the head,
the coder's alphabet (installed spans, not ids) and Q1 (a pack the model never
trained on). Note carefully: P0's linchpin tested `W_q = I` on a *frozen* base
with a *classifier* gate; it did not test a trained pointer over installed
spans. Do not inherit that negative here.

## 1b. Ideas from r1-r7 that the plan dropped and should take back

| idea | sketch | why it belongs now | slot | cost |
|---|---|---|---|---|
| lexicon-agnostic training: rotate the pack per step (2-4 parses, held-out packages) | r3 | install = data only if the model can read a pack it never saw | M3 | +1 h |
| local copy symbols: repeat from the last 1-4 kB (LZ-style distance+length codes) | r2 copy bypass | code repeats identifiers inside a file; a static pack cannot | M3 | +1 h |
| re-parse with the model's prices (Viterbi over spans under the pointer's log-probs) | r4/r1 DP | longer units where the model is confident; the parse is the headroom | M4 | +1.5 h |
| market admission: admit a span iff measured saving > its charge | r5 | picks the pack at fixed bytes instead of a count heuristic | M4 | +1.5 h |
| two clocks readout: steps/byte, FLOPs/byte vs pack size | r6/r5 | the speed half of "smaller model"; it is a measurement, not a build | M4 | +0.5 h |
| librarian: anti-unification lifts a template with a hole from repeated lines | r7 | Q1 needs a template, not a hand-written slot split | M7 | +1 d |

Left out on purpose: the charge term in the training loss (r2) — the equal-bytes
table does the selection for now; safe-mode verification (r3/r1) — a generation
concern, not a codec one; joint EM with parse sampling (r4) — the single
re-parse above is the 80%; alignment KL (r5) — a PPL regularizer, no codec role.

## 2. Efficiency rules for this build

- **Encode at scale, decode a proof slice.** Encoding is batched (286k units/s,
  1 MB in ~2 s). Decoding is causal (6.9 ms/unit at 20M, ~2 ms at 5M): prove
  50k-100k units once per arm, never per iteration.
- **Top-k CDF.** The coder's alphabet is the trunk's top 256 units + one byte
  escape (257 symbols), not 16k. Measure the rate loss once; if >0.02 bpb, use
  k=512.
- **One table.** `reports/RECORDED.md`, generated by `cml recorded`; `GATES`
  reads it. Delete `saved` (oracle) from the gate metrics; `recorded` replaces
  it. No second accounting.
- **One file per seam.** Split `cml/codec.py` (coder + counts + arms) because
  the CLI, the tests and the recorded table are three callers. `cml.py` keeps
  data, model, pack and the CLI.
- **No new manifest per arm.** `cml codec --arm/--corpus/--bytes` is the single
  source of truth; every run writes `runs/codec-<arm>-<corpus>-<bytes>/`.
- Ruff format; `ruff check --fix` on changed files; tests only for the
  invariants below; atomic writes; the header carries the SHAs.

## 3. The corpus (one source of truth)

`code` = the repo's `*.py` plus the installed venv's `site-packages/**.py`,
sorted by path, frozen to `~/.cache/hncm/code-<sha12>.bin` on first use (the
existing `src_bytes` freeze, generalized). Cap at 8 MB. Split by **whole
files**: the last 10% by path is the test, so the test shares no file with the
train. Record `corpus_sha` in every header and metrics.
- Drop `test`/`__pycache__`/vendored duplicates by path filter; keep the SHA as
  the truth. Target 5-8 MB, ≥1500 files. The first pin is 8.0 MB / **555
  files**: the path-sorted cap spends the budget on a few large packages. The
  file-count target is not met; a deliberate small-file selection is the next
  option, and the actual count is recorded rather than hidden.

## 4. Milestones (each ships a runnable number)

| M | ships | acceptance | cost |
|---|---|---|---|
| M0 ✅ | `cml/codec.py`: `ArithEncoder/Decoder`, `Counts`, `quant`, `cml coder --test` | 10k random symbols round-trip; 200 kB byte codec round-trip | 1.5 h |
| M1 ✅ | byte arm: order-2 counts, escape to order-1; `cml codec --arm byte --corpus code` | exact round-trip; recorded bpb on 1 MB | 1.5 h |
| M2 ✅ | lexicon + greedy parse: `lexicon(code, 8192, min_count 16)`; OOV → byte escape; header carries the lexicon | re-expand == input; ≥60% bytes covered by units; avg unit ≥3 B | 1.5 h |
| M3 ✅ | the pointer head: 5M trunk (d256, 6 layers, seq 256), content span embeddings + `W_q`, byte head, gate; rotating lexicon; local copy symbols; `--arm concept` | exact round-trip on 50k units; a pack the model never trained on still decodes; recorded bpb < byte arm | 4.5 h |
| M4 ✗ | the equal-bytes table + the levers: 5M byte vs 5M pointer + 90 kB lexicon; re-parse under the model's prices; market admission; steps/byte and FLOPs/byte | the unit arm wins ≥0.2 bpb at equal total bytes, else record the negative | 4 h |
| M5 ◐ | ops `verify/trace/diff/merge/rm` ✅; Q2/Q3 demos ✅; the adapter arm as a 30-min negative control (flag+index+byte) and a rent λ sweep remain | provenance 100%; `rm` localizes the delta and still round-trips | 2.5 h |
| M6 | text control: the P1 trunk concept codec on the 10 MB parse (recorded, vs PPMd 1.992) | recorded bpb within 0.05 of 1.920 + header | 2 h |
| M7 | library with holes: the librarian lifts a template by anti-unification; slots fill from other units; Q1 | ≥0.1 bpb on slot tokens vs the byte arm | 1-2 d |

Do not start M7 before M4 prints a table. Do not tune M3 before M1 records a
number.

## 5. The coder (M0), exact interface

```python
class Frequencies(Protocol):
    n: int
    def cums(self) -> np.ndarray: ...       # int64, len n+1
    def find(self, target: int) -> int: ... # bisect_right(cums, target) - 1
    def update(self, sym: int) -> None: ...

class ArithEncoder:  # WNC 32-bit, E1/E2/E3, bit output, zero-padded tail
    def encode(self, lo: int, hi: int, total: int) -> None: ...
    def finish(self) -> bytes: ...
class ArithDecoder:
    def __init__(self, blob: bytes) -> None: ...
    def target(self, total: int) -> int: ...
    def update(self, lo: int, hi: int, total: int) -> None: ...
```

- `Counts(n, cap=1<<16)`: adaptive, `+16` per update, halve at cap; cached
  `cums` invalidated on update (one cumsum per symbol, 256 or 257 long).
- Neural CDF: `q = clamp(round(p * (1<<20)), 1)`, `cums = cumsum(q)`; the
  ~0.012 bits/symbol quantization loss is acceptable; k=256 keeps it cheap.
- The bit-level coder is ~80 lines and is the only piece that must be exact on
  the first try: test it against random frequency tables and random symbol
  streams before any arm uses it.

## 6. The record (M4) and the gates

```
payload = 8·blob_bytes / n
shared  = payload + 8·(header + pack)/n
self    = shared  + 8·base_bytes/n      # base_bytes = params/2 (4-bit)
```

`cml recorded` writes one row per `(arm, corpus, bytes)` with those three
numbers, the header's SHAs, and the round-trip status. `GATES` P0a/P1b/P2a-c
read `recorded`/`shared`/`self`; the CE metrics stay in the run store as
`ce_bpb` and never appear in the table.

## 7. The qualitative tests (the new qualities)

- **Q1 install.** Two packs: one the model trained with, one induced on a
  package it never saw. Encode a held-out file with each, no gradient: report
  bpb vs the byte model and the *pack-agreement* gap. Bar: ≥0.2 bpb at equal
  total bytes, and the unseen pack within 0.15 bpb of the seen one.
- **Q2 provenance.** Every decoded byte has `byte → unit id → pack offset`;
  `trace A:B` prints it; 100% coverage. Measured: 756,752 B / 511,933 elements,
  contiguous, on both the unit arm and the pointer arm. This is the auditable-memory claim, and
  no byte codec can produce it.
- **Q3 editing.** `rm --prefix X` deletes a concept; the affected spans become
  escapes and the delta is localized to their byte ranges; print the ranges and
  the bpb delta. Measured: `rm self` removes 27 units, hits 4 spans / 18 B,
  +2 B payload on a 70k slice, round-trip OK.
- **Q4 capacity split.** Break the recorded bits down by token class
  (identifier, keyword, operator, literal, whitespace, comment). The claim is
  that the unit arm's win concentrates on identifiers and literals — the small
  model spends its weights on structure. If the win is on whitespace, the
  lexicon is a whitespace compressor; say so.

## 8. Traps, already paid for

- **The oracle trap.** `adapter_eval` charged `entropy(pi)` and no name; the
  name cost 0.918 bpb at 3.27 bits/span. Any copy names its entry.
- **The future-information trap.** A gate on `length` is decodable only if the
  encoder sends the length; the flag+index codec does.
- **The moving-corpus trap.** Freeze the corpus; key the lexicon by the train
  SHA and the header by every SHA.
- **The timing trap.** P0d repeats 2.2-4.3x (synseg) and 1.05-1.40x (code); use
  ≥3 repeats and the median, or no speed claim.
- **The CE trap.** `eval_bits` is a scheduler; only `recorded` is a claim.
- **The identity-cosine trap.** P0's cos AUC 0.427 measured `W_q = I` on a
  frozen base with a *classifier* gate. A trained pointer over installed spans
  is a different object; do not cite the linchpin against it, and do not build
  the pointer with the identity map either.

## 9. Start

```bash
cd cml
uv sync && uv run pytest -q
uv run cml coder --test
uv run cml codec --arm byte --corpus code --bytes 1000000
uv run cml codec --arm byte-model --corpus code --bytes 756752 --steps 3000
uv run cml codec --arm lexicon --corpus code --bytes 200000
uv run cml codec --arm concept --corpus code --bytes 756752 --proof 50000 --steps 8000 --rotate 8 --full-every 8
uv run cml codec --arm concept --corpus code --bytes 756752 --proof 50000 --steps 8000 --rotate 8 --full-every 8 --active 2048
uv run cml recorded
uv run cml pack verify codec-concept-code-756752
uv run cml pack trace codec-concept-code-756752 --range 0:256
uv run cml pack diff codec-lexicon-code-200000 codec-concept-code-756752
uv run cml pack merge pack-merged codec-lexicon-code-200000 codec-concept-code-756752
uv run cml pack rm codec-lexicon-code-200000 --prefix self
```

The concept arm codes in fp16 for speed; training stays fp32. Its model is
cached by `(train_sha, active, rotate, full_every, steps, seed)` and copied
into the run directory. Encode groups 32 windows of 256 elements; each window
is one arithmetic segment, both sides run the same batched step, and the
segment table is charged in `header`. The 50k-unit proof decodes in seconds.

References: `cml/PLAN.md` §11; `n10.py:241` (`build_full`); `runs/v9/CODEC.md`
(the header and charge); `reports/GATES.md` (the record).
