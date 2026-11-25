# TODO7 — the end-to-end codec, then optimize

This supersedes the analysis phase. The next session builds one runnable system:
a real range coder, every arm selectable, a byte-exact round-trip, and the
recorded rate as the only judge. Then it optimizes the recorded rate and the
qualitative tests. No CE-to-bpb claim survives this file.

## 0. State: what is measured, what is dead

The codebase is `cml/` (one file, `cml/cml.py`, ~2900 lines; the plan's rule is
one file until P4). Tests: `cml/tests/test_invariants.py` (16 pass). The run
store is `cml/runs/<id>/`, the record is `cml/reports/GATES.md`, the plan is
`cml/PLAN.md` §11.

| fact | number | where |
|---|---|---|
| P1 unit trunk, 10 MB parse | **1.920 bpb** payload, 1.912 ± 0.008 seeds 0-2 | `runs/p1-class-10m-3084fd95-s0/` |
| P1 checkpoint | vocab 16219, d 384, 8 layers, 8 heads, seq 128, 20.46M params | `artifacts/model.pt` |
| P1 test slice | 366,020 units per 1 MB | `metrics.json` |
| P0 adapter, oracle (name free) | 0.997 bpb, +0.758 net | `runs/p0-adapter-4e989907-s0/` |
| P0 adapter, **charged** (count prior) | 3.97 bpb, **−2.224 net** → P0a fails | same |
| P0 on code, charged | **−0.403 net** (oracle +1.621) | `runs/p0-code-79525d79-s0/` |
| the linchpin | tied = counts (0.510/0.510), gate w0 −0.04, cos AUC 0.427 | `runs/p0-linchpin-5fb03736-s0/` |
| zero-shot pack | +0.422 bpb vs a domain pack | `runs/p0-zeroshot-a5704c19-s0/` |
| LoRA at equal charge | −0.033 net (the only decodable P0 row) | `runs/p0-lora-4ab0bed1-s0/` |

Three lessons from the analysis phase, now invariants:

1. **Never charge a decision, always charge the message.** A copy must name the
   entry; the name is what killed P0. Any arm that emits a symbol pays for it.
2. **Features must be in the decoder's state.** `length` and `match` are future
   information; a gate may use them only if the codec transmits them.
3. **CE is a scheduler metric.** The claim is `8·(blob + header + pack + base)/n`.

## 1. The build order (each milestone ships something runnable)

| M | ships | acceptance | cost |
|---|---|---|---|
| M1 coder | `ArithEncoder`/`ArithDecoder`, `Counts`, `cml coder --test` | 10k random symbols round-trip; 200 kB byte-codec round-trip | 2 h |
| M2 byte arm | `cml codec --arm byte` | round-trip exact; `bpb < 3` on enwik8 | 1 h |
| M3 concept arm | `cml codec --arm concept` over the cached parse | round-trip exact; `bpb` within 0.05 of 1.920 | 3 h |
| M4 adapter arm | `cml codec --arm adapter` with the entry name charged | round-trip exact; `bpb` within 0.05 of the −2.224 estimate | 2 h |
| M5 pack ops | `cml pack verify/trace/diff/merge/rm` | provenance 100%; `rm` shrinks and still round-trips | 2 h |
| M6 the table | `cml recorded` → `reports/RECORDED.md`; gates read recorded rates | the table lists all arms × corpora × conventions | 1 h |
| M7 ledger arm | rent sweep, admission/eviction, per-byte provenance | recorded rate vs rent λ; delete a concept and localize the delta | 3 h |
| M8 library arm | templates with typed holes, slot fills | Q1-Q3 below | 1-2 days |

Do not start M7/M8 before M1-M6 pass; do not optimize before a round-trip exists.

## 2. The coder (M1)

32-bit arithmetic coder, Witten-Neal-Cleary with E1/E2/E3 and pending bits.
Bit output, zero-padded tail; the decoder reads zero past the end.

```python
class Frequencies(Protocol):
    n: int
    def cums(self) -> np.ndarray: ...      # int64, len n+1, cums[0]=0
    def find(self, target: int) -> int: ...# bisect_right(cums, target) - 1
    def update(self, sym: int) -> None: ...

class ArithEncoder:
    def encode(self, lo: int, hi: int, total: int) -> None: ...
    def finish(self) -> bytes: ...

class ArithDecoder:
    def __init__(self, blob: bytes) -> None: ...
    def target(self, total: int) -> int: ...
    def update(self, lo: int, hi: int, total: int) -> None: ...
```

- `Counts(n, cap=1<<16)`: adaptive table, `+16` per update, halve at `cap`.
- Quantized neural tables: `q = clamp(round(p * 1<<20), 1)`, `cums =
  cumsum(q)`, `total = cums[-1]`. This costs ~0.012 bits/symbol; acceptable.
- Test with random symbol sequences and random tables; assert byte equality.

## 3. The arms (M2-M4)

**byte** (`arm=byte`). Order-1 adaptive bytes: 256 `Counts` keyed by the
previous byte, uniform CDF for the first byte. Header: `{arm, n, corpus_sha}`.

**concept** (`arm=concept`). The alphabet is the lexicon's units (16,219). The
CDF is the P1 trunk's softmax, quantized as above. Encode: one batched forward
per window of `seq` units (the P1 eval layout: windows of `seq+1`, positions
0..seq-1); decode: one `win_step` per unit with a fresh cache per window. The
first unit is sent with a uniform CDF. Header: `{arm, model_sha, lexicon_sha,
n_units, parse="cached"}`.
- Decode must re-expand via `lexicon.strings[id]` and equal the original bytes.
- The parse is *given* by the cached pickle; the header says `parse=cached`.
  A self-produced parse is P3's joint EM, not M3.
- Use `runs/v9/fit-seed-16384-16384-10000000.pkl` and checkpoint
  `runs/p1-class-10m-3084fd95-s0/artifacts/model.pt`. Decode 20k units ≈ 1 min.

**adapter** (`arm=adapter`). Greedy longest match over the pack trie.
- copy: flag 1 (adaptive binary, context = previous flag) + the entry index
  coded with a **static count-prior CDF** shipped in the pack.
- fallback: flag 0 + the byte from the byte arm's model.
- The decoder mirrors: read the flag, the index, or the byte. This is the honest
  name charge; its measured rate is the number the P0 oracle hid.
- Header: `{arm, pack_sha, n_bytes}`. Encode 200 kB ≈ 30 s.

## 4. Interfaces and CLI

Add to `cml.py` (no new modules until P4):

```python
def run_codec(arm: str, corpus: str, n_bytes: int, **kw) -> dict
def codec_file(arm: str, path: Path, out: Path) -> Path   # blob + header
def decode_file(path: Path) -> bytes                      # no side channels
```

```
cml coder --test
cml codec --arm byte|concept|adapter --corpus enwik8_10m --bytes N
cml codec --arm concept --units N --model runs/p1-.../artifacts/model.pt
cml pack verify|trace|diff|merge|rm <pack> [--prefix X] [--range A:B]
cml recorded            # the table: arm × corpus × rate/shared/self
```

`recorded` writes `reports/RECORDED.md` with, for each arm:
`payload = 8·blob/n`, `shared = payload + 8·pack/n`, `self = shared + 8·base/n`,
plus `header` bytes and the provenance rate. Gates P0a/P1b/P2* read these.

## 5. Pack ops (M5, the P4 seed)

- `verify`: every entry's `fit:<offset>` matches the corpus at that offset
  (SHA over the re-extracted bytes); prints the count and the pack SHA.
- `trace A:B`: for a byte range in the concept arm, one line per unit:
  `unit_id, provenance, bytes, span`. This is the auditable output.
- `diff P Q`: added/removed/changed entries and the byte delta.
- `merge P Q`: counts summed, deduplicated; report the new size and SHA.
- `rm --prefix X`: delete entries, re-run `verify`, and re-run the adapter
  codec on 200 kB to show the rate delta (the quantitative edit).

## 6. After the round-trip: the optimization order

1. **Coder precision**: total bits 1<<18 vs 1<<20 vs 1<<22 — the delta is the
   only free win; measure it once.
2. **Byte model**: order-1 vs order-2 with an escape; report both.
3. **Concept**: the window layout (seq 64/128/256) and the CDF temperature;
   then the parse as a variable (greedy vs DP) — the parse is the headroom.
4. **Adapter**: pack size at scale (dict 1024/4096/16384 at 650 kB, 10 MB,
   100 MB); the name charge scales as `8·pack/n`, so a repo-scale stream is the
   only place the arm can pay.
5. **Ledger (M7)**: rent λ sweep, admission/eviction; the recorded rate vs λ.
6. **Library (M8)**: holes and slot fills; Q1-Q3.

## 7. The qualitative tests (the reason to build concepts)

These are the claims a byte codec cannot make; run them at M8, report them next
to the rate.

- **Q1 composition.** A held-out slot split: a template whose slot varies over
  50 values. Measure bpb on slot tokens for the library arm against the byte
  arm; bar: ≥0.1 bpb on those tokens.
- **Q2 re-expansion.** Every decoded byte has a provenance chain
  `byte → unit id → pack offset`; `trace` prints it; 100% coverage.
- **Q3 editing.** `rm --prefix X` deletes a concept; the affected spans become
  fallback bytes and the delta is localized to their byte ranges (report them).
- **Q4 planning.** An event-level beam vs a byte-level beam at equal FLOPs
  (later; needs M8's events).

## 8. Traps, already paid for

- **The oracle trap.** `adapter_eval` charged `entropy(pi)` and no name. With
  the count prior the arm is −2.224 on text; the codec must transmit the index.
- **The future-information trap.** A gate on `length` is decodable only if the
  encoder sends the length; the codec's flag+index does send it.
- **The moving-corpus trap.** `src` is frozen as `~/.cache/hncm/src-<sha>.bin`;
  pack keys include the train SHA. Never key a pack on the config alone.
- **The timing trap.** P0d repeats at 2.2-4.3x (synseg) and 1.05-1.40x (code);
  use ≥3 repeats and the median, or no speed gate.
- **The CE trap.** `eval_bits` is a scheduler; a claim needs `recorded`.

## 9. Start here

```bash
cd cml
uv sync
uv run cml selftest
uv run pytest -q
uv run cml codec --arm byte --corpus enwik8_10m --bytes 200000
uv run cml codec --arm adapter --corpus enwik8_10m --bytes 200000
uv run cml codec --arm concept --units 20000 \
  --model runs/p1-class-10m-3084fd95-s0/artifacts/model.pt
uv run cml recorded
```

References: `cml/PLAN.md` §11 (the corrected accounting), `n10.py:241`
(`build_full`, the 90 MB parse), `runs/v9/CODEC.md` (the v9 codec's header and
charge), `cml/reports/GATES.md` (the record).
