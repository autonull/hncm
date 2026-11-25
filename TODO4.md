# TODO4 — H-NCM v9: the coverage and context levers, then the 100 MB verdict

Status: v8 is measured end to end. `codec.py` round-trips every corpus; the
recorded sizes are **3.472** (Shakespeare), **1.389** (synseg), **2.098**
(syngrid) at 1.2 MB, and **3.035** on enwik8 at 10 MB. The structured win is
real (synseg 0.111 below PPMd); the enwik8 text gap is 0.875 bpb and it is the
tail. The v7 estimates (2.723/3.006) are corrected lower bounds.

**One-sentence state.** The phrasebook + concept model + shared fallback is one
artifact that pays on structured data and loses on natural text at 10 MB; the
covered part is already below PPMd per covered byte, the 24% uncovered at 5.6 bpb
is the gap, and the three remaining levers are coverage (the union vocabulary),
the tail's context (order 2), and the charge (100 MB, 0.02 bpb).

## 0. Start here

```bash
python codec.py selftest
python concept_lm.py selftest
python probe.py selftest
python ncm.py selftest
python corpora.py fetch --corpus enwik8_10m   # cached
python codec.py run --corpora shakespeare,synseg,syngrid --try 3
python probe.py run --corpus enwik8_10m --vocab substr --sizes 8192,16384
```

Artifacts: `runs/v8/{codec.json,COVERAGE.md,ENWIK8.md,ncm10m.json}`. The
candidate scan is cached at `~/.cache/hncm/counts-<corpus>-<vocab>-<sha>.npz`,
so a 9 min scan is seconds on rerun.

## 1. What is established (measured; do not re-derive)

- **The bitstream.** The model is packed as counts; the CDF is derived
  (`round(2^24 p)`); a 32-bit arithmetic coder writes the DP's path; the
  decoder never reruns the DP. Recorded = the code's entropy + the charge within
  **0.0013 bpb** in both conventions (`shared`, `self`).
- **The v7 estimate is a lower bound.** Its `covered = total - sum(byte unigram)`
  subtracts 5.05 bpb for bytes the concept path had already priced at the bigram
  row (3.83). The record is 3.472 on Shakespeare, not 3.006.
- **The concept arm is the architecture.** One normalized model (every token is
  coded by the row of the previous token, so the H-Net rows are the tail's
  context) beats the hard split by 0.09 at half the model on enwik8 (3.035 vs
  3.126) and by 0.27 at 1 MB (3.671 vs 3.939).
- **The 10 MB coverage.** The dictionary covers 0.545/0.629/0.698/0.756 at
  2k/4k/8k/16k substrings, 0.414-0.536 at word onsets. The covered tokens
  cost ~2.0 per covered byte (below PPMd 2.16); the uncovered 24% at 5.6 bpb
  is the gap.
- **The tail's context.** The SVD restarts at 256; the real preceding bytes
  give order-1 4.17, order-2 **3.83**, order-3 3.73 on the 1 MB tail (the SVD
  4.48); the val ceilings are 3.80/2.75/1.62. The 25 kB 3-byte NCM reaches
  4.307 on the 10 MB tail against the SVD's 4.459 (+0.04 net).
- **The structured win.** synseg 1.389 against PPMd 1.500 and bz2 1.513. The
  syngrid is 2.098 against lzma 2.005. The K-file curve is flat; the per-file
  codecs rise.

## 2. The work, in order

### N8 — the union vocabulary (30 min) — **do this first**

*Question:* does one vocabulary of word tokens **and** substring tokens raise
coverage at the same charge?
*How:* in `concept_lm.py`, merge `counts_onsets` and `counts_substr` into one
candidate map, one EM unigram, and one DP over the union. Measure enwik8_10m at
the 8k/16k/32k dict (table 16k), the coverage, the covered rate, the tail and
the recorded. The codec is unchanged.
*Gate:* coverage ≥ 0.80 and the covered rate ≤ 2.0 at no more model bytes; the
recorded 10 MB ≤ 2.94.
*Stop:* if the coverage gain is eaten by the charge, the vocabulary is not the
lever; record it and go to N9.

### N9 — the order-2 tail in the concept arm (1 h)

*Question:* does the real two-byte context replace the concept row on the
uncovered spans at the same charge?
*How:* a pruned order-2 count table (the actual preceding bytes, `3-4 B/entry`)
with the order-1 backoff, used only when the previous token is a single byte.
The span's first byte uses the concept row; the rest the order-2 table. Select
the table size on the held-out 10%; charge it.
*Gate:* the tail ≤ 4.0 and the recorded 10 MB ≤ 2.94.
*Stop:* if the counts do not beat the row at 10 MB, the text claim is a negative
at this size; report it.

### N9b — the NCM vs the counts at 10 MB (30 min)

The 25 kB 3-byte NCM is 4.307 against the SVD's 4.459 but the count order-2 is
3.83 at 1 MB. If the count table wins at 10 MB, it is the tail of record; if
the NCM wins, charge it and use it. Either way, one tail model per stream size,
chosen on the held-out 10%.

### N10 — the 100 MB run (2-3 h) — **the verdict**

*How:* `enwik8_100m`; the split scan cached; the dict 32k/64k; the table
16k/32k; the order-2 tail; the codecs on the 1-10 MB val; the chunked DP; the
charge 0.018-0.04; the calls 20x+.
*Gate:* the recorded ≤ 1.90, or the crossing in MB, or the negative. Report
PPMd at 10/100 MB on the same val.
*Stop:* if the recorded is behind PPMd by more than 0.10, the text claim is a
negative; the structured win is the product.

### N11 — the structured table of record (1 h)

synseg, syngrid, Shakespeare at 1.2 MB: the recorded size, the charge, the
calls, the coverage, the encode/decode throughput, both conventions, the seven
codecs (PPMd included). The claim: synseg pays; syngrid is 0.09 behind lzma;
Shakespeare is behind PPMd.

### N12 — the factorial `I_perf` (1 h, 3 seeds)

A byte-AR, B the dense codec (v6), C the dictionary + count model + DP, D the
dictionary + concept arm + NCM, at a matched budget, on synseg (the win) and
enwik8_1m (the loss). `I_perf = (D-C) - (B-A)`. The per-seed spread.

### N13 — delivery (1 h)

`runs/v9/` with the model files, the config JSON and the hashes; the README
headline; `REPORT.md`; one command.

## 3. Risks and cheap checks

| risk | check | cost |
|---|---|---|
| the union's charge eats its coverage | N8's recorded 10 MB | 30 min |
| the order-2 table is too big at 10 MB | 95 kB = 0.076 bpb; the recorded | 1 h |
| the 100 MB scan is 1-2 h | the counts cache | once |
| PPMd wins at every text size | the codec table at 10/100 MB | free |
| the 1 MB/10 MB result is an artifact | N10's 100 MB | 2-3 h |
| the shared model only wins at K=1 | `transfer.py` with the v8 charge | 15 min |

## 4. Stop rules

1. The recorded size is the judge (rule 17); a rate estimate is not.
2. Every budget and table size is chosen on the held-out 10%, never the val.
3. The model's size is charged at its packed bytes.
4. If the union and the order-2 tail both fail, the text claim is a negative at
   10 MB; publish the structured win and the clean negative.
5. If the 100 MB is behind PPMd by > 0.10, the text claim is a negative.
6. The structure claims (the model paid once, the flat K curve, 6.5-20x fewer
   calls) stand regardless of the rate verdict.

## 5. The big picture

The mechanism is one model, many streams: an exact phrasebook (the dictionary
index), a first-order concept model (the decoder-side cost), and a shared
fallback (the spans the phrasebook lacks). The model is paid once. On structured
sources the phrasebook carries the structure and the model wins at 1.2 MB
(synseg 1.389 against PPMd 1.500). On natural text the covered part is already
cheaper than PPMd per covered byte; the uncovered 24% at 5.6 bpb is the gap, and
the order-2 context is the lever. At 100 MB the model charge falls to 0.02 bpb
and the coverage rises; that is where the text claim is decided. If it works, a
single small model compresses a mixed archive, paid once, with 6.5-20x fewer
decoder calls and a flat K-file curve; if it does not, the structured win and
the measured negative are the result.

## 6. Time

| step | cost |
|---|---|
| N8 the union vocabulary | 30 min |
| N9 the order-2 tail | 1 h |
| N9b the NCM vs the counts | 30 min |
| N10 the 100 MB verdict | 2-3 h |
| N11 the structured table | 1 h |
| N12 the factorial | 1 h |
| N13 delivery | 1 h |

Minimal decisive run: **N8 + N9 + N10 (4-5 h)**. The rest is the record.
