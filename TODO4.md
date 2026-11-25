# TODO4 — H-NCM v9: the coverage and context levers, then the 100 MB verdict

Status: v9's N8, N9 and N10 are measured. The union vocabulary raises the
10 MB coverage to 0.815 at 32k and the codec's record from 3.035 to 2.987 (a
0.048 win; the 2.94 gate misses by 0.047). The order-2 tail on the DP path is
4.329 against the row's 4.738; on the NCM's exact tail it is 4.151 against
4.307 (+0.113 net), but the 4.0 gate misses. At 100 MB the charged estimate is
2.503 against PPMd's 1.834 (coverage 0.825, charge 0.040): the text claim is a
negative at 100 MB, and the structured win stands.

## 0. Start here

```bash
python codec.py selftest
python concept_lm.py selftest
python probe.py selftest
python ncm.py selftest
python n10.py selftest
python corpora.py fetch --corpus enwik8_100m   # cached
python codec.py run --corpora shakespeare,synseg,syngrid --try 3
python concept_lm.py run --corpora enwik8_10m --vocab union \
  --dicts 8192,16384,32768 --prunes 8192,16384,32768 --out runs/v9/union.json
python probe.py run --corpus enwik8_10m --vocab union \
  --sizes 8192,16384,32768 --out runs/v9/coverage-union.json
python n10.py scan --min-count 32 --out runs/v9/scan100.npz
python n10.py run --dict 32768 --table 16384 --full \
  --out runs/v9/enwik8_100m_full.json
```

Artifacts: `runs/v9/` (the tables of record in `LEVERS.md` and
`ENWIK8-100M.md`). The 10 MB candidate scan is cached at
`~/.cache/hncm/counts-<corpus>-<vocab>-<sha>.npz`, so a 9 min scan is seconds
on rerun; the 90 MB scan is `runs/v9/scan100.npz` (64 MB, outside git; the log
is committed).

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

## 7. Results: N8, N9, N10

Artifacts: `runs/v9/{union.json,coverage-union.json,codec-union.json,
codec-ncmrect.json,enwik8_100m.json,enwik8_100m_full.json,LEVERS.md,
ENWIK8-100M.md,scan100.log}`. The candidate scan is cached at
`runs/v9/scan100.npz` (64 MB, outside git; the log records it), and the
candidate counts for the 10 MB stream at `~/.cache/hncm/counts-*`, so a rerun is
seconds. Every `selftest` passes; the union map reuses the two cached EM
tilings, so N8 needs no new scan. The codec is the judge.

**N8, the union vocabulary.** At the 32k dict the unigram probe covers 0.815 of
the val (8k 0.699, 16k 0.766); the first-order DP rate is 2.607 / 2.499 / 2.425
at the 32k table, and the selection rule picks 16,384/16,384 (estimate 2.751).
The codec records:

| budget | arm | rate | model kB | shared | calls |
|---|---|---:|---:|---:|---:|
| union 16k/16k | concept | 2.739 | 309.6 | **2.987** | 247,688 |
| union 16k/16k | tail | 2.778 | 342.4 | 3.052 | 247,688 |
| substr 8k/16k (v8) | concept | 2.857 | 222.5 | 3.035 | 56,592 |

N8 misses its recorded gate (2.940) by 0.047, but it is a 0.048 win on the record
and 0.011 below zlib; PPMd is 2.160. The estimate fell 0.284 while the record
fell 0.048: the 0.186 G2r gap is the derived CDF's loss against the model it
packs. The coverage gate's three legs do not hold at one budget: 0.815 costs
347.7 kB (more than v8's 222.5) and its covered rate is 2.129 (> 2.0).

**N9, the order-2 tail on the DP path.** The pruned order-2 counts (real two
preceding bytes, order-1 backoff, top chosen on the held-out 10%, charged) beat
the concept row on the same tail bytes:

| rect | tail bytes | ord2 kB | ord2 bpb | row bpb | gate (4.0) |
|---|---:|---:|---:|---:|---|
| union 16k/16k | 247,687 | 80.5 | 4.329 | 4.738 | miss 0.329 |
| substr 8k/4k | 310,677 | 80.9 | 4.151 | 4.625 | miss 0.151 |

**N9b, the NCM against the counts on the same tail.** On the identical 310,677
bytes: the row 4.625, the order-2 counts 4.151 (80.9 kB), the NCM 4.307 (27.4
kB). The counts win the rate by 0.156, pay 0.043 bpb more at 10 MB, and net
+0.113; the count table is the tail of record, the NCM the smallest model.
The NCM's earlier 4.307 against the SVD's 4.459 is reproduced.

**The 10 MB record with that tail.** 2.987 - (4.738 - 4.329) * 247,687 / 1e6 +
8 * 80.5e3 / 10e6 = 2.950, still 0.010 above 2.94. It is a projection: the
codec does not implement the ord2 arm yet.

**N10, the verdict (measured).** The 90 MB train / 10 MB val split. The 10 MB
union vocabulary is the baseline; the full row is the proper 100 MB vocabulary
(the 64k seed plus the 90 MB heavy hitters, one EM unigram on the 81 MB fit).
The table is fitted on the first 90% of the train, chosen on the held-out 9 MB,
and charged over 100 MB:

| vocab | dict | table | coverage | covered bpb | tail bpb | rate | model kB | charge | charged | sel charged | ord2 projected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 MB seed | 32,768 | 16,384 | 0.825 | 1.961 | 4.831 | 2.464 | 498.7 | 0.040 | **2.503** | 2.522 | 2.446 |
| 90 MB full | 32,768 | 16,384 | **0.912** | 2.164 | 5.781 | 2.484 | 451.9 | 0.036 | **2.520** | 2.540 | 2.475 |

zlib 2.903, bz2 2.281, lzma 2.144, PPMd **1.834** on the same 10 MB val.
`covered bpb` is derived: `(rate - (1-cov) * tail) / cov`. The model is 0.669
(seed) / 0.686 (full) behind PPMd and misses the 1.90 gate by 0.603 / 0.619.
Coverage is not the binding lever: the full vocabulary adds
0.087 of coverage but the covered tokens cost 2.164 against the seed's 1.961,
and the tail (5.781) is the remaining 8.8%. The covered tokens are already
0.127 (seed) and 0.330 (full) above PPMd's 1.834, so the gap is the concept
*rate*, not the coverage. The order-2 tail (349 kB, 0.028 of charge) projects
2.446 / 2.475, still 0.61 behind. The stop rule: the text claim is a
**negative at 100 MB**; the structured win (synseg 1.389 against PPMd 1.500) is
the product. The codec record at 100 MB is not run; the 10 MB record is
0.19-0.24 above the rate estimate, so the record is >= 2.69.

The 90 MB scan found 2,702,205 substrings with count >= 32
(`runs/v9/scan100.npz`, 64 MB, not committed; the log is). The GPU unique
pass is the exact replacement for the CPU `counts_substr` that needs ~10x the
memory.

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
| N8 the union vocabulary | 30 min | measured (2.987; 2.94 missed by 0.047) |
| N9 the order-2 tail | 1 h | measured (4.329; 4.0 missed) |
| N9b the NCM vs the counts | 30 min | measured (counts net +0.113) |
| N10 the 100 MB verdict | 2-3 h | measured: 2.503 vs PPMd 1.834 (negative) |
| N11 the structured table | 1 h | |
| N12 the factorial | 1 h | |
| N13 delivery | 1 h | v9 artifacts written |

Minimal decisive run: **N8 + N9 + N10**. All three are measured; the text claim
is a negative at 100 MB, and the structured win is the product.
