# H-NCM v9: N8 (the union vocabulary), N9 (the order-2 tail), N9b

Corpus: `enwik8_10m`, 9 MB train / 1 MB val, SHA-256
`5985c81c39d927ae0e169625790ca4d9e7d1531270c8b09ad73176a375bb3d97`.
The dictionary and the concept table are fitted on the first 90% of the train; the
budget is chosen on the held-out last 10% of the train; the val only reports.
The union map is the merge of the two cached EM tilings (substr + word onsets),
so no new scan was needed for N8. The codec is `codec.py` (the recorded judge).

Provenance:

```bash
python concept_lm.py run --corpora enwik8_10m --vocab union \
  --dicts 8192,16384,32768 --prunes 8192,16384,32768 --out runs/v9/union.json
python probe.py run --corpus enwik8_10m --vocab union \
  --sizes 8192,16384,32768 --out runs/v9/coverage-union.json
python codec.py run --corpora enwik8_10m --rect runs/v9/union.json \
  --try 1 --out runs/v9/codec-union.json
python codec.py run --corpora enwik8_10m --rect /tmp/rect-ncm.json \
  --try 1 --out runs/v9/codec-ncmrect.json   # substr 8192/4096, the NCM's tail
python n10.py scan --min-count 32 --out runs/v9/scan100.npz
python n10.py run --dict 32768 --table 16384 --out runs/v9/enwik8_100m.json
python n10.py run --dict 32768 --table 16384 --full \
  --out runs/v9/enwik8_100m_full.json
```

Times: the concept grid ~9 min (the union map is cached), the probe ~3 min, each
codec fit ~4 min. Encode/decode are 0.19/0.21 MB/s in Python. The NCM is the
GPU arm (`ncm.py`, RTX 3080), 3 epochs on the 10 MB DP tail.

## N8: the union vocabulary

The probe's unigram cover (a bound on the first-order DP) and the concept grid's
first-order rate at the largest table (32,768 entries):

| dict | coverage | covered bpb | tail (SVD) | ord2 (unigram path) | probe charge | DP rate | DP charged | model kB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 0.699 | 2.162 | 5.602 | 5.236 | 0.100 | 2.607 | 2.838 | 288.3 |
| 16,384 | 0.766 | 2.146 | 5.758 | 5.464 | 0.179 | 2.499 | 2.809 | 388.0 |
| 32,768 | **0.815** | 2.129 | 5.922 | 5.693 | 0.347 | **2.425** | 2.902 | 597.0 |

The selection rule (charged on the held-out 10%) picks 16,384/16,384:
rate 2.519, charged 2.751, model 289.7 kB. The codec, the judge, records:

| budget | arm | rate | model kB | shared | calls | RT |
|---|---|---:|---:|---:|---:|---|
| union 16k/16k | concept | 2.739 | 309.6 | **2.987** | 247,688 | ok |
| union 16k/16k | tail | 2.778 | 342.4 | 3.052 | 247,688 | ok |
| substr 8k/16k (v8) | concept | 2.857 | 222.5 | 3.035 | 56,592 | ok |

zlib on the same val is 2.998; PPMd is 2.160. The union is a 0.048 win on the
record against v8 and 0.011 against zlib, but the 2.940 gate is missed by
0.047. The estimate moved 0.284 (3.035 -> 2.751); the record moves 0.048,
because the 0.186 rate gap G2r reports is the derived CDF's loss against the
model it packs (the codec's entropy 2.739 against the model's 2.553), and it
charges 309.6 kB against the estimate's 289.7.

The coverage gate: 0.815 at the 32k dict passes 0.80, but that budget costs
347.7 kB (more than v8's 222.5) and its covered rate is 2.129, above 2.0. The
gate's three legs do not hold at one budget.

## N9: the order-2 tail on the DP path

`ord2` is the pruned order-2 byte table (the real two preceding bytes) with the
order-1 backoff, fitted on the fit DP tail, top chosen on the held-out 10%,
charged at its packed size. `row` is the concept row's price for the same
single-byte tokens.

| rect | tail bytes | ord2 top | ord2 kB | ord2 bpb | row bpb | gain |
|---|---:|---:|---:|---:|---:|---:|
| union 16k/16k | 247,687 | 1,024 | 80.5 | **4.329** | 4.738 | 0.409 |
| substr 8k/4k | 310,677 | 1,024 | 80.9 | **4.151** | 4.625 | 0.474 |

The N9 gate (tail <= 4.0) is missed by 0.329 and 0.151. The ord2 table beats
the row at both budgets, but the 10 MB tail is harder than the 1 MB tail (3.83):
the val ceilings are 3.80/2.75/1.62, and the charged table reaches 4.15.

## N9b: the NCM against the counts on the same tail

Both codecs price the identical 310,677 bytes (substr 8k/4k DP tail):

| tail arm | bpb | model kB | charge at 10 MB | net |
|---|---:|---:|---:|---:|
| row (concept) | 4.625 | in the table | 0.000 | - |
| order-2 counts | **4.151** | 80.9 | 0.065 | **0.409** |
| NCM (3-byte + class + position) | 4.307 | 27.4 | 0.022 | 0.296 |

The count table wins the tail by 0.156 on the rate, pays 53.5 kB (0.043 bpb) more,
and nets +0.113 over the NCM. It is the tail of record on the net; the NCM is the
smallest model. The NCM's earlier 4.307 against the SVD's 4.459 is reproduced.

## The 10 MB record with the tail of record

The recorded concept arm is 2.987. Replacing the tail's row price with `ord2` on
the union path gains (4.738 - 4.329) * 247,687 / 8 = 12,673 B = 0.101 bpb and
pays 80.5 kB = 0.064 bpb: a projected **2.950**, still 0.010 above 2.94. This is
a projection; the codec does not yet implement the ord2 arm, so it is not the
record.

## N10: the 100 MB verdict

`n10.py` reads the 100 MB dump as bytes (not 800 MB of torch tensors), refits
the 32k vocabulary's counts on the 90 MB train with an exact GPU scan, fits the
table on the first 90% (81 MB), chooses on the held-out 9 MB, and charges over
100 MB. The val is 10 MB.

| vocab | dict | table | coverage | covered bpb | tail bpb | rate | model kB | charge | charged | sel charged | ord2 projected | PPMd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 MB seed | 32,768 | 16,384 | 0.825 | 1.961 | 4.831 | 2.464 | 498.7 | 0.040 | **2.503** | 2.522 | 2.446 | **1.834** |
| 90 MB full | 32,768 | 16,384 | **0.912** | 2.164 | 5.781 | 2.484 | 451.9 | 0.036 | **2.520** | 2.540 | 2.475 | **1.834** |

zlib 2.903, bz2 2.281, lzma 2.144, PPMd 1.834 on the same 10 MB val.
`covered bpb` is derived: `(rate - (1-cov) * tail) / cov`. The model is
**0.669 behind PPMd** (seed) / 0.686 (full) and misses the 1.90 gate by
0.603 / 0.619. Coverage is not the binding lever: the full vocabulary adds
0.087 of coverage but the covered tokens cost 2.164 against the seed's 1.961;
the covered tokens themselves are 0.127 / 0.330 above PPMd. The gap is the
concept **rate**, not coverage. The order-2 tail (349 kB, 0.028 of charge)
projects 2.446 / 2.475, still 0.61 behind. The text claim is a **negative at
100 MB**; the structured win stands.

The scan found 2,702,205 substrings with count >= 32 on the 90 MB train
(`runs/v9/scan100.npz`, 64 MB, not committed; the log is). The 10 MB-selected
vocabulary is the baseline; the full row is the proper 100 MB vocabulary (the
64k seed plus the 90 MB heavy hitters, one EM unigram on the 81 MB fit). The
codec record at 100 MB is not run: the 10 MB val encode is out of Python's
throughput, and the 10 MB record is 0.19-0.24 above the rate estimate, so the
record is >= 2.69.
