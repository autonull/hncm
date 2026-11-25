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
python n10.py proof --val 1000000 --dict 16384 --table 16384 --seed \
  --out runs/v9/proof-100m-seed.json
python n10.py proof --val 10000000 --dict 16384 --table 16384 --seed \
  --out runs/v9/proof-100m-seed-10m.json
python n10.py proof --val 1000000 --dict 16384 --table 16384 \
  --out runs/v9/proof-100m-16k.json
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
| union 16k/16k | concept | 2.729 | 309.6 | **2.977** | 253,853 | ok |
| union 16k/16k | tail | 2.772 | 342.4 | 3.046 | 253,853 | ok |
| substr 8k/16k (v8) | concept | 2.857 | 222.5 | 3.035 | 56,592 | ok |

zlib on the same val is 2.998; PPMd is 2.160. The union is a **0.058 win on the
record** against v8 and 0.021 below zlib; the 2.940 gate misses by 0.037. The
estimate moved 0.284 (3.035 -> 2.751), the record 0.058. The remaining G2r gap
is 0.044: the packed counts are log-quantized to 2 bytes, so the CDF costs 0.044
over the unquantized model. The coverage gate's three legs do not hold at one
budget: 0.815 costs 347.7 kB and its covered rate is 2.129 (> 2.0).

## N9: the order-2 tail on the DP path

`ord2` is the pruned order-2 byte table (the real two preceding bytes) with the
order-1 backoff, fitted on the fit DP tail, top chosen on the held-out 10%,
charged at its packed size. `row` is the concept row's price for the same
single-byte tokens.

| rect | tail bytes | ord2 top | ord2 kB | ord2 bpb | row bpb | gain |
|---|---:|---:|---:|---:|---:|---:|
| union 16k/16k | 253,852 | 1,024 | 81.0 | **4.303** | 4.854 | 0.551 |
| substr 8k/4k | 317,741 | 1,024 | 81.2 | **4.140** | 4.721 | 0.581 |

The N9 gate (tail <= 4.0) misses by 0.303 and 0.140. The ord2 table beats the
row at both budgets, but the 10 MB tail is harder than the 1 MB tail (3.83).

## N9b: the NCM against the counts

The NCM (3-byte context, H-Net class, position; 27.4 kB, GPU) is 4.307 on the
old substr DP tail (310,677 B). The corrected substr tail is 317,741 B: the row
4.721, the order-2 counts **4.140** (81.2 kB). On the old identical bytes the
counts were 4.151 against the NCM's 4.307; the corrected model keeps the
counts ahead by ~0.17 on the rate. The NCM is the smallest model; the counts
pay 53 kB more and win the net at 10 MB.

## N10: the end-to-end proof

`n10.py` reads the 100 MB dump as bytes, refits the 10 MB union vocabulary's
counts on the 81 MB fit (one EM round), fits the table on 81 MB, chooses on the
held-out 9 MB, and charges over 100 MB. The full model adds the 90 MB heavy
hitters. The bitstream round-trips:

| vocab | model kB | val | payload B | rate | shared | calls | RT |
|---|---:|---:|---:|---:|---:|---:|---|
| seed | 300.9 | 1 MB | 327,389 | 2.619 | **2.643** | 233,504 | ok |
| seed | 300.9 | 10 MB | 3,300,098 | 2.640 | **2.664** | 2,393,110 | ok |
| full | 278.7 | 1 MB | 532,626 | 4.261 | 4.283 | 131,766 | ok |

PPMd on the same 10 MB val is **1.834** (zlib 2.903, bz2 2.281, lzma 2.144).
The seed is 0.830 behind PPMd on the 10 MB held-out; the 1.90 gate misses. The
full vocabulary covers 0.912 but its table's unseen transitions cost 4.261:
coverage is not the lever. The text claim is a **negative at 100 MB**; the
structured win (synseg 1.389 against PPMd 1.500) is the product. The earlier
unnormalized estimate (2.503 / 2.520) was 0.14 / 1.76 optimistic; the encoded,
round-tripping records are 2.643 / 4.283.

The scan found 2,702,205 substrings with count >= 32 on the 90 MB train
(`runs/v9/scan100.npz`, 64 MB, not committed; the log is). The GPU unique
pass is the exact replacement for the CPU `counts_substr` that needs ~10x the
memory.
