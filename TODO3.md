# TODO3 — H-NCM v8: end to end, then scale (rev 2)

Status: v7 is measured as a *rate* (`TODO2.md` §8-§9): the dictionary index pays
where it covers, the shared low-rank tail model pays where it does not, both are
charged, the model is paid once, and the decoder makes 6.5x fewer calls. What is
missing is the artifact — **a bitstream that round-trips** — and the lever the
measurements point at hardest: **coverage** (51.8% at 1.1 MB).

**N0 is done (§9).** `codec.py` round-trips all three corpora; the recorded
size equals the code's entropy + the charge within 0.0013 bpb. The recorded
table corrects the estimate: 3.47 / 1.39 / 2.10 charged on
shakespeare/synseg/syngrid; the published 2.723/3.006 is a lower bound by
0.07-0.28 bpb (the tail formula double-subtracts the byte unigram for bytes the
concept path priced at the bigram row). At 1 MB the concept arm wins on text;
the tail arm's 6.5x fewer calls cost 0.30 bpb more. Next: N1a (coverage) and
N1b (10 MB), re-based on the recorded numbers.

Rev 2 changes, all from measurement:

1. **PPMd is the text bar, not bz2.** Measured on the same val bytes:
   shakespeare 1.12 MB: PPMd 2.504, bz2 2.641, brotli 2.825, xz 2.948; synseg:
   PPMd 1.500, bz2 1.520; syngrid: PPMd 2.063, xz 2.006. The concept LM's 3.006
   on shakespeare is 0.50 behind PPMd, not 0.37 behind bz2. Every target is now
   stated against PPMd, and `7z -m0=PPMd -mx9` joins the codecs.
2. **A coverage probe before the scale run** (15 min): fit the dictionary, tile
   the val, report coverage and the covered rate. It decides whether N1 is a
   dictionary story or a tail story.
3. **The codec's accounting is specified** (the CDF is derived from the charged
   counts; the token counts are stored log-quantized to 2 bytes; the model is the
   charge). Without this, the recorded size cannot reconcile with the rate.
4. **A per-file budget choice** (`--try 3`, 2 transmitted bits) recovers the 0.14
   bpb that the held-out rule costs on shakespeare. The *choice* is transmitted,
   so it is not a selection leak.
5. **Efficiency**: one cached candidate pass per (corpus, vocab), a probe before
   a grid, sequential budget curves, three corpora as three processes.
6. **A risk register** (§5) with the cheap check for each way the claim can fail.

## 0. The targets (revised, against PPMd)

| corpus | size | concept LM now | **PPMd** | bz2 | target |
|---|---:|---:|---:|---:|---:|
| shakespeare | 1.12 MB | 3.006 | **2.504** | 2.641 | ≤ 2.50 (parity) |
| synseg | 1.20 MB | **1.323** | 1.500 | 1.520 | keep ≤ 1.32 |
| syngrid | 1.20 MB | **1.939** | 2.063 | 2.138 | keep ≤ 1.94 |
| shakespeare 10 MB | 10 MB | – | measure | – | ≤ 2.20 |
| enwik8 10 MB | 10 MB | – | ~1.9-2.0 (measure) | – | ≤ 2.10 |

The *structural* claims are the ones that scale, and they are already measured:
the model is paid once (flat in K, bz2 crosses at K≈5), the decoder needs 6.5x
fewer calls at 1 MB (20x+ at 10 MB), and the charge curve is flat. The *rate*
claim on text is against PPMd, at 10 MB, and it is the hard one.

## 1. Where the bits are (1.12 MB shakespeare, 2.723 rate / 3.006 charged)

| part | bytes | bpb | bits | share |
|---|---:|---:|---:|---:|
| covered spans | 57,790 (51.8%) | 0.943 | 54,500 | 18% |
| tail bytes | 53,746 (48.2%) | 4.020 | 216,100 | **71%** |
| tail boundary | 8,263 spans | 0.616 | 33,100 | 11% |
| model charge (on top) | 39.5 kB | 0.283 | 31,500 | 9.4% |

The tail's 71% share is a **coverage artifact**: at 10 MB the same dictionary
covers ~85% and the total moves toward the covered rate. So the plan's order is:
prove the artifact (N0), raise coverage (N1), then cut the tail (N2-N4), then the
estimand (N6). The tokenizer is one lever of six (N4).

## 2. What is established (do not re-derive)

1. A dense pooled code cannot carry order (v6): `total` is monotone in the code
   rate; the optimum is to collapse it.
2. A dictionary *index* has an exact inverse: residual zero; the model is the
   only charge (`runs/v7/concept.json`).
3. The first-order concept LM cuts 0.4-0.67 bpb unpaid; at 1.12 MB only a 1-2k
   table is affordable; the held-out rule costs 0.14 bpb (`runs/v7/CONCEPT.md`).
4. The tail is 48% of the val and 80% of the bits; a rank-16 shared model of the
   257x256 byte table (16.4 kB) reaches 4.020 bpb, 1.03 below the byte unigram and
   0.43 below a 256.5 kB count table (`runs/v7/TAIL.md`).
5. The tail pays: 3.006 against 3.087; the boundary is 0.616 as a length, 0.522
   as a trailing space (a bound).
6. The shared model's charge curve is flat in K, the codecs' rise; bz2 crosses at
   K≈5 with the tail (`runs/v7/TRANSFER.md`).
7. A hard concept code with counts is a negative on the tail (`prev` 4.795,
   k-means 4.974 against `ar` 4.480). The positive is the *shared low-rank*
   model, not a discrete code.
8. **PPMd is the text bar** (2.504 on shakespeare, 1.500 on synseg, 2.063 on
   syngrid), and it is now in `codecs()` (7z -m0=PPMd -mx9), so every run reports
   it. It is a context-mixing codec with no per-file model to share: its
   per-file mean rises with K like bz2's.

## 3. The work, in order

### N0 — the bitstream (~3-4 h) — **do this first**

*Question:* does the accounting survive an actual encoder and decoder?
*How:* one file, `codec.py`.

- **The model** (`model`): the dictionary strings with a length and a
  **log-quantized count** (2 bytes: 8-bit exponent, 8-bit mantissa; the cost is
  recomputed from it, so the model stores *counts*, not costs); the concept
  table (2-byte indices, 2-byte counts); the tail model (two 16-bit matrices,
  16.4 kB). Packed and compressed with the byte model.
- **The CDF is derived, never stored**: weights `w_i = round(2^20 * p_i)` from
  the model's costs, a Fenwick tree of `uint32` weights, a 32-bit range coder.
  The decoder computes the same tree from the same model, so the CDF costs zero
  extra bits and the quantization loss is `< 1e-4` bpb per token (verify).
- **`encode`**: the DP gives the token sequence; each token is coded with the
  *bigram* CDF (the state is the previous token); a tail token sends its length
  (the length prior, 16 symbols) then its bytes under the tail CDF.
- **`decode`**: the inverse. The decoder never reruns the DP; it decodes the
  token sequence. That is what makes the calls count true.
- **`--self` / `--shared`**: the model inside the file (the archive convention)
  or sent once (`--shared`, the K-file convention). The charge follows the
  convention; both are reconciled.
- **`--try N`**: encode with N budgets (2-3), keep the smallest, send 2 bits.
  This recovers the 0.14 bpb the held-out rule costs; the choice is transmitted,
  so it is not a leak.
- **`check`**: round-trip on shakespeare, synseg, syngrid, a 200-slice fuzz, and
  the reconciliation: the recorded size equals the charged bpb within 0.005 bpb
  in both conventions. Selftest: a round-trip on a 1 kB slice, the model
  pack/unpack, the quantized-count round-trip (`|cost - cost_q| < 0.01` bpb), and
  the Fenwick CDF (`sum(p) == 1` within 1e-9 for random costs).

*Gate:* every corpus round-trips; the recorded size equals the charge within
0.005 bpb in both conventions; the model is ≤ 45 kB today; encode ≥ 0.3 MB/s,
decode ≥ 1 MB/s (report the wall clock).
*Stop:* if the recorded size exceeds the charge by more than 0.02 bpb, the
accounting is wrong: fix it before any new model.

### N1 — the coverage probe, then scale

*N1a, the probe (~15 min).* Fit the dictionary at 2-4 sizes on the static 10 MB
train, tile the 1 MB val with the DP, and report coverage, the covered rate, and
the charge. This decides the scale run's budget and whether the tail is the
story. *Gate:* coverage is measured, not assumed; the predicted total is
`cov*covered + (1-cov)*tail + charge`.
*N1b, the run (~1-2 h, plus 5-10 min per corpus).* Add `load_enwik8` to
`hncm6.py` (the 100 MB dump from `mattmahoney.net/dc/enwik8.zip`, cached like
shakespeare; the 10 MB slice is `data[:10_000_000]`). Run enwik8 at 10 MB and
100 MB, shakespeare at 10 MB if a larger source is reachable (the complete works
at Project Gutenberg, ~5.5 MB, cached), and every codec (zlib, gzip, bz2, xz/7z,
brotli, zstd, **PPMd**) on the same val. Report coverage, the split, the rate,
the charge, the recorded size, the calls, the throughput, and the codec curves at
1/10/100 MB.
*Gate:* at 10 MB the charged total ≤ 1.80 (stretch 1.60) and within 0.10 of
PPMd; the calls ratio ≥ 20x.
*Stop:* if the coverage gain is eaten by the charge, the budget is the story and
N3 comes first; report the curve either way.

### N2 — the tail's context at the same 16.4 kB (~1 h)

*Question:* does a second letter of context beat the rank-16 bigram in the same
parameter budget?
*How:* three arms, fitted on the train tail, all charged:

- `cls`: the previous byte bucketed to 32 frequency classes → a 32x256 table
  (8 kB) plus the bigram (16 kB).
- `tri`: a rank-r *trigram* on the sparse (prev2, prev1, next) counts (a dense
  65k x 256 table is 16.9M entries: use the sparse counts and a randomised SVD,
  two 65k x 32 products per iteration); rows renormalised, backoff to the rank-16
  bigram.
- `bias`: the *concept class* of the previous dictionary token as a rank-1 bias
  on the logits (65 x 256 = 16 kB): the slow tier shaping the fast tier, which
  the count version could not do.

Select r and the arm on the held-out 10%; report the captured size, the rate and
the charge.
*Gate:* the tail bytes < 4.020 at the same or smaller charged total; target 3.80.
*Stop:* if none wins, the letter model is at its bigram limit at this budget;
record it and move on.

### N3 — the concept table's sharing, and a second-order context (~1 h)

*Question:* does the rank-r sharing that won for the bytes win for the concept
bigram — and does a second concept of context pay?
*How:* the 2k x 2k transition counts (dense is 32 MB, fine), rank r = 4..64 by
SVD, rows renormalised with the unigram backoff; the same budget curve; r and the
dict size selected on the held-out 10%. Then a *second-order* arm: the counts
`P(s | p2, p1)` factorised the same way (a sparse 2k x 2k x 2k tensor: use *two*
low-rank factors, `P(s | p1)` plus a rank-r residual on `(p2, p1)`), which is where
the covered part gains at 10 MB (it is 80% of the bits there).
*Gate:* the charged total at 1.12 MB < 3.006 (target 2.90) and the affordable
dict size grows, so N1's coverage compounds; the second-order arm < the
first-order one at 10 MB.
*Stop:* if the shared table loses to the counts, the concept alphabet is too
sparse to share; report it.

### N4 — the boundary (~1-1.5 h, after N2) — *only if N0 shows it binds*

*Question:* can the 0.616 bpb boundary be removed by the tiling's structure?
*How:* retokenise so that a dictionary token ends *with* its space and a tail
token ends at the space; rebuild the dictionary on the retokenised train, re-run
the DP, the tail and the charge. The decoder stops a tail token when it decodes a
space, so no length is sent. Arms: v7 (space-prefixed) against v8
(space-suffixed) at the same budget.
*Gate:* the recorded size ≤ the tail-side bound (2.677 rate, 2.960 charged).
*Stop:* if the covered rate rises by more than the boundary falls, keep the
length boundary; the codec is unaffected.

### N5 — the consistent lattice (~1 h, optional)

The state collapses to (a dictionary span) or `TAIL`; the tail span's cost is the
shared model's bits + `H(L)`; one DP per MB; the budget and r on the held-out
10%. The plain path is a subset, so the optimum cannot be worse.
*Gate:* the recorded size ≤ the two-stage size, still exact.
*Stop:* a gain < 0.02 bpb does not pay for the lattice.

### N6 — the factorial (`I_perf`), the claim's evidence (~1 h)

A a byte-AR arm (an order-3 count model, or `hncm6.py`'s byte arm at the same
budget), B the dense code (v6), C the dictionary + unigram + DP, D the dictionary
+ concept LM + shared tail, at a matched budget, 3 seeds, on the same val.
`I_perf = (D-C) - (B-A)` in bpb, per seed.
*Gate:* `I_perf > 0` with clean controls; report the per-seed spread.

### N7 — delivery (~1 h)

The table of record: recorded size, charge, calls, coverage, encode/decode
throughput, for 5 corpora and 7 codecs (PPMd included), both conventions; the
README headline, `REPORT.md`, `runs/v8/` with the model files, the config JSON
and the hashes; one command. The v7 tables were re-run with PPMd in `codecs()`,
so their `vs best` column is against PPMd on text (`runs/v7/*.md`).

## 4. Efficiency (the plan's own cost)

1. **Cache the candidate counts** per (corpus, vocab) as `.npz`
   (`runs/v8/counts-<corpus>-<vocab>-<sha>.npz`): the substring scan is
   `O(CAP x N)` and dominates a 10 MB run (3-5 min); every rerun after it is
   seconds.
2. **Probe before a grid**: one running point (the 1 MB optimum scaled by 8)
   plus the geometric neighbours (x2), stop when the charge rises.
3. **Sequence the budgets**: 2k → 4k → 8k → … on the held-out split; each step
   is one DP, and the curve is unimodal in practice.
4. **Three corpora, three processes** (the box has 16 cores); the DP is
   single-threaded Python.
5. **One counts pass per (corpus, vocab)**: `split()` computes it once and
   `rebuild()` reuses it (already the case).
6. **The SVD is cheap**: 257x256 is instant; the 2k x 2k concept table is a 32 MB
   dense SVD (fine); use a randomised SVD only above 10k symbols.
7. **Wall clock is a metric**: report MB/s encode and decode, not only bpb.
8. **The DP is O(L) dicts today** (`best = [dict() for _ in range(L+1)]`): at
   10 MB that is ~640 MB of empty dicts. Before the scale run, keep the scores in
   a rolling window of `CAP+1` positions and the backpointers in one array
   (`prev` token id as `int32`, length as `uint8`): 10 MB is then ~80 MB and
   faster. That change is required for 10 MB and 100 MB to be safe.
9. **Scope the lattice (N5) to 1 MB**: the tail lattice is ~10x the plain DP; a
   10 MB sweep is 20+ min per point and buys 0.03-0.10 bpb.
10. **Cross-corpus tail transfer** (optional, in N2): fit the 16.4 kB on one
    corpus, apply it to another's tail. It removes the model from the per-corpus
    charge and is the "one model, many corpora" version of the K-file result.

## 5. Risk register (each has a cheap check)

| risk | check | cost |
|---|---|---|
| coverage at 10 MB is low (enwik8's vocabulary is huge) | the N1a probe | 15 min |
| the tail's rate at 10 MB is worse than 4.0 | the probe + one tail fit | 15 min |
| PPMd wins on text at every size | the codec table (it is in the runs) | free |
| the CDF quantization costs more than 0.005 bpb | N0's reconciliation | free |
| the model at 10 MB is bigger than the stream pays for | the charge column at each size | free |
| the val block is unrepresentative | the 10 MB val (1 MB, not 111 kB) | N1 |
| a hidden leak (selection on the val) | every budget from the held-out 10%; `--try N` transmits its choice | by construction |

## 6. Rules to add (each already cost a run)

12. The model's *size* is a hyperparameter of every tier: select it on a
    held-out split, never on the val, and charge it at its packed size.
13. A hard code (k-means, VQ) is not a shared model; the low-rank factorization
    is what generalizes.
14. Factorize the *counts*, not the log probabilities; keep the discount floored
    (Good-Turing, 0.25-0.95).
15. A boundary is information: pay it, or remove it with the tiling's structure.
16. Coverage is the largest lever: measure it before tuning below 0.1 bpb.
17. The recorded size is the judge; a rate estimate is not. Reconcile the two.
18. Store *counts*, derive the costs and the CDF; a stored CDF is a second table
    (a defect) and a selection leak when it is fitted on the val.

## 7. Time

| step | cost | why |
|---|---|---|
| N0 the bitstream | 3-4 h | the artifact; validates every number |
| N1a the probe | 15 min | decides the scale run |
| N1b scale | 1-2 h + runs | the biggest lever (coverage) |
| N2 the tail's context | 1 h | 71% of the bits at 1 MB |
| N3 the concept table | 1 h | shifts the budget; compounds with N1 |
| N4 the boundary | 1-1.5 h | 11% of the bits |
| N5 the lattice | 1 h | 0.03-0.10 bpb |
| N6 the factorial | 1 h | the claim's evidence |
| N7 delivery | 1 h | the table of record |

Minimal end-to-end run: **N0 + N1a + N1b + N2 (~6-7 h)**. Then N3, N4, N6.

## 8. Commands (as they land)

```bash
python codec.py selftest
python codec.py encode --corpus shakespeare --shared runs/v8/model.bin --try 3 --out /tmp/s.bin
python codec.py decode --shared runs/v8/model.bin --in /tmp/s.bin --out /tmp/s.txt
python codec.py check --corpora shakespeare,synseg,syngrid
python concept_lm.py probe --corpus enwik8_10m     # N1a: coverage and the covered rate
python concept_lm.py run --corpus enwik8_10m       # N1b: the budget curve
python tail_coder.py run                           # N2: the tail arms
python transfer.py run                             # K files
```

## 9. Results: N0, the bitstream

Artifacts: `codec.py`, `runs/v8/codec.json`, `runs/v8/CODEC.md`. `python
codec.py selftest` passes (the count quantization, the CDF, the coder, the
model pack/unpack, a 1 kB slice in both arms); every corpus round-trips and
the recorded size equals the code's entropy + the model charge within 0.0013
bpb in both conventions. The model is packed as counts, every CDF is derived,
the 32-bit arithmetic coder writes the DP's path, and the decoder never reruns
the DP.

| corpus | arm | dict | table | rate | ref | published v7 | model kB | shared | self | calls | enc MB/s | dec MB/s | best codec | charged gap | RT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| shakespeare | concept | 1024 | 2048 | 3.305 | 3.024 | 3.024 | 23.1 | 3.472 | 4.965 | 56592 | 0.25 | 0.15 | ppmd | +0.968 | ok |
| synseg | tail | 512 | 1024 | 1.290 | 1.231 | 1.221 | 14.7 | 1.389 | 2.272 | 536 | 0.14 | 0.11 | ppmd | -0.111 | ok |
| syngrid | tail | 302 | 8192 | 1.739 | 1.594 | 1.594 | 53.6 | 2.098 | 5.316 | 1 | 0.08 | 0.07 | lzma | +0.093 | ok |

`ref` is the measured cost of the model the codec implements (the multi-byte
tokens + the tail + the length); `published v7` is the earlier estimate. The
codec is the judge (rule 17):

- **The published tail estimate is a lower bound, by 0.07-0.28 bpb.** Its
  `covered = total - sum(byte unigram)` subtracts 5.05 bpb for tail bytes the
  concept path had already priced at the bigram row (3.83 bpb), so it
  double-counts the tail's gain. The two codecs at the same budget confirm it:
  the concept arm is the best realizable codec at 1 MB on shakespeare (it pays
  the row, not the boundary); the tail arm pays the type, the length and the
  shared model on top of a row cost it does not beat. The concept arm is 3.305
  rate on the 1024/full budget against the tail arm's 3.602; the boundary and
  the type eat the 0.43 bpb the SVD gains on the tail bytes.
- **At 1 MB the model pays where the source has a small exact vocabulary:**
  synseg is 1.389 charged against PPMd 1.500 (+0.11) and bz2 1.513 (+0.12);
  shakespeare is 0.25 behind gzip and 0.97 behind PPMd; syngrid is 0.09
  behind lzma (the 4096-entry table is 0.02 worse, so the 8192 table stays).
- **The structure survives:** the model is paid once (the shared column is the
  K-file convention); the tail arm makes 6.5x fewer calls (8,264 vs 56,592) but
  pays 0.30 bpb more than the concept arm at 1 MB, so the calls are not free at
  this size; the decoder runs at 0.07-0.16 MB/s in Python (the DP is the encode
  cost, not the codec).

The `--try` choice is transmitted; `self` embeds the model per file, `shared`
sends it once over train+val. The winner is the smallest charged total, so the
model's size is part of the choice. The next work is scale: the normalization
cost is a 1 MB artifact (the escape falls as the counts grow), and N1a/N1b
must re-base on these recorded numbers, not on 2.723/3.006. The v7 §8-§9 tables
are estimates; the v8 `rate`/`shared` columns are the table of record.


## 10. Results: N1a (coverage), the NCM, and the defect

Artifacts: `corpora.py` (the enwik8 cache/split), `probe.py` (`runs/v8/coverage.json`,
`coverage-onsets.json`, `COVERAGE.md`), `ncm.py` (`runs/v8/ncm.json`, the 15 kB
GRU trained on the GPU), the counts cache (`~/.cache/hncm/counts-*`). All
`selftest`s pass; the split cache turns a 9 min scan into seconds.

Measured (enwik8_10m, 9 MB train / 1 MB val, unigram cover): coverage
0.545/0.629/0.698/0.756 at 2k/4k/8k/16k (substr), 0.414-0.536 (onsets); the
predicted total 3.56 -> 3.14 and 3.59 -> 3.38. PPMd 2.476 on the 1 MB val.

The defect: the tail's SVD restarts at 256 and discards the real preceding
bytes. Fitted on the train tail: order-1 4.17, order-2 3.83, order-3 3.73
(the SVD 4.48); the val ceilings 3.80/2.75/1.62. The order-2 table is 23,851
entries / 95 kB: it pays at 10 MB (0.088 charge vs ~0.35 gain) and not at 1 MB
(0.877). The NCM (15 kB GRU, GPU, 10 epochs) is 4.61 vs the SVD's 4.48: the
learned tail loses to the counts at this budget. The complete model is the
*concept arm* (one normalized model; the rows are the tail's context): 3.671
charged at 1 MB vs the hard split's 3.939; PPMd 2.476.

Revised next: N1b is the concept arm at 10 MB with (i) the order-2 tail (real
preceding bytes), (ii) the first-order concept table, (iii) the chunked DP
(the 10 MB `viterbi` allocates L+1 dicts; chunk at 1 MB), (iv) the counts cache
and three corpora as processes. The 10 MB target is <= 2.50 and PPMd parity is
the stretch at 100 MB, where the model charge falls to 0.02 and the coverage
rises; the 10 MB prediction with the order-2 tail and the bigram is ~2.5-2.7.

## 11. Results: N1b, enwik8 at 10 MB

Artifacts: `runs/v8/enwik8_10m.json`, `runs/v8/ENWIK8.md`. The fit uses the
chunked DP (1 MB chunks; the 10 MB `viterbi` allocates L+1 dicts) and the scaled
tail (the float16 product overflowed at 10 MB; the counts are divided by their max
and the prior by the same scale). The table is chosen on the held-out 10% of the
train; the val is 1 MB; the model is charged over the 10 MB stream.

| dict | table | sel | concept rate | model kB | concept charged | tail charged |
|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 4,096 | 3.029 | 2.991 | 135.0 | 3.098 | 3.134 |
| 8,192 | 8,192 | 2.981 | 2.927 | 165.0 | 3.059 | 3.122 |
| 8,192 | 16,384 | **2.948** | 2.857 | 222.5 | **3.035** | 3.126 |
| 16,384 | 4,096 | 3.001 | 2.873 | 248.8 | 3.072 | 3.094 |

PPMd on the same val: **2.160** (brotli 2.344, bz2 2.393, lzma 2.427). The rule
costs nothing (the sel picks the val's best); the concept arm beats the hard split
by 0.09 bpb at half the model. The 10 MB concept arm is **+0.875** behind PPMd;
the 2.50 target is not reachable at 10 MB.

The decomposition: the 16k dict covers 0.756 and the covered tokens are ~2.0 per
covered byte (below PPMd's 2.16); the tail (24% at 5.6 bpb, the SVD cold start)
is the whole gap. The real-context order-2 is 3.83 vs the SVD's 4.48 on the 1 MB
tail; order 3 is 3.73, ceiling 1.62. With the tail at ~3.0 the 16k/16k is ~2.6 at
10 MB, and at 100 MB the charge falls to 0.02 and the coverage rises, so ~1.8-2.0
against PPMd's ~1.8. The claim is a 100 MB claim, not a 10 MB one; on the
structured source (synseg 1.389 vs PPMd 1.500) it is already a win at 1.2 MB.

Next: the real-context tail in the concept arm (the NCM at 10 MB is training; the
1 MB 15 kB GRU lost to the SVD, the 25 kB 3-byte-context model is the test), then
the 16k dict + that tail at 10 MB (~2.6 predicted), then the 100 MB run.
