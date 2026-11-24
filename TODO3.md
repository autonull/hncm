# TODO3 — H-NCM v8: end to end, then scale

Status: v7 is measured end to end as a *rate* (`TODO2.md` §8-§9): the dictionary
index pays where it covers, the shared low-rank tail model pays where it does
not, both are charged, the model is paid once, and the decoder makes 6.5x fewer
calls. What is missing is the thing itself: **a bitstream that round-trips**,
and the one lever that the measurements point at hardest: **scale** (coverage is
51.8% at 1.1 MB, and the tail's 71% share of the bits is a coverage artifact).

The target is not a better tokenizer. It is a working hierarchical codec whose
recorded output size beats bz2/xz on real text, with every model charged and
selected on held-out splits, and whose decoder needs 6-20x fewer calls.

## 0. One paragraph

**Best possible end-to-end result.** A single command compresses a file to a
bitstream and decodes it back byte-exactly, with a model artifact (dictionary +
concept table + shared tail model) that is charged at its packed size. At 1.12 MB
of shakespeare it charges ≤ 2.50 bpb (stretch 2.30, bz2 2.635); at 10 MB it
charges ≤ 1.80 bpb (stretch 1.60) against bz2/xz measured on the same bytes (the
codec numbers at 10 MB are unknown until N1 runs; the targets are the plan, not
measurements); on synseg/syngrid it stays 0.06-0.19 bpb below the best codec; and
above K ≈ 5 files the shared model beats bz2 per file by construction. Every
number is the *file size on disk*, not a rate estimate.

## 1. Where the bits are, and what each lever can buy

shakespeare, 1.12 MB, current: **rate 2.723, charged 3.006**.

| part | bytes | bpb | bits | share |
|---|---:|---:|---:|---:|
| covered spans | 57,790 (51.8%) | 0.943 | 54,500 | 18% |
| tail bytes | 53,746 (48.2%) | 4.020 | 216,100 | **71%** |
| tail boundary | 8,263 spans | 0.616 | 33,100 | 11% |
| model charge (on top) | 39.5 kB | 0.283 | 31,500 | 9.4% |

So 1 bpb on the tail bytes is 0.48 bpb on the total; the boundary is 0.30; 1 bpb
on the covered part is 0.09. But the largest end-to-end lever is not in this
table at all: **coverage**. At 1.1 MB the dictionary covers half the bytes
because the val's rare words are not in a 1 MB train. At 10 MB the same dictionary
covers ~85%, and the total moves toward the covered rate (~0.7-1.0 bpb).

| lever | mechanism | expected at 1.12 MB | expected at 10 MB |
|---|---|---:|---:|
| **N0 the bitstream** | the artifact; <= 4 bits of coder flush | +0.0003 | +0.0003 |
| **N1 scale** | coverage 0.52 -> ~0.85 | n/a | **-1.0 to -1.4** |
| **N2 the tail's context** | two letters at the same 16 kB | -0.10 to -0.20 | -0.10 |
| **N3 the concept table** | the same low-rank sharing; the dict budget shifts | -0.05 to -0.10 | -0.10 |
| **N4 the boundary** | the space, or a space-suffixed tiling | -0.05 to -0.30 | -0.10 |
| **N5 the lattice** | the tail span as a first-class DP option | -0.03 to -0.10 | -0.03 |
| **N6 the factorial** | the estimand `I_perf`, the claim's evidence | 0 | 0 |

## 2. What is established (do not re-derive)

1. A dense pooled code cannot carry order (v6): `total` is monotone in the code
   rate, and the optimum is to collapse it.
2. A dictionary *index* has an exact inverse: residual zero, and the model is the
   only charge (`runs/v7/concept.json`).
3. The first-order concept LM over the indices cuts 0.4-0.67 bpb unpaid; at
   1.12 MB only a 1-2k-entry table is affordable, and the held-out rule costs
   0.14 bpb on shakespeare (`runs/v7/CONCEPT.md`).
4. The tail is 48% of the val and 80% of the bits; a **rank-16 shared model of
   the 257x256 byte table (16.4 kB)** reaches 4.020 bpb there, 1.03 below the
   byte unigram and 0.43 below a 256.5 kB count table (`runs/v7/TAIL.md`).
5. The tail now pays: 3.006 charged against 3.087; the boundary is 0.616 bpb as
   a length and 0.522 as a trailing space (a bound).
6. The shared model's charge curve is flat in K and every codec's rises; bz2
   crosses at K ~ 5 with the tail, K = 8 without (`runs/v7/TRANSFER.md`).
7. A *hard* concept code with counts is a negative on the tail (`prev` 4.795,
   k-means 4.974 against `ar` 4.480). The positive is the *shared* low-rank
   model, not a discrete code.

## 3. The work, in order

### N0 — the bitstream, and the round-trip (~2-3 h) — **do this first**

*Question:* does the accounting survive an actual encoder and decoder?
*How:* one file, `codec.py`:

- `model`: the dictionary (lengths, quantised costs), the concept table (2-byte
  indices and counts) and the tail model (two 16-bit float matrices), packed and
  compressed with the byte model. It is 39.5 kB today, and it is the charge.
- `encode --shared MODEL`: the stream only (the model is sent once for K files);
  `encode --self`: the model inside the file (the archive convention). Both are
  reported; the charge follows the convention.
- `encode`: the DP over the shard with the concept costs; each token is coded as
  its index with a range coder on the *bigram* CDF; a tail token is coded as its
  length (the length prior) and then its bytes under the tail model's CDF.
- `decode`: the inverse. The decoder has the model file and the stream; it never
  reruns the encoder's DP, it decodes the token sequence (this is what makes the
  calls count true).
- `check`: the round-trip on shakespeare, synseg, syngrid, plus a fuzz of random
  slices; the *recorded* size must equal the charged bpb within 0.005 bpb (that
  equality is the proof of the accounting, including the space and header bits).
- Selftest: a round-trip on a 1 kB slice for every corpus, a model pack/unpack
  round-trip, and the token index -> cost tables.

*Gate:* every corpus round-trips; the recorded size equals the reported charge
within 0.005 bpb, in *both* conventions; the model file is <= 45 kB (today
39.5 kB).
*Stop:* if the recorded size exceeds the charge by more than 0.02 bpb, the
accounting is wrong: fix it before any new model.

### N1 — scale: the coverage lever (~1-2 h, plus runs) — **the headline**

*Question:* does coverage turn the parity of §0 into a win?
*How:* add `load_enwik8` to `hncm6.py` (the 100 MB dump from
`mattmahoney.net/dc/enwik8.zip`, cached like shakespeare; the 10 MB slice is
`data[:10_000_000]`), then run the same pipeline at 10 MB and 100 MB. The
substring candidates are `O(CAP x N)` numpy: budget 3-5 min per corpus at 10 MB
and cache them. The budget and the rank are chosen on a held-out 10% of the
*static train*; the codecs are measured on the same val. Report: coverage, the
covered/tail split, the rate, the charge, the recorded size, the calls, and the
codec curve at 1/10/100 MB.
*Gate:* at 10 MB the charged total <= 1.80 bpb (stretch 1.60) and <= the best
codec; the calls ratio >= 20x.
*Stop:* if the covered rate does not fall with the dictionary (i.e. the coverage
gain is eaten by the charge), the *budget* is the story and N3 (the table) is the
next lever; report the curve either way.

### N2 — the tail's context at the same 16.4 kB (~45 min)

*Question:* does a second letter of context beat the rank-16 bigram inside the
same parameter budget?
*How:* three arms, all fitted on the train tail by counts + SVD, all charged:

- `cls`: the previous byte bucketed to 32 frequency classes -> a 32x256 table
  (8 kB) plus the bigram (16 kB).
- `tri`: a rank-r *trigram*: the sparse (prev2, prev1, next) counts, a randomised
  SVD on the 65k x 256 table, rows renormalised, backoff to the rank-16 bigram.
- `bias`: the *concept* class of the previous dictionary token as a rank-1 bias on
  the logits (65 x 256 = 16 kB): the slow tier shaping the fast tier, which the
  count version could not do.

Select r on the held-out 10%; report the captured size, the rate and the charge.
*Gate:* the tail bytes < 4.020 at the same or smaller charged total; target 3.80.
*Stop:* if none wins, the letter model is at its bigram limit at this budget;
record it and move to N3.

### N3 — the concept table's sharing (~45 min)

*Question:* does the rank-r sharing that won for the bytes win for the concept
bigram?
*How:* the 2k x 2k transition counts (sparse), rank r = 4..64 by SVD, the rows
renormalised with the unigram backoff, the same budget curve, r and the dict size
selected on the held-out 10%. Endpoint: the charged total at 1.12 MB and the
affordable dict size.
*Gate:* the charged total < 3.006 (target 2.90) *and* the affordable dict size
grows, so N1's coverage gain compounds.
*Stop:* if the shared concept table loses to the counts, the concept alphabet is
too sparse to share; report it.

### N4 — the boundary (~1-1.5 h, after N2)

*Question:* can the 0.616 bpb boundary be removed by the tiling's structure?
*How:* retokenise so that a dictionary token ends *with* its space (the space
joins the preceding token, and a tail token ends at the space), rebuild the
dictionary on the retokenised train, re-run the DP, the tail, and the charge.
Arms: v7 (space-prefixed) against v8 (space-suffixed) at the same budget; the
decoder stops a tail token when it decodes a space, so no length is sent.
*Gate:* the recorded size <= the tail-side bound (2.677 rate, 2.960 charged).
*Stop:* if the covered rate rises by more than the boundary falls, report it and
keep the length boundary; the codec (N0) is unaffected.

### N5 — the consistent lattice (~1 h, optional)

*Question:* does the DP gain when the tail span is a lattice option?
*How:* the state collapses to (a dictionary span) or `TAIL`; the tail span's cost
is the shared model's bits + `H(L)`; one DP per MB, no EM rounds; the budget and
r on the held-out 10%. The plain path is a subset, so the optimum cannot be worse.
*Gate:* the recorded size <= the two-stage size and still exact.
*Stop:* a gain < 0.02 bpb does not pay for the extra lattice: keep two-stage.

### N6 — the factorial (`I_perf`), the claim's evidence (~1 h)

*Question:* is the gain an interaction or two main effects?
*How:* A a byte-AR arm (an order-3 count model, or `hncm6.py`'s byte arm at the
same budget), B the dense code (v6), C the dictionary + unigram + DP, D the
dictionary + concept LM + shared tail, at a matched budget, 3 seeds, on the same
val. `I_perf = (D-C) - (B-A)` in bpb, per seed.
*Gate:* `I_perf > 0` with clean controls; report the per-seed spread.

### N7 — delivery (~1 h)

The table of record: recorded size, charge, calls, coverage, for 5 corpora
(synseg, syngrid, shakespeare, shakespeare-10MB, enwik8-10MB) and 6 codecs; the
README headline and `REPORT.md`; the artifact (`runs/v8/`), the model files, and
one command. Every claim cites a file.

## 4. What would make this a result

- **The bitstream exists** and its size equals the charge: no fantasy, the
  decoder has every input by construction (`N0`).
- **It wins where it should**: structured corpora by 0.06-0.19 bpb against the
  best codec; real text at 10 MB by >= 0.2 bpb against bz2/xz; per file above
  K ~ 5.
- **It is cheap to decode**: 6.5x fewer calls at 1 MB, >= 20x at 10 MB.
- **It is honest**: every model charged, every budget selected on a held-out
  split, the codecs on the same bytes, the pitfalls in TODO2 §5 and §9.

## 5. Rules to add (each already cost a run)

12. The model's *size* is a hyperparameter of every tier: select it on a
    held-out split, never on the val, and charge it at its packed size.
13. A hard code (k-means, VQ) is not a shared model. The low-rank factorization
    is what generalizes; the hard code was a measured negative.
14. Factorize the *counts*, not the log probabilities; keep the discount floored
    (Good-Turing, 0.25-0.95).
15. A boundary is information: pay it, or remove it with the tiling's structure.
    It is 11% of the bits at 1 MB.
16. Coverage is the largest lever (`51.8%` at 1.1 MB): measure at 10 MB before
    tuning anything below 0.1 bpb.
17. The recorded size is the judge; a rate estimate is not. Reconcile the two.

## 6. Commands (as they land)

```bash
python codec.py selftest
python codec.py encode --corpus shakespeare --model runs/v8/model.bin --out /tmp/s.bin
python codec.py decode --model runs/v8/model.bin --in /tmp/s.bin --out /tmp/s.txt
python tail_coder.py run                 # the tail arms (N2)
python concept_lm.py run                 # the concept budget (N3)
python transfer.py run                   # K files
```

## 7. Time

| step | cost | why |
|---|---|---|
| N0 the bitstream | 2-3 h | the artifact; validates every number |
| N1 scale | 1-2 h + runs | the biggest lever (coverage) |
| N2 the tail's context | 45 min | 71% of the bits at 1 MB |
| N3 the concept table | 45 min | shifts the budget; compounds with N1 |
| N4 the boundary | 1-1.5 h | 11% of the bits, 0.30 bpb |
| N5 the lattice | 1 h | 0.03-0.10 bpb |
| N6 the factorial | 1 h | the claim's evidence |
| N7 delivery | 1 h | the table of record |

Minimal end-to-end run: **N0 + N1 + N2 (~4-5 h)**. Then N3, N4, N6.
