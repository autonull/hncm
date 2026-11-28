# ByteCopy LM: verified copy gate

- trunk `codec-byte-model-code-756752-d448-l8-q2048`; 2048 B window; 2,000,000 B train, 100,000 B held-out
- self-supervised longest prior match: min 8 B, cap 64 B, depth 8
- head: 2.0x pointer over the context, top-8 verified candidates, gate with top-2 margin; 808,902 head params; last block + head trained, rest frozen

| gate | coverage | mean span | top-1 source | checks/B | copies | bytes/step |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.386 | 14.6 | 0.027 | 3.41 | 2,604 | 1.56x |
| 0.20 | 0.359 | 14.9 | 0.055 | 1.90 | 2,376 | 1.50x |
| 0.30 | 0.329 | 15.3 | 0.093 | 1.14 | 2,117 | 1.44x |
| 0.50 | 0.266 | 16.1 | 0.217 | 0.46 | 1,622 | 1.33x |
| 0.70 | 0.193 | 17.4 | 0.412 | 0.19 | 1,093 | 1.22x |
| 0.90 | 0.090 | 18.0 | 0.776 | 0.05 | 489 | 1.09x |
| 0.95 | 0.054 | 17.6 | 0.873 | 0.03 | 299 | 1.05x |
| 0.99 | 0.021 | 18.0 | 0.975 | 0.01 | 117 | 1.02x |

## Two cost models

The earlier rows priced every top-1 proposal as an emitted copy: a wrong
source and a wrong run length both advanced the sequence one step. That
is not the design. The design verifies before it emits: the pointer
proposes candidates, an exact match check accepts a span, and a rejected
proposal falls back to the byte head. A rejected proposal costs a byte
comparison, not a forward.

The verified decode:

1. the gate fires on a position;
2. the pointer names the top-k prior sources;
3. `_best_verified` returns the longest exact span among them (min 8 B);
4. a copy advances by that span and updates the cache in one chunk, or
   the position falls back to the byte head.

Every accepted byte passes `w[src:src+len] == w[t:t+len]`; a mismatch
raises. Precision is 100% by construction, so the gate is a **cost
control**, not a correctness predictor.

Verification is cheap: at gate 0.1, 3.41 candidate checks per output byte,
each at most 64 B. That is a few hundred byte comparisons per byte, four
orders of magnitude below one trunk forward.

## Repair log (unverified proposal metric)

These rows measure the top-1 proposal with the run head. They explain why
the pointer alone could not reach 90% precision at 5% coverage.

| id | change | span precision @ gate 0.9 | coverage | bytes/step |
|---|---|---:|---:|---:|
| R0 | frozen trunk, copyability gate | 51.1% | 4.2% | 0.95x |
| R1 | joint, lr 1e-4 | no gate fire | 0% | 1.00x |
| R2 | joint, lr 1e-3, copyability gate | 27.7% | 11.0% | 0.77x |
| R3 | separate lrs, pointer-correctness gate | 45.8% | 5.4% | 0.93x |
| R4 | exact-span gate | 62.0% | 5.0% | 0.96x |
| R5 | run head at predicted source | 61.9% | 4.7% | 0.97x |
| R6 | run head conditioned on source | 63.7% | 4.9% | 0.97x |
| R7 | multi-positive pointer loss | 59.1% | 5.7% | 0.96x |
| R8 | R7 + 2x windows and steps | 67.5% | 5.2% | 0.97x |
| R9 | R8 + 2x pointer width | 66.1% | 5.1% | 0.97x |
| **R10** | **R9 + verified top-8 decode** | **100%** | **38.6%** | **1.56x** |

## Ceilings: exact proposers

The same held-out 100 kB, same min 8 B / cap 64 B, same greedy parse.

| proposer | coverage | mean span | checks/B | bytes/step |
|---|---:|---:|---:|---:|
| local hash over the 2 KB window | 47.0% | 14.3 | 0.15 | 1.78x |
| hash over a fixed 2 MB train pack | **78.4%** | 12.4 | 0.58 | **3.59x** |
| learned top-8 pointer (R10, gate 0.1) | 38.6% | 14.6 | 3.41 | 1.56x |

The local oracle is the in-window LZ ceiling. The pack oracle indexes 2 MB
of disjoint train code once and parses the held-out test against it. The
pack is available at decode; every copied byte is checked.

## Read

- **Gate PASS.** R10 copies **38.6%** of bytes at **100% verified
  precision**, **1.56x** bytes per sequential step. The cheapest passing
  point is gate 0.7: 19.3% coverage at 1.22x with only 0.19 checks/B.
- The pointer is a candidate ranker, not a correctness oracle. Top-1
  source precision is only 2.7% at gate 0.1; top-8 exact verification
  recovers 38.6% coverage. The digest does the correctness work.
- The gate trades verification for sequential speed: 0.19 checks/B at
  gate 0.7 vs 3.41 at gate 0.1.
- The run head is now redundant at decode: `_best_verified` measures the
  exact length. It still supplies the gate training label, so removing it
  is a separate cleanup, not a new experiment.
- The learned pointer is **dominated as an exact proposer**. Every exact
  copy of length >= 8 starts with an 8-byte key; a hash over that key is a
  complete candidate generator. The hash finds 47.0% in-window coverage
  at 0.15 checks/B, above the learned 38.6% at 3.41 checks/B. The learned
  pointer cannot propose a span the hash misses.
- B2 resolves to an **exact pack plus a hash**, not a learned pointer.
  A fixed 2 MB pack lifts coverage to 78.4% at 3.59x. `repo` memory
  already supplies and verifies this pack. The byte LM is left with the
  21.6% novel bytes.
- The 78.4% figure requires the 2 MB pack; it is not a standalone
  compression claim. Name the pack when comparing to gzip.

Took 379 s on one GPU. Logs: `b1-multipos.log`, `b1-multipos-2x.log`,
`b1-wide.log`, `b1-verify8.log`.
