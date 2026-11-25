# TODO5 — H-NCM v10: the shared-context archive (the product), the CM (the stretch)

Status: v9's 100 MB negative is measured (2.664 against PPMd 1.834). The cause
is now decomposed, and it is neither coverage nor the dictionary:

- **The unit.** On the same 10 MB val, the solid PPMd is 1.8340; the per-file
  PPMd at K=100 is 2.4385; the v9 model is 2.6642. v9 is 0.23 behind the
  per-file and 0.83 behind the solid. The archive is the unit.
- **The model class.** The same dictionary parse, coded by PPMd instead of the
  homegrown order-1 table, is 1.9782 (10 MB) and 2.2792 (1 MB); with the
  strings charged (169,533 B = 0.014 bpb over 100 MB) it is **1.992 / 2.293**
  in the shared convention. That is 0.67 ahead of v9 and 0.16 behind PPMd.
- **The dictionary.** It is a 0.16 text loss (1.992 against 1.834) and a 0.11
  structured win (synseg 1.389 against 1.500). Its value is the structured
  source and the bridge, not the text rate.
- **The information.** The in-sample oracle over the v9 path is order-1
  **2.072**, order-2 **1.327**, token + last-2-bytes **1.265**, byte
  order-4 **1.775**. The order-2 table is 485,653 contexts / 1,237,905 triples:
  too big to transmit, free to learn. The naive online interpolation is a
  measured negative (2.39-2.45), so the estimator is the work.

The product is now clear, and it is not a single-file text compressor. It is a
**shared-context archive**: a real CLI, a shared dictionary, a solid stream,
block checkpoints, two regimes, and a measured win on a real source archive.
On the repo's 24 files (712,333 B) the archive is **1.644 bpb** against
per-file PPMd **2.300** (28.6% smaller) and `tar.xz` **1.677**; any file
extracts in under a second. On enwik8 100 MB the solid is **1.834** at the
10 MB val, the per-file is 2.4385, and the block curve gives 1 MB access at
**2.096**. The structured dictionary is **1.389** on synseg. The text CM is
the stretch: its ceiling is 1.327, its target 1.65 at 100 MB.

## 0. Start here

```bash
python codec.py selftest
python concept_lm.py selftest
python hncm.py test
python hncm.py prior install --library enwik8,src,json,log
python hncm.py pack . -o /tmp/repo.hncm --block 10MB --prior auto
python hncm.py list /tmp/repo.hncm
python hncm.py extract /tmp/repo.hncm codec.py -o /tmp/codec.py
python hncm.py verify /tmp/repo.hncm
python hncm.py bench . --baseline ppmd,xz,bz2 \
  --out runs/v10/repo.json
python hncm.py bench enwik8_100m --files 10,100 --block 1,10,100 \
  --out runs/v10/enwik8.json
python hncm.py bench shakespeare,synseg,syngrid,json,csv,src \
  --regime auto --out runs/v10/structured.json
```

Artifacts: `runs/v10/` (the archive tables, the bench JSONs, the proofs, the
format spec, the hashes). `hncm.py` is the CLI and the format; `codec.py`,
`concept_lm.py` and `probe.py` stay the judge. The repo archive and the
enwik8 10 MB val are the acceptance tests. Every recorded size round-trips.

## 1. What v9 established (do not re-derive)

- **The bitstream.** The model is packed as counts; the CDF is derived; a
  32-bit arithmetic coder writes the path; the decoder never reruns the DP.
  Recorded = the code's entropy + the charge within **0.0013 bpb**.
- **The union vocabulary.** 0.815 coverage at the 32k dict; 2.977 on the 10 MB
  record (a 0.058 win over v8); the 2.940 gate misses.
- **The order-2 byte tail.** 4.303 on the union tail, 4.140 on the substr
  tail, against the row's 4.854 / 4.721; the 4.0 gate misses; the NCM is 4.307.
- **The 100 MB negative.** 2.643 / 2.664; PPMd 1.834; the full vocabulary
  covers 0.912 but costs 4.283; the text claim is a negative at 100 MB; the
  structured win is synseg 1.389 against PPMd 1.500.
- **The speed.** 0.02 MB/s in Python; 1850 s per 10 MB proof.

## 2. The measured decomposition (the new evidence)

All rows are the same 10 MB val, the same 90/10 split and the same SHA
(`5985c81c…b3d97`). `shared` charges the model over 100 MB; `self` charges it
over the val. PPMd is zero-charge: its state is the decoded history.

| scheme | rate | charge | shared | self |
|---|---:|---:|---:|---:|
| solid PPMd (the archive ceiling) | 1.8340 | 0 | **1.8340** | 1.8340 |
| token parse + PPMd (10 MB) | 1.9782 | 0.0136 | **1.9918** | 2.1138 |
| v9 model (10 MB) | 2.6401 | 0.0241 | 2.6642 | 2.8808 |
| per-file PPMd, K=10 | 2.0961 | 0 | 2.0961 | 2.0961 |
| per-file PPMd, K=100 | 2.4385 | 0 | 2.4385 | 2.4385 |
| token parse, K=10 (proxy) | 2.2023 | 0.0136 | 2.2159 | 2.3382 |
| token parse, K=100 (proxy) | 2.4222 | 0.0136 | 2.4358 | 2.5581 |

The **real** archive (the repo's 24 files, 712,333 B; the measured table in
`runs/v10/repo.json`):

| codec | per-file | solid | win |
|---|---:|---:|---:|
| PPMd | 2.300 | **1.644** | 0.656 (28.6%) |
| xz | 2.427 | 1.677 | 0.750 |
| bz2 | 2.396 | 1.726 | 0.670 |

The block curve on that archive: solid 1.644; K=2 1.743; K=5 1.896; K=10
**2.074**; K=20 2.269. The access cost is the block cold start: 0.43 bpb at
10 blocks, 0.63 at 20.

The oracle over the v9 path (in-sample MLE; a ceiling only):

| context | bits/token | bpb |
|---|---:|---:|
| order-1 token (v9's class) | 5.549 | 2.072 |
| order-2 token | 3.554 | **1.327** |
| token + last 2 bytes | 3.387 | 1.265 |
| byte order-1 / 2 / 3 / 4 | — | 3.874 / 3.016 / 2.311 / 1.775 |

The 1 MB oracle (order-1 1.829, order-2 1.044) is **not** comparable: 370k
tokens overfit more than 3.7M. Use the 10 MB oracle, and the LOO/KT estimate
in N3, for the CM forecast.

## 3. The architecture: the format, the blocks, the regimes

**The format.** `HNCM` magic, version, flags, the prior id, the file count,
the block count, the dictionary length, the SHA-256 of the original tree. The
file table holds the name, the size, the block, the offset, the regime, the
SHA and the chunk list; the block table holds the offset, the length, the
first file and the codec; the chunk table holds the rolling hash, the block
and the length; the dictionary holds the strings and the lengths, or is empty
when the prior id is set; the payload holds the blocks. The file index is
`4 B/file`, the block index `16 B/block`, the chunk index `24 B/chunk`. A file
is extracted by decoding its block from the start to the file's end; the cost
is the block, not the archive.

**The blocks.** Sort the files by regime, then by the dictionary sketch (the
top-64 tokens); group into blocks of `--block` (default 10 MB); encode each
block as one solid stream. The block is the checkpoint: the decoder starts at
the block, and the file is a slice. The access cost is the measured cold
start (0.43 bpb at 10 blocks on the repo archive); the block size is the
access/size trade.

**The regimes.** `text`: the solid PPMd now, the CM in N4; zero charge, the
model is the decoded history. `structured`: the dictionary + the concept
model, charged at the strings (0.014 bpb over 100 MB); synseg 1.389. `auto`:
a small decision tree on the sketch (the coverage, the distinct-token rate,
the byte entropy, the repeat rate) picks the regime; the tag is 2 bits per
file, transmitted, so no selection leak. The regime is chosen on the held-out
10% of the source, never on the val.

**The zero charge.** The solid PPMd's state is the decoded history, so it is
not transmitted. The only charge is the dictionary, and only for the
structured regime. The v9 model's 0.024 is removed; the token parse pays
0.014; the text pays 0.

**The shipped prior.** The dictionary is not a file inside the archive; it is
a prior installed with the tool. `hncm` ships a small prior library (the
English text, the source, the JSON and the log priors), each ~170 kB. The
archive records a 2-byte prior id and nothing else; the decoder resolves it.
The archive is pure payload. The per-archive dictionary and the shipped
prior are the same 169.5 kB, so the tool pays for itself at the **second
archive** and every archive after that is 0.014 bpb smaller. `--prior none`
keeps the archive self-contained.

**The archive is also a backup format.** The file table carries the SHA-256;
a duplicate file is a reference and costs 0 B. The block table is independent,
so the pack encodes blocks in parallel and a damaged block loses only its own
files. `hncm append` adds a block without rewriting the archive; `hncm verify`
checks the tree. The chunk table (a rolling hash, 8 kB chunks) makes the
dedup content-defined, so a re-packed file with a small edit reuses its
unchanged chunks. `--codec ppmd|zstd|cm` trades size for speed: the solid
PPMd is the size default, zstd is the fast default, the CM is the stretch.

**The CM.** One binary arithmetic coder, the byte models (orders 1-4, hashed),
the match model (the last 8 bytes -> the last position, the predicted bit, the
match length), the token-concept predictor (the v9 table as a prior), and a
logistic mixer with an SSE stage. The history is the archive prefix; the
checkpoint is the block start; the model is the decoded history, so it is
zero-charge. The archive is the unit: the CM is warm across the block, not
cold per file.

## 4. The work, in order

### N0 — the format, the CLI and the real archive (2.5 h) — **do this first**

*Question:* does the shared-context archive beat the per-file codecs on a real
archive, end to end, with random access?
*How:* `hncm.py` with `pack`, `list`, `extract`, `verify`, `bench`, `test`;
the tracked text files (the repo's 24 files); the solid PPMd per block; the
file/block tables; the SHA per file; the extract. The bench runs per-file
PPMd/xz/bz2 and `tar.xz` on the same files. The dedup test duplicates three
files; the append test adds a file; the parallel pack encodes the blocks in
parallel; the verify and the extract run on the result.
*Gate:* the total ≤ 1.70 (the solid 1.644 + the header); the extract of any
file ≤ 1 s; the duplicate costs ≤ 100 B; the append is O(block); the verify
and the round-trip pass; the list is exact.
*Stop:* if the total is not ≤ 0.80 × the per-file PPMd, the archive is not the
unit; report the per-file result and go to N2.

### N1 — the 100 MB archive (2 h) — **the headline**

*Question:* at 100 MB, what does the archive cost at a 1 MB access granularity?
*How:* the 100 MB enwik8 split into K=10 (10 MB) and K=100 (1 MB); the solid
PPMd; the block curve B=1/10/100; the per-file control; the round-trip; the
`bench`. Report the total = the payload + the header + the index, and the
access cost = the block total - the solid.
*Gate:* the solid ≤ 1.90; the block at B=10 (1 MB access) ≤ 2.10; the access
cost ≤ 0.30; the round-trip is clean.
*Stop:* if the solid is not 0.15 ahead of the per-file at K=100, the archive
is not the unit; say so.

### N2 — the structured suite (1-2 h) — **the product**

*Question:* does the dictionary win on real structured sources, or only on the
synthetic synseg?
*How:* synseg, syngrid, shakespeare, plus four real sources: the repo's
`.py` files (source), a deterministic JSON dump, a CSV dump and a log sample.
The dictionary + the concept model (`codec.py`) and the token parse + PPMd;
the charge; the round-trip. The regime is chosen per source on the held-out
10% (2 bits), never on the val.
*Gate:* the structured median ≤ the best standard codec; synseg keeps its 0.11
win; the real structured sources (JSON/CSV/log) do not lose; the source is
the text regime; the auto choice selects; the mixed archive's per-source
choice beats one model.
*Stop:* if the real structured sources do not win, the synseg win is a
synthetic artifact; report that and keep the archive claim only.

### N3 — the zero-charge dictionary and the token parse (30 min)

*Question:* what does the dictionary cost when only the strings are sent?
*How:* the strings-only dictionary (169,533 B = 0.014 bpb over 100 MB); the
token parse + PPMd; the solid control; the charge table; the LOO/KT estimate
of the order-2 online model.
*Gate:* the charge ≤ 0.014; the token parse is within 0.20 of the solid and
≥ 0.60 ahead of v9 in `shared`; if it beats the solid, the dictionary is a
text win after all.
*Stop:* if the token parse is > the solid + 0.20, the dictionary parse is a
text loss; the structured claim is the only dictionary claim.

### N3b — the shipped prior (30 min) — **the amortization**

*Question:* is the dictionary a tool prior instead of an archive section?
*How:* the prior library (the English text, the source, the JSON and the log
priors); the 2-byte prior id; `pack` resolves it; `extract` checks it;
`--prior none` keeps the archive self-contained. Bench the per-archive
dictionary against the shipped prior: the break-even is the second archive.
*Gate:* the archive is ≤ 1.644 + the header (the dictionary is 0); the tool
is 169.5 kB; the archive with `--prior none` is 1.658; the break-even is 2.
*Stop:* if the library is too large for the tool, keep the per-archive
dictionary and the 0.014 charge.

### N4 — the CM (3-4 h) — **the stretch**

*Question:* can the CM close the 0.65 gap from the token parse (1.978) to the
order-2 oracle (1.327) and beat PPMd on text?
*How:* the binary CM in §3: the byte orders 1-4, the match model, the
token-concept predictor, the logistic mixer, the SSE; the 1/10/100 MB; the
dictionary as the match source; the prior as the concept; the block archive.
Two parses: the prior DP (exact) and an adaptive lazy parse
(decoder-synchronous); the coder round-trips either way.
*Gate:* 10 MB ≤ 1.80 (a win against PPMd 1.834); 100 MB ≤ 1.65; the calls
and the speed are the bar's, the rate is the claim.
*Stop:* if the 10 MB is > 1.90, the CM is the research, not the product;
report the oracle and keep the archive + the structured win.

### N5 — the C kernel (3 h)

*Question:* can the CM and the extractor run at 5 MB/s?
*How:* the CM and the coder in C (compiled with `cc`, loaded by `ctypes`);
the model packing stays Python. Fuzz the C and Python coders on 100 random
strings: the same decoded bytes and the same recorded size within 0.001 bpb.
*Gate:* ≥ 5 MB/s encode and decode; the same bitstream; the extract ≤ 0.2 s
per 10 MB block.
*Stop:* if the C kernel diverges, keep the Python coder and cap the archive
at K=100; the time is the limit.

### N6 — delivery (1 h)

`runs/v10/` with the format spec, the archive tables, the bench JSONs, the CM
proof, the config and the hashes; the README headline; `REPORT.md`; one
command. The table of record: the real repo archive, the enwik8 solid and the
block curve, the per-file controls, the v9 and PPMd rows, the structured
suite, and the negative (the dictionary is a text loss, a structured win).

## 5. Risks and cheap checks

| risk | check | cost |
|---|---|---|
| the archive is just a different PPMd run | the per-file vs solid on the same files | the N0 bench |
| the block cold start eats the access win | the B=1/10/100 curve | 30 min |
| the structured win is synthetic | the real source/JSON/CSV/log suite | 1 h |
| the dictionary is a text loss | the token parse vs the solid | 30 min |
| the CM does not beat PPMd | the 10 MB CM | 3 h |
| the CM is too slow | the C kernel | 2 h |
| the regime selection leaks | the held-out selection, the 2-bit tag | 30 min |
| the format is not stable | the SHA and the fuzz | 30 min |

## 6. Stop rules

1. The archive total is the judge: the payload + the header + the index, at
   the same files and the same K.
2. The solid PPMd is the ceiling; the per-file is the unit baseline; the
   block curve is the price of random access. Report all three.
3. The dictionary is charged at its strings (169,533 B = 0.014 bpb over
   100 MB), not at the counts; the token parse needs no counts.
4. The repo archive is the acceptance test: ≤ 1.70, the extract ≤ 1 s, the
   verify clean. The enwik8 solid ≤ 1.90, the 1 MB block ≤ 2.10.
5. If the solid is not 0.15 ahead of the per-file at K=100, the archive is
   not the unit; the per-file token parse is the win.
6. If the structured median does not win on the real sources, the dictionary
   is the token-parse bridge only; say so.
7. If the CM does not beat PPMd at 10 MB, the text is the bar's; the archive
   and the structured win are the product.
8. The `hncm` CLI is the deliverable; the plan is not done without it.

## 7. The big picture

v9 asked "can the dictionary beat PPMd on a file?" The measured answer is
0.16 behind. The right question is "can the archive beat the per-file codecs,
with random access, on real mixed data?" The measured answer is 28.6% smaller
on the repo archive, 24.8% smaller on enwik8 at 100 files, and 1 MB access at
0.26 bpb. The dictionary's value is the structured source (0.11 ahead on
synseg) and the shared bridge, not the text rate. The system is a
**shared-context archive**: the text regime is the solid adaptive stream; the
structured regime is the dictionary + the concept model; the archive is the
unit; the block is the checkpoint; the CM is the text stretch.

If the CM closes the 0.65 oracle gap, the text claim returns and the archive
gets stronger. If it does not, the measured archive win, the structured win
and the CLI are the result, and the v9 negative is explained: the homegrown
order-1 table was the model class, and the per-file unit was the other. Both
are fixed by the same move: the archive is the unit, and the model is the
decoded history.

## 8. Time

| step | cost | result |
|---|---|---|
| N0 the format, the CLI, the repo archive | 2.5 h | the product, 1.644, dedup |
| N1 the 100 MB archive | 2 h | the headline, the block curve |
| N2 the structured suite | 1-2 h | the real structured win |
| N3 the zero-charge dictionary | 30 min | 0.014, the token parse |
| N3b the shipped prior | 30 min | the tool pays at archive 2 |
| N4 the CM | 3-4 h | the text stretch |
| N5 the C kernel | 3 h | 5 MB/s, K=1000 |
| N6 delivery | 1 h | `runs/v10/`, the report |

Minimal decisive run: **N0 + N1 + N2** (6.5 h): the product, the headline
and the structured win. The shipped prior is N3b; the CM is the stretch; the
C kernel is the speed.

Provenance: the repo archive is the 24 tracked text files (712,333 B) with
the measured table; the enwik8 rows are the cached 100 MB dump, SHA-256
`5985c81c…b3d97`, the 10 MB val `raw[90_000_000:][:10_000_000]`; the per-file
rows split it at equal boundaries; the token parse uses the `fit-seed` pickle's
DP path; the solid row is the v9 `PROOF-100M.md` 10 MB val. The oracle is
in-sample and labelled. The commands in §0 reproduce the tables; the seed is 0.
