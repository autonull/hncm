# hncm — products and research

## Products

| product | what | status |
|---|---|---|
| [`products/codec`](products/codec/README.md) | a lossless byte codec for code: `cml compress` / `cml decompress` | exact **1.266 bpb (6.3x)**; priced-parse frontier **0.852** |
| [`products/delta`](products/delta/README.md) | exact version compression from an old file: `cml delta` / `cml undelta` | exact **56.6x** on hunk edits, **27.8x** scattered; CPU-only |
| [`products/memory`](products/memory/README.md) | exact history compression: `cml chain` / `cml unchain` | exact **40.0x** over 1,823 real versions; **6.5x better than gzip** |
| [`products/repo-memory`](products/repo-memory/README.md) | exact searchable repo index: `cml repo` | exact **20.7x** on 80 files; `path:line` search |
| [`products/memory-mcp`](products/memory-mcp/README.md) | MCP server for the repo index | 5 exact tools: files, search, read, versions, diff |
| [`products/patch-guard`](products/patch-guard/README.md) | apply a diff only if byte-exact | exit 0 on exact, hard error otherwise |
| [`products/repo-delta`](products/repo-delta/README.md) | exact delta between two commits | **85.8x** on `README.md`, verified |
| [`products/log-delta`](products/log-delta/README.md) | exact CI log history | **27.3x**, 1.8x better than gzip |
| [`products/pack-doctor`](products/pack-doctor/README.md) | rank spans by net bits at the measured bpb | 1,024 → 498 paying spans |
| [`products/copy-lm`](products/copy-lm/README.md) | byte LM with an exact copy head | research prototype; next: joint gate training |

Each product is self-contained in its own directory and uses the shared
utilities in `cml/`. Start with [`products/README.md`](products/README.md).

# H-NCM: one model, many streams

## The idea in plain terms

A phrasebook plus a letter-by-letter fallback, both paid once.

- **The phrasebook** is a dictionary of exact byte spans. The code is an index;
  the decoder writes the exact bytes, so there is no residual.
- **The concept model** prices the indices: the previous index determines the
  next one, so a first-order Viterbi finds the cheapest tokenization.
- **The fallback** codes spans the dictionary lacks with a small shared context
  model. Those are the bytes the phrasebook missed.
- **The charge** is the model itself: the dictionary, the concept table and the
  fallback. It is data, counted in bytes, and amortized over the stream.

Per-file compressors (gzip, bz2, PPMd, xz) pay their model on every file and
start cold. Here the model is paid once and reused across files, corpora and
sizes. The decoder also needs far fewer model calls: one to two orders of
magnitude fewer than a byte-at-a-time model, depending on how much the
phrasebook covers. That is the bet: a single file does not need to win for the
archive to win.

## The status

The mechanism works end to end. The bitstream round-trips every corpus; the
recorded size equals the code's entropy plus the charged model (the exact
reconciliation is in the report of record), so the accounting is checked by an encoder
and decoder, not by an estimate.

What the mechanism already shows:

- **The model is paid once.** The archive charge falls as the stream grows while
  per-file codecs' mean rises; the flat-K curve is the structural result.
- **The decoder is cheap.** Coverage determines the calls: the covered spans are
  one index each, the uncovered bytes are the fallback.
- **Structured sources pay.** Where the phrasebook fits the source, the model is
  at or ahead of the best standard codec; on at least one structured corpus it
  wins. The exact table is in the report of record.
- **The coverage lever pays at the margin.** The union vocabulary raises the
  10 MB coverage to 0.815 at a 32k dictionary and moves the recorded size from
  3.035 to **2.977**, a 0.058 win; the 2.94 gate misses by 0.037.
- **The tail's context is the count table.** On the substr 10 MB DP tail,
  order-2 counts reach 4.140 against the NCM's 4.307 and the row's 4.721; on
  the union tail 4.303 against 4.854.
- **The end-to-end proof.** The 100 MB model round-trips at **2.643** (1 MB
  held-out) and **2.664** (10 MB held-out), with a 300.9 kB model and 2.39M
  calls; PPMd is 1.834. The text claim is a negative at 100 MB; the structured
  win is the product.

The measured tables are the record of the current run. They
will change as the levers land; this README states the mechanism and the claim,
not the bit rates.

## The potential at stake

If the text lever closes at 100 MB, one small model compresses a mixed archive
— structured data, code, records, prose — with one charge, far fewer decoder
calls, and a flat K-file curve. The archive, not the single file, is the unit
that pays. If the lever does not close, the structured result is the product
and the text negative is the finding: an exact dictionary index is not cheaper
than a strong byte-context model on natural language at these sizes. Both are
useful results; the controls are already measured.

The mechanism is also a template. The phrasebook is a discrete, exact
representation; the concept model is a cheap decoder-side cost; the fallback is
a learned context model. Any one of the three can be replaced without touching
the other two, and the bitstream remains the judge.

## The open question

Does the text lever close at 100 MB? The answer will be a win, a measured
crossing, or a clean negative; all three are publishable, and the structural
claims survive either way. The plan and its gates are kept with the current work
plan, not on this page.

## What we do not claim

- Not a text win yet. The 1-10 MB text results are behind the best standard
  codec and are reported as such.
- Not a trained index. The dictionary and the concept table are counts; the
  fallback may be learned or counted, selected on a held-out split.
- Not the earlier estimates. The pre-bitstream numbers were lower bounds; the
  recorded sizes are the record, and the report of record keeps both.

## The design is not pinned to the numbers

The dictionary size, the concept table, the fallback and the budget are
hyperparameters. Each is chosen on a held-out split, charged at its packed
size, and replaceable at the same interface. The bitstream derives the CDF from
the packed model and reconciles the recorded size to the charge, so an
accounting error is visible. That is what lets the model improve without
rewriting the claim.

## Where the evidence lives

The bitstream, the solver, the loader, the probe and the fallback are separate
modules. The measured tables, the reconciliation and the current work plan are
kept with the run artifacts, so the numbers can change without rewriting this
page. This README states the mechanism, the claim and the open question; the
report of record states the numbers for the current run.

## History

- **v6 (negative).** A dense latent code is a permutation-invariant mean; it
  cannot carry byte order. `total = code + residual` is monotone in the code
  rate, and the optimizer collapses the code. The code, not the decoder, is the
  limit.
- **v7 (the redesign).** The code becomes an exact dictionary index; the
  residual is zero; the dictionary and the concept table are charged. The
  first-order concept LM cuts the rate, and the tail is measured separately.
- **v8 (the artifact).** The bitstream packs the model as counts and derives
  the CDF. It corrects the earlier estimates: the published tail account had
  double-subtracted the byte unigram for bytes the concept path had already
  priced. The concept arm is the architecture; the coverage probe and the
  learned/count fallback are the 10 MB measurements. The structured win stands;
  the text verdict moves to 100 MB.
- **v9 (the levers).** The union vocabulary is a measured 0.048 win on the
  10 MB record (2.987 against 3.035) and raises coverage to 0.815 at 32k; the
  order-2 count tail beats both the concept row and the 25 kB NCM on the same
  bytes, but the 2.94 and 4.0 gates miss. At 100 MB the end-to-end proof
  round-trips at 2.643 / 2.664 against PPMd's 1.834; the full vocabulary
  covers 0.912 but its table's unseen transitions cost 4.283. The text claim
  is a measured negative at 100 MB; the structured win is the product.

The details of each stage are in the report of record and the run artifacts.
