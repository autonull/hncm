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
reconciliation is in `REPORT.md`), so the accounting is checked by an encoder
and decoder, not by an estimate.

What the mechanism already shows:

- **The model is paid once.** The archive charge falls as the stream grows while
  per-file codecs' mean rises; the flat-K curve is the structural result.
- **The decoder is cheap.** Coverage determines the calls: the covered spans are
  one index each, the uncovered bytes are the fallback.
- **Structured sources pay.** Where the phrasebook fits the source, the model is
  at or ahead of the best standard codec; on at least one structured corpus it
  wins. The exact table is in `REPORT.md`.
- **Natural text is still behind.** On English text the model currently trails
  PPMd at 1-10 MB, and the gap is the uncovered part: the covered tokens are
  already competitive, the fallback is not. The verdict at 100 MB is not yet
  measured.

The tables in `REPORT.md` and `runs/` are the record of the current run. They
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

Does the text lever close at 100 MB? The plan and the gates are in `TODO4.md`.
The answer will be a win, a measured crossing, or a clean negative; all three
are publishable, and the structural claims survive either way.

## What we do not claim

- Not a text win yet. The 1-10 MB text results are behind the best standard
  codec and are reported as such.
- Not a trained index. The dictionary and the concept table are counts; the
  fallback may be learned or counted, selected on a held-out split.
- Not the earlier estimates. The pre-bitstream numbers were lower bounds; the
  recorded sizes are the record, and `REPORT.md` keeps both.

## The design is not pinned to the numbers

The dictionary size, the concept table, the fallback and the budget are
hyperparameters. Each is chosen on a held-out split, charged at its packed
size, and replaceable at the same interface. The bitstream derives the CDF from
the packed model and reconciles the recorded size to the charge, so an
accounting error is visible. That is what lets the model improve without
rewriting the claim.

## The record

| file | what it holds |
|---|---|
| `codec.py` | the bitstream: the model, the CDF, the arithmetic coder, both conventions |
| `concept_lm.py`, `hncm7.py` | the dictionary, the concept LM and the first-order solver |
| `corpora.py` | the cached corpora and the deterministic split |
| `probe.py` | the coverage probe |
| `ncm.py` | the learned fallback (the context model) |
| `tail_coder.py`, `transfer.py` | the fallback arms and the K-file charge |
| `REPORT.md` | the measured tables (the record) |
| `TODO4.md` | the remaining levers and the 100 MB verdict |

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

The details of each stage are in `REPORT.md`, `TODO1.md`, `TODO2.md` and
`runs/`.
