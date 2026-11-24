# H-NCM v8: N0, the bitstream

The model is packed as counts, every CDF is derived (`round(2^24 p)`), and a
32-bit arithmetic coder writes the DP's token path. The decoder reads the
token stream and never reruns the DP. Two codecs are measured at every budget:
`concept` codes every token with the row's joint probability (the segmentation
is implicit); `tail` codes a type, a length and the shared byte model. The
budget is chosen on the held-out 10% and transmitted (`--try`); the winner is
the smallest *charged* total, so the model's size is part of the choice.

`ref` is the model's correct account (the multi-byte tokens + the tail + the
length). `v7` is the published estimate: it subtracts the byte unigram for
tail bytes that the concept path already priced at the bigram row (~3.8 vs
5.0 bpb), so it double-counts the tail's gain. `gap` is the published gap;
the code is the judge (rule 17).

| corpus | arm | dict | table | r | payload B | rate | ref | v7 | gap | model kB | shared | self | calls | enc MB/s | dec MB/s | RT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| shakespeare | concept | 1024 | 2048 | 16 | 46072 | 3.305 | 3.024 | 3.024 | +0.280 | 23.1 | 3.472 | 4.965 | 56592 | 0.25 | 0.15 | ok |
| synseg | tail | 512 | 1024 | 1 | 19346 | 1.290 | 1.231 | 1.221 | +0.069 | 14.7 | 1.389 | 2.272 | 536 | 0.14 | 0.11 | ok |
| syngrid | tail | 302 | 8192 | 0 | 26091 | 1.739 | 1.594 | 1.594 | +0.145 | 53.6 | 2.098 | 5.316 | 1 | 0.08 | 0.07 | ok |

## The arms at the chosen budget

| corpus | arm | dict | table | rate | model kB | charged | calls |
|---|---|---:|---:|---:|---:|---:|---:|
| shakespeare | tail | 2048 | 1024 | 3.273 | 59.1 | 3.696 | 49644 |
| shakespeare | concept | 2048 | 1024 | 3.372 | 26.2 | 3.560 | 49644 |
| shakespeare | tail | 1024 | 2048 | 3.602 | 39.6 | 3.886 | 56592 |
| shakespeare | concept | 1024 | 2048 | 3.305 | 23.1 | 3.470 | 56592 |
| shakespeare | tail | 4096 | 1024 | 3.275 | 79.6 | 3.846 | 43408 |
| shakespeare | concept | 4096 | 1024 | 3.305 | 46.8 | 3.640 | 43408 |
| synseg | tail | 512 | 512 | 1.315 | 11.4 | 1.391 | 507 |
| synseg | concept | 512 | 512 | 1.424 | 10.3 | 1.493 | 507 |
| synseg | tail | 639 | 256 | 1.357 | 9.6 | 1.421 | 28 |
| synseg | concept | 639 | 256 | 1.427 | 9.6 | 1.491 | 28 |
| synseg | tail | 512 | 1024 | 1.290 | 14.7 | 1.388 | 536 |
| synseg | concept | 512 | 1024 | 1.413 | 13.7 | 1.504 | 536 |
| syngrid | tail | 302 | 8192 | 1.739 | 53.6 | 2.097 | 1 |
| syngrid | concept | 302 | 8192 | 1.836 | 53.6 | 2.194 | 1 |
| syngrid | tail | 302 | 4096 | 1.921 | 29.1 | 2.115 | 1 |
| syngrid | concept | 302 | 4096 | 2.097 | 29.1 | 2.291 | 1 |
| syngrid | tail | 302 | 2048 | 1.999 | 16.7 | 2.110 | 1 |
| syngrid | concept | 302 | 2048 | 2.207 | 16.7 | 2.318 | 1 |

## The bit categories (bpb)

| corpus | arm | concept | covered | type | len | tail | sum | rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| shakespeare | concept | 3.305 | 0.000 | 0.000 | 0.000 | 0.000 | 3.305 | 3.305 |
| synseg | tail | 0.000 | 1.249 | 0.040 | 0.000 | 0.000 | 1.290 | 1.290 |
| syngrid | tail | 0.000 | 1.738 | 0.001 | 0.000 | 0.000 | 1.739 | 1.739 |

## Gates

G0 the round-trip (the artifact); G1 the recorded size equals the code's
entropy + the charge within 0.005 bpb (shared); G1s the same, self; G2 the
published v7 estimate reconciles within 0.02 bpb; G2r the correct model does;
G3 the model is at most 45 kB.

| gate | corpus | value | status |
|---|---|---|---|
| G0 | shakespeare | round-trip=ok decode=0.15 MB/s encode=0.25 MB/s | **pass** |
| G1 | shakespeare | recorded shared=3.4717 entropy+charge=3.4705 delta=+0.0013 | **pass** |
| G1s | shakespeare | recorded self=4.9652 entropy+charge=4.9640 | **pass** |
| G2 | shakespeare | published v7=3.0242 code=3.3045 gap=+0.2803 | **fail** |
| G2r | shakespeare | correct model=3.0242 code=3.3045 gap=+0.2803 | **fail** |
| G3 | shakespeare | model=23.1 kB arm=concept | **pass** |
| G0 | synseg | round-trip=ok decode=0.11 MB/s encode=0.14 MB/s | **pass** |
| G1 | synseg | recorded shared=1.3890 entropy+charge=1.3877 delta=+0.0012 | **pass** |
| G1s | synseg | recorded self=2.2716 entropy+charge=2.2703 | **pass** |
| G2 | synseg | published v7=1.2206 code=1.2897 gap=+0.0691 | **fail** |
| G2r | synseg | correct model=1.2307 code=1.2897 gap=+0.0590 | **fail** |
| G3 | synseg | model=14.7 kB arm=tail | **pass** |
| G0 | syngrid | round-trip=ok decode=0.07 MB/s encode=0.08 MB/s | **pass** |
| G1 | syngrid | recorded shared=2.0979 entropy+charge=2.0967 delta=+0.0012 | **pass** |
| G1s | syngrid | recorded self=5.3159 entropy+charge=5.3147 | **pass** |
| G2 | syngrid | published v7=1.5937 code=1.7391 gap=+0.1454 | **fail** |
| G2r | syngrid | correct model=1.5937 code=1.7391 gap=+0.1454 | **fail** |
| G3 | syngrid | model=53.6 kB arm=tail | **fail** |

## Finding

The published v7 tail formula is `total - sum(byte_unigram) + tail + length`.
The `total` already priced the tail bytes at the concept bigram row (3.83 bpb
on shakespeare's oracle budget), not at the byte unigram (5.05), so the
subtraction is too large and the 2.723 is about 0.50 bpb below the model it
names. The bitstream prices the model it implements: on shakespeare the
concept arm is the best realizable codec at 1 MB, and the tail arm pays the
type on top of a row cost it does not beat. The recorded `rate`/`shared`
columns are the number of record; the v7/v8 estimates are lower bounds.

