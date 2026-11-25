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
| enwik8_10m | concept | 8192 | 4096 | 32 | 373820 | 2.991 | 2.748 | 2.748 | +0.242 | 134.8 | 3.099 | 4.069 | 310678 | 0.27 | 0.19 | ok |

## N9: the order-2 tail on the DP path

`ord2` prices the DP tail bytes with the two real preceding bytes (order-1
backoff, pruned, charged); `row` is the concept row's price for the same
single-byte tokens. A tail that does not beat the row does not pay.

| corpus | ord2 top | ord2 kB | ord2 bpb | row bpb |
|---|---:|---:|---:|---:|
| enwik8_10m | 1024 | 80.9 | 4.151 | 4.625 |

## The arms at the chosen budget

| corpus | arm | dict | table | rate | model kB | charged | calls |
|---|---|---:|---:|---:|---:|---:|---:|
| enwik8_10m | tail | 8192 | 4096 | 3.000 | 167.6 | 3.134 | 310678 |
| enwik8_10m | concept | 8192 | 4096 | 2.991 | 134.8 | 3.098 | 310678 |

## The bit categories (bpb)

| corpus | arm | concept | covered | type | len | tail | sum | rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| enwik8_10m | concept | 2.991 | 0.000 | 0.000 | 0.000 | 0.000 | 2.991 | 2.991 |

## Gates

G0 the round-trip (the artifact); G1 the recorded size equals the code's
entropy + the charge within 0.005 bpb (shared); G1s the same, self; G2 the
published v7 estimate reconciles within 0.02 bpb; G2r the correct model does;
G3 the model is at most 45 kB.

| gate | corpus | value | status |
|---|---|---|---|
| G0 | enwik8_10m | round-trip=ok decode=0.19 MB/s encode=0.27 MB/s | **pass** |
| G1 | enwik8_10m | recorded shared=3.0985 entropy+charge=3.0984 delta=+0.0001 | **pass** |
| G1s | enwik8_10m | recorded self=4.0691 entropy+charge=4.0690 | **pass** |
| G2 | enwik8_10m | published v7=2.7484 code=2.9906 gap=+0.2422 | **fail** |
| G2r | enwik8_10m | correct model=2.7484 code=2.9906 gap=+0.2422 | **fail** |
| G3 | enwik8_10m | model=134.8 kB arm=concept | **fail** |
| G4 | enwik8_10m | ord2=4.151 row=4.625 top=1024 size=80.9 kB | **pass** |

## Finding

The published v7 tail formula is `total - sum(byte_unigram) + tail + length`.
The `total` already priced the tail bytes at the concept bigram row (3.83 bpb
on shakespeare's oracle budget), not at the byte unigram (5.05), so the
subtraction is too large and the 2.723 is about 0.50 bpb below the model it
names. The bitstream prices the model it implements: on shakespeare the
concept arm is the best realizable codec at 1 MB, and the tail arm pays the
type on top of a row cost it does not beat. The recorded `rate`/`shared`
columns are the number of record; the v7/v8 estimates are lower bounds.

