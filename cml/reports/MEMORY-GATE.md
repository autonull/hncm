# A0.2: fitted gate with priced unit ids

- run `codec-byte-model-code-756752-d448-l8-q2048` on `code` sha `7ae9de7b4a15`, held-out 756,752 B
- base 0.970 bpb; LZ copy arm `0.852`
- pack 1,024 units, 15,429 B (content + 6 B/unit for count and length); charge 0.163 bpb
- gate features: tied cosine + 0.5 log count, length/16, log count
- softmax temperature 2; fit on 1,000,000 B train, 250,000 sampled candidates

| arm | id | accept | coverage | payload bpb | charge | total | vs stream | vs LZ | calls/B | speedup | held bpb |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tied-oracle | oracle | 0.893 | 0.653 | 0.749 | 0.163 | 0.912 | +0.058 | +0.060 | 0.545 | 1.84x | 0.252 |
| tied-count | count | 0.893 | 0.653 | 2.605 | 0.163 | 2.768 | -1.798 | +1.916 | 0.545 | 1.84x | 2.945 |
| tied-index | index | 0.893 | 0.653 | 2.732 | 0.163 | 2.895 | -1.925 | +2.043 | 0.545 | 1.84x | 3.053 |
| tied-softmax | tied | 0.893 | 0.653 | 2.743 | 0.163 | 2.906 | -1.936 | +2.054 | 0.545 | 1.84x | 3.062 |
| counts-count | count | 0.893 | 0.653 | 2.610 | 0.163 | 2.773 | -1.803 | +1.921 | 0.545 | 1.84x | 2.949 |
| tied-shuffle | count | 0.890 | 0.653 | 3.019 | 0.163 | 3.182 | -2.212 | +2.330 | 0.545 | 1.83x | 3.508 |
| softmax-shuffle | tied | 0.890 | 0.653 | 2.773 | 0.163 | 2.936 | -1.966 | +2.084 | 0.545 | 1.83x | 3.080 |

## Read

- best arm **tied-oracle** at 0.912 bpb; best decodable **tied-count** at 2.768, -1.798 against the stream and +1.916 against LZ.
- tied embeddings vs counts: +0.004 bpb in favour of tied.
- A0b shuffle costs: count **+0.414**, softmax **+0.030**; chosen **+0.414** (pass: ids are load-bearing).

**A0.2 bits fail.** No decodable id code keeps a ≥0.1 bpb win over the stream. The gate still gives 1.84x decode at 100% provenance and the chosen shuffle costs +0.414 bpb, so identity is load-bearing; the blocker is the id price plus the pack charge, not the copy decision. Record the negative; do not start A1 for bits. The next experiment is a large-stream amortization test (or a trained pointer head), not another 756 KB run.

Took 220 s on one GPU.
