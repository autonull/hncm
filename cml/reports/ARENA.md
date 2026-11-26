# Design arena

One row per design on the same frozen slice, pack and proof.

| design | span | head | payload | shared | self | units | coverage | proof | enc s | run |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| content-gate | content | gate | 2.139 | 2.461 | 286.347 | 256 | 0.416 | 50000 | 14.2 | `codec-concept-code-70000` |
| content-joint | content | joint | 2.039 | 2.361 | 286.232 | 256 | 0.416 | 50000 | 6.5 | `codec-concept-code-70000-cj` |
| ngram-gate | ngram | gate | 2.069 | 2.392 | 300.025 | 256 | 0.416 | 50000 | 14.4 | `codec-concept-code-70000-ng` |
| ngram-joint | ngram | joint | 1.946 | 2.268 | 299.887 | 256 | 0.416 | 50000 | 6.7 | `codec-concept-code-70000-nj` |

Best payload: **codec-concept-code-70000-nj** (1.946 bpb).
