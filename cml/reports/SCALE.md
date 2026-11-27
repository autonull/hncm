# R3 scale (T2)

The pinned T2 test is unchanged; only the pointer and the train code
grow.  The 256-unit pack is the promoted codec combination.  `5m` and
`10m` are parameter targets, not exact counts.

- train code: 23,999,981 B (base 7,243,247 B), test 756,752 B
- pack: 256 units / 2,771 B
- steps 8000, seed 0, ngram+joint
- `seconds` is the encode+eval wall time after a model-cache hit;
  the first 8k-step train pass is in the run log.

| arm | params | payload | shared | teacher | round-trip | provenance | seconds |
|---|---:|---:|---:|---:|---|---|---:|
| `concept-pointer-code-p5m-s8000` | 5.21M | 1.692 | 1.722 | 1.600 | OK | pointer-codec | 99.9 |
| `codec-scale-code-756752-p5m` | 5.21M | 1.583 | 1.613 | 1.490 | OK | pointer-codec | 120.7 |
| `codec-scale-code-756752-p10m` | 9.17M | 1.538 | 1.568 | 1.445 | OK | pointer-codec | 103.0 |

## Verdict

- best scale arm `codec-scale-code-756752-p10m`: 1.538 payload / 1.568 shared bpb.
- versus the 5M base arm: -0.154 payload; promote.
- T2 gate (≤1.75 payload and shared, round-trip): PASS.
