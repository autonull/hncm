# R3 scale (T2)

The pinned T2 test is unchanged; only the pointer and the train code
grow.  The 256-unit pack is the promoted codec combination.  `5m` and
`10m` are parameter targets, not exact counts.

- train code: base 7,243,247 B; caps [24.0, 48.0] MB, test 756,752 B
- pack: 256 units / 2,771 B
- steps 8000, ngram+joint; seeds [0, 1, 2, 3, 4]
- `seconds` is the encode+eval wall time after a model-cache hit;
  the first 8k-step train pass is in the run log.

| arm | seed | target | train MB | params | payload | shared | teacher | round-trip | provenance | seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| `concept-pointer-code-p5m-s8000` | 0 | 5m | 7.2 | 5.21M | 1.692 | 1.722 | 1.600 | OK | pointer-codec | 99.9 |
| `codec-scale-code-756752-p5m` | 0 | 5m | 24.0 | 5.21M | 1.583 | 1.613 | 1.490 | OK | pointer-codec | 120.7 |
| `codec-scale-code-756752-p5m-n1` | 1 | 5m | 24.0 | 5.21M | 1.602 | 1.632 | 1.509 | OK | pointer-codec | 491.2 |
| `codec-scale-code-756752-p5m-n2` | 2 | 5m | 24.0 | 5.21M | 1.599 | 1.629 | 1.506 | OK | pointer-codec | 421.4 |
| `codec-scale-code-756752-p5m-n3` | 3 | 5m | 24.0 | 5.21M | 1.597 | 1.627 | 1.504 | OK | pointer-codec | 421.6 |
| `codec-scale-code-756752-p10m` | 0 | 10m | 24.0 | 9.17M | 1.538 | 1.568 | 1.445 | OK | pointer-codec | 103.0 |
| `codec-scale-code-756752-p10m-c48m` | 0 | 10m | 48.0 | 9.17M | 1.569 | 1.599 | 1.476 | OK | pointer-codec | 657.2 |
| `codec-scale-code-756752-p10m-n1` | 1 | 10m | 24.0 | 9.17M | 1.541 | 1.571 | 1.447 | OK | pointer-codec | 631.1 |
| `codec-scale-code-756752-p10m-n2` | 2 | 10m | 24.0 | 9.17M | 1.557 | 1.587 | 1.464 | OK | pointer-codec | 586.1 |
| `codec-scale-code-756752-p10m-n3` | 3 | 10m | 24.0 | 9.17M | 1.546 | 1.576 | 1.452 | OK | pointer-codec | 586.5 |
| `codec-scale-code-756752-p10m-n4` | 4 | 10m | 24.0 | 9.17M | 1.533 | 1.563 | 1.440 | OK | pointer-codec | 617.5 |
| `codec-scale-code-756752-p20m-c48m` | 0 | 20m | 48.0 | 20.02M | 1.488 | 1.518 | 1.395 | OK | pointer-codec | 988.6 |

## Verdict

- best scale arm `codec-scale-code-756752-p20m-c48m`: 1.488 payload / 1.518 shared bpb.
- versus the 5M base arm: -0.204 payload; promote.
- T2 gate (≤1.75 payload and shared, round-trip): PASS.
- 5m @ 24 MB seed spread: 0.019 payload over seeds [0, 1, 2, 3]; best `codec-scale-code-756752-p5m` 1.583.
- 10m @ 24 MB seed spread: 0.024 payload over seeds [0, 1, 2, 3, 4]; best `codec-scale-code-756752-p10m-n4` 1.533.
- 10m @ 24 MB: all seeds beat the 5M base: True.
- 20m versus 10m @ 48 MB: -0.081 payload; promote 20m.
