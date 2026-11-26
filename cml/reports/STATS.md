# CML stats

- metrics files: 44
- codec runs: 19; text runs: 1; P0a runs: 1; install runs: 1; transfer runs: 2

## Text LM

| run | params | train B | test B | steps | bpb |
|---|---:|---:|---:|---:|---:|
| `text-enwik8_100m-d384l8-s30000` | 20,461,440 | 90,000,000 | 9,999,707 | 30,000 | **1.517** |

## Best codec row per slice (payload)

| corpus | n | payload | shared | units | coverage | run |
|---|---:|---:|---:|---:|---:|---|
| shakespeare | 5,000 | **3.930** | 6.264 | 256 | 0.694 | `codec-concept-shakespeare-5000-nj` |
| code | 756,752 | **1.692** | 1.722 | 256 | 0.453 | `codec-concept-code-756752-nj` |
| code | 200,000 | **2.321** | 5.911 | 8192 | 0.919 | `codec-lexicon-code-200000` |
| code | 70,000 | **1.916** | 1.922 | 0 | 0.000 | `codec-byte-model-code-70000` |
| code | 30,000 | **2.385** | 3.137 | 256 | 0.403 | `codec-concept-code-30000` |
| code | 5,000 | **1.725** | 2.054 | 8 | 0.013 | `codec-hybrid-code-5000` |
| code | 1,000 | **1.824** | 2.224 | 0 | 0.000 | `codec-byte-model-code-1000` |

## P0a (Shakespeare prompt)

| run | params | pack B | payload bpb | train loss | coverage |
|---|---:|---:|---:|---:|---:|
| `lm-shakespeare-tiny-s2000` | 1,143,680 | 1,409 | **2.775** | 2.944 | 0.690 |

## P0b install / transfer

| run | kind | (b)-(a) | (c)-(a) | gate |
|---|---|---:|---:|---|
| `install-code-networkx-a256-s3000` | oracle | +0.088 | +2.959 | PASS |
| `install-transfer-code-networkx-a256-s3000` | A->B | +0.212 | +3.007 | FAIL |
| `install-transfer-code-networkx-a256-s3000` | B->A | +0.150 | +2.913 | PASS |
| `install-transfer-code-networkx-a256-s8000` | A->B | +0.280 | +3.771 | FAIL |
| `install-transfer-code-networkx-a256-s8000` | B->A | +0.209 | +3.584 | FAIL |
