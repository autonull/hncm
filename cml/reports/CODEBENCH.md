# Code bench: exact structural memory

- root `/home/me/hncm/cml`; train 33 files/711,525 B; held-out 1 files/26,850 B
- pack 8,888 templates, 312,136 B table

| item | value |
|---|---:|
| structural coverage | 0.243 |
| fully exact held-out lines | 0.176 |
| literal share | 0.757 |
| literal bytes | 20,319 |
| fill copy local / pack / literal | 0.074 / 0.038 / 0.188 |
| decisions/B | 0.591 |
| bytes/decision | 1.69 |
| byte trunk bpb | 1.143 |
| hybrid pack bpb | 1.186 |

## Read

Structural coverage is the fraction of held-out bytes that are
fixed template bytes or exact fills. Fully exact lines are the
strong LLM capability: no literal byte on the line. The hybrid
bpb charges the frozen trunk for literals and a flat id cost for
events, so it is an estimate, not a trained model result.
The table is 43.9% of train bytes and is **not** in the hybrid bpb; prune or share it before a product claim.
