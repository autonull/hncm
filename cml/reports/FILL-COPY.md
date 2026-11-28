# Fill-copy codec: long fills, exact pointers

- code corpus `7ae9de7b4a15`, train 2,000,000 B, held-out 100,000 B
- tokenizer: min fill 16 B, cap 1024 B; templates 16+ occurrences, <= 2 holes, >= 16 fixed B
- library 300 templates, 11,929 B table

| item | bytes | bytes/byte |
|---|---:|---:|
| blob | 99,331 | 0.993 |
| blob 1 bpb literals | 12,643 | 0.126 |
| blob + table | 111,260 | 1.113 |
| gzip -9 | 24,954 | 0.250 |
| byte trunk 0.97 | 12,125 | 0.121 |

## Stream

- fixed 0.007, fill 0.017, literal 0.976
- fill copy: local 0.002, pack 0.000, literal 0.015
- templates 0.001 decisions/B; 1020.41 B/decision
- round-trip OK; table charge 11,929 B

## Read

Fixed syntax is free after the table charge. Fills are exact pointers into earlier output or the train pack; a fill that is not present is sent literally. The one-bpb row replaces literal bytes with the byte trunk's measured rate, so it is an estimate.

Took 0.1 s CPU.
