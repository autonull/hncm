# Fill-copy codec: long fills, exact pointers

- code corpus `7ae9de7b4a15`, train 2,000,000 B, held-out 100,000 B
- tokenizer: min fill 12 B, cap 1024 B; templates 3+ occurrences
- library 1,638 templates, 48,933 B table

| item | bytes | bytes/byte |
|---|---:|---:|
| blob | 97,047 | 0.970 |
| blob 1 bpb literals | 13,478 | 0.135 |
| blob + table | 145,980 | 1.460 |
| gzip -9 | 24,954 | 0.250 |
| byte trunk 0.97 | 12,125 | 0.121 |

## Stream

- fixed 0.023, fill 0.067, literal 0.911
- fill copy: local 0.022, pack 0.000, literal 0.044
- templates 0.005 decisions/B; 184.50 B/decision
- round-trip OK; table charge 48,933 B

## Read

Fixed syntax is free after the table charge. Fills are exact pointers into earlier output or the train pack; a fill that is not present is sent literally. The one-bpb row replaces literal bytes with the byte trunk's measured rate, so it is an estimate.

Took 0.1 s CPU.
