# Per-file: the model charge, once per file

- run `codec-byte-model-code-756752` on `code` sha `1ccd9ffc7f68`, 2,432,000 B model (fp16), 4,864,000 params
- each file is coded alone, and the model is charged once per file
  (`self = payload + model`). `install` charges the model once over
  all files coded, the shared-installation alternative in the plan.

| file B | files | payload B | payload bpb | self bpb |
|---:|---:|---:|---:|---:|
| 1,024 | 739 | 160,591 | 1.698 | 19,001.7 |
| 4,096 | 184 | 159,615 | 1.694 | 4,751.7 |
| 10,240 | 73 | 158,219 | 1.693 | 1,901.7 |
| 102,400 | 7 | 151,966 | 1.696 | 191.7 |

## Read

- charging the model once per file costs thousands of bpb at 1-100 KB
  files: G5 fails for a per-file product, as predicted.
- install-once over 2,974,720 B costs `8.236` bpb: the
  codec is a shared installation, not a per-file pack.
- To make it a product, cut the model (a smaller trunk) or share it
  across a larger corpus; the copy arm then lowers the payload.
