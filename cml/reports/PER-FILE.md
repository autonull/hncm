# Per-file: the pack charge, amortized

- corpus `code` sha `1ccd9ffc7f68`, held-out 756,752 B
- each file is coded alone; the pack is charged once per file:
  `shared = payload + pack_bytes`.
- the byte column is the order-2 arm on the same files.

| file B | files | units | pack B | shared mean | shared min | shared max | byte mean | delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 739 | 64 | 764 | 9.587 | 7.789 | 10.945 | 3.661 | +5.926 |
| 1,024 | 739 | 256 | 2,771 | 25.542 | 23.641 | 26.922 | 3.661 | +21.881 |
| 4,096 | 184 | 64 | 764 | 4.532 | 2.873 | 5.463 | 3.023 | +1.509 |
| 4,096 | 184 | 256 | 2,771 | 8.808 | 7.170 | 9.631 | 3.023 | +5.785 |
| 10,240 | 73 | 64 | 764 | 3.368 | 2.373 | 3.968 | 2.725 | +0.643 |
| 10,240 | 73 | 256 | 2,771 | 5.331 | 4.305 | 5.950 | 2.725 | +2.606 |
| 102,400 | 7 | 64 | 764 | 2.625 | 2.372 | 2.758 | 2.477 | +0.148 |
| 102,400 | 7 | 256 | 2,771 | 3.217 | 2.973 | 3.362 | 2.477 | +0.740 |

## Read

- best case 102,400 B with 64 units still loses 0.148 bpb; G5 fails.
- worst case 1,024 B with 256 units loses 21.881 bpb.
- The charge is per file: a static pack cannot pay below its own
  size, so the product needs an adaptive pack, a delta codec, or a
  shared installation that is charged once (`reports/COPY.md`).
