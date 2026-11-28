# Exact Edit Bench: C1 copy-SFT

- arm **C1 copy-SFT**; split `val`; n = 50
- seed `0`; steps `1000`; 5,175,552 params; one window ≤1024 B; samples 8
- manifest sha256 `02c8c56542235773`

## Task

Emit a serialized copy/literal action program that rebuilds `new`
from `old`, given `old` and the commit message. Exact match is the
label; an invalid program scores zero. Only files that fit one
window are in this pilot. `exact@K` samples K programs and keeps
any that assemble exactly (`K = samples`).

## Results

| arm | n | exact@1 | exact@K | valid | line distance | output bytes |
|---|---:|---:|---:|---:|---:|---:|
| C1 copy-SFT | 50 | 0.000 | 0.000 | 0.960 | 0.228 | 0.057 |

## Gate GC1

GC1 fails: C1 is exact on `0` files. Every baseline has exact
rate ≥ 0, so a +10-point lift is impossible from here. The
full-file arm is not needed to reject this arm.

## Data

| split | files |
|---|---:|
| train | 596 |
| eval | 50 |

## Reading

C1 is not exact on any file. The action grammar and decoder
work for valid programs only if `valid` is high; that is not
an edit capability. Do not start DPO from this arm. The next
lever is the target (shorter action grammar) or a copy-aware
decoder, not more steps.
