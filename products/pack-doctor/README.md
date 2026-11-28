# `pack-doctor` — rank spans by net bits, not bytes

A diagnostic for span packs in front of a low-bpb byte trunk.

```bash
uv run cml pack fit runs/a02-1024/artifacts/pack.bin \
  --bpb 0.974 --id-bits 9 --units 1024 --min-len 8
```

## The defect it fixes

The shipped pack gain is `count*(L-1) - (L+4)` — byte savings at 8 bpb.
At a ~1 bpb trunk, a copy of `L` bytes saves `L*bpb` bits but pays an id.
A 3-byte span saves ~3 bits and a 9-bit id loses 6; only long spans pay.

`pack fit` re-ranks a counted pack by:

```
net = count*(L*bpb - id_bits) - 8*(L+6)
```

and keeps only positive-net spans.

## Result

A0.2 pack, 1,024 spans, 0.974 bpb, 9-bit id, min length 8:
**498 spans kept**, mean **12.9 B**, 13,389 B pack.

## Use it when

- a pack was induced for a high-bpb model and is used with a low-bpb trunk;
- you need to know which spans actually pay after the id;
- you are designing a local phrasebook.
