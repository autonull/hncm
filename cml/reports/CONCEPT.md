# Concept LM (M1)

Tokenize the code corpus with the 256-unit codec pack, then train on
unit ids. `pointer` is content-addressed; `trunk` needs a new row for
a new unit and is the fixed-vocab control.

| model | params | steps | bits/byte | payload | shared | B/decision | span B/dec | coverage | round-trip | pack B | run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| pointer | 5.21M | 8000 | 1.600 | 1.692 | 1.722 | 1.478 | 3.498 | 0.453 | pointer-codec | 2,771 | `concept-pointer-code-p5m-s8000` |
| trunk | 4.93M | 8000 | 1.516 | - | - | 1.478 | 3.498 | 0.453 | parse | 2,771 | `concept-trunk-code-p5m-s8000` |

## H1

- **bits/byte** 1.600 (payload 1.692, shared 1.722) ≤ 1.75: PASS (kill at 1.9)
- **round-trip** 50,000 units / 81,531 B: PASS (pointer-codec)
- **B/decision** 1.478 ≥ 2.5: FAIL
- **span-only B/decision** 3.498 ≥ 2.5: PASS
- literal fraction 0.809; the 256-unit pack covers only 45% of test bytes, so an escape costs one decision for one byte.

The conjunction is the issue: at 256 units the payload gate passes, but the decision gate cannot. The span-only figure is the action-
space view; the overall figure charges the byte escape. Do not
reinterpret the gate silently; either accept the escape charge or
choose a larger pack for RLFP and pay its header.

## Controls and next

- `trunk` is the fixed-vocabulary control: the same parse, integer ids.
- M2 needs 10k span-repair pairs and an SFT concept policy; no `cml rl`
  until the concept LM prints.
- M3 compares concept/byte/word DPO at equal pairs and compute.
