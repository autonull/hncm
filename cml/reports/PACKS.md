# Pack size: end-to-end pointer T1

- corpus `code` sha `1ccd9ffc7f68`, held-out 70,000 B, 200 fixed byte spans, seed 0
- one pointer per active pack at 8000 steps, seq 256
- the same spans come from the 256 parse; each pack covers them
  with its own units. `dec/span` is that cover size.
- exact and bits use the full pointer (concept or byte escape).

| active | exact | bits/span | dec/span | parse dec/B | span B/dec | coverage | pack B |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.375 | 3.578 | 1.67 | 0.796 | 4.02 | 0.272 | 1,449 |
| 256 | 0.415 | 3.133 | 1.00 | 0.716 | 3.16 | 0.416 | 2,771 |
| 512 | 0.395 | 4.269 | 1.18 | 0.610 | 2.90 | 0.595 | 5,443 |
| 1024 | 0.300 | 5.359 | 1.31 | 0.539 | 2.96 | 0.696 | 10,942 |

## Read

- best: active **256**, exact 0.415 at 3.133 bits/span, 2,771 B pack.
- The trained 256 pack wins all three T1 axes: 128 pays literal
  escapes, 512/1024 pay harder concept decisions and a wider header.
  Larger packs raise coverage, but coverage alone does not pay.
- Park 128/512/1024 unless a combination makes their coverage useful.

Grid took 3.6 s.
