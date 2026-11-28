# Literal LM: event-conditioned byte fallback

- code corpus `7ae9de7b4a15`, train 2,000,000 B, held-out 100,000 B
- library 300 templates; vocab 3,010; 97,608 held-out literal bytes
- trunk d192 x 4 layers x 4 heads, seq 256, batch 32, 2000 steps
- mixed: template and fill ids precede each literal run; ablated: one EVENT token occupies the same positions

| arm | literal bpb | params | seconds |
|---|---:|---:|---:|
| mixed | 2.9614 | 2,403,840 | 28.0 |
| ablated | 2.9680 | 2,403,840 | 24.6 |

## Read

- event context changes literal bits by **+0.0067 bpb** (positive means the mixed arm is better).
- gate: mixed literal bpb at least 0.03 below ablated. MISS.
- this is a small control model, not the byte trunk; the ablation isolates the event identity at equal length and cost.

Took 53 s on one GPU.
