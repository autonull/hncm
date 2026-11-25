# TODO6 — H-NCM v11

The plan moved to `cml/PLAN.md`. The new codebase is `cml/`; the old scripts
are frozen references. P1 is measured (1.920 bpb at seed 0, 1.912 ± 0.008 over
seeds 0-2, on the held-out 1 MB of the 10 MB parse); P0 is measured (base
1.827 → adapter 0.997 bpb, net +0.758, P0a-g pass, P0d 2.22x on structured);
the equal-charge rows are measured (no-D +0.000, counts-only +0.758, kNN +0.516,
LoRA −0.033). The linchpin is negative: on the held-out 10% of the pack the tied
gate equals counts-only (gap 0.000, embedding weight −0.04), so the gate is a
length/count classifier and the exact-span copy has no use for embeddings. A
foreign pack is domain-local (+0.422 bpb), and on 650 kB of Python the adapter
saves +1.932 rate (net +1.648) at 1.05x latency. Next: speculative
verification, P1d, P2, holes/fills. `cml/PLAN.md` §11 lists them with blockers
and cost. The record is `cml/reports/GATES.md`. Start with:

```bash
cd cml
uv sync
uv run cml selftest
uv run cml run experiments/p0-adapter.toml --dry
```
