# TODO6 — H-NCM v11

The plan moved to `cml/PLAN.md`. The new codebase is `cml/`; the old scripts
are frozen references. P1 is measured (1.920 bpb at seed 0, 1.912 ± 0.008 over
seeds 0-2, on the held-out 1 MB of the 10 MB parse); P0 is measured (base
1.827 → adapter 0.997 bpb, net +0.758, P0a-g pass, P0d 2.22x on structured);
the equal-charge rows are measured (no-D +0.000, counts-only +0.758, kNN +0.516,
LoRA −0.033). Counts-only ties the tied gate on seen spans, so the held-out-span
linchpin, P1d/P1e and P2+ are next: `cml/PLAN.md` §11 lists them with blockers
and cost. The record is `cml/reports/GATES.md`. Start with:

```bash
cd cml
uv sync
uv run cml selftest
uv run cml run experiments/p0-adapter.toml --dry
```
