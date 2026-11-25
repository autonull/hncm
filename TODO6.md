# TODO6 — H-NCM v11

The plan moved to `cml/PLAN.md`. The new codebase is `cml/`; the old scripts
are frozen references. P1 is measured (1.920 bpb at seed 0, 1.912 ± 0.008 over
seeds 0-2, on the held-out 1 MB of the 10 MB parse); the P0 core is measured
(base 1.827 → adapter 0.997 bpb, net +0.758, P0a/b/c/e/f/g pass). P0d, the
equal-charge baselines, P1d/P1e and P2+ are next. The record is
`cml/reports/GATES.md`. Start with:

```bash
cd cml
uv sync
uv run cml selftest
uv run cml run experiments/p0-adapter.toml --dry
```
