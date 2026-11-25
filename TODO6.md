# TODO6 — H-NCM v11

The plan moved to `cml/PLAN.md`. The new codebase is `cml/`; the old scripts
are frozen references. P1 is built and measured (1.920 bpb at seed 0, 1.912 ±
0.008 over seeds 0-2, on the held-out 1 MB of the 10 MB parse); P0 is next.
The record is `cml/reports/GATES.md`. Start with:

```bash
cd cml
uv sync
uv run cml selftest
uv run cml run experiments/p0-adapter.toml --dry
```
