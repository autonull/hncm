# TODO6 — H-NCM v11

The plan moved to `cml/PLAN.md`. The new codebase is `cml/`; the old scripts
are frozen references. Start with:

```bash
cd cml
uv sync
uv run cml selftest
uv run cml run experiments/p0-adapter.toml --dry
```
