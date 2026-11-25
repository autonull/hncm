# TODO6 — H-NCM v11

The plan moved to `cml/PLAN.md`. The new codebase is `cml/`; the old scripts
are frozen references. P1 is measured (1.920 bpb at seed 0, 1.912 ± 0.008 over
seeds 0-2, on the held-out 1 MB of the 10 MB parse); that is the honest rate,
because a model that emits unit ids pays for them through its own
probabilities. P0's 0.997 bpb is an **oracle** rate: it does not charge the
span's name. Charged with the count prior the adapter is 3.97 bpb (net −2.224)
on text and −0.403 on the repo's code, so P0a now gates `saved_charged` and
fails. The linchpin is null (0.510/0.510, embedding weight −0.04) and a foreign
pack is domain-local (+0.422); both are moot until a name is paid. Next: a
charged P0d, P1d, P2 and holes/fills. `cml/PLAN.md` §11 has the correction and
the cost. The record is `cml/reports/GATES.md`. Start with:

```bash
cd cml
uv sync
uv run cml selftest
uv run cml run experiments/p0-adapter.toml --dry
```
