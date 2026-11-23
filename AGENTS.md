# Code Guidelines

Working functionality beats coverage; coverage beats cosmetic lint. Optimize for
iteration speed, not ceremony.

- Elegant, consolidated, organized.
- DRY: one source of truth per rule. A second table/filter/cache is a defect; feed the
  first or delete it.
- Modular when reuse is real. One file is fine for exploratory research; split only when
  a second caller appears.
- Backwards compatibility: NONE.
- Few comments; self-documenting code. `# TODO(name): ...` for deferred work.
- Prefer GPU over CPU where appropriate.
- Be skeptical of low-performing experiments; suspect an implementation defect before
  tuning.
- Reproducibility: seed, log the config, save artifacts. A run without provenance is
  not evidence.

## Toolchain

- Python 3.14+.
- `uv` for the environment, dependencies, and tasks. One `uv.lock`; no
  `requirements.txt`.
- Ruff format always (cheap). `ruff check --fix` on changed files only; fix
  correctness findings (`F`, `E9`), ignore cosmetic findings.
- No pre-commit hook by default. Add one only if it saves more time than it costs.
- Line length: Ruff default. Use `# noqa` freely when a split hurts readability.
- Pyright/mypy: optional and non-blocking. Type hints on public APIs and tricky
  code; do not annotate everything.
- Tests: only invariants that have already broken (controls, cancellation, JSON
  round-trip, resume, shared init). No coverage target.

## Types and Data

- Built-in generics and `X | None`. Do not import `List`, `Dict`, `Optional`, or
  `Union`.
- `Protocol` for seams; `StrEnum`/`Literal` when they prevent string bugs.
- Frozen dataclasses/slots for value objects; mutable is fine for scratch code.
- Pydantic only at untrusted I/O boundaries; plain dataclasses/dicts otherwise.
- Avoid gratuitous `Any`; use `object`/generics when easy. Do not block on it.

## Control Flow

- Guard clauses; avoid deep nesting.
- `match`/`case` when it clarifies state routing.
- Never `return`/`break`/`continue` in a `finally` block.
- No bare `except:`. Catch specific exceptions; chain when adding context.
- One source of truth for gates, metrics, config, and capability flags.

## Modules, I/O, Logging

- No heavy work or I/O at import time. Use a `main()`/CLI entry point.
- Atomic writes for results; resume from checkpoints. Never lose a run to a crash.
- `logging` or `print` are both fine for research scripts; include enough context
  to debug. Parameterize SQL; never interpolate untrusted input into a query.
- Context managers for files, arrays, and GPU resources.
- Short docstrings on public modules/APIs.

## Async

- Use only when I/O concurrency is the bottleneck. `TaskGroup` if used;
  `asyncio.to_thread` for blocking calls. Not needed for single-GPU training.
