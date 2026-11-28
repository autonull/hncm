# TODO10 — exact memory products + ByteCopy LM

Status: 2026-11-28. Continuation of `TODO9.md`. `TODO9` is the research
record; this file is the working plan. Keep it short and update it at each
commit.

## 1. Shipped — do not regress

| product | command | measured | tests |
|---|---|---|---|
| repo-memory | `cml repo` | 80 files, 9.84 MB → 476,080 B = **20.7x**; 7.2x per-file gzip; 34 s | `test_repo_memory.py` |
| memory-mcp | `cml mcp --index` | 5 exact tools: files, search, read, versions, diff | `test_mcp.py` |
| patch-guard | `cml patch` | exact apply or hard error | `test_tools.py` |
| repo-delta | `cml repo-delta` | 22,830 → 266 B = **85.8x**, verified | manual |
| log-delta | `cml logdelta` | 5 logs, 504,465 → 18,445 B = **27.3x**, 1.8x gzip | `test_tools.py` |
| pack-doctor | `cml pack fit` | 1,024 → 498 paying spans, mean 12.9 B | manual |
| ByteCopy LM | `cml copy-lm` | 403k head, exact copies; 51% precision at gate 0.9, 0.96x | `test_copylm.py` |
| repo global mode | `cml repo --global` | 80 files → 390,024 B = **25.2x**, 8.8x per-file gzip | manual |
| repo incremental | `cml repo --update` | appends new versions; per-file chains byte-identical to a full rebuild; unchanged chains are not re-encoded | `test_repo_memory.py` |
| MCP resources | `cml mcp --index` | 5 exact tools + `repo://path[@n]` resources and a version template | `test_mcp.py` |
| global checkpoints | `cml repo --global --checkpoint 32` | 80 files → 482,502 B = **20.4x**; 41 snapshots, access ≤31 deltas; cold latest 0.25 s vs 20.9 s full decode | `test_repo_memory.py` |
| context search | `cml repo --search q --context 2 --hits-json` | ordered `path:line` hits, exact context lines, deduped per line | `test_repo_memory.py` |
| log-delta real logs | `cml logdelta --dir <logs>` | near 32,190 → 1,773 B = **18.2x**, changed 8,548 → 3,268 B = **2.6x**; gzip is 2.6x/3.1x smaller, so no compression claim | `LOG-DELTA-REAL.md` |

`69 passed`. The bytes for the repo index are stored gzip-wrapped when
smaller; `memory.load_chain` reads both.

## 2. The model invention: ByteCopy LM

**Design.** A byte-level trunk has three heads: a byte head for novel
bytes, a **copy head** (pointer over the context), and a **run head**
(span length). A gate picks copy or compute. Training is self-supervised:
the label is the longest prior match in the context (min 8 B, cap 64 B).
A copied span is exact; the digest is the guard, not the model. At
inference a copy updates the cache in one chunked forward, so the small
model spends capacity on novel bytes. No tokenizer, no flat span id, no
pack charge.

**Why a smaller model can win.** Code repeats long exact spans. A 20M
model that delegates them to a pointer keeps capacity for novel bytes,
gets exact recall for free, and avoids a tokenizer mismatch.

**Current bottleneck.** The first B1 run (lr 1e-4) did not fire the gate;
joint lr 1e-3 degraded the pointer. The gate now trains on exact-span
correctness and the run head on the length at the predicted source. At
gate 0.90 the pointer source is 75.7% and the exact span 63.7%; at 0.99
the source is 100% and the span 96.3% at 0.85% coverage. The exact-copy
path works; pointer precision at usable coverage is the open term.

Pass gate: **>90% exact-span precision at ≥5% coverage and >1.2x
bytes/step**, with every copied byte exact. Trunk `-q2048`; 2 MB code,
2,000 steps per run.

**Repair loop (one variable per run).**

| id | change | gate 0.9 span precision | coverage | bytes/step |
|---|---|---:|---:|---:|
| R0 | frozen trunk, copyability gate | 51.1% | 4.2% | 0.95x |
| R1 | joint block, copyability gate, lr 1e-4 | 0% (no fire) | 0% | 1.00x |
| R2 | separate lrs, pointer-correctness gate | 45.8% | 5.4% | 0.93x |
| R3 | exact-span gate | 62.0% | 5.0% | 0.96x |
| R4 | run head at predicted source | 61.9% | 4.7% | 0.97x |
| R5 | run head conditioned on source | 63.7% | 4.9% | 0.97x |

- R1 is an underfit/control failure, not a verdict: the head can fire at
  lr 1e-3, and the block needs a smaller rate than the head.
- R3 gains precision by labeling the gate with the exact copy outcome.
- R4/R5 remove the run-length gap at high confidence: at gate 0.99 the
  span is 96.3% and the source is 100%, but coverage is 0.85%.

**Next repair: pointer coverage.** The target precision exists at gate
0.99, but not at ≥5% coverage. Try, in order: (a) multi-positive pointer
loss over all exact prior matches, (b) more windows/steps, (c) wider
pointer. Keep the run head fixed. B2 still tests the retrieval-context
path. One GPU job at a time; no new base.

**Repair rules.** A failed run identifies a term, not the interface.
Change one term, record a row, rerun. Stop only after two independent
repairs cannot exceed 1.1x at ≥5% coverage, and the retrieval-context path
also fails.

## 3. Product improvements (each ≤1 day, in order)

1. ~~**repo-memory incremental (`--update`).**~~ **Shipped.** Reads the
   manifest HEAD; unchanged HEAD is a no-op. Per-file chains append only
   the new delta payloads and are byte-identical to a full rebuild. Global
   mode appends at the tail and records non-contiguous `ranges`; exact
   bytes, slightly worse ratio than a rebuild. History rewrites stop with a
   clear error.
2. ~~**MCP resources and prompts.**~~ **Shipped.** Each file is a
   `repo://path` resource; `repo://{path}@{version}` is the template; the
   five tools are unchanged. Test: `tests/test_mcp.py`; config and limits
   in `cml/reports/MCP.md`.
3. ~~**repo-memory context search.**~~ **Shipped.** `--context N` prints
   grep-style context; `--hits-json PATH` writes ordered hits; hits are
   deduped per line and capped by `--max-hits`. `--context 1` on the
   FastAPI index returned exact surrounding lines.
4. ~~**repo global checkpoints.**~~ **Shipped.** `--global --checkpoint K`
   writes a gzip snapshot every K versions; decode starts at the nearest
   snapshot. At K=32 the 80-file sample is 20.4x with 41 snapshots and
   access ≤31 deltas. Rebuilding an index with `--update --checkpoint K`
   adds snapshots without re-encoding the chain.
5. ~~**log-delta on real logs.**~~ **Recorded stop.** Real pytest logs:
   near-identical 32,190 → 1,773 B = 18.2x, changed 8,548 → 3,268 B =
   2.6x. Both raw-ratio gates pass, but gzip over the concatenation is
   2.6x/3.1x smaller. Keep the chain for exact version replay only; do not
   claim compression. Details: `cml/reports/LOG-DELTA-REAL.md`.
6. **pack-doctor end-to-end.** Fit the pack, then run
   `cml memory-probe --oracle` and the count-prior priced oracle on the
   held-out test. Gate: positive net on the test.
7. **repo-delta multi-file.** `cml repo-delta --before a --after b`
   walks the tree, emits one exact patch set and a manifest. Gate: apply
  + verify on 100 random commit pairs.
8. **patch-guard breadth.** `git apply` mode, binary patches, 100 random
   diffs. Gate: exact or a clear refusal, never a silent wrong byte.

## 4. Small experiment queue

Run one row at a time, ≤1 GPU-h or ≤1 CPU-day, record a row here.

| id | hypothesis | command | pass signal | cost |
|---|---|---|---|---|
| **B1** | joint copy head — repair loop | `cml copy-lm --joint` | 96.3% span at 0.85%; next: pointer at ≥5% | 1 GPU-h |
| **B2** | pointer into a retrieved exact pack | `cml copy-lm --pack <repo-index>` | exact recall + >1.2x | 1 GPU-h |
| ~~**R1**~~ | incremental index — **shipped** | `cml repo --update` | exact, no rebuild | 1 day |
| ~~**R2**~~ | global checkpoints — **shipped** | `cml repo --global --checkpoint 32` | 20.4x, access ≤31 | 1 day |
| ~~**M1**~~ | MCP resources — **shipped** | `cml mcp --index` | client lists/reads resources | 1 day |
| ~~**L1**~~ | real CI logs — **done; gzip wins** | `cml logdelta --dir <logs>` | 18.2x near / 2.6x changed; stop compression claim | 1 day |
| **P1** | fitted pack pays | `cml pack fit` + `memory-probe` | priced total < stream | 3 h |

## 5. What not to do

- No more flat span-id bpb probes. A0.2 killed that interface; keep the
  result in `TODO9`.
- No 3-5 day run before a ≤1 h control passes.
- No new base training; use the `-q2048` trunk.
- No claim without a digest or an exact round-trip.
- No product headline that loses to the fair baseline; name the unit
  (per-file vs global) when comparing to gzip.

## 6. Definition of done

Shipped this session: **R1/M1/R2 + context search + L1** — incremental
updates, MCP resources, global checkpoints (20.4x, access ≤31), ordered
exact search with context and JSON hits, and real-log evidence. L1 is a
recorded stop: the ratio gate passes, but gzip wins the fair baseline.

Next session, one of:
1. **B1 repair loop**: pointer coverage at ≥5%. Try the multi-positive
   pointer loss first; keep the run head fixed. Do not rerun the old B1.
2. **B2**: pointer into a retrieved exact pack, after B1 reaches the gate.
3. **pack-doctor end-to-end** (P1): fit, then price the held-out test.
4. **repo-delta multi-file**: exact patch sets on 100 commit pairs.

Then: commit, update this file's status, and move the result into
`TODO9.md` if it is a research finding.

## 7. Commands

```bash
cd cml
uv run pytest tests
uv run cml repo --repo ~/.cache/hncm/edit/repos/tiangolo__fastapi.git \
  --out /tmp/fastapi-index --limit 80
uv run cml repo --repo ~/.cache/hncm/edit/repos/tiangolo__fastapi.git \
  --out /tmp/fastapi-index --update
uv run cml repo --index /tmp/fastapi-index --search "FastAPI"
uv run cml repo --index /tmp/fastapi-index --search "FastAPI" \
  --context 2 --max-hits 20 --hits-json /tmp/hits.json
uv run cml repo --repo ... --out /tmp/g --global --limit 80
uv run cml repo --repo ... --out /tmp/g --global --checkpoint 32 --limit 80
uv run cml mcp --index /tmp/fastapi-index
uv run cml copy-lm --steps 5000 --train-bytes 2000000 \
  --eval-bytes 100000 --windows 200
uv run cml copy-lm --joint --steps 2000 --lr 0.001 --block-lr 1e-5 \
  --train-bytes 2000000 --eval-bytes 100000 --windows 200
uv run cml pack fit runs/a02-1024/artifacts/pack.bin \
  --bpb 0.974 --id-bits 9 --units 1024 --min-len 8
uv run cml logdelta --dir /tmp/logs-demo --out /tmp/logs.cmlh
```

## References

- `TODO9.md`: research record; A0.0/A0.1/A0.2, the id-price defect.
- `products/README.md`: the product map.
- `products/copy-lm/README.md`: ByteCopy LM design and prototype.
- `cml/reports/COPY-LM.md`, `cml/reports/COPY-LM-JOINT.md`,
  `cml/reports/MEMORY-GATE.md`.
- `cml/reports/REPO-INDEX.md`, `cml/reports/REPO-GLOBAL.md`,
  `cml/reports/REPO-CHECKPOINT.md`, `cml/reports/LOG-DELTA-REAL.md`,
  `cml/reports/PACK-FIT.md`.
- `cml/reports/MCP.md`: tools, resources, OpenCode config, and limits.
