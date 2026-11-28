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

`68 passed`. The bytes for the repo index are stored gzip-wrapped when
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

**Current bottleneck.** The head is trained on frozen last states. At gate
0.9 it reaches 51% copy precision and 0.96x bytes/step; the gate is not
calibrated. The exact copy path itself works.

**Next experiment B1 (≤1 GPU-h, one job).**
- Trunk `codec-byte-model-code-756752-d448-l8-q2048`.
- Unfreeze the last block + copy head; keep the rest frozen.
- Add the relative-position bias (already in `copylm.CopyHead`) and a
  top-2 pointer-margin feature to the gate.
- Train 2,000 steps on 2 MB code, batch 4 windows, lr 1e-4.
- Gate: **>90% copy precision at ≥5% coverage**, and **>1.2x bytes/step**
  with all copied bytes exact. Cost: 1 GPU-h.
- If B1 fails: stop the head-only formulation. Do not scale the head.
- If B1 passes: B2 appends 4k exact spans from `repo-memory` to the
  context, points into both, and measures exact recall and speed at 2K
  context. Cost: 1 GPU-h.

**Kill rules.** No >1.1x at ≥5% coverage after B1 and B2 → the copy-head
path is not a win; keep the retrieval-context path (`memory-mcp`) and stop
model work. One GPU job at a time; no new base.

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
5. **log-delta on real logs.** Mine or generate pytest CI logs; gate
   ≥10x on near-identical and ≥2x on changed logs. Synthetic sample is
   27.3x; this is the real test.
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
| **B1** | joint last block + margin gate calibrates copy | `cml copy-lm --joint` | >90% precision at ≥5%, >1.2x | 1 GPU-h |
| **B2** | pointer into a retrieved exact pack | `cml copy-lm --pack <repo-index>` | exact recall + >1.2x | 1 GPU-h |
| ~~**R1**~~ | incremental index — **shipped** | `cml repo --update` | exact, no rebuild | 1 day |
| ~~**R2**~~ | global checkpoints — **shipped** | `cml repo --global --checkpoint 32` | 20.4x, access ≤31 | 1 day |
| ~~**M1**~~ | MCP resources — **shipped** | `cml mcp --index` | client lists/reads resources | 1 day |
| **L1** | real CI logs | `cml logdelta --dir <logs>` | ≥10x near-identical | 1 day |
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

Shipped this session: **R1/M1/R2 + context search** — incremental updates,
MCP resources, global checkpoints (20.4x, access ≤31), and ordered exact
search with context and JSON hits.

Next session, one of:
1. **L1**: real-log gate pass (≥10x near-identical, ≥2x changed). Gather
   pytest CI logs first; the synthetic 27.3x is not evidence.
2. **B1/B2**: the ByteCopy gate, unchanged. B1 is the cheap control; run
   it before any head scaling.
3. **pack-doctor end-to-end** (P1): fit, then price the held-out test.

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
uv run cml pack fit runs/a02-1024/artifacts/pack.bin \
  --bpb 0.974 --id-bits 9 --units 1024 --min-len 8
uv run cml logdelta --dir /tmp/logs-demo --out /tmp/logs.cmlh
```

## References

- `TODO9.md`: research record; A0.0/A0.1/A0.2, the id-price defect.
- `products/README.md`: the product map.
- `products/copy-lm/README.md`: ByteCopy LM design and prototype.
- `cml/reports/COPY-LM.md`, `cml/reports/MEMORY-GATE.md`.
- `cml/reports/REPO-INDEX.md`, `cml/reports/REPO-GLOBAL.md`,
  `cml/reports/REPO-CHECKPOINT.md`, `cml/reports/LOG-DELTA.md`,
  `cml/reports/PACK-FIT.md`.
- `cml/reports/MCP.md`: tools, resources, OpenCode config, and limits.
