# TODO11 — template products and the exact memory suite

Status: 2026-11-28. Supersedes `TODO10.md` as the working plan. `TODO10`
stays as the record of B1/T0-T1d/TP1. `TODO9.md` is the research record.

## 0. Fresh context: start here

- `85 passed`; last code commit `db2d3d7`.
- Run `cd cml && uv run pytest tests` before touching anything.
- Read, in order: `TODO10.md` §2 (copy/template results), `cml/reports/
  FILL-COPY.md`, `LITERAL-LM.md`, `TPACK.md`, and
  `products/template-pack/README.md`.
- Do **not** rerun T1a-T1d. Their rows are recorded.
- First task: **P2, Template Pack MCP + diff** (CPU, ≤1 day).

The exact-memory products are green. The template line has a working
exact codec and a shipped product. The open work is productization,
install transfer, and two long-standing code tasks.

## 1. Shipped — do not regress

| product | command | measured | tests |
|---|---|---|---|
| repo-memory | `cml repo` | 80 files, 9.84 MB → 476,080 B = **20.7x** | `test_repo_memory.py` |
| repo global | `cml repo --global` | 80 files → 390,024 B = **25.2x** | manual |
| checkpoints | `cml repo --global --checkpoint 32` | 20.4x, 41 snapshots, access ≤31 | `test_repo_memory.py` |
| memory-mcp | `cml mcp --index` | 5 exact tools + `repo://` resources | `test_mcp.py` |
| patch-guard | `cml patch` | exact apply or hard error | `test_tools.py` |
| repo-delta | `cml repo-delta` | 22,830 → 266 B = **85.8x**, verified | manual |
| log-delta | `cml logdelta` | 27.3x raw; gzip wins the fair baseline | `LOG-DELTA-REAL.md` |
| pack-doctor | `cml pack fit` | 1,024 → 498 paying spans, mean 12.9 B | manual |
| verified copy | `cml copy-lm --verify-topk` | 38.6% at 100% exact, **1.56x** | `test_copylm.py` |
| template codec | `cml fill-copy` | exact; 0.122 B/B at 0.968 literals; 97.6% literal | `test_fillcopy.py` |
| Template Pack | `cml tpack` | 2,990 templates, 7,318 fills, 844 KB; search + provenance | `test_tpack.py` |

## 2. What the research established

- Exact copying is a **data-structure** problem: an 8-byte hash is a
  complete in-window proposer. The learned pointer is dominated.
- A fixed 2 MB pack reaches **78.4% coverage at 3.59x**; the pack is a
  required artifact, not a standalone compression claim.
- The useful template unit is **exact fixed syntax plus typed holes**:
  22.4% fixed, 27.8% holes, 4.58 B/decision, 29.7% of hole bytes are new
  slot combinations.
- The event model has a compositional signal: **4.8% new-combination
  accuracy vs 1.4% type-count**.
- The exact long-fill codec is at size parity on the blob (**0.122-0.126
  B/B**) and compute-cheap; the table charge is the remaining product
  term.
- Event-conditioned literals gain only **+0.0067 bpb** at 2.4M params;
  the 0.03 gate needs scale or a stronger byte head.
- The shipped product is **Template Pack**: exact structure, provenance,
  search. It is an audit/install artifact, not a compressor.

## 3. Plan (ordered)

1. **P2 — Template Pack MCP + diff** (CPU, 1 day).
   - Expose each pack as `template://path` and a `template://{path}@{hash}`
     version template; add a pack list/search resource.
   - Add `cml tpack --diff a.tpack b.tpack`. Gate: exact added/removed/
     changed template sets; MCP client lists and reads resources.
   - Files: `cml/tpack.py`, `cml/mcp_server.py`, `tests/test_tpack.py`,
     `tests/test_mcp.py`. Keep the five repo tools unchanged.
2. **T2 — install transfer** (GPU, 1 day).
   - Build a pack on repo A; parse repo B with A's templates/fills.
   - Compare literal bpb against B's own pack and empty. Gate: ≥0.15 bpb
     from installation alone; parse exact.
   - Planned CLI: `cml tpack --score --pack a.tpack --repo b`.
3. **T1e — literal scale** (GPU, 1 day), only if T2 shows transfer.
   - Mixed vs ablated at 8M/12 layers/4k steps. Gate: +0.03 bpb.
   - If two scales miss, stop the size claim and keep the pack product.
4. **P1 — pack-doctor end to end** (3 h).
   - `cml pack fit` the shipped counted pack, then price the held-out
     test. Note: `memory-probe --oracle` uses a corpus lexicon, not the
     fitted pack; add a pack-probe path before claiming the gate.
   - Gate: positive net on the test.
5. **RD1 — repo-delta multi-file** (1 day).
   - `cml repo-delta --repo R --path P --before A --after B --out O`.
   - Gate: apply + verify on 100 random commit pairs.
6. **PG1 — patch-guard breadth** (1 day).
   - `git apply` mode, binary patches, 100 random diffs.
   - Gate: exact or a clear refusal, never a silent wrong byte.

## 4. Queue

| id | hypothesis | command | pass signal | cost |
|---|---|---|---|---|
| **P2** | Template Pack over MCP | `cml tpack --diff` + MCP | list/read + exact diff | 1 CPU-day |
| **T2** | pack install transfer | `cml tpack --score --pack a --repo b` | ≥0.15 bpb, exact | 1 GPU-day |
| **T1e** | literal model scales | `cml literal-lm --steps 4000 --d 384` | +0.03 bpb vs ablated | 1 GPU-day |
| **P1** | fitted pack pays | `cml pack fit` + pack probe | positive net | 3 h |
| **RD1** | multi-file repo delta | `cml repo-delta --repo --path` | 100 pairs apply+verify | 1 CPU-day |
| **PG1** | patch-guard breadth | `cml patch --base --diff` | exact or refusal | 1 CPU-day |
| ~~**TP1**~~ | Template Pack | `cml tpack` | **shipped** | done |
| ~~**T1a-d**~~ | event/template pilots | `cml event-lm`, `literal-lm`, `fill-copy` | **recorded** | done |
| ~~**B1/B2**~~ | copy head / exact pack | `cml copy-lm` | **recorded; hash dominates** | done |

## 5. What not to do

- No more pointer capacity or pointer training; the hash dominates exact
  spans.
- Do not call the 844 KB Template Pack a compressor; name the pack and
  the table charge.
- Do not claim the 78.4% pack coverage without the 2 MB pack.
- Do not charge a repo table per 100 KB file; state the unit.
- No new base training; use the `-q2048` trunk.
- No 3-5 day run before a ≤1 h control passes.
- No claim without an exact round-trip or a digest.

## 6. Definition of done

This session shipped **T0/T1a/T1b/T1c/T1d + Template Pack** and closed the
copy-head line. The exact codec is at blob parity; the pack is a working
product. The next session is **P2**, then **T2**; P1/RD1/PG1 remain.

If T2 and T1e both miss, record the template line as a compute/audit
product, not a size win, and keep the exact memory suite as the main
product.

## 7. Commands

```bash
cd cml
uv run pytest tests
uv run cml tpack --repo /home/me/hncm/cml --out /tmp/cml.tpack --search "def "
uv run cml tpack --diff /tmp/a.tpack /tmp/b.tpack        # P2, to add
uv run cml literal-lm --train-bytes 2000000 --test-bytes 100000 \
  --steps 2000 --seq 256 --d 192 --layers 4 --batch 32
uv run cml fill-copy --train-bytes 2000000 --test-bytes 100000 \
  --min-fill 16 --min-count 16 --max-n 10 --max-holes 2 --min-fixed 16
uv run cml event-lm --train-bytes 2000000 --test-bytes 100000 \
  --steps 500 --seq 128 --d 192 --layers 4 --batch 32
uv run cml pack fit runs/a02-1024/artifacts/pack.bin \
  --bpb 0.974 --id-bits 9 --units 1024 --min-len 8
uv run cml memory-probe --oracle --bytes 756752 --sizes 256,1024,4096
uv run cml repo-delta --repo R --path P --before A --after B --out O
uv run cml patch --base B --diff D --expect E
```

## References

- `TODO10.md`: previous plan and the B1/T0-T1d/TP1 record.
- `TODO9.md` §15-16: copy-head and template-fill research.
- `cml/reports/TEMPLATES.md`, `EVENT-LM.md`, `FILL-COPY.md`,
  `LITERAL-LM.md`, `TPACK.md`.
- `products/template-pack/README.md`: product surface and limits.
- `cml/reports/MCP.md`, `REPO-INDEX.md`, `PACK-FIT.md`.
- `cml/reports/COPY-LM-JOINT.md`: verified copy result.
