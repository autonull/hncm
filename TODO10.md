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

`79 passed`. The bytes for the repo index are stored gzip-wrapped when
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

**Resolved: verify, do not predict.** The unverified pointer tops out at
66-67% span precision at ~5% coverage (R0-R9): the gate cannot predict
correctness above source precision. R10 makes the pointer a candidate
ranker: top-8 proposals, an exact match check, and a copy only for
verified spans. Gate **PASS**: 38.6% coverage at 100% verified
precision, 1.56x; the cheap point is gate 0.7 at 19.3%, 1.22x. A
rejected proposal costs a byte comparison, not a forward.

Pass gate: **>90% exact-span precision at ≥5% coverage and >1.2x
bytes/step**, with every copied byte exact. Trunk `-q2048`; 2 MB code,
2,000 steps per run.

**Repair loop (one variable per run).**

| id | change | gate 0.9 span precision | coverage | bytes/step |
|---|---|---:|---:|---:|
| R0 | frozen trunk, copyability gate | 51.1% | 4.2% | 0.95x |
| R1 | joint, lr 1e-4 | 0% (no fire) | 0% | 1.00x |
| R2 | joint, lr 1e-3, copyability gate | 27.7% | 11.0% | 0.77x |
| R3 | separate lrs, pointer-correctness gate | 45.8% | 5.4% | 0.93x |
| R4 | exact-span gate | 62.0% | 5.0% | 0.96x |
| R5 | run head at predicted source | 61.9% | 4.7% | 0.97x |
| R6 | run head conditioned on source | 63.7% | 4.9% | 0.97x |

- R1 is an underfit/control failure, not a verdict: the head can fire at
  lr 1e-3, and the block needs a smaller rate than the head.
- R4 gains precision by labeling the gate with the exact copy outcome.
- R5/R6 remove the run-length gap at high confidence: at gate 0.99 the
  span is 96.3% and the source is 100%, but coverage is 0.85%.

**Repair result (R7-R9).** Three coverage repairs miss the gate:

| id | change | span precision @ gate 0.9 | coverage | bytes/step |
|---|---|---:|---:|---:|
| R7 | multi-positive pointer loss | 59.1% | 5.7% | 0.96x |
| R8 | R7 + 2x windows and steps | 67.5% | 5.2% | 0.97x |
| R9 | R8 + 2x pointer width | 66.1% | 5.1% | 0.97x |

At gate 0.99, R9 is 98.0% span at 98.1% source, 1.2% coverage, 1.00x.
The unverified operating point stalls at 66-67% precision at ~5%.

**Verified decode (R10) — gate PASS.** Propose top-8, verify, copy only
exact spans. A wrong proposal costs a byte comparison, not a forward.

| gate | coverage | mean span | top-1 source | checks/B | bytes/step |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 38.6% | 14.6 | 2.7% | 3.41 | 1.56x |
| 0.50 | 26.6% | 16.1 | 21.7% | 0.46 | 1.33x |
| 0.70 | 19.3% | 17.4 | 41.2% | 0.19 | 1.22x |
| 0.90 | 9.0% | 18.0 | 77.6% | 0.05 | 1.09x |

The pointer is a candidate ranker; exact verification does the
correctness work. **B1 closes with a pass.** The run head is redundant at
decode; it only supplies the gate training label now.

**B2 resolved by exact proposers (same held-out test).**

| proposer | coverage | checks/B | bytes/step |
|---|---:|---:|---:|
| hash over the 2 KB window | 47.0% | 0.15 | 1.78x |
| hash over a fixed 2 MB train pack | **78.4%** | 0.58 | **3.59x** |
| learned top-8 pointer | 38.6% | 3.41 | 1.56x |

Any exact copy of >=8 B shares an 8-byte key, so a hash over that key is a
complete in-window proposer. The learned pointer is dominated. A 2 MB pack
of disjoint train code lifts coverage to 78.4% at 3.59x. **B2 is an exact
pack plus a hash**, which `repo` memory already supplies; do not train a
bigger pointer. The byte LM keeps the 21.6% novel bytes. The pack is
required for the headline; do not claim 78.4% standalone.

### Breakthrough path: TF-LM

An exact span is one event. The broader unit is a token template with
typed holes: fixed syntax is exact, atoms are fills. Probe
(`cml template-probe`, `cml/reports/TEMPLATES.md`):

- **22.4%** of held-out bytes is fixed syntax, **27.8%** holes,
  **49.8%** literal; **0.218 decisions/B**, **4.58 B/decision**.
- 18,925 templates + 15,873 fills, 894 KB table, exact round-trip.
- Of hole bytes, **29.7% are new slot combinations with a known fill**,
  12.1% repeat a seen slot, 58.2% are novel values. The new combinations
  are the capability target: a flat memorizer must learn each pair; a
  template model shares the slot.

**Architecture.** Bytes -> tokenizer -> template parse -> events
`T(key, fills)` or `L(token)`. A decoder-only transformer runs on events:
template embedding from the fixed tokens + hole types, fill embedding
from content, heads for next template / per-slot fill / byte fallback.
Exact output by lookup; a digest verifies each event. One forward per
event, not per byte.

**Hyperparameters.** d512, 12 layers, 8 heads, 512 events, batch 8,
AdamW lr 3e-4, 20k steps on 8 MB code; library 16k templates + 16k
fills; anti-unification lifts multi-line templates after T1.

**Decisive gate T1.** Hold out 20% of (template, slot, fill) pairs while
each held-out fill still appears in other slots. Gate: 100% exact
round-trip, <=0.03 bpb regression vs the byte trunk, >=+10 points exact
fill accuracy on held-out combinations, >=2x bytes/decision. T2: install
a repo-A library on repo B, empty-library control, >=0.15 bpb.

**T1 pilot (500 steps, `cml event-lm`).** Template event model reaches
**4.8% new-combination accuracy vs 1.4% for type-count** and 13.0%
overall fill accuracy. Bits/byte is **1.77** on template-covered bytes:
predicting an exact fill ID is expensive. The arm has the capability
signal but not the size. The next lever is **long fills (>=8 B) with a
local copy pointer**, not a larger fill softmax. A 500 KB probe estimate:
8 B fill gate, 5.12 B/decision, **0.957 bpb** with an 11-bit local
pointer and a 1 bpb literal fallback. That is a narrow pass; implement
the exact codec before claiming it.

**Repair rules.** A failed run identifies a term, not the interface.
Change one term, record a row, rerun. Price a rejected proposal as a byte
comparison, not a forward: a proposal is not an emitted copy. When a
learned proposer loses to an exact one, record the domination and stop.

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
| ~~**B1**~~ | joint copy head — repair loop | `cml copy-lm --joint` | **shipped; gate PASS** on verified top-8: 38.6% at 1.56x, 100% exact | 1 GPU-h |
| ~~**B2**~~ | exact pack proposer | `_pack_oracle` + `repo` | **resolved; hash dominates**: 78.4% at 3.59x with a 2 MB pack | CPU |
| ~~**T0**~~ | template-fill headroom | `cml template-probe` | **shipped**: 4.58 B/decision, 29.7% new combinations | 1 CPU-day |
| ~~**T1a**~~ | event LM pilot | `cml event-lm` | **signal**: 4.8% new combos vs 1.4%; bpb 1.77, fill IDs expensive | 1 GPU-h |
| **T1b** | long-fill copy codec | `cml event-lm --min-fill 8 --copy` | estimate 0.957 bpb; gate <=0.03 regression, exact | 1 GPU-day |
| **T2** | library install transfer | `cml template-lm --install` | >=0.15 bpb from installation | 1 GPU-day |
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

Shipped this session: **T0 template probe + T1a event pilot**. The new
unit is a template with typed holes. The event arm reaches **4.8% new-
combination accuracy vs 1.4% for type-count**, so template context has a
signal. Its bits/byte is 1.77 because exact fill IDs are expensive; the
next lever is **long fills with a local copy pointer**, not a larger fill
softmax. A 500 KB estimate puts the long-fill codec at 0.957 bpb, near
the 0.97 byte trunk.

Next session, one of:
1. **T1b**: long-fill copy codec. Gate: exact round-trip, <=0.03 bpb
   regression, >=2x bytes/decision. This is the decisive size test.
2. **T2**: install a repo-A library on repo B; >=0.15 bpb.
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
uv run cml copy-lm --joint --steps 3000 --lr 0.001 --block-lr 1e-5 \
  --pointer-scale 2 --verify-topk 8 --train-bytes 2000000 \
  --eval-bytes 100000 --windows 400
uv run cml template-probe --train-bytes 2000000 --test-bytes 100000
uv run cml event-lm --train-bytes 2000000 --test-bytes 100000 \
  --steps 500 --seq 128 --d 192 --layers 4 --batch 32
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
