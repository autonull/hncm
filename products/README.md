# Products

Each product is self-contained: its own README explains what it is, how to
run it, what it costs, and what it does not do. Products share utilities
under `cml/` (the codec, the trunk and the CLI) but not documentation.

| product | directory | one line |
|---|---|---|
| codec | [`codec/`](codec/README.md) | lossless byte codec for code, 6.3x exact, 9.4x frontier |
| delta | [`delta/`](delta/README.md) | exact version compression, 56.6x on hunk edits, CPU-only |
| memory | [`memory/`](memory/README.md) | exact history compression, 40x over 1,823 real versions, CPU-only |
| repo-memory | [`repo-memory/`](repo-memory/README.md) | exact searchable repo index, 20.7x on 80 files, `path:line` hits |
| memory-mcp | [`memory-mcp/`](memory-mcp/README.md) | MCP server: repo search, read, versions, diff |
| patch-guard | [`patch-guard/`](patch-guard/README.md) | apply a diff, refuse unless byte-exact |
| repo-delta | [`repo-delta/`](repo-delta/README.md) | exact delta between two commits, 85.8x on `README.md` |
| log-delta | [`log-delta/`](log-delta/README.md) | exact CI log history, 27.3x, 1.8x better than gzip |
| pack-doctor | [`pack-doctor/`](pack-doctor/README.md) | rank spans by net bits at the measured bpb |
| copy-lm | [`copy-lm/`](copy-lm/README.md) | byte LM with an exact copy head (research prototype) |
