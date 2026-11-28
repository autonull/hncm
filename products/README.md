# Products

Each product is self-contained: its own README explains what it is, how to
run it, what it costs, and what it does not do. Products share utilities
under `cml/` (the codec, the trunk and the CLI) but not documentation.

| product | directory | one line |
|---|---|---|
| codec | [`codec/`](codec/README.md) | lossless byte codec for code, 6.3x exact, 9.4x frontier |
| delta | [`delta/`](delta/README.md) | exact version compression, 56.6x on hunk edits, CPU-only |
| memory | [`memory/`](memory/README.md) | exact history compression, 40x over 1,823 real versions, CPU-only |
| repo-memory | [`repo-memory/`](repo-memory/README.md) | exact searchable repo index, 12.5x on 80 files, `path:line` hits |
