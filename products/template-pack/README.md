# Template Pack: exact structural memory for a repo

A `.tpack` stores the repeated **structure** of a code repository:

- template keys: exact fixed bytes plus typed holes (identifiers, numbers,
  strings, dotted names, bracket groups);
- counts and provenance: which files use each template;
- fill values: the exact atoms that fill the holes.

The pack is inspectable and searchable. It is not a compressor: the
research codec reaches size parity only at a narrow frontier, and the pack
can be a large fraction of the source. The product value is **audit,
search, and installable structure**, not a smaller archive.

## Build

```bash
uv run cml tpack --repo path/to/repo --out repo.tpack
uv run cml tpack --repo path/to/repo --out repo.tpack --search "def "
```

## What it is for

- **Audit**: read the exact repeated shapes in a codebase and where they
  occur.
- **Style and boilerplate**: find the high-count templates that dominate
  a repo.
- **Installable structure**: the pack is a file you can diff, merge, and
  ship next to a frozen model.
- **Provenance**: every template lists source files; every fill is exact.

## Current status

Prototype. The pack is JSON, the search is literal, and the table is not
MDL-pruned per file. The next step is to expose the pack as an MCP
resource and to add a template-level diff between two versions of a repo.
