"""Template Pack: exact code structure as an inspectable, searchable artifact.

A pack stores template keys (fixed bytes and typed holes), their counts,
provenance, and the fill values seen in a repo.  It is not a compression
artifact; it is a structural memory for audit, search, and install.
"""

from __future__ import annotations

import base64
import json
from collections import Counter, defaultdict
from pathlib import Path

import cml
import fillcopy
import templates

TEXT = {".py", ".md", ".toml", ".json", ".txt", ".cfg", ".ini", ".yaml", ".yml"}
SKIP = {
    "runs",
    "reports",
    "__pycache__",
    ".git",
    ".venv",
    ".pytest_cache",
    "node_modules",
}


def _files(root: Path, max_bytes: int) -> list[Path]:
    out: list[Path] = []
    total = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT:
            continue
        if any(part in SKIP or part.startswith(".") for part in path.parts):
            continue
        size = path.stat().st_size
        if total + size > max_bytes:
            break
        out.append(path)
        total += size
    return out


def _enc(key: tuple[bytes, ...]) -> list[dict[str, str]]:
    return [
        {"f": base64.b64encode(e[1:]).decode("ascii")}
        if e.startswith(b"F")
        else {"h": e[1:].decode("ascii")}
        for e in key
    ]


def _dec(e: dict[str, str]) -> bytes:
    if "f" in e:
        return base64.b64decode(e["f"])
    return b"H" + e["h"].encode("ascii")


def build(
    root: Path,
    min_fill: int = 8,
    min_count: int = 3,
    max_n: int = 6,
    max_bytes: int = 2_000_000,
) -> dict:
    files = _files(root, max_bytes)
    if not files:
        raise SystemExit(f"tpack: no text files under {root}")
    raws = {rel: (root / rel).read_bytes() for rel in files}
    train = b"\n".join(raws.values())
    lib = fillcopy.induce(train, min_fill, min_count, max_n)
    counts: Counter[tuple[bytes, ...]] = Counter()
    prov: dict[tuple[bytes, ...], set[str]] = defaultdict(set)
    fills: Counter[bytes] = Counter()
    for rel, raw in raws.items():
        name = str(rel)
        for event in fillcopy.parse(raw, lib, min_fill, max_n):
            if not isinstance(event, templates.Hit):
                continue
            counts[event.key] += 1
            prov[event.key].add(name)
            fills.update(event.fills)
    order = sorted(counts, key=lambda key: (-counts[key], key))
    return {
        "root": str(root),
        "files": len(files),
        "bytes": len(train),
        "templates": [
            {
                "key": _enc(key),
                "count": counts[key],
                "files": sorted(prov[key])[:8],
            }
            for key in order
        ],
        "fills": [
            {"value": value.decode("latin-1"), "count": count}
            for value, count in fills.most_common(20_000)
        ],
    }


def _show(key: list[dict[str, str]]) -> str:
    out = []
    for element in key:
        if "f" in element:
            out.append(_dec(element).decode("utf-8", "replace"))
        else:
            out.append(f"<{element['h']}>")
    return "".join(out)


def _score(row: dict) -> int:
    fixed = b"".join(_dec(e) for e in row["key"] if "f" in e)
    return row["count"] * len(fixed.strip())


def save(pack: dict, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pack, separators=(",", ":")) + "\n")
    return out.stat().st_size


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def search(pack: dict, query: bytes, limit: int = 20) -> list[dict]:
    hits = []
    for row in pack["templates"]:
        fixed = b"".join(_dec(e) for e in row["key"] if "f" in e)
        if query in fixed:
            hits.append(
                {
                    "kind": "template",
                    "text": fixed[:160],
                    "count": row["count"],
                    "files": row["files"],
                }
            )
        if len(hits) >= limit:
            break
    for row in pack["fills"]:
        value = row["value"].encode("latin-1")
        if query in value:
            hits.append(
                {
                    "kind": "fill",
                    "text": value[:160],
                    "count": row["count"],
                    "files": [],
                }
            )
        if len(hits) >= limit:
            break
    return hits[:limit]


def run(
    repo: Path,
    out: Path,
    search_text: str = "",
    min_fill: int = 8,
    min_count: int = 3,
    max_n: int = 6,
    max_bytes: int = 2_000_000,
) -> int:
    pack = build(repo, min_fill, min_count, max_n, max_bytes)
    size = save(pack, out)
    hits = search(pack, search_text.encode(), 20) if search_text else []
    top = sorted(pack["templates"], key=_score, reverse=True)[:20]
    lines = [
        "# Template Pack",
        "",
        f"- root `{pack['root']}`: {pack['files']:,} files, {pack['bytes']:,} B",
        f"- templates {len(pack['templates']):,}; fills {len(pack['fills']):,}; "
        f"pack {size:,} B",
        f"- min fill {min_fill} B, min count {min_count}, max n {max_n}, "
        f"budget {max_bytes:,} B",
        "",
        "| count | fixed text | files |",
        "|---:|---|---|",
        *[
            f"| {row['count']} | `{_show(row['key'])[:100]}` | "
            f"{', '.join(row['files'][:3])} |"
            for row in top
        ],
        "",
    ]
    if search_text:
        lines += ["## Search", ""]
        for hit in hits:
            lines.append(
                f"- **{hit['kind']}** {hit['text']!r} x{hit['count']} "
                f"{', '.join(hit['files'][:3])}"
            )
        lines += [""]
    text = "\n".join(lines)
    cml.write_text(cml.REPORTS / "TPACK.md", text)
    cml.write_json(cml.REPORTS / "tpack.json", {"pack": str(out), "size": size})
    print(text)
    return 0
