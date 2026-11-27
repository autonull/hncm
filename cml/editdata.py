"""Exact Edit Bench data: mine real old/new commit pairs, split by repository.

One source of truth is the manifest, which records the mining config and each
repository head SHA.  Blobs are content-addressed, so a run is reproducible
from the manifest plus the cache.
"""

from __future__ import annotations

import difflib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REPOS = (
    "psf/requests",
    "pallets/flask",
    "pytest-dev/pytest",
    "encode/httpx",
    "tiangolo/fastapi",
)

ALIASES = {
    "requests": "psf/requests",
    "flask": "pallets/flask",
    "pytest": "pytest-dev/pytest",
    "httpx": "encode/httpx",
    "fastapi": "tiangolo/fastapi",
}


def canonical(repo: str) -> str:
    return ALIASES.get(repo, repo)


# Generated or vendored paths are not edits we want the model to learn.
BAD_PARTS = {
    ".git",
    ".tox",
    "build",
    "dist",
    "node_modules",
    "site-packages",
    "third_party",
    "_vendor",
    "vendor",
    "vendored",
}
BAD_SUFFIX = (
    ".lock",
    ".map",
    ".min.css",
    ".min.js",
    ".pyc",
    ".so",
    ".whl",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".pdf",
    ".zip",
    ".gz",
    ".woff",
    ".woff2",
    ".ttf",
)


def cache_root(path: Path | str | None = None) -> Path:
    if path is None:
        path = "~/.cache/hncm/edit"
    return Path(path).expanduser()


def repo_splits(repos: list[str] | tuple[str, ...]) -> dict[str, str]:
    """Split by repository: first n-2 train, then val, then test.

    Alphabetical order is the seed, so the split is deterministic and no
    repository crosses a split.
    """
    order = sorted(set(repos))
    if len(order) < 3:
        raise ValueError("need at least three repositories for train/val/test")
    train = order[:-2]
    return {
        repo: ("train" if repo in train else "val" if repo == order[-2] else "test")
        for repo in order
    }


def _git(args: list[str], cwd: Path | None = None, data: bytes | None = None) -> bytes:
    p = subprocess.run(
        ["git", *args], cwd=cwd, input=data, capture_output=True, check=False
    )
    if p.returncode:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {p.stderr.decode('utf-8', 'replace')}"
        )
    return p.stdout


def ensure_repo(remote: str, cache: Path, refresh: bool = False) -> Path:
    """Clone a remote once, then reuse it.  Local paths are used as they are."""
    local = Path(remote).expanduser()
    if local.exists():
        return local
    slug = remote.strip("/").removesuffix(".git").replace("/", "__")
    dst = cache / "repos" / f"{slug}.git"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = remote if "://" in remote else f"https://github.com/{remote}.git"
        _git(["clone", "--bare", "--quiet", url, str(dst)])
    elif refresh:
        _git(["fetch", "--all", "--prune", "--quiet"], cwd=dst)
    return dst


def repo_head(repo: Path) -> str:
    return _git(["rev-parse", "HEAD"], cwd=repo).decode().strip()


def accept_path(path: str) -> bool:
    parts = path.split("/")
    if any(part in BAD_PARTS for part in parts[:-1]):
        return False
    name = parts[-1]
    return not name.endswith(BAD_SUFFIX)


def changed_lines(old: bytes, new: bytes) -> int:
    """Changed line count: inserted plus deleted lines, from a line diff."""
    a = old.splitlines(keepends=True)
    b = new.splitlines(keepends=True)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    changed = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        changed += (i2 - i1) + (j2 - j1)
    return changed


def text_ok(old: bytes, new: bytes) -> bool:
    if not new or b"\x00" in old or b"\x00" in new:
        return False
    try:
        old.decode("utf-8")
        new.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def diff_fraction(old: bytes, new: bytes) -> float:
    lines = len(new.splitlines())
    return changed_lines(old, new) / max(lines, 1)


def accept_blob(
    path: str, old: bytes, new: bytes, max_file: int, max_diff: float
) -> bool:
    if not accept_path(path):
        return False
    if not text_ok(old, new):
        return False
    if len(old) > max_file or not 0 < len(new) <= max_file:
        return False
    if old == new:
        return False
    return diff_fraction(old, new) <= max_diff


def _parse_log(raw: bytes) -> list[tuple[str, str, str, list[bytes]]]:
    """Parse `git log --format=%x01%H%x01%P%x01%s --name-only` output."""
    out: list[tuple[str, str, str, list[bytes]]] = []
    cur: tuple[str, str, str, list[bytes]] | None = None
    for line in raw.split(b"\n"):
        if line.startswith(b"\x01"):
            fields = line[1:].split(b"\x01")
            if len(fields) != 3:
                cur = None
                continue
            sha, parents, subject = fields
            parent = parents.decode().split()[0] if parents.strip() else ""
            cur = (
                sha.decode(),
                parent,
                subject.decode("utf-8", "replace").strip(),
                [],
            )
            out.append(cur)
        elif line and cur is not None:
            cur[3].append(line)
    return out


def _cat_blobs(repo: Path, shas: list[str]) -> dict[str, bytes]:
    if not shas:
        return {}
    raw = _git(
        ["cat-file", "--batch"], cwd=repo, data=("\n".join(shas) + "\n").encode()
    )
    blobs: dict[str, bytes] = {}
    i = 0
    while i < len(raw):
        nl = raw.index(b"\n", i)
        header = raw[i:nl].split()
        if len(header) != 3:
            break
        sha, kind, size = header[0], header[1], int(header[2])
        start = nl + 1
        if kind != b"blob":
            i = start
            continue
        blobs[sha.decode()] = raw[start : start + size]
        i = start + size + 1
    return blobs


@dataclass(frozen=True, slots=True)
class MineConfig:
    repo: str
    max_file: int = 64 * 1024
    max_diff: float = 0.2
    limit: int = 0


def mine_repo(repo: Path, cfg: MineConfig) -> list[dict]:
    """Mine one repository: one text file changed, size and diff filters."""
    raw = _git(
        [
            "log",
            "--no-merges",
            "--diff-filter=M",
            "--name-only",
            "--format=%x01%H%x01%P%x01%s",
        ],
        cwd=repo,
    )
    cands: list[tuple[str, str, str, str]] = []
    for sha, parent, subject, paths in _parse_log(raw):
        if not parent or not subject or len(paths) != 1:
            continue
        pb = paths[0]
        if pb.startswith(b'"'):
            continue  # quoted path: rare, skip rather than unquote
        try:
            path = pb.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if accept_path(path):
            cands.append((sha, parent, subject, path))

    names = []
    for sha, parent, _, path in cands:
        names.append(f"{parent}:{path}".encode())
        names.append(f"{sha}:{path}".encode())
    if not names:
        return []
    raw = _git(["cat-file", "--batch-check"], cwd=repo, data=b"\n".join(names) + b"\n")
    lines = raw.splitlines()
    if len(lines) != len(names):
        raise RuntimeError(f"{repo}: cat-file returned {len(lines)} of {len(names)}")

    keep: list[tuple[str, str, str, str, str, str]] = []
    for i, (sha, parent, subject, path) in enumerate(cands):
        old_meta = lines[2 * i].split()
        new_meta = lines[2 * i + 1].split()
        if len(old_meta) != 3 or len(new_meta) != 3:
            continue
        if old_meta[1] != b"blob" or new_meta[1] != b"blob":
            continue
        old_size, new_size = int(old_meta[2]), int(new_meta[2])
        if old_size > cfg.max_file or not 0 < new_size <= cfg.max_file:
            continue
        if old_meta[0] == new_meta[0]:
            continue
        keep.append(
            (
                sha,
                parent,
                subject,
                path,
                old_meta[0].decode(),
                new_meta[0].decode(),
            )
        )
    if not keep:
        return []

    blobs = _cat_blobs(repo, sorted({*[k[4] for k in keep], *[k[5] for k in keep]}))
    seen: set[tuple[str, str]] = set()
    records: list[dict] = []
    for sha, parent, subject, path, old_sha, new_sha in keep:
        key = (old_sha, new_sha)
        if key in seen:
            continue
        old = blobs.get(old_sha)
        new = blobs.get(new_sha)
        if old is None or new is None:
            continue
        if not accept_blob(path, old, new, cfg.max_file, cfg.max_diff):
            continue
        seen.add(key)
        records.append(
            {
                "id": f"{cfg.repo.split('/')[-1]}-{sha[:12]}",
                "repo": cfg.repo,
                "parent": parent,
                "commit": sha,
                "path": path,
                "old": old_sha,
                "new": new_sha,
                "old_size": len(old),
                "new_size": len(new),
                "diff_lines": changed_lines(old, new),
                "diff_frac": round(diff_fraction(old, new), 4),
                "message": subject,
            }
        )
    records.sort(key=lambda r: (r["commit"], r["path"]))
    if cfg.limit:
        records = records[: cfg.limit]
    return records


def write_blobs(cache: Path, records: list[dict], repo: Path) -> None:
    shas = sorted({*[r["old"] for r in records], *[r["new"] for r in records]})
    blobs = _cat_blobs(repo, shas)
    for sha, data in blobs.items():
        path = cache / "blobs" / sha[:2] / sha
        if path.exists() and path.stat().st_size == len(data):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)


def blob_path(cache: Path, sha: str) -> Path:
    return cache / "blobs" / sha[:2] / sha


def read_blob(cache: Path, sha: str) -> bytes:
    return blob_path(cache, sha).read_bytes()


def build(
    repos: list[str],
    cache: Path,
    max_file: int = 64 * 1024,
    max_diff: float = 0.2,
    limit: int = 0,
    refresh: bool = False,
) -> dict:
    """Mine every repository and build the manifest, deduplicated per split."""
    repos = [canonical(r) for r in repos]
    split = repo_splits(repos)
    heads: dict[str, str] = {}
    records: list[dict] = []
    for remote in sorted(set(repos)):
        path = ensure_repo(remote, cache, refresh)
        head = repo_head(path)
        heads[remote] = head
        cfg = MineConfig(remote, max_file, max_diff, limit)
        mined = mine_repo(path, cfg)
        write_blobs(cache, mined, path)
        records.extend(mined)
    for rec in records:
        rec["split"] = split[rec["repo"]]

    # Deduplicate identical (old, new) content within a split, not across it.
    kept: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for rec in sorted(records, key=lambda r: (r["split"], r["repo"], r["commit"])):
        key = (rec["split"], rec["old"], rec["new"])
        if key in seen:
            continue
        seen.add(key)
        kept.append(rec)
    kept.sort(key=lambda r: (r["split"], r["repo"], r["commit"], r["path"]))
    counts = {
        name: sum(1 for r in kept if r["split"] == name)
        for name in ("train", "val", "test")
    }
    return {
        "version": 1,
        "config": {
            "repos": sorted(set(repos)),
            "max_file": max_file,
            "max_diff": max_diff,
            "diff_metric": "changed lines / new lines",
            "limit": limit,
            "refresh": refresh,
        },
        "heads": heads,
        "splits": {
            name: sorted(r for r, s in split.items() if s == name)
            for name in ("train", "val", "test")
        },
        "counts": counts,
        "edits": kept,
    }


def summary(manifest: dict) -> str:
    counts = manifest["counts"]
    n = sum(counts.values())
    repos = len(manifest["heads"])
    return (
        f"editdata: {repos} repos -> {n} edits "
        f"(train {counts['train']}, val {counts['val']}, test {counts['test']})"
    )


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1) + "\n")
    os.replace(tmp, path)


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def load_edits(
    manifest: dict, cache: Path, split: str
) -> list[tuple[dict, bytes, bytes]]:
    """Read old and new bytes for every edit in a split, in manifest order."""
    out: list[tuple[dict, bytes, bytes]] = []
    for rec in manifest["edits"]:
        if split != "all" and rec["split"] != split:
            continue
        old = read_blob(cache, rec["old"])
        new = read_blob(cache, rec["new"])
        if len(old) != rec["old_size"] or len(new) != rec["new_size"]:
            raise ValueError(f"blob size mismatch for {rec['id']}")
        out.append((rec, old, new))
    return out


def run(
    repos: str,
    cache: Path,
    out: Path,
    max_file: int,
    max_diff: float,
    limit: int,
    refresh: bool,
) -> int:
    names = [r.strip() for r in repos.split(",") if r.strip()]
    if len(names) < 3:
        raise SystemExit("editdata: pass at least three --repos")
    manifest = build(names, cache, max_file, max_diff, limit, refresh)
    write_manifest(out, manifest)
    print(summary(manifest))
    print(f"editdata: manifest {out}")
    return 0
