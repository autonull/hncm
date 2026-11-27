"""Concepts: typed holes, recursive composition, and exact byte realization.

A `Template` is a reusable byte pattern with typed holes.  A `Term` applies one
template to arguments; a `Literal` fills a hole.  `realize` writes the exact
bytes, so a term is an editable representation, never a second encoding of the
bytes.  `Library` grows only when a template's MDL gain pays its charge.  The
`benchmark` prices a held-out composition against the byte arm: the concept
gain is a representation gain, not a reasoning result.

The span lexicon in `codec.py` is a compression dictionary of exact byte
sequences.  These terms are different in kind: an operation is a template with
holes, and `ADD(MULTIPLY(x, y), z)` is a term the library built by composing
two templates, not a string that happened to repeat.
"""

from __future__ import annotations

import random
import re
import struct
from dataclasses import dataclass

import numpy as np

import codec

BANNED = frozenset({("ADD", "MUL")})
DIGITS = b"0123456789"
HOLE = re.compile(rb"\{(\d+)\}")


@dataclass(frozen=True, slots=True)
class Hole:
    name: str
    type: str


@dataclass(frozen=True, slots=True)
class Template:
    """A typed operation: `pattern` with `{0}`, `{1}` ... holes, in order."""

    op: str
    pattern: bytes
    holes: tuple[Hole, ...]

    def arity(self) -> int:
        return len(self.holes)

    def blob(self) -> bytes:
        out = struct.pack("<H", len(self.op)) + self.op.encode()
        out += struct.pack("<H", len(self.pattern)) + self.pattern
        out += struct.pack("<H", len(self.holes))
        for hole in self.holes:
            out += struct.pack("<H", len(hole.name)) + hole.name.encode()
            out += struct.pack("<H", len(hole.type)) + hole.type.encode()
        return out


@dataclass(frozen=True, slots=True)
class Literal:
    type: str
    data: bytes


@dataclass(frozen=True, slots=True)
class Term:
    op: str
    args: tuple[Term | Literal, ...]


def arithmetic_library() -> Library:
    """The four operation templates; a digit literal is the only leaf."""
    ints = (Hole("left", "Int"), Hole("right", "Int"))
    return Library(
        (
            Template("ADD", b"ADD({0}, {1})", ints),
            Template("MUL", b"MUL({0}, {1})", ints),
            Template("SUB", b"SUB({0}, {1})", ints),
            Template("NEG", b"NEG({0})", ints[:1]),
        )
    )


class Library:
    """Templates keyed by op, plus the gate that keeps the library honest."""

    def __init__(self, templates: tuple[Template, ...] = ()) -> None:
        self.templates: dict[str, Template] = {t.op: t for t in templates}
        self.rejected: int = 0

    def offer(self, template: Template, gain_bits: float) -> bool:
        """Add a template only if its gain pays its bytes; otherwise reject.

        `gain_bits` is what the template saves over coding the pattern as
        bytes; the charge is its own blob at 8 bits per byte.
        """
        cost = 8 * len(template.blob())
        if gain_bits <= cost:
            self.rejected += 1
            return False
        self.templates[template.op] = template
        return True

    def blob(self) -> bytes:
        out = bytearray(b"CMLT" + struct.pack("<I", len(self.templates)))
        for op in sorted(self.templates):
            out += self.templates[op].blob()
        return bytes(out)


def realize(term: Term | Literal, lib: Library) -> bytes:
    """Substitute the term into its templates; the bytes are exact."""
    if isinstance(term, Literal):
        return term.data
    template = lib.templates.get(term.op)
    if template is None:
        raise KeyError(f"concept: unknown op {term.op}")
    if len(term.args) != template.arity():
        raise ValueError(f"concept: {term.op} takes {template.arity()} args")
    values = [realize(arg, lib) for arg in term.args]
    out = bytearray()
    pos = 0
    for match in HOLE.finditer(template.pattern):
        out += template.pattern[pos : match.start()]
        out += values[int(match.group(1))]
        pos = match.end()
    out += template.pattern[pos:]
    return bytes(out)


def size(term: Term | Literal) -> int:
    """Nodes in the term, the unit the library charges for."""
    if isinstance(term, Literal):
        return 1
    return 1 + sum(size(arg) for arg in term.args)


def pairs(term: Term | Literal, parent: str | None = None) -> set[tuple[str, str]]:
    """Every parent -> child op edge, with `@root` for the top op."""
    if isinstance(term, Literal):
        return set()
    out = {(parent or "@root", term.op)}
    for arg in term.args:
        out |= pairs(arg, term.op)
    return out


def has_ban(term: Term | Literal) -> bool:
    return bool(pairs(term) & BANNED)


def twin(term: Term | Literal) -> Term | Literal:
    """The same tree with the banned ADD -> MUL edge rewritten to ADD -> SUB.

    The control has the identical shape and op count, so the two splits differ
    only in the edge: their cost difference is the cost of that edge.
    """
    if isinstance(term, Literal):
        return term
    args = []
    for arg in term.args:
        if term.op == "ADD" and isinstance(arg, Term) and arg.op == "MUL":
            args.append(Term("SUB", arg.args))
        else:
            args.append(twin(arg))
    return Term(term.op, tuple(args))


class ConceptArm:
    """Pre-order term code: an op row per parent op, digits per literal."""

    def __init__(self, lib: Library) -> None:
        self.lib = lib
        self.ops = tuple(sorted(lib.templates))
        self.index = {op: i for i, op in enumerate(self.ops)}
        n = len(self.ops)
        self.kind = codec.Counts(2, init=np.ones(2, dtype=np.int64))
        self.ops0 = codec.Counts(n, init=np.ones(n, dtype=np.int64))
        self.ops1: dict[str, codec.Counts] = {}
        self.digits = codec.Counts(
            len(DIGITS), init=np.ones(len(DIGITS), dtype=np.int64)
        )

    def _rungs(self, parent: str | None) -> list[codec.Counts]:
        if parent is None:
            return [self.ops0]
        row = self.ops1.get(parent)
        if row is None:
            row = self.ops1[parent] = codec.new_row(len(self.ops) + 1, len(self.ops))
        return [row, self.ops0]

    def _digit(self, term: Literal) -> int:
        if len(term.data) != 1 or term.data[0] not in DIGITS:
            raise ValueError(f"concept arm: not a digit literal {term.data!r}")
        return DIGITS.index(term.data[0])

    def _encode_term(self, term: Term | Literal, parent: str | None, enc) -> None:
        if isinstance(term, Literal):
            codec.emit([self.kind], 0, 2, enc)
            codec.emit([self.digits], self._digit(term), len(DIGITS), enc)
            return
        codec.emit([self.kind], 1, 2, enc)
        codec.emit(self._rungs(parent), self.index[term.op], len(self.ops), enc)
        for arg in term.args:
            self._encode_term(arg, term.op, enc)

    def _decode_term(self, parent: str | None, dec) -> Term | Literal:
        if codec.take([self.kind], 2, dec) == 0:
            d = codec.take([self.digits], len(DIGITS), dec)
            return Literal("Int", bytes([DIGITS[d]]))
        op = self.ops[codec.take(self._rungs(parent), len(self.ops), dec)]
        template = self.lib.templates[op]
        return Term(
            op, tuple(self._decode_term(op, dec) for _ in range(template.arity()))
        )

    def encode(self, terms: list[Term | Literal]) -> bytes:
        enc = codec.ArithEncoder()
        for term in terms:
            self._encode_term(term, None, enc)
        return struct.pack("<I", len(terms)) + enc.finish()

    def decode(self, blob: bytes) -> list[Term | Literal]:
        count = struct.unpack_from("<I", blob)[0]
        dec = codec.ArithDecoder(blob[4:])
        return [self._decode_term(None, dec) for _ in range(count)]


def term(rng: random.Random, depth: int, lib: Library) -> Term | Literal:
    """A random term of at most `depth` nested operations; leaves are digits."""
    if depth <= 0 or rng.random() < 0.3:
        return Literal("Int", bytes([DIGITS[rng.randrange(len(DIGITS))]]))
    op = rng.choice(tuple(lib.templates))
    template = lib.templates[op]
    return Term(op, tuple(term(rng, depth - 1, lib) for _ in range(template.arity())))


def split(lib: Library, n: int, depth: int, seed: int, ban: bool) -> list[Term]:
    """Rejection-sample terms that do or do not contain the banned edge."""
    rng = random.Random(seed)
    out: list[Term] = []
    tries = 0
    while len(out) < n:
        tries += 1
        candidate = term(rng, depth, lib)
        if not isinstance(candidate, Term):
            continue
        if has_ban(candidate) == ban:
            out.append(candidate)
        if tries > 100 * n:
            raise RuntimeError("concept split: cannot satisfy the ban")
    return out


def benchmark(
    n: int = 500, depth: int = 3, seed: int = 0, lib: Library | None = None
) -> dict:
    """Three splits: train, seen structures, banned (unseen) structures."""
    lib = lib or arithmetic_library()
    train = split(lib, n, depth, seed, ban=False)
    seen = split(lib, n, depth, seed + 1, ban=False)
    unseen = split(lib, n, depth, seed + 2, ban=True)
    twins = [twin(t) for t in unseen]
    rows = []
    for name, terms in (
        ("train", train),
        ("seen", seen),
        ("unseen", unseen),
        ("twin", twins),
    ):
        raw = b"\n".join(realize(t, lib) for t in terms)
        payload = ConceptArm(lib).encode(terms)
        back = ConceptArm(lib).decode(payload)
        if b"\n".join(realize(t, lib) for t in back) != raw:
            raise ValueError("concept arm: round-trip failed")
        base = codec.ByteArm().encode(raw)
        shared = len(payload) + len(lib.blob())
        rows.append(
            {
                "split": name,
                "terms": len(terms),
                "nodes": sum(size(t) for t in terms),
                "bytes": len(raw),
                "payload_bytes": len(payload),
                "shared_bytes": shared,
                "concept_bpb": 8 * len(payload) / max(len(raw), 1),
                "shared_bpb": 8 * shared / max(len(raw), 1),
                "byte_bpb": 8 * len(base) / max(len(raw), 1),
                "ban": name == "unseen",
            }
        )
    return {
        "rows": rows,
        "library_bytes": len(lib.blob()),
        "library": len(lib.templates),
        "rejected": lib.rejected,
        "banned": sorted("->".join(edge) for edge in BANNED),
        "seed": seed,
        "depth": depth,
        "n": n,
    }


def render(report: dict) -> str:
    rows = {r["split"]: r for r in report["rows"]}
    unseen, control = rows["unseen"], rows["twin"]
    penalty = unseen["concept_bpb"] - control["concept_bpb"]
    byte_penalty = unseen["byte_bpb"] - control["byte_bpb"]
    lines = [
        "# Composition: typed holes and exact realization",
        "",
        f"- {report['library']} templates, {report['library_bytes']:,} B library, "
        f"depth {report['depth']}, seed {report['seed']}",
        f"- banned edge `{report['banned'][0]}`: train and seen never contain it",
        "- `concept` codes the term stream; `bytes` is the order-2 byte arm on",
        "  the same realizations. Both adapt on the split they code.",
        "- `twin` is the unseen split with ADD -> SUB: the same shape and op",
        "  count, so `unseen - twin` is the price of that one edge.",
        "- `shared` charges the library once; `bpb` divides by realized bytes.",
        "",
        "| split | terms | nodes | bytes | concept | shared | bytes arm |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("train", "seen", "unseen", "twin"):
        r = rows[name]
        lines.append(
            f"| {name} | {r['terms']:,} | {r['nodes']:,} | {r['bytes']:,} | "
            f"{r['concept_bpb']:.3f} | {r['shared_bpb']:.3f} | "
            f"{r['byte_bpb']:.3f} |"
        )
    lines += [
        "",
        "## Read",
        "",
        "- every realization round-trips exactly; the term is the message and",
        "  the bytes are derived, so the representation is editable.",
        f"- the novel edge costs {penalty:+.3f} bpb for the term code and",
        f"  {byte_penalty:+.3f} bpb for the byte arm, at matched shape.",
        "- Both arms generalize this edge: at this grain the term gain is",
        "  symbolization, not composition. That is a pilot limit, not a gate.",
        f"- the term code still pays {unseen['byte_bpb'] - unseen['concept_bpb']:.3f} bpb "
        "less than bytes on unseen",
        "  compositions, with exact realization.",
        "- This prices the representation, not reasoning: the encoder and",
        "  decoder both hold the term. The separating test needs a harder",
        "  composition task and a matched byte Transformer (P5e/P5f).",
        "",
    ]
    return "\n".join(lines)
