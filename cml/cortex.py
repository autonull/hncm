"""Cortex: predict the term stream, not the bytes.

`concepts.py` shows the representation is exact and cheaper than bytes.  This
module asks the separating question: at matched parameters, data and steps,
does a model that predicts the term stream beat a byte model that predicts the
realization?  The held-out edge is never in train.  A pilot, not a reasoning
result: the task is synthetic arithmetic, not code or algorithms.
"""

from __future__ import annotations

import math


import torch
import torch.nn.functional as F

import cml
import concepts

OPS = ("ADD", "MUL", "SUB", "NEG", "MAX", "MIN", "XOR", "AND")
BINARY = tuple(op for op in OPS if op != "NEG")
LITERALS = tuple(bytes([b]) for b in b"0123456789xyz")
SEP = len(OPS) + len(LITERALS)
VOCAB = SEP + 1
BANNED_EDGE = ("ADD", "XOR")
BANNED = frozenset({BANNED_EDGE})
SWAP = ("ADD", "XOR", "AND")
OP_INDEX = {op: i for i, op in enumerate(OPS)}


def library() -> concepts.Library:
    """Eight operations; NEG is the only unary one."""
    pair = (concepts.Hole("left", "Int"), concepts.Hole("right", "Int"))
    templates = [
        concepts.Template(op, op.encode() + b"({0}, {1})", pair) for op in BINARY
    ]
    templates.append(concepts.Template("NEG", b"NEG({0})", pair[:1]))
    return concepts.Library(tuple(templates))


def term_ids(term: concepts.Term | concepts.Literal) -> list[int]:
    """Pre-order ids: an op symbol, then its arguments; a literal is one id."""
    if isinstance(term, concepts.Literal):
        return [len(OPS) + LITERALS.index(term.data)]
    out = [OP_INDEX[term.op]]
    for arg in term.args:
        out += term_ids(arg)
    return out


def pack(terms: list[concepts.Term], lib: concepts.Library) -> tuple[list[int], int]:
    """The id stream and the realized bytes; the two describe the same terms."""
    ids: list[int] = []
    for term in terms:
        ids += term_ids(term)
        ids.append(SEP)
    raw = b"\n".join(concepts.realize(t, lib) for t in terms)
    return ids, len(raw)


def unpacked(ids: list[int], lib: concepts.Library, count: int) -> list[concepts.Term]:
    """Parse a symbol stream back to terms; raises on a malformed stream."""
    pos = 0

    def read() -> concepts.Term | concepts.Literal:
        nonlocal pos
        if pos >= len(ids) or ids[pos] == SEP:
            raise ValueError("cortex: truncated stream")
        sym = ids[pos]
        pos += 1
        if sym >= len(OPS):
            return concepts.Literal("Int", LITERALS[sym - len(OPS)])
        op = OPS[sym]
        template = lib.templates[op]
        return concepts.Term(op, tuple(read() for _ in range(template.arity())))

    terms: list[concepts.Term] = []
    while len(terms) < count:
        node = read()
        if not isinstance(node, concepts.Term):
            raise ValueError("cortex: a top-level literal is not a term")
        terms.append(node)
        if pos >= len(ids) or ids[pos] != SEP:
            raise ValueError("cortex: missing separator")
        pos += 1
    if pos != len(ids):
        raise ValueError("cortex: trailing symbols")
    return terms


@torch.no_grad()
def stream_bits(
    model: cml.Trunk, ids: list[int], seq: int, batch: int = 64
) -> tuple[float, int]:
    """Teacher-forced bits over a continuous stream; both arms use it."""
    model.eval()
    x = torch.tensor(ids, dtype=torch.long).unfold(0, seq + 1, seq).cuda()
    bits = 0.0
    n = 0
    for chunk in x.split(batch):
        logits = model(chunk[:, :-1])
        target = chunk[:, 1:]
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction="none"
        )
        bits += float(ce.double().sum()) / math.log(2)
        n += int(target.numel())
    return bits, n


def train(
    ids: list[int], vocab: int, cfg: dict, seq: int, steps: int, batch: int, seed: int
) -> tuple[cml.Trunk, float]:
    torch.manual_seed(seed)
    x = torch.tensor(ids, dtype=torch.long)
    model = cml.Trunk(vocab, cfg["d"], cfg["layers"], cfg["heads"], seq).cuda()
    win = cml.windows(x, seq)
    secs = cml.train_loop(model, win, steps, batch, 3e-4)
    return model, secs


def pilot(
    pairs: int = 4000,
    depth: int = 4,
    steps: int = 1200,
    d: int = 128,
    layers: int = 4,
    heads: int = 4,
    seq: int = 128,
    batch: int = 32,
    seed: int = 0,
) -> dict:
    if not torch.cuda.is_available():
        raise SystemExit("cortex: the pilot needs a GPU")
    lib = library()
    cfg = {"d": d, "layers": layers, "heads": heads}
    train_terms = concepts.split(lib, pairs, depth, seed, False, BANNED)
    splits = {
        "seen": concepts.split(lib, pairs, depth, seed + 1, False, BANNED),
        "unseen": concepts.split(lib, pairs, depth, seed + 2, True, BANNED),
    }
    splits["twin"] = [
        concepts.twin(t, SWAP[0], SWAP[1], SWAP[2]) for t in splits["unseen"]
    ]
    train_ids, _ = pack(train_terms, lib)
    cortex, cortex_secs = train(train_ids, VOCAB, cfg, seq, steps, batch, seed)
    byte_ids: list[int] = []
    for term in train_terms:
        byte_ids.extend(concepts.realize(term, lib))
        byte_ids.append(10)
    byte, byte_secs = train(byte_ids, 256, cfg, seq, steps, batch, seed)
    rows = []
    for name, terms in splits.items():
        ids, nbytes = pack(terms, lib)
        back = unpacked(ids, lib, len(terms))
        if [concepts.realize(t, lib) for t in back] != [
            concepts.realize(t, lib) for t in terms
        ]:
            raise ValueError("cortex: stream does not round-trip")
        cb, _ = stream_bits(cortex, ids, seq)
        bb, _ = stream_bits(byte, byte_ids_of(terms, lib), seq)
        rows.append(
            {
                "split": name,
                "terms": len(terms),
                "nodes": sum(concepts.size(t) for t in terms),
                "bytes": nbytes,
                "cortex_bpb": cb / max(nbytes, 1),
                "byte_bpb": bb / max(nbytes, 1),
            }
        )
    return {
        "rows": rows,
        "pairs": pairs,
        "depth": depth,
        "steps": steps,
        "seq": seq,
        "batch": batch,
        "seed": seed,
        "cfg": cfg,
        "params": sum(p.numel() for p in cortex.parameters()),
        "library_bytes": len(lib.blob()),
        "banned": "->".join(BANNED_EDGE),
        "seconds": round(cortex_secs + byte_secs, 1),
    }


def byte_ids_of(terms: list[concepts.Term], lib: concepts.Library) -> list[int]:
    out: list[int] = []
    for term in terms:
        out.extend(concepts.realize(term, lib))
        out.append(10)
    return out


def render(report: dict) -> str:
    rows = {r["split"]: r for r in report["rows"]}
    unseen, twin = rows["unseen"], rows["twin"]
    penalty = unseen["cortex_bpb"] - twin["cortex_bpb"]
    byte_penalty = unseen["byte_bpb"] - twin["byte_bpb"]
    delta = unseen["cortex_bpb"] - unseen["byte_bpb"]
    gap = penalty - byte_penalty
    if abs(delta) <= 0.02 and abs(gap) <= 0.01:
        verdict = "The pilot does not separate the two: the term representation is"
        verdict += " cheaper when the term is given, not when it is predicted."
    elif delta < 0 and gap < 0:
        verdict = "The term model wins on both the rate and the novel edge."
    elif delta > 0 and gap > 0:
        verdict = "The byte arm wins on both the rate and the novel edge."
    else:
        verdict = "The arms split: one wins the rate, the other the novel edge."
    lines = [
        "# Cortex: predict the term stream, not the bytes",
        "",
        f"- {report['pairs']} terms per split, depth {report['depth']}, "
        f"seed {report['seed']}",
        f"- both arms: d{report['cfg']['d']} l{report['cfg']['layers']} "
        f"h{report['cfg']['heads']} seq{report['seq']}, "
        f"{report['steps']} steps, batch {report['batch']}, "
        f"{report['params']:,} params",
        f"- held-out edge `{report['banned']}`: train and seen never contain it;",
        "  `twin` rewrites it to an allowed edge at the same shape",
        "- `cortex` has a 22-symbol vocabulary, `byte` has 256; both code the",
        "  same bytes and the table divides by realized bytes, so `bpb` is",
        "  comparable at matched params, data and steps.",
        "",
        "| split | terms | nodes | bytes | cortex bpb | byte bpb | delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("seen", "unseen", "twin"):
        r = rows[name]
        lines.append(
            f"| {name} | {r['terms']:,} | {r['nodes']:,} | {r['bytes']:,} | "
            f"{r['cortex_bpb']:.3f} | {r['byte_bpb']:.3f} | "
            f"{r['cortex_bpb'] - r['byte_bpb']:+.3f} |"
        )
    lines += [
        "",
        "## Read",
        "",
        f"- the novel edge costs {penalty:+.3f} bpb for the cortex and",
        f"  {byte_penalty:+.3f} bpb for the byte arm, at matched shape.",
        f"- on unseen compositions the cortex is {delta:+.3f} bpb against the",
        f"  byte arm; its edge pay is {gap:+.3f} relative to bytes.",
        f"- {verdict}",
        "- This is a synthetic arithmetic pilot with matched params and steps:",
        "  it tests representation and prediction, not reasoning. It does not",
        "  decide the program; a tie is a reason to fall back to",
        "  `TODO8.compression.md`.",
        "",
        f"Training took {report['seconds']:.0f} s on one GPU.",
        "",
    ]
    return "\n".join(lines)
