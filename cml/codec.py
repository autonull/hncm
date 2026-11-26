"""The codec seam: arithmetic coder, adaptive counts, byte and unit arms.

M0 is the only piece that must be exact: `ArithEncoder`/`ArithDecoder` implement
the WNC 32-bit coder (E1/E2/E3) and `Counts` is the adaptive table that both
sides rebuild from the same symbol stream.  `cml coder --test` is the invariant.

The byte arm is an order-2 PPM with an order-1 and an order-0 fallback.  The
unit arm parses the input with the installed lexicon (longest match) and codes
unit ids with the same escape idea; literal bytes escape to an independent byte
arm.  The frame carries the lexicon, so install is data, never a gradient step.
"""

from __future__ import annotations

import random
import struct
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

TOP = 1 << 32
MASK = TOP - 1
HALF = 1 << 31
QTR = 1 << 30
THREE_QTR = 3 << 30

FRAME_MAGIC = b"CMLC"
FRAME_VERSION = 1
LEX_MAGIC = b"CMLX"
ESC = 256  # the byte-arm escape
UNIT_ESC = -1  # an invented symbol for a literal in the unit arm
QUANT = 1 << 20


class Frequencies(Protocol):
    n: int

    def cums(self) -> np.ndarray: ...

    def find(self, target: int) -> int: ...

    def update(self, sym: int) -> None: ...


def quant(p: np.ndarray | list[float], total: int = QUANT) -> np.ndarray:
    """A neural CDF as positive integer counts (at least one per symbol)."""
    a = np.asarray(p, dtype=np.float64).ravel()
    return np.maximum(np.rint(a * total).astype(np.int64), 1)


class Counts:
    """An adaptive frequency table.  +16 per update, halve at `cap`."""

    __slots__ = ("n", "cap", "counts", "total", "_cums")

    def __init__(self, n: int, cap: int = 1 << 16, init: np.ndarray | None = None):
        self.n = n
        self.cap = cap
        self.counts = np.zeros(n, dtype=np.int64)
        if init is not None:
            self.counts[:] = init
        self.total = int(self.counts.sum())
        self._cums: np.ndarray | None = None

    def cums(self) -> np.ndarray:
        if self._cums is None:
            self._cums = np.concatenate(([0], np.cumsum(self.counts)))
        return self._cums

    def find(self, target: int) -> int:
        return int(np.searchsorted(self.cums(), target, side="right")) - 1

    def update(self, sym: int) -> None:
        self.counts[sym] += 16
        self.total += 16
        if self.total >= self.cap:
            self._halve()
        else:
            self._cums = None

    def _halve(self) -> None:
        self.counts = np.where(self.counts > 0, np.maximum(self.counts // 2, 1), 0)
        self.total = int(self.counts.sum())
        self._cums = None


def _bits(blob: bytes):
    for byte in blob:
        for shift in range(7, -1, -1):
            yield (byte >> shift) & 1


class ArithEncoder:
    """WNC 32-bit arithmetic encoder: E1/E2/E3, bit output, padded tail."""

    __slots__ = ("low", "high", "pending", "out", "_acc", "_n")

    def __init__(self) -> None:
        self.low = 0
        self.high = MASK
        self.pending = 0
        self.out = bytearray()
        self._acc = 0
        self._n = 0

    def _bit(self, bit: int) -> None:
        self._acc = (self._acc << 1) | bit
        self._n += 1
        if self._n == 8:
            self.out.append(self._acc)
            self._acc = 0
            self._n = 0

    def _emit(self, bit: int) -> None:
        self._bit(bit)
        for _ in range(self.pending):
            self._bit(bit ^ 1)
        self.pending = 0

    def encode(self, lo: int, hi: int, total: int) -> None:
        if not 0 <= lo < hi <= total:
            raise ValueError(f"bad range: {lo}:{hi}/{total}")
        width = self.high - self.low + 1
        self.high = self.low + width * hi // total - 1
        self.low += width * lo // total
        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QTR and self.high < THREE_QTR:
                self.pending += 1
                self.low -= QTR
                self.high -= QTR
            else:
                break
            self.low = (self.low << 1) & MASK
            self.high = ((self.high << 1) & MASK) | 1

    def finish(self) -> bytes:
        self.pending += 1
        self._emit(0 if self.low < QTR else 1)
        if self._n:
            self._acc <<= 8 - self._n
            self.out.append(self._acc)
        return bytes(self.out)


class ArithDecoder:
    """The inverse of `ArithEncoder`; absent tail bits read as zero."""

    __slots__ = ("low", "high", "value", "bits")

    def __init__(self, blob: bytes) -> None:
        self.low = 0
        self.high = MASK
        self.bits = _bits(blob)
        self.value = 0
        for _ in range(32):
            self.value = (self.value << 1) | self._bit()

    def _bit(self) -> int:
        return next(self.bits, 0)

    def target(self, total: int) -> int:
        width = self.high - self.low + 1
        return min(((self.value - self.low + 1) * total - 1) // width, total - 1)

    def update(self, lo: int, hi: int, total: int) -> None:
        if not 0 <= lo < hi <= total:
            raise ValueError(f"bad range: {lo}:{hi}/{total}")
        width = self.high - self.low + 1
        self.high = self.low + width * hi // total - 1
        self.low += width * lo // total
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.value -= HALF
            elif self.low >= QTR and self.high < THREE_QTR:
                self.low -= QTR
                self.high -= QTR
                self.value -= QTR
            else:
                break
            self.low = (self.low << 1) & MASK
            self.high = ((self.high << 1) & MASK) | 1
            self.value = ((self.value << 1) & MASK) | self._bit()


def _encode_in(row: Counts, sym: int, enc: ArithEncoder) -> None:
    cums = row.cums()
    enc.encode(int(cums[sym]), int(cums[sym + 1]), int(cums[-1]))


def _decode_in(row: Counts, dec: ArithDecoder) -> int:
    cums = row.cums()
    target = dec.target(int(cums[-1]))
    sym = row.find(target)
    dec.update(int(cums[sym]), int(cums[sym + 1]), int(cums[-1]))
    return sym


def _emit(rungs: list[Counts], sym: int, esc: int, enc: ArithEncoder) -> None:
    """Encode `sym` through ordered rows, escaping toward the last row."""
    if sym == esc:
        for row in rungs:
            _encode_in(row, esc, enc)
            row.update(esc)
        return
    found = False
    for row in rungs[:-1]:
        if row.counts[sym] > 0:
            _encode_in(row, sym, enc)
            found = True
            break
        _encode_in(row, esc, enc)
        row.update(esc)
    if not found:
        _encode_in(rungs[-1], sym, enc)
    for row in rungs:
        row.update(sym)


def _take(rungs: list[Counts], esc: int, dec: ArithDecoder) -> int:
    for row in rungs[:-1]:
        sym = _decode_in(row, dec)
        if sym != esc:
            for r in rungs:
                r.update(sym)
            return sym
        row.update(esc)
    sym = _decode_in(rungs[-1], dec)
    if sym == esc:
        rungs[-1].update(esc)
        return esc
    for row in rungs:
        row.update(sym)
    return sym


def _row(n: int, escape: int, init: int = 0) -> Counts:
    table = np.full(n, init, dtype=np.int64)
    table[escape] = 1
    return Counts(n, init=table)


class ByteArm:
    """Order-2 adaptive byte codec with order-1 and order-0 escapes."""

    __slots__ = ("o2", "o1", "o0", "prev2", "prev1")

    def __init__(self) -> None:
        self.o2: dict[int, Counts] = {}
        self.o1: dict[int, Counts] = {}
        self.o0 = Counts(256, init=np.ones(256, dtype=np.int64))
        self.prev2 = -1
        self.prev1 = -1

    def _rungs(self) -> list[Counts]:
        if self.prev1 >= 0:
            one = self.o1.get(self.prev1)
            if one is None:
                one = self.o1[self.prev1] = _row(257, ESC)
            if self.prev2 >= 0:
                key = self.prev2 << 8 | self.prev1
                row = self.o2.get(key)
                if row is None:
                    row = self.o2[key] = _row(257, ESC)
                return [row, one, self.o0]
            return [one, self.o0]
        return [self.o0]

    def _push(self, sym: int) -> None:
        self.prev2, self.prev1 = self.prev1, sym

    def encode_one(self, sym: int, enc: ArithEncoder) -> None:
        _emit(self._rungs(), sym, ESC, enc)
        self._push(sym)

    def decode_one(self, dec: ArithDecoder) -> int:
        sym = _take(self._rungs(), ESC, dec)
        self._push(sym)
        return sym

    def encode(self, data: bytes) -> bytes:
        enc = ArithEncoder()
        for sym in data:
            self.encode_one(sym, enc)
        return enc.finish()

    def decode(self, blob: bytes, n: int) -> bytes:
        dec = ArithDecoder(blob)
        return bytes(self.decode_one(dec) for _ in range(n))


@dataclass(frozen=True, slots=True)
class Lexicon:
    units: tuple[bytes, ...]
    sha: str = ""
    index: dict[bytes, int] = field(default=None, repr=False, compare=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.index is None:
            object.__setattr__(self, "index", {u: i for i, u in enumerate(self.units)})

    def blob(self) -> bytes:
        out = bytearray(LEX_MAGIC + struct.pack("<II", FRAME_VERSION, len(self.units)))
        for unit in self.units:
            if len(unit) > 0xFFFF:
                raise ValueError("lexicon unit too long")
            out += struct.pack("<H", len(unit))
            out += unit
        return bytes(out)

    @classmethod
    def from_blob(cls, blob: bytes) -> Lexicon:
        if blob[:4] != LEX_MAGIC:
            raise ValueError("lexicon: bad magic")
        _, count = struct.unpack_from("<II", blob, 4)
        pos = 12
        units = []
        for _ in range(count):
            n = struct.unpack_from("<H", blob, pos)[0]
            pos += 2
            units.append(blob[pos : pos + n])
            pos += n
        return cls(tuple(units), sha=_sha(blob))

    def offsets(self) -> list[int]:
        """The byte offset of each unit's data inside `blob()`."""
        out = []
        pos = 12
        for unit in self.units:
            out.append(pos + 2)
            pos += 2 + len(unit)
        return out

    def drop_prefix(self, prefix: bytes) -> "Lexicon":
        return Lexicon(tuple(u for u in self.units if not u.startswith(prefix)))


def _sha(blob: bytes) -> str:
    import hashlib

    return hashlib.sha256(blob).hexdigest()


def make_trie(units: tuple[bytes, ...]) -> dict:
    root: dict = {}
    for i, unit in enumerate(units):
        node = root
        for b in unit:
            node = node.setdefault(b, {})
        node["#"] = i
    return root


def greedy_parse(data: bytes, lex: Lexicon, cap: int = 16) -> tuple[list[int], int]:
    """Longest-match parse.  A literal byte is `-1 - byte`."""
    trie = make_trie(lex.units)
    out: list[int] = []
    covered = 0
    i = 0
    n = len(data)
    while i < n:
        node = trie
        best = 0
        best_id = -1
        for j in range(i, min(n, i + cap)):
            node = node.get(data[j])
            if node is None:
                break
            hit = node.get("#")
            if hit is not None:
                best = j - i + 1
                best_id = hit
        if best_id >= 0:
            out.append(best_id)
            covered += best
            i += best
        else:
            out.append(-1 - data[i])
            i += 1
    return out, covered


def parse_stats(
    parse: list[int], lex: Lexicon, nbytes: int, covered: int | None = None
) -> dict[str, float | int]:
    lens = [len(lex.units[s]) if s >= 0 else 1 for s in parse]
    hit = (
        covered
        if covered is not None
        else sum(n for s, n in zip(parse, lens) if s >= 0)
    )
    return {
        "units": len(parse),
        "covered": int(hit),
        "coverage": hit / max(nbytes, 1),
        "mean_unit": sum(lens) / max(len(lens), 1),
        "literals": sum(s < 0 for s in parse),
    }


class UnitArm:
    """Codec over installed unit ids; a literal escapes to the byte arm."""

    __slots__ = ("lex", "rows", "o0", "prev", "bytes")

    def __init__(self, lex: Lexicon) -> None:
        self.lex = lex
        n = len(lex.units) + 1
        self.rows: dict[int, Counts] = {}
        self.o0 = Counts(n, init=np.ones(n, dtype=np.int64))
        self.o0.counts[-1] = 1
        self.prev = -1
        self.bytes = ByteArm()

    @property
    def esc(self) -> int:
        return len(self.lex.units)

    def _rungs(self) -> list[Counts]:
        if self.prev < 0:
            return [self.o0]
        row = self.rows.get(self.prev)
        if row is None:
            row = self.rows[self.prev] = _row(self.esc + 1, self.esc)
        return [row, self.o0]

    def _push(self, sym: int) -> None:
        self.prev = sym

    def encode_one(self, sym: int, enc: ArithEncoder) -> None:
        if sym < 0:
            self._emit_literal(enc)
            self.bytes.encode_one(-1 - sym, enc)
            return
        _emit(self._rungs(), sym, self.esc, enc)
        self._push(sym)

    def _emit_literal(self, enc: ArithEncoder) -> None:
        for row in self._rungs():
            _encode_in(row, self.esc, enc)
            row.update(self.esc)
        self._push(self.esc)

    def decode_one(self, dec: ArithDecoder) -> int:
        sym = _take(self._rungs(), self.esc, dec)
        if sym == self.esc:
            self._push(self.esc)
            return -1 - self.bytes.decode_one(dec)
        self._push(sym)
        return sym

    def encode(self, data: bytes, parse: list[int] | None = None) -> bytes:
        if parse is None:
            parse, _ = greedy_parse(data, self.lex)
        enc = ArithEncoder()
        for sym in parse:
            self.encode_one(sym, enc)
        return enc.finish()

    def decode(self, blob: bytes, n: int) -> bytes:
        dec = ArithDecoder(blob)
        out = bytearray()
        while len(out) < n:
            sym = self.decode_one(dec)
            if sym < 0:
                out.append(-1 - sym)
            else:
                out += self.lex.units[sym]
            if len(out) > n:
                raise ValueError("unit arm overshot the frame length")
        return bytes(out)

    def trace(self, blob: bytes, n: int) -> tuple[bytes, list[tuple[int, int, int]]]:
        """Decode with `(start, end, sym)` provenance for every element."""
        dec = ArithDecoder(blob)
        out = bytearray()
        spans: list[tuple[int, int, int]] = []
        while len(out) < n:
            start = len(out)
            sym = self.decode_one(dec)
            if sym < 0:
                out.append(-1 - sym)
            else:
                out += self.lex.units[sym]
            if len(out) > n:
                raise ValueError("unit arm overshot the frame length")
            spans.append((start, len(out), sym))
        return bytes(out), spans


def build_lexicon(
    train: bytes, size: int = 8192, min_count: int = 16, cap: int = 16
) -> Lexicon:
    """Induce the lexicon from substring counts; score is bytes saved."""
    a = np.frombuffer(train, dtype=np.uint8).astype(np.uint64)
    counts: dict[bytes, int] = {}
    for ell in range(2, cap + 1):
        m = len(a) - ell + 1
        if m <= 0:
            break
        h = np.zeros(m, dtype=np.uint64)
        for j in range(ell):
            h *= np.uint64(257)
            h += a[j : j + m]
        _, idx, cnt = np.unique(h, return_index=True, return_counts=True)
        keep = cnt >= min_count
        for i, c in zip(idx[keep].tolist(), cnt[keep].tolist()):
            s = bytes(train[i : i + ell])
            counts[s] = int(c)
    scored = [(c * (len(s) - 1) - (len(s) + 4), s) for s, c in counts.items()]
    scored.sort(key=lambda x: (-x[0], x[1]))
    units = tuple(s for gain, s in scored[:size] if gain > 0)
    return Lexicon(units, sha=_sha(b"".join(units)))


@dataclass(frozen=True, slots=True)
class Frame:
    arm: int
    n: int
    corpus_sha: str
    train_sha: str
    lexicon: Lexicon | None
    payload: bytes
    header_bytes: int


def pack_frame(
    arm: int,
    n: int,
    payload: bytes,
    corpus_sha: str,
    train_sha: str,
    lexicon: Lexicon | None = None,
) -> bytes:
    lex_blob = lexicon.blob() if lexicon is not None else b""
    cs = corpus_sha.encode()[:12].ljust(12, b"\0")
    ts = train_sha.encode()[:12].ljust(12, b"\0")
    head = struct.pack(
        "<4sBBQQ12s12sI",
        FRAME_MAGIC,
        FRAME_VERSION,
        arm,
        n,
        len(payload),
        cs,
        ts,
        len(lex_blob),
    )
    return head + lex_blob + payload


def unpack_frame(blob: bytes) -> Frame:
    fields = struct.unpack_from("<4sBBQQ12s12sI", blob, 0)
    magic, version, arm, n, payload_len, cs, ts, lex_len = fields
    if magic != FRAME_MAGIC:
        raise ValueError("frame: bad magic")
    if version != FRAME_VERSION:
        raise ValueError(f"frame: bad version {version}")
    pos = 50
    lex = None
    if lex_len:
        lex = Lexicon.from_blob(blob[pos : pos + lex_len])
        pos += lex_len
    payload = blob[pos : pos + payload_len]
    if len(payload) != payload_len:
        raise ValueError("frame: truncated payload")
    return Frame(
        arm,
        n,
        cs.rstrip(b"\0").decode(),
        ts.rstrip(b"\0").decode(),
        lex,
        payload,
        pos,
    )


def encode_byte(data: bytes, corpus_sha: str = "") -> bytes:
    payload = ByteArm().encode(data)
    return pack_frame(0, len(data), payload, corpus_sha, corpus_sha)


def decode_frame(blob: bytes) -> bytes:
    frame = unpack_frame(blob)
    if frame.arm == 0:
        return ByteArm().decode(frame.payload, frame.n)
    if frame.arm == 1:
        if frame.lexicon is None:
            raise ValueError("frame: unit arm needs a lexicon")
        return UnitArm(frame.lexicon).decode(frame.payload, frame.n)
    raise ValueError(f"frame: unknown arm {frame.arm}")


def _random_freq(rng: random.Random, n: int) -> np.ndarray:
    raw = [rng.randint(1, 1000) for _ in range(n)]
    return np.asarray(raw, dtype=np.int64)


def selftest() -> dict[str, float | int]:
    """10k random symbols and a 200 kB byte round-trip."""
    import time

    t0 = time.perf_counter()
    rng = random.Random(0)
    fixed = 0
    for trial in range(20):
        n = rng.randint(2, 300)
        freqs = _random_freq(rng, n)
        cums = np.concatenate(([0], np.cumsum(freqs)))
        total = int(cums[-1])
        syms = []
        for _ in range(500):
            target = rng.randrange(total)
            sym = int(np.searchsorted(cums, target, side="right")) - 1
            syms.append(sym)
        enc = ArithEncoder()
        for sym in syms:
            enc.encode(int(cums[sym]), int(cums[sym + 1]), total)
        blob = enc.finish()
        dec = ArithDecoder(blob)
        out = []
        for _ in syms:
            target = dec.target(total)
            sym = int(np.searchsorted(cums, target, side="right")) - 1
            dec.update(int(cums[sym]), int(cums[sym + 1]), total)
            out.append(sym)
        if out != syms:
            raise AssertionError(f"fixed table {trial} failed")
        fixed += len(syms)

    adaptive = 0
    for trial in range(10):
        n = rng.randint(2, 257)
        init = _random_freq(rng, n)
        syms = [rng.randrange(n) for _ in range(1000)]
        enc = ArithEncoder()
        row = Counts(n, init=init)
        for sym in syms:
            _encode_in(row, sym, enc)
            row.update(sym)
        blob = enc.finish()
        dec = ArithDecoder(blob)
        row = Counts(n, init=init)
        out = []
        for _ in syms:
            sym = _decode_in(row, dec)
            row.update(sym)
            out.append(sym)
        if out != syms:
            raise AssertionError(f"adaptive table {trial} failed")
        adaptive += len(syms)

    text = bytes((i * 37 + i // 7) & 0xFF for i in range(200_000))
    blob = ByteArm().encode(text)
    if ByteArm().decode(blob, len(text)) != text:
        raise AssertionError("200 kB byte round-trip failed")
    elapsed = time.perf_counter() - t0
    return {
        "fixed_symbols": fixed,
        "adaptive_symbols": adaptive,
        "byte_bytes": len(text),
        "byte_bpb": 8 * len(blob) / len(text),
        "elapsed_s": elapsed,
    }
