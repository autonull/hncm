import torch

import eventlm

DATA = b"def a(x):\n    return x+1\ndef b(y):\n    return y+2\n" * 3


def test_library_events_and_alignment() -> None:
    lib = eventlm.build_library(DATA, min_count=1, max_n=8)
    stream = eventlm.events(DATA, lib, max_n=8)
    assert lib.keys and lib.fills and stream
    ids, kinds, lens, slots, fids, tids = eventlm.flatten(stream, lib)
    assert ids.shape == kinds.shape == lens.shape == slots.shape == fids.shape
    assert ids.shape == tids.shape
    limit = len(lib.keys)
    for token, kind in zip(ids.tolist(), kinds.tolist()):
        assert (kind == 0 and token < limit) or (
            kind != 0 and limit <= token < limit + len(lib.fills)
        )


def test_masked_ce_runs_on_cpu() -> None:
    lib = eventlm.build_library(DATA, min_count=1, max_n=8)
    stream = eventlm.events(DATA, lib, max_n=8)
    ids, kinds, _, _, _, _ = eventlm.flatten(stream, lib)
    logits = torch.randn(1, len(ids) - 1, lib.vocab)
    total, count = eventlm.masked_ce_sum(
        logits, ids[1:].unsqueeze(0), kinds[1:].unsqueeze(0), lib
    )
    assert count == len(ids) - 1
    assert torch.isfinite(total)


def test_split_keeps_every_event() -> None:
    lib = eventlm.build_library(DATA, min_count=1, max_n=8)
    stream = eventlm.events(DATA, lib, max_n=8)
    keep, held, seen = eventlm.split(stream, stream)
    assert keep == stream
    assert not held
    assert seen
