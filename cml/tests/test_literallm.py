import literallm

DATA = b"def foo(self, x):\n    return self.value + x\n" * 4
TEST = b"# one-off comment\n" + DATA


def torch_equal(a, b) -> bool:
    import torch

    return torch.equal(a, b)


def test_stream_keeps_length_and_byte_mask() -> None:
    vocab, lib = literallm.build_vocab(
        DATA, min_fill=4, min_count=1, max_n=8, max_holes=4, min_fixed=6
    )
    ids, mask = literallm.stream(TEST, vocab, lib, False, min_fill=4, max_n=8)
    ab_ids, ab_mask = literallm.stream(TEST, vocab, lib, True, min_fill=4, max_n=8)
    assert ids.shape == ab_ids.shape == mask.shape == ab_mask.shape
    assert bool(mask.any())
    assert not torch_equal(mask, ab_mask) or ids.tolist() != ab_ids.tolist()


def test_ablation_changes_event_tokens_only() -> None:
    import torch

    vocab, lib = literallm.build_vocab(
        DATA, min_fill=4, min_count=1, max_n=8, max_holes=4, min_fixed=6
    )
    ids, mask = literallm.stream(TEST, vocab, lib, False, min_fill=4, max_n=8)
    ab_ids, ab_mask = literallm.stream(TEST, vocab, lib, True, min_fill=4, max_n=8)
    assert torch.equal(mask, ab_mask)
    event = ~mask
    assert int((ids[event] != ab_ids[event]).sum()) > 0
    assert torch.equal(ids[mask], ab_ids[mask])
