import torch

from trlm.memory import ScratchpadMemory


def test_empty_memory_has_expected_shapes() -> None:
    memory = ScratchpadMemory(d_model=8, slots=3)
    values, occupied = memory.empty(2, device=torch.device("cpu"), dtype=torch.float32)
    assert values.shape == (2, 3, 8)
    assert occupied.shape == (2, 3)


def test_first_write_is_fully_novel() -> None:
    memory = ScratchpadMemory(d_model=8, slots=3)
    hidden = torch.randn(2, 4, 8)
    values, occupied = memory.empty(2, device=hidden.device, dtype=hidden.dtype)
    _, occupied, raw, novelty, strength = memory.write(hidden, values, occupied, 0)
    assert torch.allclose(raw, torch.ones_like(raw))
    assert torch.allclose(novelty, torch.ones_like(novelty))
    assert occupied[:, 0].all()
    assert ((0.0 <= strength) & (strength <= 1.0)).all()


def test_soft_novelty_is_less_harsh() -> None:
    raw = torch.tensor([0.01, 0.04, 0.25, 1.0])
    softened = raw.pow(0.5)
    assert torch.all(softened[:-1] > raw[:-1])
    assert softened[-1] == raw[-1]


def test_recurrent_idea_merges_into_existing_slot() -> None:
    memory = ScratchpadMemory(d_model=8, slots=3, merge_threshold=0.15)
    hidden = torch.randn(2, 4, 8)
    values, occupied = memory.empty(2, device=hidden.device, dtype=hidden.dtype)
    values, occupied, *_ = memory.write(hidden, values, occupied, 0)
    _, occupied, *_ = memory.write(hidden, values, occupied, 1)
    assert torch.equal(occupied.sum(dim=1), torch.ones(2, dtype=torch.long))


def test_chunked_write_preserves_multiple_local_summaries() -> None:
    memory = ScratchpadMemory(d_model=8, slots=6, chunks=3, merge_threshold=0.0)
    hidden = torch.randn(2, 12, 8)
    values, occupied = memory.empty(2, device=hidden.device, dtype=hidden.dtype)
    values, occupied, raw, novelty, strength = memory.write(hidden, values, occupied, 0)
    assert occupied.sum(dim=1).tolist() == [3, 3]
    assert values.shape == (2, 6, 8)
    assert raw.shape == novelty.shape == strength.shape == (2,)
