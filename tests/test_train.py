import numpy as np
import pytest
import torch
from yue2.protocol import ABC_END, CODEC_OFFSET, MUSIC_END, MUSIC_START

from yue2_lora.train import _sequence


class TinyTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(value) for value in text]


def _item() -> dict:
    return {
        "name": "song",
        "style": "style",
        "lyrics": "",
        "abc": "X:1\nK:C\nCDEF|",
        "codec": np.asarray([3, 7, 11], dtype=np.int32),
    }


def test_full_sequence_supervises_abc_and_audio_separately() -> None:
    sequence = _sequence(_item(), TinyTokenizer(), 512, False, "cpu", cot="full")
    ids = sequence.ids[0].tolist()
    assert sequence.abc_targets is not None
    abc_start, abc_stop = sequence.abc_targets
    audio_start, audio_stop = sequence.audio_targets
    assert ids[abc_stop - 1] == ABC_END
    assert ids[abc_stop] == MUSIC_START
    assert ids[abc_start : abc_stop - 1] == TinyTokenizer().encode(_item()["abc"])
    assert ids[audio_start:audio_stop] == [CODEC_OFFSET + 3, CODEC_OFFSET + 7, CODEC_OFFSET + 11, MUSIC_END]


def test_off_sequence_has_no_abc_targets() -> None:
    sequence = _sequence(_item(), TinyTokenizer(), 512, False, "cpu", cot="off")
    assert sequence.abc_targets is None
    audio_start, _ = sequence.audio_targets
    assert sequence.ids[0, audio_start - 1].item() == MUSIC_START


def test_abc_only_stops_at_abc_end_without_reading_codec() -> None:
    item = _item()
    del item["codec"]
    sequence = _sequence(item, TinyTokenizer(), 512, False, "cpu", cot="full", abc_only=True)
    assert sequence.audio_targets is None
    assert sequence.ids[0, -1] == ABC_END
    assert MUSIC_START not in sequence.ids[0]
    with pytest.raises(ValueError, match="requires"):
        _sequence(item, TinyTokenizer(), 512, False, "cpu", abc_only=True)


def test_abc_only_loss_has_no_audio_gradient(monkeypatch) -> None:
    import yue2_lora.train as training

    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(3, 8)
    hidden = torch.randn(1, 4, 3, requires_grad=True)
    monkeypatch.setattr(training, "_hidden", lambda *args: hidden)
    sequence = training.SupervisedSequence(torch.tensor([[1, 2, 3, 4]]), (1, 3), None)
    parts = training._loss(model, sequence, False, 1.0, 1.0)
    parts.total.backward()
    assert parts.audio is None
    assert parts.total is parts.abc
    assert hidden.grad[0, :2].abs().sum() > 0
    assert hidden.grad[0, 2:].abs().sum() == 0


def test_truncated_audio_does_not_train_false_end_of_song() -> None:
    full = _sequence(_item(), TinyTokenizer(), 512, False, "cpu")
    budget = full.audio_targets[0] + 3
    truncated = _sequence(_item(), TinyTokenizer(), budget, True, "cpu")
    assert truncated.ids[0, -1] != MUSIC_END
    with pytest.raises(ValueError, match="exceed"):
        _sequence(_item(), TinyTokenizer(), budget, False, "cpu")
