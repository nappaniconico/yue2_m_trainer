import numpy as np
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
