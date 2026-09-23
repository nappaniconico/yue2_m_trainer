import runpy
from pathlib import Path

SCRIPT = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "transcribe_abc.py"), run_name="test_module")


def test_template_is_replaceable_but_reviewed_abc_is_protected(tmp_path: Path) -> None:
    target = tmp_path / "song.abc"
    assert SCRIPT["_needs_transcription"](target, False)
    target.write_text("X:1\nT:REPLACE WITH REVIEWED SHEETSAGE2 ABC\n", encoding="utf-8")
    assert SCRIPT["_needs_transcription"](target, False)
    target.write_text("X:1\nK:C\nCDEF|\n", encoding="utf-8")
    assert not SCRIPT["_needs_transcription"](target, False)
    assert SCRIPT["_needs_transcription"](target, True)


def test_extract_and_atomic_write_abc(tmp_path: Path) -> None:
    target = tmp_path / "song.abc"
    abc = SCRIPT["_extract_abc"]({"abc": "X:1\nK:C\nCDEF|"})
    SCRIPT["_write_abc"](target, abc)
    assert target.read_text(encoding="utf-8") == "X:1\nK:C\nCDEF|\n"
    assert not (tmp_path / ".song.abc.tmp").exists()
