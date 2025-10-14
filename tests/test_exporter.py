import pytest
from app.core.exporter import Exporter
import math
import mido


@pytest.fixture
def exporter(tmp_path):
    return Exporter(output_dir=str(tmp_path))


def test_constructor_creates_dir(tmp_path):
    out_dir = tmp_path / "subdir"
    exporter = Exporter(output_dir=str(out_dir))
    assert out_dir.exists()


def test_to_json(exporter, tmp_path):
    data = [{"time": 0.0, "note": "A4", "freq": 440.0}]
    path = exporter.to_json(data, "test.json")
    assert path.exists()
    content = path.read_text()
    assert '"note": "A4"' in content


def test_to_csv(exporter, tmp_path):
    with pytest.raises(ValueError):
        exporter.to_csv([])


def test_freq_to_midi():
    from app.core.exporter import Exporter
    midi = Exporter._freq_to_midi(440.0)
    assert midi == 69
    midi_low = Exporter._freq_to_midi(220.0)
    assert midi_low == 57
    midi_high = Exporter._freq_to_midi(880.0)
    assert midi_high == 81


def test_to_midi_basic(exporter, tmp_path):
    data = [
        {"time": 0.0, "note": "A4", "freq": 440.0},
        {"time": 0.5, "note": "B4", "freq": 493.88}
    ]
    path = exporter.to_midi(data, "test.mid")
    assert path.exists()
    mid = mido.MidiFile(str(path))
    assert len(mid.tracks[0]) >= 4


def test_to_midi_skip_without_freq(exporter):
    data = [
        {"time": 0.0, "note": "A4"},  # missing freq
        {"time": 0.5, "note": "B4", "freq": 493.88}
    ]
    path = exporter.to_midi(data,"test_skip.midi")
    mid = mido.MidiFile(str(path))
    notes = [msg for msg in mid.tracks[0] if msg.type == "note_on"]
    assert len(notes) == 1

