import pytest
import numpy as np
from app.core.tab_mapping import GuitarTabMapper


@pytest.fixture
def mapper():
    return GuitarTabMapper(frets=12)


def test_fretboard_generation(mapper):
    fb = mapper.fretboard

    assert fb.shape == (6, 13)

    assert np.isclose(fb[0, 0], 82.41)

    assert fb[0, -1] > fb[0, 0]


def test_map_note_zero_or_negative(mapper):
    result = mapper.map_note(0)
    assert result == []

    result = mapper.map_note(-100)
    assert result == []


def test_map_note_exact_string(mapper):
    result = mapper.map_note(82.41, tolerance_cents=0.1)
    assert len(result) >= 1
    assert result[0]['string'] == 1
    assert result[0]['fret'] == 0
    assert abs(result[0]['error_cents']) < 0.2


def test_map_note_near_fret(mapper):
    result = mapper.map_note(82.5, tolerance_cents=5)
    assert any(r['string'] == 1 for r in result)


def test_map_note_no_match(mapper):
    result = mapper.map_note(1000, tolerance_cents=5)
    assert result == []


def test_map_sequence_empty(mapper):
    result = mapper.map_sequence([])
    assert result == []


def test_map_sequence_single_note(mapper):
    note_seq = [{'time': 0.0, 'freq': 82.41, 'note': 'E2'}]
    mapped = mapper.map_sequence(note_seq)
    assert len(mapped) == 1
    assert mapped[0]['string'] == 1
    assert mapped[0]['fret'] == 0


def test_map_sequence_multiple_notes(mapper):
    note_seq = [
        {'time': 0.0, 'freq': 82.41, 'note': 'E2'},  # string 1
        {'time': 0.5, 'freq': 110.0, 'note': 'A2'},  # string 2
        {'time': 1.0, 'freq': 146.83, 'note': 'D3'}  # string 3
    ]
    mapped = mapper.map_sequence(note_seq)
    assert len(mapped) == 3
    strings = [m['string'] for m in mapped]
    assert strings == sorted(strings)


def test_map_sequence_closest_string(mapper):
    note_seq = [
        {'time': 0.0, 'freq': 82.41, 'note': 'E2'},  # string 1
        {'time': 0.5, 'freq': 87.31, 'note': 'F2'}  # could be string 1 fret 1 or string 2 fret 0
    ]
    mapped = mapper.map_sequence(note_seq)
    assert mapped[1]['string'] == 1
