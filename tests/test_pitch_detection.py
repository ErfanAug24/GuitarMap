import pytest
import numpy as np
import librosa
from app.core.pitch_detection import PitchDetector
from app.models.segment import NoteSegment


@pytest.fixture
def detector():
    return PitchDetector(sr=22050, fmin=80, fmax=1200)


def test_detect_empty_wave(detector):
    audio = np.array([])
    result = detector.detect(audio)
    # Should return an empty list of NoteSegment
    assert isinstance(result, list)
    assert len(result) == 0


def test_detect_zero_wave(detector):
    audio = np.zeros(1024, dtype=np.float32)
    result = detector.detect(audio)
    assert isinstance(result, list)
    assert len(result) == 0


def test_detect_single_sine(detector):
    sr = detector.sr
    t = np.linspace(0, 1, sr, endpoint=False)
    freq = 440
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    result = detector.detect(audio)
    assert isinstance(result, list)
    assert all(isinstance(x, NoteSegment) for x in result)
    # Expect at least one detected note near A4 (440 Hz)
    assert any(abs(x.freq - 440) < 5 for x in result)


def test_freq_to_note_name(detector):
    note = detector.freq_to_note_name(440)
    assert note.startswith('A') and note.endswith('4')

    note = detector.freq_to_note_name(0)
    assert note is None

    note = detector.freq_to_note_name(-10)
    assert note is None


def test_short_waveform(detector):
    audio = np.random.randn(10)
    result = detector.detect(audio)
    assert isinstance(result, list)


def test_no_pitch_detected(monkeypatch, detector):
    def fake_piptrack(*args, **kwargs):
        return np.zeros((1025, 5)), np.zeros((1025, 5))

    monkeypatch.setattr(librosa, 'piptrack', fake_piptrack)

    audio = np.random.randn(1024)
    result = detector.detect(audio)
    assert result == []


def test_multiple_frames(monkeypatch):
    import librosa

    detector = PitchDetector(silence_threshold=0.0, smoothing_window=1)

    pitches = np.zeros((1025, 3))
    magnitudes = np.zeros((1025, 3))
    pitches[10, 0] = 440  # A4
    magnitudes[10, 0] = 1.0
    pitches[20, 1] = 493.88  # B4
    magnitudes[20, 1] = 1.0
    pitches[30, 2] = 523.25  # C5
    magnitudes[30, 2] = 1.0

    def fake_piptrack(*args, **kwargs):
        return pitches, magnitudes

    monkeypatch.setattr(librosa, 'piptrack', fake_piptrack)

    # Use a dummy audio array of correct length
    audio = np.random.randn(1024)
    result = detector.detect(audio)

    note_names = {seg.note for seg in result}
    assert note_names == {"A4", "B4", "C4"}
