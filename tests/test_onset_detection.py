import pytest
import numpy as np
from app.core.onset_detection import OnsetDetector


@pytest.fixture
def default_detector():
    return OnsetDetector(sr=22050)


@pytest.fixture
def custom_detector():
    return OnsetDetector(sr=22050, backtrack=False, sensitivity=0.5)


def test_detect_normal_wave(default_detector):
    sr = 22050
    t = np.linspace(0, 1, sr)

    audio = np.zeros(sr)
    audio[int(sr * 0.5)] = 1.0
    onsets = default_detector.detect(audio)
    assert isinstance(onsets, np.ndarray)
    assert len(onsets) >= 1
    assert np.all((onsets >= 0) & (onsets <= 1))


def test_detect_empty_wave(default_detector):
    audio = np.array([])
    with pytest.raises(ValueError, match="Empty or invalid waveform"):
        default_detector.detect(audio)


def test_detect_short_wave(default_detector):
    audio = np.array([0.1, 0.5, 0.2], dtype=np.float32)
    onsets = default_detector.detect(audio)
    assert isinstance(onsets, np.ndarray)
    assert len(onsets) >= 0


def test_detect_multiple_spikes(default_detector):
    sr = 22050
    audio = np.zeros(sr)
    spikes = [0.2, 0.5, 0.7]
    audio[[int(s * sr) for s in spikes]] = 1.0
    onsets = default_detector.detect(audio)
    assert len(onsets) >= len(spikes) - 1


def test_detect_custom_parameters(custom_detector):
    sr = 22050
    audio = np.zeros(sr)
    audio[int(sr * 0.25)] = 1.0
    audio[int(sr * 0.75)] = 1.0
    onsets = custom_detector.detect(audio)
    assert isinstance(onsets, np.ndarray)
    assert len(onsets) >= 1
    assert np.all((onsets >= 0) & (onsets <= 1))


def test_detect_backtrack_effect():
    sr = 22050
    audio = np.zeros(sr)
    audio[int(sr * 0.25)] = 1.0

    det1 = OnsetDetector(sr=sr, backtrack=True)
    det2 = OnsetDetector(sr=sr, backtrack=False)
    o1 = det1.detect(audio)
    o2 = det2.detect(audio)

    assert np.all(o1 <= o2 + 0.01)

