import pytest
import numpy as np
from app.core.audio_loader import AudioLoader
import soundfile as sf
from pathlib import Path
import os

TEST_FILE = "tests/test_files/Faust.mp3"
INVALID_FILE = "tests/test_files/non_existent.wav"


@pytest.fixture
def loader():
    return AudioLoader(target_sr=22050, mono=True)


def test_load_valid_file(loader):
    audio, sr = loader.load(TEST_FILE)
    assert isinstance(audio, np.ndarray)
    assert sr == 22050
    assert len(audio) > 0
    assert np.max(np.abs(audio)) <= 1.0


def test_load_nonexistent_file(loader):
    with pytest.raises(FileNotFoundError):
        loader.load(INVALID_FILE)


def test_stereo_conversion(tmp_path):
    path = tmp_path / "stereo.wav"
    sr = 22050
    waveform = np.random.randn(2, sr).astype(np.float32)
    import soundfile as sf
    sf.write(path, waveform.T, sr)

    loader_stereo = AudioLoader(target_sr=sr, mono=True)
    audio, sr_loaded = loader_stereo.load(path)

    assert audio.ndim == 1
    assert sr_loaded == sr


def test_no_normalization_zero_signal(tmp_path):
    path = tmp_path / "zeros.wav"
    sr = 22050
    waveform = np.zeros(sr).astype(np.float32)
    import soundfile as sf
    sf.write(path, waveform, sr)
    loader_zero = AudioLoader(target_sr=sr, mono=True)
    audio, sr_loaded = loader_zero.load(path)
    assert np.all(np.abs(audio) < 1e-8)
    assert sr_loaded == sr


def test_mono_false(tmp_path):
    sr = 22050
    waveform = np.zeros((sr, 2), dtype=np.float32)
    path = tmp_path / "stereo2.wav"
    sf.write(path, waveform, sr)
    loader_stereo = AudioLoader(target_sr=sr, mono=False)
    audio, sr_loaded = loader_stereo.load(path)

    assert audio.ndim == 2
    assert audio.shape[0] == 2
    assert audio.shape[1] == sr
    assert sr_loaded == sr

