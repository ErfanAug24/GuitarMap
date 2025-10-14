import pytest
import numpy as np
from pathlib import Path
from app.core.separation import AudioSeparator
import torch


@pytest.fixture
def dummy_file(tmp_path):
    path = tmp_path / "dummy.wav"
    path.write_bytes(b"dummy")
    return str(path)


@pytest.fixture
def separator(monkeypatch):
    class DummyModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        sources = ["guitar", "bass", "drums", "vocals"]

    monkeypatch.setattr("app.core.separation.pretrained.get_model", lambda name: DummyModel())

    def fake_apply_model(model, wav, split=True, progress=True):
        import torch
        # simulate batch, stems, samples
        return torch.randn(len(model.sources), wav.shape[2]) * 0.1

    monkeypatch.setattr("app.core.separation.apply_model", fake_apply_model)

    def fake_load(file_path):
        import torch
        wav = torch.randn(2, 1024)
        sr = 22050
        return wav, sr

    monkeypatch.setattr("torchaudio.load", fake_load)
    monkeypatch.setattr("torchaudio.save", lambda *a, **kw: None)

    return AudioSeparator(model_name="dummy")


def test_path_exists(tmp_path, separator):
    file_path = tmp_path / "exists.wav"
    file_path.write_bytes(b"dummy")
    result = separator._path(str(file_path))
    assert result.exists()


def test_path_not_exists(separator):
    import pytest
    with pytest.raises(FileNotFoundError):
        separator._path("nonexistent.wav")


def test_separate_default_dir(dummy_file, separator, monkeypatch):
    output = separator.separate(dummy_file)
    assert set(output.keys()) == set(separator.model.sources)

    for path in output.values():
        assert isinstance(path, str)


def test_separate_custom_dir(dummy_file, separator, tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    output = separator.separate(dummy_file, output_dir=str(output_dir))
    assert set(output.keys()) == set(separator.model.sources)

    for path in output.values():
        assert Path(path).parent == output_dir


def test_device_attribute(separator):
    assert separator.device in ["cpu", "cuda"]
