import librosa
import numpy as np
from pathlib import Path
from typing import Tuple


class AudioLoader:
    """
    Handles loading and preprocessing of audio files.
    Converts to mono, normalizes amplitude, and ensures consistent sample rate.
    """

    def __init__(self, target_sr=44100, mono=True) -> None:
        self.target_sr = target_sr
        self.mono = mono

    @staticmethod
    def _path(file_path: str) -> Path:
        """
        Check if file exists.
        :param file_path: path to the audio file (.wav, .mp3, etc.)
        :return: Path(file_path)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        return file_path

    def load(self, file_path: str) -> Tuple[np.ndarray, float]:
        """
        Load an audio file into a normalized numpy array.
        :param file_path: path to the audio file (.wav, .mp3, etc.)
        :return: tuple(audio_data, sample_rate)
        """
        file_path = self._path(file_path)
        try:
            audio_data, sr = librosa.load(file_path, sr=self.target_sr, mono=self.mono)

            # Safe normalization: avoid division by zero
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val

            return audio_data, sr
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file {file_path}: {e}")
