import librosa
import numpy as np
from pathlib import Path
from typing import Tuple

from app.utils.logging_config import get_logger
from app.utils.timing import timing
from app.config.settings import settings

logger = get_logger(__name__)


class AudioLoader:
    """
    Handles loading and preprocessing of audio files.
    Converts to mono, normalizes amplitude, and ensures consistent sample rate.
    """

    def __init__(self, target_sr: int = 44100, mono: bool = True) -> None:
        self.target_sr = target_sr or settings.SAMPLE_RATE
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
        if not file_path.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
            logger.warning(f"Suspicious file format for {file_path.name}. Attempting to load anyway.")
        return file_path

    @timing
    def load(self, file_path: str) -> Tuple[np.ndarray, float]:
        """
        Load an audio file into a normalized numpy array.
        :param file_path: path to the audio file (.wav, .mp3, etc.)
        :return: tuple(audio_data, sample_rate)
        """
        file_path = self._path(file_path)
        logger.info(f"Loading audio file: {file_path.name}")
        try:
            audio_data, sr = librosa.load(file_path, sr=self.target_sr, mono=self.mono)
            logger.debug(f"Loaded file {file_path.name} at sample rate {sr}, shape={audio_data.shape}")
            # Safe normalization: avoid division by zero
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
                logger.debug(f"Normalized audio with peak amplitude {max_val:.4f}")
            else:
                logger.warning(f"Audio file {file_path.name} is silent (max amplitude = 0).")

            return audio_data, sr
        except Exception as e:
            logger.error(f"Failed to load audio file {file_path}: {e}")
            raise RuntimeError(f"Failed to load audio file {file_path}: {e}")
