import librosa
import numpy as np

from app.utils.logging_config import get_logger
from app.utils.timing import timing
from app.config.settings import settings


class OnsetDetector:
    """
    Detects note onsets (beginnings of notes or riffs) from and audio waveform
    """

    def __init__(self, sr: int = 44100, backtrack: bool = True, sensitivity: float = 1.0,
                 use_adaptive: bool = True) -> None:
        """

        :param sr: (int) sample rate of audio input
        :param backtrack: (bool) if True, moves onset slightly earlier for precisions
        :param sensitivity: (float) Multiplier for onset thresholding (lower = more sensitive)
        """
        cfg = settings.ONSET if hasattr(settings, "ONSET") else {}
        self.sr = sr or getattr(settings, "SAMPLE_RATE", 44100)
        self.backtrack = backtrack if backtrack is not None else cfg.get("BACKTRACK", True)
        self.sensitivity = sensitivity if sensitivity is not None else cfg.get("SENSITIVITY", 1.0)
        self.use_adaptive = use_adaptive

        self.logger = get_logger(__name__)
        self.logger.debug(f"OnsetDetector initialized with sr={self.sr}, backtrack={self.backtrack}, "
                          f"sensitivity={self.sensitivity}, adaptive={self.use_adaptive}")

    @timing
    def detect(self, waveform: np.ndarray) -> np.ndarray:
        """
        Detect onset times in seconds.
        :param waveform: (np.ndarray) Loaded an normalized audio signal
        :return: (np.ndarray) Array of onset times in seconds
        """
        if waveform is None or len(waveform) == 0:
            raise ValueError("Empty or invalid waveform provided to OnsetDetector.")

        self.logger.info("Starting onset detection...")
        onset_env = librosa.onset.onset_strength(y=waveform, sr=self.sr)
        threshold = self._adaptive_threshold(onset_env) if self.use_adaptive else self.sensitivity * 0.2
        self.logger.debug(f"Computed onset threshold: {threshold:.4f}")
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env,
                                                  sr=self.sr,
                                                  backtrack=self.backtrack,
                                                  pre_max=int(0.03 * self.sr / 512),
                                                  post_max=int(0.03 * self.sr / 512),
                                                  pre_avg=int(0.1 * self.sr / 512),
                                                  post_avg=int(0.1 * self.sr / 512),
                                                  delta=self.sensitivity * 0.2,
                                                  wait=int(0.03 * self.sr / 512)
                                                  )
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sr)
        self.logger.info(f"Detected {len(onset_times)} onsets.")
        return onset_times

    def _adaptive_threshold(self, onset_env: np.ndarray) -> float:
        """
        Dynamically adjusts threshold based on the energy range of the signal.
        :param onset_env: Onset envelope energy array
        :return: Adaptive delta threshold
        """
        energy_mean = np.mean(onset_env)
        energy_std = np.std(onset_env)
        threshold = max(0.05, min(0.5, (energy_std / (energy_mean + 1e-6)) * 0.2 * self.sensitivity))
        self.logger.debug(
            f"Adaptive threshold computed (mean={energy_mean:.4f}, std={energy_std:.4f}): {threshold:.4f}")
        return threshold
