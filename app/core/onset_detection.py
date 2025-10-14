import librosa
import numpy as np


class OnsetDetector:
    """
    Detects note onsets (beginnings of notes or riffs) from and audio waveform
    """

    def __init__(self, sr: int = 44100, backtrack: bool = True, sensitivity: float = 1.0) -> None:
        """

        :param sr: (int) sample rate of audio input
        :param backtrack: (bool) if True, moves onset slightly earlier for precisions
        :param sensitivity: (float) Multiplier for onset thresholding (lower = more sensitive)
        """
        self.sr = sr
        self.backtrack = backtrack
        self.sensitivity = sensitivity

    def detect(self, waveform: np.ndarray) -> np.ndarray:
        """
        Detect onset times in seconds.
        :param waveform: (np.ndarray) Loaded an normalized audio signal
        :return: (np.ndarray) Array of onset times in seconds
        """
        onset_env = librosa.onset.onset_strength(y=waveform, sr=self.sr)
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
        return onset_times
