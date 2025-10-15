import numpy as np
import librosa
from typing import List, Union, Optional

from app.config.settings import settings
from app.utils.logging_config import get_logger
from app.utils.timing import timing
from app.models.segment import NoteSegment

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']


class PitchDetector:
    """
    Detects fundamental frequencies and converts them into musical notes.
    Designed for monophonic guitar signals.
    """

    def __init__(
            self,
            sr: Optional[int] = None,
            fmin: Optional[float] = None,
            fmax: Optional[float] = None,
            silence_threshold: float = 0.01,
            smoothing_window: int = 3
    ):
        cfg_sr = sr or settings.SAMPLE_RATE
        self.sr = cfg_sr
        self.fmin = fmin or 80.0
        self.fmax = fmax or 1200.0
        self.silence_threshold = silence_threshold
        self.smoothing_window = smoothing_window

        self.log = get_logger(__name__)
        self.log.debug(f"PitchDetector initialized: sr={self.sr}, fmin={self.fmin}, fmax={self.fmax}")

    @timing
    def detect(self, waveform: np.ndarray, frame_length: int = 2048, hop_length: int = 256) -> List[NoteSegment]:
        """
        Perform continuous pitch detection over the audio.

        :param waveform: (np.ndarray) Input mono audio.
        :param frame_length: (int) Analysis window size.
        :param hop_length:  (int) Step size between analysis frames.
        :return: (list[dict]) Each dict contains {'time': float, 'freq': float, 'note': str}
        """
        if waveform.size == 0:
            return []
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)
            self.log.debug("Converted stereo to mono for pitch detection.")

            # Normalize waveform safely
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val

        self.log.info(f"Running pitch detection (fmin={self.fmin}, fmax={self.fmax}, sr={self.sr})")

        pitches, magnitudes = librosa.piptrack(y=waveform, sr=self.sr, fmin=self.fmin, fmax=self.fmax,
                                               hop_length=hop_length)
        rms = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)[0]
        frequencies = np.zeros(pitches.shape[1])
        for i in range(pitches.shape[1]):
            if rms[i] > self.silence_threshold:
                idx = magnitudes[:, i].argmax()
                freq = pitches[idx, i]
                frequencies[i] = freq if freq > 0 else 0

        if self.smoothing_window > 1:
            frequencies = self._smooth(frequencies, self.smoothing_window)

        segments: List[NoteSegment] = []
        for i, freq in enumerate(frequencies):
            if freq > 0:
                note_name = self.freq_to_note_name(freq)
                time = librosa.frames_to_time(i, sr=self.sr, hop_length=hop_length)
                note_seg = NoteSegment(time=float(time), freq=float(freq), note=note_name)
                segments.append(note_seg)

        self.log.info(f"Detected {len(segments)} note events.")
        return segments

    @staticmethod
    def freq_to_note_name(freq: float) -> Union[str, None]:
        if freq <= 0:
            return None
        note_num = 12 * np.log2(freq / 440.0) + 69
        note_index = int(round(note_num)) % 12
        octave = int(note_num // 12) - 1
        return f"{NOTE_NAMES[note_index]}{octave}"

    @staticmethod
    def _smooth(values: np.ndarray, window: int) -> np.ndarray:
        """Simple moving average smoothing."""
        if len(values) < window:
            return values
        return np.convolve(values, np.ones(window) / window, mode="same")
