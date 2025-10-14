import numpy as np
import librosa
from typing import List, Union

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']


class PitchDetector:
    """
    Detects fundamental frequencies and converts them into musical notes.
    Designed for monophonic guitar signals.
    """

    def __init__(self, sr: int = 44100, fmin: float = 80.0, fmax: float = 1200.0):
        """
        :param sr: (int) Sample rate of the audio.
        :param fmin: (float) Minimum frequency to consider (E2 ≈ 82 Hz for guitar).
        :param fmax: (float) Maximum frequency to consider (around 1200 Hz covers high frets).
        """
        self.sr = sr
        self.fmin = fmin
        self.fmax = fmax

    def detect(self, waveform: np.ndarray, frame_length: int = 2048, hop_length: int = 256) -> List[dict]:
        """
        Perform continuous pitch detection over the audio.

        :param waveform: (np.ndarray) Input mono audio.
        :param frame_length: (int) Analysis window size.
        :param hop_length:  (int) Step size between analysis frames.
        :return: (list[dict]) Each dict contains {'time': float, 'freq': float, 'note': str}
        """
        pitches, magnitudes = librosa.piptrack(y=waveform, sr=self.sr, fmin=self.fmin, fmax=self.fmax,
                                               hop_length=hop_length)
        pitch_data = []
        for i in range(pitches.shape[1]):
            index = magnitudes[:, i].argmax()
            freq = pitches[index, i]
            if freq > 0:
                note_name = self.freq_to_note_name(freq)
                time = librosa.frames_to_time(i, sr=self.sr, hop_length=hop_length)
                pitch_data.append({'time': float(time), 'freq': float(freq), 'note': note_name})
        return pitch_data

    @staticmethod
    def freq_to_note_name(freq: float) -> Union[str, None]:
        """
        Convert a frequency to its nearest note name.
        :param freq: (float) Frequency to convert.
        :return: (str) Note name.
        """
        if freq <= 0:
            return None
        note_num = 12 * np.log2(freq / 440.0) + 69
        note_index = int(round(note_num)) % 12
        octave = int(round(note_num / 12)) - 1
        return f"{NOTE_NAMES[note_index]}{octave}"
