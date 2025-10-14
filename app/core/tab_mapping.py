import numpy as np
from typing import List

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']


class GuitarTabMapper:
    """
    Maps detects pitches (frequencies or note names) to guitar fret positions.
    Supports standard tuning by default (E2, A2, D3, G3, B3, E4).
    """

    def __init__(self, tuning=None, frets=22) -> None:
        self.tuning = tuning or [
            ('E2', 82.41),
            ('A2', 110.00),
            ('D3', 146.83),
            ('G3', 196.00),
            ('B3', 246.94),
            ('E4', 329.63)
        ]
        self.frets = frets
        self.fretboard = self._generate_fretboard()

    def _generate_fretboard(self) -> np.ndarray:
        fretboard = []
        for string_index, (note_name, freq) in enumerate(self.tuning, start=1):
            freqs = [freq * (2 ** (f / 12)) for f in range(self.frets + 1)]
            fretboard.append(freqs)
        return np.array(fretboard)

    def map_note(self, freq: float, tolerance_cents: float = 25) -> List[dict]:
        """
        Map a detected frequency to possible string-fret positions.

        :param freq: (float) Frequency to detect note.
        :param tolerance_cents: (float) Allowed pitch deviation (in cents).
        :return: (list[dict]) [{ 'string': int, 'fret': int, 'error_cents': float }]
        """
        results = []
        if freq <= 0:
            return results
        for string_idx, freqs in enumerate(self.fretboard, start=1):
            for fret_idx, fret_freq in enumerate(freqs):
                cents_diff = 1200 * np.log2(freq / fret_freq)
                if abs(cents_diff) <= tolerance_cents:
                    results.append({'string': string_idx,
                                    'fret': fret_idx,
                                    'error_cents': float(cents_diff)})
        return results

    def map_sequence(self, note_sequence):
        """
        Map a list of detected notes to best fretboard positions.
        :param note_sequence: (list[dict]) [{'time': float, 'freq': float, 'note': str}]
        :return: (list[dict]) [{'time': float, 'freq': float, 'note': str,
                          'string': int, 'fret': int}]
        """
        mapped = []
        prev_string = None
        for note in note_sequence:
            options = self.map_note(note['freq'])
            if not options:
                continue

            if prev_string:
                options.sort(key=lambda x: abs(x['string'] - prev_string))
            best = options[0]
            prev_string = best['string']
            mapped.append({
                **note,
                'string': best['string'],
                'fret': best['fret']
            })
        return mapped
