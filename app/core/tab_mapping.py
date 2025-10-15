import numpy as np
from typing import List, Optional, Dict
from app.utils.logging_config import get_logger
from app.utils.timing import timing
from app.models.segment import NoteSegment

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']


class GuitarTabMapper:
    """
    Maps detects pitches (frequencies or note names) to guitar fret positions.
    Supports standard tuning by default (E2, A2, D3, G3, B3, E4).
    """

    def __init__(self, tuning: Optional[List[tuple]] = None, frets: int = 22) -> None:
        self.log = get_logger(__name__)
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
        self.log.info(f"GuitarTabMapper initialized with {len(self.tuning)} strings and {self.frets} frets.")

    def _generate_fretboard(self) -> np.ndarray:
        """
        Precompute all possible fret frequencies for each string.
        :return: 2D numpy array (strings x frets)
        """
        fretboard = []
        for _, freq in self.tuning:
            freqs = [freq * (2 ** (f / 12)) for f in range(self.frets + 1)]
            fretboard.append(freqs)
        return np.array(fretboard, dtype=float)

    @timing
    def map_note(self, freq: float, tolerance_cents: float = 25.0) -> List[dict]:
        """
        Map a detected frequency to possible string-fret positions.

        :param freq: (float) Frequency to detect note.
        :param tolerance_cents: (float) Allowed pitch deviation (in cents).
        :return: (list[dict]) [{ 'string': int, 'fret': int, 'error_cents': float }]
        """
        results = []
        if freq <= 0:
            self.log.warning(f"Ignored non-positive frequency: {freq}")
            return results
        for string_idx, freqs in enumerate(self.fretboard, start=1):
            cents_diff = 1200 * np.log2(freq / freqs)
            mask = np.abs(cents_diff) <= tolerance_cents
            valid_frets = np.where(mask)[0]

            for fret_idx in valid_frets:
                results.append({
                    'string': string_idx,
                    'fret': int(fret_idx),
                    'error_cents': float(cents_diff[fret_idx])
                })

        if not results:
            self.log.debug(f"No fret match found for {freq:.2f} Hz within {tolerance_cents} cents.")
        return results

    @timing
    def map_sequence(self, note_sequence: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """
        Map a sequence of detected notes to best-fit fretboard positions.
        Tries to minimize string jumps between consecutive notes.

        :param note_sequence: [{'time': float, 'freq': float, 'note': str}]
        :return: [{'time': float, 'freq': float, 'note': str, 'string': int, 'fret': int}]
        """
        mapped = []
        prev_string = None

        for note in note_sequence:
            freq = getattr(note, 'freq', 0)
            options = self.map_note(note.freq)
            if not options:
                continue

            if prev_string is not None:
                options.sort(key=lambda x: abs(x['string'] - prev_string))

            best = options[0]
            prev_string = best['string']

            mapped.append(
                NoteSegment(
                    time=note.time,
                    freq=note.freq,
                    note=note.note,
                    string=best['string'],
                    fret=best['fret'],
                    velocity=getattr(note, 'velocity', 90)  # preserve optional fields
                )
            )

        self.log.info(f"Mapped {len(mapped)} of {len(note_sequence)} notes to fretboard positions.")
        return mapped
