import json
import csv
from pathlib import Path
import mido
from typing import List
import math


class Exporter:
    """
    Handles exporting detected notes, tabs,
    and metadata into different formats (JSON, CSV, MIDI)
    """

    def __init__(self, output_dir: str = "exports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def to_json(self, data: List[dict], filename: str = "result.json") -> Path:
        """
        Save a list of note/tab dictionaries to JSON.
        :param data: (list[dict])
        :param filename: (str) Output filename
        :return: (Path(filename))
        """
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path

    def to_csv(self, data: List[dict], filename: str = "result.csv") -> Path:
        """
        Export the same data as a CSV file.
        :param data: (list[dict])
        :param filename: (str) Output filename
        :return: (Path(filename))
        """
        if not data:
            raise ValueError("No data to export.")
        path = self.output_dir / filename
        fieldnames = list(data[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
        return path

    def to_midi(self, data: List[dict], filename: str = "result.midi", tempo=500000) -> Path:
        """
        Convert detected notes into a simple monophonic MIDI file.
        :param data: (list[dict]): [{'time': float, 'note': str, 'freq': float}]
        :param filename: (str) MIDI output name.
        :param tempo: (int) Microseconds per beat (default 120 BPM).
        :return: (Path(filename))
        """

        path = self.output_dir / filename
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=tempo))

        for note in data:
            if "freq" not in note:
                continue
            midi_num = self._freq_to_midi(note["freq"])
            tick = int(note["time"] * mid.ticks_per_beat)
            track.append(mido.Message("note_on", note=midi_num, velocity=90, time=tick))
            track.append(mido.Message("note_off", note=midi_num, velocity=90, time=int(mid.ticks_per_beat / 2)))

        mid.save(path)
        return path

    @staticmethod
    def _freq_to_midi(freq) -> int:
        """
        Convert frequency (Hz) to MIDI note number.
        :param freq: Frequency
        :return: (int)
        """
        return int(round(69 + 12 * (math.log2(freq / 440.0))))
