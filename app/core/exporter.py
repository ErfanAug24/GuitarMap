import json
import csv
from pathlib import Path
import mido
from typing import List, Union, Dict
import math

from app.utils.logging_config import get_logger
from app.utils.timing import timing
from app.config.settings import settings

logger = get_logger(__name__)


class Exporter:
    """
    Handles exporting detected notes, tabs,
    and metadata into different formats (JSON, CSV, MIDI)
    """

    def __init__(self, output_dir: Union[str, Path] = None) -> None:
        self.output_dir = Path(output_dir or settings.EXPORTS_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Exporter initialized with directory: {self.output_dir}")

    def _validate_data(self, data: List[Dict]) -> None:
        if not data:
            raise ValueError("No data to export.")
        if not isinstance(data[0], dict):
            raise TypeError("Data must be a list of dictionaries.")

    def _resolve_filename(self, filename: str, extension: str) -> Path:
        """
        Build and sanitize the output filename.
        """
        if not filename.endswith(f".{extension}"):
            filename += f".{extension}"
        path = self.output_dir / filename
        logger.debug(f"Resolved export path: {path}")
        return path

    @timing
    def to_json(self, data: List[dict], filename: str = "result.json") -> Path:
        """
        Save a list of note/tab dictionaries to JSON.
        :param data: (list[dict])
        :param filename: (str) Output filename
        :return: (Path(filename))
        """
        self._validate_data(data)
        path = self._resolve_filename(filename or settings.DEFAULT_EXPORT_NAME, "json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                logger.info(f"Exported {len(data)} items to JSON: {path.name}")
            return path
        except Exception as e:
            logger.error(f"Failed to export JSON: {e}")
            raise

    @timing
    def to_csv(self, data: List[dict], filename: str = "result.csv") -> Path:
        """
        Export the same data as a CSV file.
        :param data: (list[dict])
        :param filename: (str) Output filename
        :return: (Path(filename))
        """
        self._validate_data(data)
        path = self._resolve_filename(filename or settings.DEFAULT_EXPORT_NAME, "csv")
        fieldnames = list(data[0].keys())
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            logger.info(f"Exported {len(data)} items to CSV: {path.name}")
            return path
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
            raise

    @timing
    def to_midi(self, data: List[Dict], filename: str = None, tempo=500000) -> Path:
        """
        Convert detected notes into a simple monophonic MIDI file.
        :param data: (list[dict]): [{'time': float, 'note': str, 'freq': float}]
        :param filename: (str) MIDI output name.
        :param tempo: (int) Microseconds per beat (default 120 BPM).
        :return: (Path(filename))
        """

        self._validate_data(data)
        path = self._resolve_filename(filename or settings.DEFAULT_EXPORT_NAME, "mid")
        try:
            mid = mido.MidiFile()
            track = mido.MidiTrack()
            mid.tracks.append(track)
            track.append(mido.MetaMessage("set_tempo", tempo=tempo))

            for note in data:
                if "freq" not in note or "time" not in note:
                    logger.warning(f"Skipping incomplete note entry: {note}")
                    continue
                midi_num = self._freq_to_midi(note["freq"])
                tick = int(note["time"] * mid.ticks_per_beat)
                track.append(mido.Message("note_on", note=midi_num, velocity=90, time=tick))
                track.append(mido.Message("note_off", note=midi_num, velocity=90, time=int(mid.ticks_per_beat / 2)))

            mid.save(path)
            logger.info(f"Exported {len(data)} notes to MIDI: {path.name}")
            return path
        except Exception as e:
            logger.error(f"Failed to export MIDI: {e}")
            raise

    @staticmethod
    def _freq_to_midi(freq) -> int:
        """
        Convert frequency (Hz) to MIDI note number.
        :param freq: Frequency
        :return: (int)
        """
        if freq <= 0:
            raise ValueError(f"Invalid frequency for MIDI conversion: {freq}")
        return int(round(69 + 12 * math.log2(freq / 440.0)))
