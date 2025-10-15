from .audio_loader import AudioLoader
from .separation import AudioSeparator
from .onset_detection import OnsetDetector
from .pitch_detection import PitchDetector
from .tab_mapping import GuitarTabMapper
from .exporter import Exporter
import pathlib


class GuitarAnalysisPipeline:
    """
    Full pipeline for guitar tab extraction from an audio file.
    """

    def __init__(self, sr: int = 44100, separation_model: str = "htdemucs") -> None:
        self.sr = sr
        self.loader = AudioLoader(target_sr=sr)
        self.separator = AudioSeparator(model_name=separation_model)
        self.onset_detector = OnsetDetector(sr=sr)
        self.pitch_detector = PitchDetector(sr=sr)
        self.tab_mapper = GuitarTabMapper()
        self.exporter = Exporter()

    def run(self, file_path: str, export_name: str = None) -> dict:
        """
        Run the full analysis pipeline:
        1. Load audio
        2. Separate guitar stem
        3. Detect onsets
        4. Detect pitch
        5. Map to tab
        6. Export results

        :param file_path: (str) Path to input audio file
        :param export_name: (str) Base name for exported files
        :return: (dict) {'json': path_to_json, 'csv': path_to_csv, 'tabs': list_of_notes}
        """
        print("Loading audio...")
        audio, sr = self.loader.load(file_path)

        print("Separating guitar stem...")
        stems = self.separator.separate(file_path)
        guitar_file = stems.get("other") or stems.get("guitar")
        if guitar_file:
            audio, sr = self.loader.load(guitar_file)

        print("Detecting onsets...")
        onsets = self.onset_detector.detect(audio)

        print("Detecting pitches...")
        notes = self.pitch_detector.detect(audio)

        print("Mapping notes to tab...")
        tabs = self.tab_mapper.map_sequence(notes)
        export_tabs = [tab.model_dump() for tab in tabs]
        base_name = export_name or file_path.split("/")[-1].split(".")[0]
        print("Exporting results...")
        json_path = self.exporter.to_json(export_tabs, filename=f"{base_name}.json")
        csv_path = self.exporter.to_csv(export_tabs, filename=f"{base_name}.csv")
        mid_path = self.exporter.to_midi(export_tabs, filename=f"{base_name}.mid")

        print("Pipeline complete.")
        return {"json":json_path,
                "csv":csv_path,
                "mid":mid_path,
                "tabs":tabs}

if __name__ == "__main__":
    current_dir = pathlib.Path(__file__).parent
    audio_file = current_dir.parent.parent / "exports" / "Faust.mp3"
    pipeline = GuitarAnalysisPipeline()
    pipeline.run(str(audio_file), export_name="Faust_analysis")
