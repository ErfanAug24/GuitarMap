import os
import tempfile
from pathlib import Path
import torch
import torchaudio
from demucs import pretrained
from demucs.apply import apply_model


class AudioSeparator:
    """
    Uses Demucs model to separate an audio track into its stems (guitar, bass, drums, vocals, etc.)
    """

    def __init__(self, model_name="htdemucs") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = pretrained.get_model(model_name).to(self.device)
        self.model.eval()

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
        return file_path

    def separate(self, file_path: str, output_dir: str = None) -> dict:
        """
        Separate the input song into components using Demucs.

        :param file_path: (str) Path to the input audio file.
        :param output_dir: (str, optional) Where to store separated tracks.
        :return: (dict) Keys are stem names (e.g., 'guitar', 'bass', 'drums', 'vocals')
                    Values are paths to separated audio files.
        """

        file_path = self._path(file_path)
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="demucs_")

        wav, sr = torchaudio.load(file_path)
        wav = wav.to(self.device)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()

        print(f"Separating {file_path.name} on {self.device} using Demucs...")
        sources = apply_model(self.model, wav[None], split=True, progress=True)[0]
        sources = sources.cpu()

        names = self.model.sources
        output_paths = {}
        for name, source in zip(names, sources):
            out_path = Path(output_dir) / f"{file_path.stem}_{name}.wav"
            torchaudio.save(out_path, source, sr)
            output_paths[name] = str(out_path)
        return output_paths
