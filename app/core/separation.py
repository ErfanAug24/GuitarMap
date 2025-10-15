import os
import tempfile
from pathlib import Path
import torch
import torchaudio
from demucs import pretrained
from demucs.apply import apply_model

from app.utils.logging_config import get_logger
from app.utils.timing import timing
from app.config.settings import settings


class AudioSeparator:
    """
    Uses Demucs model to separate an audio track into its stems (guitar, bass, drums, vocals, etc.)
    """

    def __init__(self, model_name="htdemucs") -> None:
        self.log = get_logger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.log.info(f"Initializing AudioSeparator on {self.device}")
        try:

            self.model = pretrained.get_model(model_name).to(self.device)
            self.model.eval()
            self.log.info(f"Loaded Demucs model: {model_name}")
        except Exception as e:
            self.log.exception(f"Failed to load Demucs model '{model_name}': {e}")
            raise RuntimeError(f"Cannot load model '{model_name}'") from e

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

    @timing
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
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            wav, sr = torchaudio.load(file_path)
            wav = wav.to(self.device)
            ref = wav.mean(0)
            wav = (wav - ref.mean()) / (ref.std() + 1e-8)
            self.log.info(f"Separating '{file_path.name}' into stems using Demucs...")

            print(f"Separating {file_path.name} on {self.device} using Demucs...")
            sources = apply_model(self.model, wav[None], split=True, progress=True)[0]
            sources = sources.cpu()
            stem_names = getattr(self.model, "sources", ["stem" + str(i) for i in range(len(sources))])

            output_paths = {}
            for name, source in zip(stem_names, sources):
                stem_file = output_dir / f"{file_path.stem}_{name}.wav"
                torchaudio.save(stem_file, source, sr)
                output_paths[name] = str(stem_file)
                self.log.debug(f"Saved {name} stem to {stem_file}")

            self.log.info(f"Separation complete. {len(output_paths)} stems generated.")
            return output_paths
        except Exception as e:
            self.log.exception(f"Failed to separate audio '{file_path}': {e}")
            raise RuntimeError(f"Separation failed for '{file_path}'") from e
