import os
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# Base directory resolution
BASE_DIR = Path(__file__).resolve().parents[2]  # points to project root, not just /app/config

class Settings:
    """
    Central configuration object.
    Automatically reads environment variables but falls back to defaults.
    """
    def __init__(self):
        # --- Directories ---
        self.BASE_DIR = BASE_DIR
        self.CORE_DIR = self.BASE_DIR / "app" / "core"
        self.DATA_DIR = self.BASE_DIR / "data"
        self.MODELS_DIR = self.BASE_DIR / "app" / "models"
        self.EXPORTS_DIR = self.BASE_DIR / "exports"

        # Ensure output folders exist
        os.makedirs(self.EXPORTS_DIR, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)

        # --- Audio processing defaults ---
        self.SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 44100))
        self.FFT_WINDOW_SIZE = int(os.getenv("FFT_WINDOW_SIZE", 2048))
        self.HOP_LENGTH = int(os.getenv("HOP_LENGTH", 512))

        # --- Onset detection ---
        self.ONSET_THRESHOLD = float(os.getenv("ONSET_THRESHOLD", 0.35))
        self.SMOOTHING = float(os.getenv("SMOOTHING", 0.5))

        # --- Export defaults ---
        self.DEFAULT_EXPORT_FORMAT = os.getenv("DEFAULT_EXPORT_FORMAT", "csv")
        self.DEFAULT_EXPORT_NAME = os.getenv("DEFAULT_EXPORT_NAME", "analysis_result")

        # --- General ---
        self.DEBUG = os.getenv("DEBUG", "true").lower() in ("1", "true", "yes")

    def __repr__(self):
        return f"<Settings DEBUG={self.DEBUG} SAMPLE_RATE={self.SAMPLE_RATE}>"


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings instance so it's loaded once per process.
    Usage:
        from app.config.settings import get_settings
        settings = get_settings()
    """
    return Settings()


# Instantiate globally for quick imports if you prefer:
settings = get_settings()
