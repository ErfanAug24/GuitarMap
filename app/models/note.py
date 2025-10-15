from pydantic import BaseModel, Field
from typing import Optional


class Note(BaseModel):
    """
    Represents a single detected or generated note event.
    """
    time_start: float = Field(..., description="Start time of the note in seconds")
    time_end: float = Field(..., description="End time of the note in seconds")
    frequency: float = Field(..., description="Fundamental frequency in Hz")
    midi: int = Field(..., description="MIDI note number (0-127)")
    name: str = Field(..., description="Musical note name, e.g., A4, C#3")
    amplitude: Optional[float] = Field(None, description="Relative loudness or confidence")
    string: Optional[int] = Field(None, description="Guitar string number (1 = high E, 6 = low E)")
    fret: Optional[int] = Field(None, description="Fret number on the guitar neck")

    @property
    def duration(self) -> float:
        """
        Returns duration in seconds.
        :return:
        """
        return round(self.time_end - self.time_start, 3)

    def to_dict(self):
        """
        Convert to a dictionary for CSV export.
        :return:
        """
        return {
            "time_start": self.time_start,
            "time_end": self.time_end,
            "duration": self.duration,
            "frequency": self.frequency,
            "midi": self.midi,
            "name": self.name,
            "amplitude": self.amplitude,
            "string": self.string,
            "fret": self.fret
        }

    def __str__(self):
        return f"{self.name} ({self.frequency:.2f} Hz) from {self.time_start:.2f}s to {self.time_end:.2f}s"
