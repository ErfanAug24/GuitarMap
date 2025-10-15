from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class NoteSegment(BaseModel):
    """
    Represents a single detected note within a segment of audio.
    """
    time: float = Field(..., description="Time of note onset in seconds")
    freq: float = Field(..., gt=0, description="Frequency in Hz")
    note: str = Field(..., description="Musical note name, e.g., A4")
    string: Optional[int] = Field(None, description="Guitar string number (1-6)")
    fret: Optional[int] = Field(None, description="Fret number on the string")
    velocity: Optional[int] = Field(90, description="MIDI velocity or intensity")

    @field_validator("note")
    def note_must_be_valid(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Note must be a non-empty string")
        return v


class Segment(BaseModel):
    """
    Represents a segment of audio, containing one or more notes.
    """
    start_time: float = Field(..., description="Segment start time in seconds")
    end_time: float = Field(..., description="Segment end time in seconds")
    notes: List[NoteSegment] = Field(default_factory=list, description="List of notes in this segment")

    @field_validator("end_time")
    def end_after_start(cls, v, values):
        start = values.get("start_time")
        if start is not None and v <= start:
            raise ValueError("end_time must be greater than start_time")
        return v

    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end_time - self.start_time

    def add_note(self, note: NoteSegment) -> None:
        """Add a note to this segment."""
        self.notes.append(note)
