from pydantic import BaseModel, Field
from typing import List,Optional
from app.models.note import Note


class ExportPaths(BaseModel):
    """Holds the file paths for exported results."""
    json: Optional[str] = Field(None, description="Path to exported JSON file")
    csv: Optional[str] = Field(None, description="Path to exported CSV file")
    midi: Optional[str] = Field(None, description="Path to exported MIDI file")


class AnalysisResponse(BaseModel):
    """Represents the full response from a guitar analysis pipeline run."""

    success: bool = Field(True, description="Whether the analysis completed successfully")
    message: Optional[str] = Field(None, description="Optional status or error message")
    exports: ExportPaths = Field(..., description="File paths for exported data")
    notes: List[Note] = Field(default_factory=list, description="List of detected notes")
    num_notes: int = Field(0, description="Count of detected notes")

    def __init__(self, **data):
        super().__init__(**data)
        self.num_notes = len(self.notes or [])

    def summary(self) -> str:
        """Generate a readable summary string."""
        return (
            f"Analysis complete. {self.num_notes} notes detected. "
            f"Exports saved to: {self.exports.json or 'N/A'}, {self.exports.csv or 'N/A'}, {self.exports.midi or 'N/A'}"
        )