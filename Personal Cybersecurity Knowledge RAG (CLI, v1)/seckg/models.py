from typing import List, Optional
from pydantic import BaseModel, Field

class Chunk(BaseModel):
    """Represents a single text chunk with metadata."""
    content: str = Field(..., description="The text content of the chunk")
    source_file: str = Field(..., description="The relative path to the source file")
    heading_path: Optional[str] = Field(None, description="The breadcrumb header path for markdown (e.g. Header 1 > Header 2)")
    page_number: Optional[int] = Field(None, description="The page number for PDFs")
    chunk_id: str = Field(..., description="Unique ID for this chunk")
    last_modified: float = Field(..., description="The last modified timestamp of the source file")

class RetrievalResult(BaseModel):
    """Represents a retrieved chunk with its similarity score."""
    chunk: Chunk
    score: float

class Answer(BaseModel):
    """Represents the grounded answer from the LLM, including relevant sources."""
    text: str
    sources: List[Chunk] = Field(default_factory=list)
