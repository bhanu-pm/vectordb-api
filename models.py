from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import datetime


# A fixed schema for metadata
class Metadata(BaseModel):
    source: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

# Main data models
class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    embedding: List[float]
    metadata: Metadata = Field(default_factory=Metadata)

class Document(BaseModel):
    document_id: str
    chunks: List[Chunk]
    metadata: Metadata = Field(default_factory=Metadata)

class Library(BaseModel):
    library_id: str
    documents: List[Document] = Field(default_factory=list)
    metadata: Metadata = Field(default_factory=Metadata)

# Input Models for API
class LibraryCreateModel(BaseModel):
    library_id: str
    metadata: Optional[Metadata] = None

class ChunkCreateModel(BaseModel):
    chunk_id: str
    text: str
    embedding: List[float]
    metadata: Optional[Metadata] = None
