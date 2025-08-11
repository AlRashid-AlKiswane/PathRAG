
from typing import Optional
from openai import BaseModel

class ChunkRequest(BaseModel):
    input_path: str
    recursive: Optional[bool]

class ChunkResponse(BaseModel):
    total_chunks: int
    inserted_chunks: int
    failed_chunks: int
    message: str
