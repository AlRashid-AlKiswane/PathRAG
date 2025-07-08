

from pydantic import BaseModel

class NERInput(BaseModel):
    """
    Input schema for NER API.
    """
    text: str
