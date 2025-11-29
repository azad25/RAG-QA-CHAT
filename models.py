from typing import Optional
from pydantic import BaseModel

class Query(BaseModel):
    question: str
    session_id: Optional[str] = "default"