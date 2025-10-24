from typing import List
from pydantic import BaseModel


class ParamPair(BaseModel):
    """Single generation parameter pair."""
    temperature: float
    top_p: float


class PromptTest(BaseModel):
    """Request body to test a prompt across parameter pairs for a given model."""
    prompt: str
    model: str
    param_pairs: List[ParamPair]
