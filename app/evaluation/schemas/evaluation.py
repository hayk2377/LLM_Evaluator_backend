from datetime import datetime
from pydantic import BaseModel


class EvaluationBase(BaseModel):
    """Evaluation metrics captured for a single generation."""
    prompt: str
    model: str
    temperature: float
    top_p: float
    lexical_diversity: float
    query_coverage: float
    flesch_kincaid_grade: float
    repetition_penalty: float


class EvaluationCreate(EvaluationBase):
    """Payload to persist a new evaluation row."""
    pass


class Evaluation(EvaluationBase):
    """Evaluation row shape returned by API."""
    id: int
    created_at: datetime

    class Config:
        from_attributes = True
