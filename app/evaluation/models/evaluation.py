from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func

from app.database import Base


class Evaluation(Base):
    """SQLAlchemy model for a single LLM evaluation run and its metrics."""

    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(String, nullable=False)
    model = Column(String, nullable=False, index=True)
    temperature = Column(Float, nullable=False, index=True)
    top_p = Column(Float, nullable=False, index=True)

    # Objective metrics
    lexical_diversity = Column(Float, nullable=False)
    query_coverage = Column(Float, nullable=False)
    flesch_kincaid_grade = Column(Float, nullable=False)
    repetition_penalty = Column(Float, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
