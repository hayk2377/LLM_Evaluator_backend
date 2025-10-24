from sqlalchemy.orm import Session
from app.evaluation.models.evaluation import Evaluation
from app.evaluation.schemas.evaluation import EvaluationCreate


def get_evaluations(db: Session, skip: int = 0, limit: int = 100):
    """Return paginated evaluations."""
    return db.query(Evaluation).offset(skip).limit(limit).all()


def create_evaluation(db: Session, evaluation: EvaluationCreate):
    """Persist an evaluation row and return it."""
    db_row = Evaluation(**evaluation.dict())
    db.add(db_row)
    db.commit()
    db.refresh(db_row)
    return db_row
