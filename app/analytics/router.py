from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.analytics.cruds.analytics import get_analytics as aggregate


router = APIRouter()


@router.get("/analytics")
def get_analytics(db: Session = Depends(get_db)):
    """Return aggregated analytics payload for dashboards."""
    return aggregate(db)
