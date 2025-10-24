from fastapi import APIRouter
from app.evaluation.router import router as evaluation_router
from app.analytics.router import router as analytics_router

router = APIRouter()
router.include_router(evaluation_router, tags=["evaluation"])
router.include_router(analytics_router, tags=["analytics"])