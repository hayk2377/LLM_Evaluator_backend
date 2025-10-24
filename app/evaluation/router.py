import asyncio
import time
import google.generativeai as genai
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.evaluation.metrics import calculate_metrics
from app.database import get_db
from app.evaluation.cruds.evaluation import create_evaluation, get_evaluations as list_evaluations
from app.evaluation.schemas.evaluation import Evaluation, EvaluationCreate
from app.evaluation.schemas.prompt import PromptTest


router = APIRouter()


async def _call_google_ai(model_name: str, prompt: str, temp: float, top_p: float) -> dict:
    """Call Gemini with generation parameters and return output + computed metrics."""
    config = genai.types.GenerationConfig(temperature=temp, top_p=top_p)
    model = genai.GenerativeModel(model_name)

    start = time.time()
    try:
        resp = await model.generate_content_async(prompt, generation_config=config)
        _ = time.time() - start  # latency captured if needed later
        text = resp.text
        metrics = calculate_metrics(prompt, text)
        return {"temperature": temp, "top_p": top_p, "output": text, "metrics": metrics, "error": None}
    except Exception as e:
        return {"temperature": temp, "top_p": top_p, "output": None, "metrics": None, "error": f"An API error occurred: {e}"}


@router.post("/test-prompt")
async def test_prompt(payload: PromptTest, db: Session = Depends(get_db)):
    """Generate responses across supplied param pairs, compute metrics, persist, and return results."""
    tasks = [
        _call_google_ai(payload.model, payload.prompt, pair.temperature, pair.top_p)
        for pair in payload.param_pairs
    ]
    results = await asyncio.gather(*tasks)

    for r in results:
        if r.get("metrics"):
            create_evaluation(
                db,
                EvaluationCreate(
                    prompt=payload.prompt,
                    model=payload.model,
                    temperature=r["temperature"],
                    top_p=r["top_p"],
                    **r["metrics"],
                ),
            )
    return results


@router.get("/evaluations", response_model=list[Evaluation])
def get_evaluations(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Paginated evaluations listing."""
    return list_evaluations(db, skip=skip, limit=limit)
