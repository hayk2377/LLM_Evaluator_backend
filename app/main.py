from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import google.generativeai as genai

from app.database import engine
from app.router import router as api_router
from app.startup import download_nltk_data, populate_db_from_csv
from app.evaluation.models.evaluation import Evaluation  # ensure model is registered
from app.evaluation.models import evaluation as _  # noqa: F401 (import side-effect)


load_dotenv()

# Configure Google Generative AI
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY not found in .env file")
genai.configure(api_key=api_key)

# Ensure NLTK resources and seed data
download_nltk_data()

# Create tables and optionally seed from CSV
from app.database import Base  # local import to avoid cycles
Base.metadata.create_all(bind=engine)
populate_db_from_csv(Evaluation)

app = FastAPI()

# CORS: allow any origin/method/header for demo/analytics usage
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount feature routers
app.include_router(api_router)
