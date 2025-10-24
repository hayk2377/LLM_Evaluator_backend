import csv
import os
import logging
import nltk
from typing import Iterable, Optional
from app.database import SessionLocal


def download_nltk_data(packages: Iterable[str] = ("punkt", "stopwords", "punkt_tab")) -> None:
    """Ensure NLTK data is present; download if missing for required packages."""
    for package in packages:
        try:
            nltk.data.find(f"tokenizers/{package}")
        except LookupError:
            try:
                nltk.data.find(f"corpora/{package}")
            except LookupError:
                nltk.download(package)


def populate_db_from_csv(model_cls, csv_path: Optional[str] = None) -> None:
    """Populate evaluations table from CSV if empty; idempotent on startup.

    - Pass SEED_CSV_PATH env var to override path; default "mock_data.csv" at project root.
    - Expects CSV columns: prompt, model, temperature, top_p, lexical_diversity, query_coverage, flesch_kincaid_grade, repetition_penalty.
    """
    logger = logging.getLogger(__name__)
    if csv_path is None:
        csv_path = os.getenv("SEED_CSV_PATH", "mock_data.csv")

    db = SessionLocal()
    try:
        if db.query(model_cls).first():
            logger.info("Database already populated; skipping CSV seed.")
            return
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                batch = []
                for row in reader:
                    batch.append(
                        model_cls(
                            prompt=row["prompt"],
                            model=row["model"],
                            temperature=float(row["temperature"]),
                            top_p=float(row["top_p"]),
                            lexical_diversity=float(row["lexical_diversity"]),
                            query_coverage=float(row["query_coverage"]),
                            flesch_kincaid_grade=float(row["flesch_kincaid_grade"]),
                            repetition_penalty=float(row["repetition_penalty"]),
                        )
                    )
                if batch:
                    db.add_all(batch)
                    db.commit()
                    logger.info("Seeded %d rows from %s", len(batch), csv_path)
                else:
                    logger.info("No rows found in %s; nothing to seed.", csv_path)
        except FileNotFoundError:
            logger.warning("Seed file %s not found; skipping CSV seed.", csv_path)
    finally:
        db.close()
