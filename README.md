# LLM Evaluator (FastAPI)

A small FastAPI backend to test the effect of temperature and top_p on LLM outputs (Gemini via Google AI Studio), store the results, and serve analytics with objective, non-LLM metrics.

## What it does

- Run prompt tests against a model with a set of (temperature, top_p) pairs
- Compute objective metrics for the outputs
  - lexical_diversity (%)
  - query_coverage (%)
  - flesch_kincaid_grade (FKGL, ~0–18)
  - repetition_penalty (% of repeated n-grams)
- Store runs in SQLite and expose analytics endpoints for charts/heatmaps
- Seed the DB from a generated mock_data.csv for demos

## Project layout

- `app/`
  - `main.py`: FastAPI app, CORS, startup (NLTK, tables, seeding)
  - `database.py`: SQLAlchemy engine, Base, SessionLocal, get_db()
  - `router.py`: Aggregates evaluation and analytics routers
  - `startup.py`: NLTK downloader + idempotent CSV -> DB seeding
  - `evaluation/` (routers, models, schemas, metrics)
  - `analytics/` (CRUD aggregations and dataset-aware normalization)
- `generate_mock_data.py`: Creates grid-variant synthetic data with varied temperature/top_p
- `mock_data.csv`: Seed file (you can regenerate)
- `Dockerfile`, `docker-compose.yml`: Containerized dev/run

Microservice-like structure and swappable services:
- The app is organized like two small services under `app/`: `evaluation/` and `analytics/`, each with its own router and CRUDs. A top-level `router.py` composes them.
- Storage is abstracted via SQLAlchemy. Default is local SQLite (`DATABASE_URL=sqlite:///./llm_evaluator.db`). You can swap to PostgreSQL/AWS RDS by updating `DATABASE_URL` (e.g., `postgresql+psycopg2://user:pass@host:5432/dbname`) and installing the appropriate driver.
- The evaluation service performs concurrent generation calls using async `gather` to boost throughput. If your provider supports bulk/batch endpoints, you can replace fan-out calls with batch requests to submit thousands in a single call.
- The analytics service pushes aggregation down to the database using SQL `AVG(...)` and `GROUP BY`, reducing Python-side computation and speeding up dashboard queries.
- Normalization is applied so metric “direction” is consistent across charts. Percent metrics pass through; FK grade is min–max normalized; repetition is min–max normalized and inverted so that higher always means better for all displayed metrics.

## Metric definitions and formulas

- Lexical diversity (0–100%): measures vocabulary variety
  - Formula: unique_tokens / total_tokens × 100
  - Tokenization: NLTK `word_tokenize` on lowercased text
  - Interpretation: higher can indicate richer wording; extremely high on short texts can be noisy

- Query coverage (0–100%): measures responsiveness to the prompt
  - Formula: |keywords_prompt ∩ words_response| / |keywords_prompt| × 100
  - Keywords: prompt tokens minus English stopwords
  - Interpretation: higher suggests better topical relevance and grounding

- Flesch–Kincaid Grade Level (FKGL): approximates readability/complexity
  - Formula: FKGL = 0.39 × (words/sentences) + 11.8 × (syllables/word) − 15.59
  - Syllables: lightweight heuristic (no heavy dictionaries)
  - Interpretation: higher implies more complex/denser prose; very high can hurt clarity

- Repetition penalty (0–100%): measures redundancy that can harm coherence
  - Formula: (total_ngrams − unique_ngrams) / total_ngrams × 100
  - Uses trigram repeats when available; falls back to bigrams for short outputs
  - Interpretation: higher raw value means more repetition (worse cohesion). In analytics we invert during normalization so “higher is better” in charts

Normalization in analytics:
- Percent metrics (lexical_diversity, query_coverage) are already 0–100% and are passed through unchanged
- flesch_kincaid_grade is min–max normalized per dataset for comparisons (higher = more complex)
- repetition_penalty is min–max normalized and inverted so that higher is better (i.e., less repetition scores higher)

## Run it

Prerequisites:
- Python 3.10+
- Docker (optional, for containerized run)

Environment:
- Copy `.env.example` to `.env` and set `GOOGLE_API_KEY`

Install dependencies (optional but recommended for the mock generator):

```
pip install -r requirements.txt
```

Generate fresh mock data (recommended before first run or after schema changes):

```
python generate_mock_data.py
```

Reset the DB (optional): delete the SQLite file `llm_evaluator.db` to recreate tables and reseed.

Run with Docker:

```
docker compose up --build
```

Or run locally without Docker:

```
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at http://localhost:8000

## Endpoints

- POST `/evaluation/test-prompt`
  - Body: prompt string + array of (temperature, top_p) pairs + model name
  - Runs async generations with Google AI Studio (Gemini), computes metrics, persists results
- GET `/evaluation/evaluations`
  - Paginated list of stored runs
- GET `/analytics/summary`
  - Aggregated metrics with dataset-aware normalization and KPIs

## Notes

- Set `GOOGLE_API_KEY` (env var) to use real generations; if unset, only analytics on seeded data will be meaningful
- `SEED_CSV_PATH` can override the default `mock_data.csv` location
- NLTK data is downloaded at startup (`punkt`, `stopwords`, `punkt_tab`)
- The mock generator uses the Hugging Face `databricks/databricks-dolly-15k` split for realistic prompts/answers

## Troubleshooting

- If you previously ran the app and changed the schema, delete `llm_evaluator.db` and restart
- If NLTK complains about missing resources, ensure your machine has internet on first run
- If the generator fails with ImportError for `datasets` or `tqdm`, re-run `pip install -r requirements.txt`
