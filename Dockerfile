
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
# Copy application code and seed data
COPY ./app /app/app
COPY ./mock_data.csv /app/mock_data.csv

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
