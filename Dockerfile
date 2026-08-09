# Deliberately lean: this image serves the API only. It does NOT contain
# lightgbm, mlflow, pandas, or the NEPSE client — those belong to the
# training/scraping jobs that run via GitHub Actions, not this service.
# The API only ever reads from Postgres (see app.py).

FROM python:3.11-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app.py .

# Render sets $PORT at runtime; 10000 is Render's own documented default,
# used here as a sane fallback for running the container locally too.
ENV PORT=10000
EXPOSE 10000

# Shell form (not exec form) is required here so ${PORT} actually gets
# expanded — exec form (["uvicorn", "app:app", ...]) would pass the
# literal string "${PORT}" to uvicorn instead of its value.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
