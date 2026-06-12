# Light FastAPI image. The model arrives from R2 at boot (not baked in); bulk
# scraping (Scrapy/Playwright) runs in CI, not here.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Install deps first for layer caching (psycopg2-binary needs no build toolchain).
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# App code, migrations, and boot scripts (.dockerignore strips Views/, *.pkl, etc.).
COPY DelphiAIApp ./DelphiAIApp
COPY scripts ./scripts
COPY yoyo.ini ./

EXPOSE 8000
CMD ["sh", "scripts/entrypoint.sh"]
