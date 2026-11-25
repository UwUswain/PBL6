FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv/api

RUN useradd --create-home --uid 1000 api && mkdir -p /srv/keys /srv/data && chown api /srv/keys /srv/data
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
USER api

EXPOSE 8000
# Apply migrations, then serve. Compose waits for the db healthcheck before starting this.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers"]
