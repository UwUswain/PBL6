# Built from the repository root (see webapp/docker-compose.yml) so the ML code in src/ and web/ is copied in.
# Dockerfile.dockerignore next to this file keeps the build context small (no datasets, no checkpoints).
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY webapp/worker/requirements.txt webapp/worker/constraints.txt /tmp/
# MONAI 1.6 metadata asks torch>=2.8, but the team trains and infers with MONAI 1.6 on torch 2.5.1: no deps.
RUN pip install -r /tmp/requirements.txt -c /tmp/constraints.txt \
 && pip install --no-deps monai==1.6.0

RUN useradd --create-home --uid 1000 worker && mkdir -p /srv/data && chown worker /srv/data
COPY src/ /srv/app/src/
COPY web/inference.py web/regions.py web/config.yaml /srv/app/web/
COPY web/atlas/ /srv/app/web/atlas/
COPY webapp/backend/app/ /srv/backend/app/
COPY webapp/worker/worker.py webapp/worker/gpu_check.py /srv/worker/
USER worker
WORKDIR /srv/worker

CMD ["python", "worker.py"]
