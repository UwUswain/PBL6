# Same torch/CUDA as the local training environment (torch 2.5.1 + cu121).
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv/worker

COPY gpu_check.py .
# shortcut: M0 only checks the GPU; the job-queue worker (M2) replaces this command.
CMD ["python", "gpu_check.py"]
