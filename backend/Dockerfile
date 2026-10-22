# Stemforge backend image (API and worker share it; the command differs).
#   CPU:  docker build --build-context samples=../samples -t stemforge-backend .
#   GPU:  docker build --build-context samples=../samples --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 -t stemforge-backend:gpu .
FROM python:3.11-slim AS base

ARG TORCH_VERSION=2.11.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    NUMBA_CACHE_DIR=/tmp/numba STEMFORGE_DATA_DIR=/data STEMFORGE_MODELS_DIR=/models \
    STEMFORGE_SAMPLES_DIR=/app/samples

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 stemforge

WORKDIR /app
RUN pip install torch==${TORCH_VERSION} torchaudio==${TORCH_VERSION} --index-url ${TORCH_INDEX}
COPY requirements/lock.txt requirements/lock.txt
# --no-deps: the lock is complete, and it keeps demucs/basic-pitch from pulling TensorFlow.
RUN pip install --no-deps -r requirements/lock.txt && pip install "psycopg[binary]>=3.2,<4"
COPY pyproject.toml ./
COPY stemforge ./stemforge
COPY scripts ./scripts
# "samples" is a named build context (docker compose sets it; plain docker: --build-context samples=../samples)
COPY --from=samples stemforge-demo.flac stemforge-demo.truth.json ./samples/
RUN pip install --no-deps . && mkdir -p /data /models && chown -R stemforge /data /models

USER stemforge
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/healthz').status==200 else 1)" || exit 1
CMD ["stemforge-api", "--host", "0.0.0.0", "--port", "8000"]
