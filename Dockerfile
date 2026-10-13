# Everything the pipeline needs, including the OCR engine, in one image.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    GSTPIPE_DATASET=/app/data/synthetic/demo \
    GSTPIPE_DB=/app/data/review.db \
    GSTPIPE_MODEL=/app/models/pipeline.joblib \
    GSTPIPE_UPLOADS=/app/data/uploads

# tesseract reads the documents; the rest are wheels that need no build tools.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Build the demo into the image so the container starts ready to use.
ARG BUSINESSES=3
RUN python -m gstpipe.webapp.cli --bootstrap --no-serve \
      --businesses ${BUSINESSES} \
      --dataset ${GSTPIPE_DATASET} --db ${GSTPIPE_DB} --model ${GSTPIPE_MODEL}

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -fsS http://localhost:${PORT}/api/health || exit 1

CMD ["sh", "-c", "python -m gstpipe.webapp.cli --host 0.0.0.0 --port ${PORT} --dataset ${GSTPIPE_DATASET} --db ${GSTPIPE_DB} --model ${GSTPIPE_MODEL}"]