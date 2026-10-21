# UrbanTrace AI - one image for the API and the ingest workers.
#
#   docker build -t urbantrace .
#
# Detector weights are not baked in: mount them at /app/models/weights.

FROM node:22-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build        # emits to ../web-dist

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/backend
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv libgl1 libglib2.0-0 libpq5 curl tini \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements.txt .
# Notebook tooling is for development, not for the service.
RUN python3 -m venv /opt/venv \
    && grep -viE '^(jupyter|notebook|ipykernel|nbformat|folium|matplotlib|pytest)' requirements.txt > /tmp/req.txt \
    && /opt/venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r /tmp/req.txt
ENV PATH=/opt/venv/bin:$PATH

COPY backend ./backend
COPY scripts ./scripts
COPY frontend ./frontend
COPY --from=web /web-dist ./web-dist

RUN useradd --system --uid 10001 urbantrace && mkdir -p data models/weights \
    && chown -R urbantrace /app/data
USER urbantrace

EXPOSE 8000 9100
ENTRYPOINT ["tini", "--"]
CMD ["uvicorn", "urbantrace.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
