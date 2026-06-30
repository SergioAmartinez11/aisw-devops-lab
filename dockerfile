# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Builder
# Install dependencies and train the model if it doesn't exist yet
# Not in production
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install dependencies first (takes advantage of Docker's cache if requirements don't change)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the model code
COPY model/ ./model/
COPY artifacts/ ./artifacts/

# IF the model doesn't exist yet, train it and export to ONNX
RUN if [ ! -f artifacts/model.onnx ]; then \
        cd model && python train.py && python export.py; \
    fi


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime
# lightweight image with only the dependencies needed for inference
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# inference dependencies only (no training dependencies)
COPY --from=builder /build/requirements.txt .
RUN pip install --no-cache-dir \
    fastapi>=0.110.0 \
    "uvicorn[standard]>=0.29.0" \
    python-multipart>=0.0.9 \
    Pillow>=10.0.0 \
    onnxruntime>=1.17.0 \
    numpy>=1.24.0

# Copy the ONNX model from the builder
COPY --from=builder /build/artifacts/model.onnx ./artifacts/model.onnx

# Copy the inference server
COPY inference/ ./inference/

# no-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "inference.server:app", "--host", "0.0.0.0", "--port", "8000"]