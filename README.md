# AISW DevOps Lab — CNN Inference Server

[![CI Pipeline](https://github.com/SergioAmartinez11/aisw-devops-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/SergioAmartinez11/aisw-devops-lab/actions/workflows/ci.yml)

An end-to-end ML DevOps lab that trains a lightweight CNN on MNIST, exports it to ONNX, and serves predictions through a FastAPI REST API — packaged in a multistage Docker container and validated by a GitHub Actions CI pipeline.

## What this lab covers

- Training a small CNN (`MiniCNN`) in PyTorch on the MNIST digit dataset
- Exporting the trained model to ONNX format for runtime-agnostic inference
- Serving predictions via a FastAPI server backed by ONNX Runtime, with lifespan-managed model loading
- Multistage Docker build: training in a builder stage, lean runtime image for serving
- Pytest test suite covering API correctness, preprocessing logic, and latency benchmarks
- GitHub Actions CI pipeline: test suite → Docker build → containerized smoke test

## Project structure

```
aisw-devops-lab/
├── .github/
│   └── workflows/
│       └── ci.yml       # CI pipeline: test → docker build → smoke test
├── model/
│   ├── cnn.py          # MiniCNN architecture (PyTorch)
│   ├── train.py        # Training loop (MNIST, 3 epochs)
│   └── export.py       # ONNX export
├── inference/
│   └── server.py       # FastAPI app with /health and /predict endpoints
├── tests/
│   ├── conftest.py         # Shared pytest fixtures (client, sample images)
│   ├── test_api.py         # Endpoint tests (schema, status codes, edge cases)
│   ├── test_preprocessing.py  # Unit tests for image preprocessing pipeline
│   └── test_benchmark.py   # Latency P95 and throughput benchmarks
├── artifacts/          # Saved model files (model.pth, model.onnx)
├── dockerfile          # Multistage Docker build
├── Makefile            # Developer convenience targets
├── pytest.ini          # Pytest configuration (asyncio mode, test discovery)
└── requirements.txt
```

## Quick start

**1. Install dependencies**

```bash
make install
```

**2. Train the model and export to ONNX**

```bash
make pipeline
```

This runs `model/train.py` (3 epochs on MNIST, ~99% val accuracy) and then `model/export.py` to produce `artifacts/model.onnx`.

**3. Start the inference server**

```bash
make serve
```

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for the interactive Swagger UI.

## API

### `GET /health`

Returns server status and loaded model providers.

```json
{
  "status": "ok",
  "model_loaded": true,
  "providers": ["CPUExecutionProvider"]
}
```

### `POST /predict`

Upload a PNG/JPEG image of a handwritten digit (any size). Returns the predicted class and confidence scores.

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@digit.png;type=image/png"
```

```json
{
  "predicted_class": "7",
  "confidence": 0.9983,
  "latency_ms": 2.341,
  "all_probabilities": {
    "0": 0.0001, "1": 0.0002, "7": 0.9983, ...
  }
}
```

The preprocessing pipeline automatically detects light backgrounds (e.g., images drawn in Paint) and inverts them to match the MNIST distribution.

## Docker

**Build the image**

```bash
make docker-build
```

The multistage build trains the model if `artifacts/model.onnx` is not present, then copies only the ONNX file and inference dependencies into the final runtime image (no PyTorch in production).

**Run the container**

```bash
make docker-run
```

The container runs as a non-root user and exposes a `HEALTHCHECK` on `/health`.

**Check image size**

```bash
make docker-size
```

## Tests

```bash
pytest tests/ -v
```

The test suite includes:

| Module | What it tests |
|---|---|
| `test_api.py` | `/health` and `/predict` endpoints — schema, status codes, error handling |
| `test_preprocessing.py` | Tensor shape, dtype, background inversion, resize, value normalization |
| `test_benchmark.py` | P95 latency < 50 ms, throughput >= 20 req/s on CPU |

## CI/CD

Every push to `main`/`develop` and every PR into `main` runs the [CI pipeline](.github/workflows/ci.yml), which chains three jobs:

1. **Test Suite** — installs dependencies (pip cache keyed on `requirements.txt`), trains + exports the model if `artifacts/` isn't already cached for the current `model/**` hash, then runs the preprocessing, API, and benchmark test modules. Test results and the ONNX model are uploaded as workflow artifacts.
2. **Docker Build** — downloads the ONNX artifact from job 1 and builds the multistage image with Buildx (GitHub Actions layer cache), only if the test suite passed.
3. **Smoke Test** — runs the built image as a container, polls `/health` until ready, then exercises `/health` and `/predict` end-to-end against the live container before tearing it down.

## Model

**MiniCNN** is a two-block convolutional network:

```
Conv2d(1→32) → ReLU → MaxPool2d
Conv2d(32→64) → ReLU → MaxPool2d
Flatten → Linear(3136→128) → ReLU → Dropout(0.3) → Linear(128→10)
```

Trained for 3 epochs with Adam (lr=1e-3) and CrossEntropyLoss. Reaches ~99% validation accuracy on MNIST. Designed to be ONNX-exportable for edge deployment (e.g., Qualcomm Snapdragon via QNN/ONNX Runtime).

## Requirements

- Python 3.11+
- Docker (for containerized workflow)
- CUDA optional — training and inference both fall back to CPU automatically
