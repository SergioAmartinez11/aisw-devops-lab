# AISW DevOps Lab

An end-to-end ML DevOps lab that trains a lightweight CNN on MNIST, exports it to ONNX, and serves predictions through a FastAPI REST API — packaged in a multistage Docker container and validated by a GitHub Actions CI/CD pipeline.

## What this lab covers

- Training a small CNN (`MiniCNN`) in PyTorch on the MNIST digit dataset
- Exporting the trained model to ONNX format for runtime-agnostic inference
- Serving predictions via a FastAPI server backed by ONNX Runtime, with lifespan-managed model loading
- Multistage Docker build: training in a builder stage, lean runtime image for serving
- Pytest test suite covering API correctness, preprocessing logic, and latency benchmarks
- GitHub Actions CI pipeline (test → Docker build → smoke test) and release pipeline (test → GitHub Release → docs deploy)

## Stack

| Layer | Technology |
|---|---|
| Model training | PyTorch |
| Model format | ONNX (opset 17) |
| Inference runtime | ONNX Runtime |
| API framework | FastAPI + Uvicorn |
| Image preprocessing | Pillow, NumPy |
| Testing | pytest, pytest-asyncio, httpx |
| Containerization | Docker (multistage build) |
| CI/CD | GitHub Actions |
| Documentation | MkDocs + Material for MkDocs + mkdocstrings |

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

**4. Run the tests**

```bash
pytest tests/ -v
```

## Benchmark numbers

Measured by `tests/test_benchmark.py` against the FastAPI test client on CPU:

| Metric | Value |
|---|---|
| P50 latency | ~0.5 ms |
| P95 latency | ~0.9 ms |
| Throughput | ~1600 req/s |

CI enforces a P95 threshold of 50 ms and a minimum throughput of 20 req/s — the numbers above reflect typical local/CI performance, well within those bounds.

## Where to go next

- [Architecture](architecture.md) — pipeline diagram, ONNX export choices, Docker build stages
- [API Reference](api.md) — endpoint docs and code reference
- [CI/CD Pipeline](ci.md) — how `ci.yml` and `release.yml` work
