# API Reference

The inference server is a FastAPI application (`inference/server.py`) that loads `artifacts/model.onnx` once at startup via a `lifespan` context manager and serves two endpoints.

## `GET /health`

Returns server status and which ONNX Runtime execution providers are active.

**Request**

```bash
curl http://localhost:8000/health
```

**Response — `200 OK`**

```json
{
  "status": "ok",
  "model_loaded": true,
  "providers": ["CPUExecutionProvider"]
}
```

If the model failed to load, `model_loaded` is `false` and `providers` is an empty list, but the endpoint itself still returns `200` — use `model_loaded` to drive readiness checks, not the HTTP status code.

## `POST /predict`

Accepts a multipart image upload of a handwritten digit (any size — it's resized to 28×28 internally) and returns the predicted class with confidence scores.

**Request**

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@digit.png;type=image/png"
```

**Response — `200 OK`**

```json
{
  "predicted_class": "7",
  "confidence": 0.9983,
  "latency_ms": 2.341,
  "all_probabilities": {
    "0": 0.0001,
    "1": 0.0002,
    "2": 0.0001,
    "3": 0.0001,
    "4": 0.0002,
    "5": 0.0001,
    "6": 0.0002,
    "7": 0.9983,
    "8": 0.0004,
    "9": 0.0003
  }
}
```

The preprocessing step (`preprocess` in `inference/server.py`) converts the image to grayscale, resizes to 28×28, normalizes to MNIST's mean/std (`0.1307`/`0.3081`), and auto-inverts light backgrounds (e.g., digits drawn on a white canvas in Paint) so they match the MNIST black-background distribution the model was trained on.

### Error codes

| Status | Condition | Example detail |
|---|---|---|
| `400` | Uploaded file's `content_type` doesn't start with `image/` | `"File must be an image"` |
| `422` | Image bytes couldn't be decoded/preprocessed (corrupt or unsupported file) | `"Preprocessing failed: ..."` |
| `503` | Model isn't loaded (session is `None`, e.g. startup still in progress or failed) | `"Model not loaded"` |

## Code reference

::: inference.server.preprocess

::: inference.server.health

::: inference.server.predict
