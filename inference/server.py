import io
import time

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pathlib import Path
from contextlib import asynccontextmanager




ONNX_PATH = str(Path(__file__).parent.parent / "artifacts" / "model.onnx")
CLASS_NAMES = [str(i) for i in range(10)]

# ── Load model once at startup ─────────────────────────────────────────────────
session: ort.InferenceSession | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session
    session = ort.InferenceSession(
        ONNX_PATH,
        providers=["CPUExecutionProvider"],
    )
    print(f"✓ ONNX model loaded | providers: {session.get_providers()}")
    yield  # server running
    session = None  # cleanup on shutdown

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AISW Inference Server",
    description="ONNX Runtime inference endpoint for MiniCNN digit classifier",
    version="0.1.0",
    lifespan=lifespan,)

# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
    img = img.resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0

    # Detect if background is light (Paint) and invert to match MNIST distribution
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    arr = (arr - 0.1307) / 0.3081                          # MNIST normalization
    return arr.reshape(1, 1, 28, 28)


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": session is not None,
        "providers": session.get_providers() if session else [],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()

    try:
        input_tensor = preprocess(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {e}")

    # ── Inference + latency tracking ───────────────────────────────────────────
    t0 = time.perf_counter()
    outputs = session.run(["output"], {"input": input_tensor})
    latency_ms = (time.perf_counter() - t0) * 1000

    logits = outputs[0][0]
    probs  = softmax(logits)
    pred   = int(np.argmax(probs))

    return JSONResponse({
        "predicted_class": CLASS_NAMES[pred],
        "confidence": round(float(probs[pred]), 4),
        "latency_ms": round(latency_ms, 3),
        "all_probabilities": {
            CLASS_NAMES[i]: round(float(p), 4)
            for i, p in enumerate(probs)
        },
    })


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()