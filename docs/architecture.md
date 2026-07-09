# Architecture

## Pipeline overview

```
┌──────────────┐     ┌───────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   MNIST      │────▶│  model/       │────▶│  model/           │────▶│  artifacts/          │
│   dataset    │     │  train.py     │     │  export.py        │     │  model.onnx          │
│              │     │  (PyTorch)    │     │  (torch.onnx)     │     │                      │
└──────────────┘     └───────────────┘     └──────────────────┘     └──────────┬───────────┘
                                                                                │
                                                                                ▼
┌──────────────┐     ┌───────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   Client     │◀───▶│  inference/   │◀───▶│  ONNX Runtime     │◀───▶│  CPUExecutionProvider│
│   (image)    │     │  server.py    │     │  InferenceSession │     │  (QNNExecutionProvider│
│              │     │  (FastAPI)    │     │                   │     │   on supported edge) │
└──────────────┘     └───────────────┘     └──────────────────┘     └─────────────────────┘
        ▲                     │
        │                     ▼
        │            ┌──────────────────┐
        └────────────│  Docker container │
                      │  (multistage)     │
                      └──────────────────┘
```

Training happens once (or is restored from CI cache); the resulting `model.onnx` is the single artifact that flows through export, testing, Docker packaging, and eventually a GitHub Release.

## Why ONNX, and why these export settings

`model/export.py` converts the trained `MiniCNN` PyTorch module to ONNX via `torch.onnx.export`. The choices made there are deliberate:

- **`opset_version=17`** — a modern opset with broad operator coverage and support across ONNX Runtime versions, while still being old enough to be well-supported by edge/embedded runtimes (a common constraint for on-device inference).
- **`do_constant_folding=True`** — folds constant subgraphs at export time, fusing operations that don't depend on runtime input. This shrinks the graph and reduces per-inference compute, which matters most on constrained (edge) hardware.
- **`dynamic_axes` on `input`/`output` batch dimension** — the model is exported with a dynamic batch size (`{0: "batch_size"}`) rather than a fixed batch of 1. This lets the same ONNX file serve single-image requests in the FastAPI server today and be reused for batched offline inference later, without re-exporting.
- **Runtime-agnostic format** — once exported, the model has no PyTorch dependency. The inference server and the production Docker image only need `onnxruntime`, not `torch`, which is what keeps the runtime image lean (see below).

### Execution providers

`inference/server.py` loads the model with `CPUExecutionProvider`, the portable default that runs anywhere Python runs. ONNX Runtime's execution-provider abstraction means the same `model.onnx` can be loaded with hardware-specific providers without any code change — notably **QNNExecutionProvider** for Qualcomm Snapdragon NPUs/DSPs, which is the intended edge deployment target referenced in the project's design (see the model card in the [Home](index.md) page). Swapping providers is a one-line change to the `providers=[...]` list passed to `ort.InferenceSession`.

## Docker multistage build

The `dockerfile` at the repo root builds in two stages so that training-only dependencies (PyTorch, torchvision, onnxscript) never ship in the runtime image.

| Stage | Base image | Purpose | Contents |
|---|---|---|---|
| `builder` | `python:3.11-slim` | Train + export the model if `artifacts/model.onnx` doesn't already exist | Full `requirements.txt` (PyTorch, ONNX, etc.), `model/` source, `artifacts/` |
| `runtime` | `python:3.11-slim` | Serve predictions | Inference-only deps (`fastapi`, `uvicorn`, `python-multipart`, `Pillow`, `onnxruntime`, `numpy`), `artifacts/model.onnx` copied from `builder`, `inference/` source |

Key properties of the runtime stage:

- Runs as a non-root `appuser` (created via `useradd -m appuser`) for defense-in-depth.
- Declares a `HEALTHCHECK` that curls `/health` internally, so container orchestrators can detect a broken server without an external probe.
- Only copies `artifacts/model.onnx` and `inference/` from the builder — no PyTorch, no training scripts, no dataset — keeping the final image small and reducing its attack surface.
