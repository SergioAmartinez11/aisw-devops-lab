.PHONY: install train export serve clean help

PYTHON   := $(CURDIR)/.venv/Scripts/python
PIP      := $(CURDIR)/.venv/Scripts/pip



IMAGE_NAME := aisw-inference
IMAGE_TAG  := latest

docker-build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .
	@echo "✓ Image built: $(IMAGE_NAME):$(IMAGE_TAG)"

docker-run:
	docker run --rm -p 8000:8000 --name aisw-server $(IMAGE_NAME):$(IMAGE_TAG)

docker-stop:
	docker stop aisw-server 2>/dev/null || true

docker-size:
	docker images $(IMAGE_NAME)


install:
	python -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✓ Environment ready"

train:
	cd model && $(PYTHON) train.py

export:
	cd model && $(PYTHON) export.py

pipeline: train export
	@echo "✓ Full ML pipeline complete: train → export"

serve:
	cd inference && $(PYTHON) -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload

clean:
	rm -rf artifacts/model.pth artifacts/model.onnx

help:
	@echo ""
	@echo "  install    → create venv + install deps"
	@echo "  train      → train MiniCNN on MNIST"
	@echo "  export     → export trained model to ONNX"
	@echo "  pipeline   → train + export in sequence"
	@echo "  serve      → start FastAPI inference server"
	@echo "  clean      → remove artifacts and cache"
	@echo ""