import pytest


class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_schema(self, client):
        data = response = client.get("/health").json()
        assert "status"       in data
        assert "model_loaded" in data
        assert "providers"    in data

    def test_health_model_is_loaded(self, client):
        data = client.get("/health").json()
        assert data["status"]       == "ok"
        assert data["model_loaded"] is True


class TestPredictEndpoint:

    def test_predict_returns_200(self, client, sample_image_bytes):
        response = client.post(
            "/predict",
            files={"file": ("digit.png", sample_image_bytes, "image/png")},
        )
        assert response.status_code == 200

    def test_predict_schema(self, client, sample_image_bytes):
        data = client.post(
            "/predict",
            files={"file": ("digit.png", sample_image_bytes, "image/png")},
        ).json()
        assert "predicted_class"    in data
        assert "confidence"         in data
        assert "latency_ms"         in data
        assert "all_probabilities"  in data

    def test_predict_confidence_range(self, client, sample_image_bytes):
        data = client.post(
            "/predict",
            files={"file": ("digit.png", sample_image_bytes, "image/png")},
        ).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_all_classes_present(self, client, sample_image_bytes):
        data = client.post(
            "/predict",
            files={"file": ("digit.png", sample_image_bytes, "image/png")},
        ).json()
        assert len(data["all_probabilities"]) == 10

    def test_predict_rejects_non_image(self, client):
        response = client.post(
            "/predict",
            files={"file": ("data.csv", b"a,b,c", "text/csv")},
        )
        assert response.status_code == 400

    def test_predict_mnist_sample(self, client, mnist_image_bytes):
        """With a real MNIST digit, confidence should be high."""
        data = client.post(
            "/predict",
            files={"file": ("mnist.png", mnist_image_bytes, "image/png")},
        ).json()
        assert data["confidence"] >= 0.5