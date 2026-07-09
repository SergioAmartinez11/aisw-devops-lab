import pytest
from fastapi.testclient import TestClient
from inference.server import app


@pytest.fixture(scope="session")
def client():
    """
    HTTP test client — starts the server in memory,
    no need to run uvicorn separately.
    """
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def sample_image_bytes():
    """Synthetic 28x28 white image — simulates Paint input."""
    from PIL import Image
    import io

    img = Image.new("L", (28, 28), color=255)   # white background
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def mnist_image_bytes():
    """First real digit from the MNIST dataset."""
    from torchvision import datasets
    import io

    ds = datasets.MNIST("data", train=False, download=True)
    img, _ = ds[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()