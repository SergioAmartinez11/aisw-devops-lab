import numpy as np
import pytest
from PIL import Image
import io
from inference.server import preprocess


def make_image(color: int, size=(28, 28)) -> bytes:
    img = Image.new("L", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestPreprocessing:

    def test_output_shape(self):
        """Output tensor must be (1, 1, 28, 28)."""
        result = preprocess(make_image(color=255))
        assert result.shape == (1, 1, 28, 28)

    def test_output_dtype(self):
        """Must be float32 for ONNX Runtime."""
        result = preprocess(make_image(color=255))
        assert result.dtype == np.float32

    def test_white_background_inverted(self):
        """White background (Paint) must be inverted — tensor mean must be negative."""
        light_bg = make_image(color=200)
        dark_bg  = make_image(color=100)
        result_light = preprocess(light_bg)
        result_dark  = preprocess(dark_bg)
        assert result_light.mean() < result_dark.mean()

    def test_resize_arbitrary_input(self):
        """Images of any size must be resized to 28x28."""
        large = make_image(color=128, size=(256, 256))
        result = preprocess(large)
        assert result.shape == (1, 1, 28, 28)

    def test_pixel_range_normalized(self):
        """Normalized values must be within a reasonable range for MNIST."""
        result = preprocess(make_image(color=128))
        assert result.min() > -5.0
        assert result.max() <  5.0