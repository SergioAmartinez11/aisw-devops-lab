import statistics
import time
import pytest


LATENCY_P95_THRESHOLD_MS = 50.0   # maximum acceptable threshold
N_REQUESTS               = 100


class TestBenchmark:

    def test_inference_latency_p95(self, client, sample_image_bytes):
        """
        Inference latency P95 must be below the threshold.
        Simulates basic load testing — comparable to what you would do
        with locust or k6 in a real CI pipeline.
        """
        latencies = []

        for _ in range(N_REQUESTS):
            start = time.perf_counter()
            response = client.post(
                "/predict",
                files={"file": ("digit.png", sample_image_bytes, "image/png")},
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert response.status_code == 200
            latencies.append(elapsed_ms)

        latencies.sort()
        p50 = statistics.median(latencies)
        p95 = latencies[int(N_REQUESTS * 0.95)]
        p99 = latencies[int(N_REQUESTS * 0.99)]

        print(f"\n── Benchmark Results ({N_REQUESTS} requests) ──")
        print(f"  P50: {p50:.2f}ms")
        print(f"  P95: {p95:.2f}ms")
        print(f"  P99: {p99:.2f}ms")
        print(f"  Threshold: {LATENCY_P95_THRESHOLD_MS}ms")

        assert p95 < LATENCY_P95_THRESHOLD_MS, (
            f"P95 latency {p95:.2f}ms exceeds threshold {LATENCY_P95_THRESHOLD_MS}ms"
        )

    def test_throughput(self, client, sample_image_bytes):
        """Minimum 20 requests per second on CPU."""
        start = time.perf_counter()
        for _ in range(50):
            client.post(
                "/predict",
                files={"file": ("digit.png", sample_image_bytes, "image/png")},
            )
        elapsed = time.perf_counter() - start
        rps = 50 / elapsed

        print(f"\n── Throughput: {rps:.1f} req/s ──")
        assert rps >= 20, f"Throughput {rps:.1f} req/s below minimum 20 req/s"