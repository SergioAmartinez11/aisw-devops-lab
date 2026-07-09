# CI/CD Pipeline

There are two GitHub Actions workflows: `ci.yml`, which validates every push/PR, and `release.yml`, which cuts a versioned release and publishes docs when a `v*` tag is pushed.

## `ci.yml` — continuous integration

Triggers on push to `main`/`develop` and on pull requests targeting `main`. Three jobs run in sequence, each gated on the previous one succeeding.

### 1. `test`

- Checks out the repo and sets up Python 3.11.
- Restores a pip cache keyed on the hash of `requirements.txt`.
- Installs dependencies.
- Restores a model-artifact cache keyed on the hash of `model/**`; if it misses, runs `model/train.py` then `model/export.py` to regenerate `artifacts/model.onnx`.
- Runs the three test modules: `test_preprocessing.py`, `test_api.py`, `test_benchmark.py`.
- Uploads `.pytest_cache/` as the `test-results` artifact and `artifacts/model.onnx` as the `onnx-model` artifact, so downstream jobs don't need to retrain.

### 2. `docker` (needs: `test`)

- Downloads the `onnx-model` artifact into `artifacts/`.
- Builds the multistage Docker image with Buildx, using GitHub Actions layer caching (`type=gha`).
- Loads the built image locally and reports its size.

### 3. `smoke-test` (needs: `docker`)

- Downloads the `onnx-model` artifact again and rebuilds the image (tagged `:smoke`).
- Starts the container, polls `/health` until it responds (up to 15 attempts, 2s apart).
- Verifies `/health` returns `status: ok` and `model_loaded: true`.
- Generates a blank 28×28 PNG and POSTs it to `/predict`, asserting the response contains `predicted_class`, `confidence`, and `latency_ms`.
- Always prints container logs and stops/removes the container afterward, even on failure.

## `release.yml` — release automation

Triggers on push of any tag matching `v*` (e.g. `v1.0.0`). Three jobs run in sequence.

### 1. `test`

Identical in spirit to the `test` job in `ci.yml`: pip cache, model cache (train + export only on a cache miss), run the full pytest suite, and upload `artifacts/model.onnx` as the `onnx-model` artifact. This guarantees a release is never cut from a model or codebase that fails its own test suite.

### 2. `release` (needs: `test`, permissions: `contents: write`)

- Downloads the `onnx-model` artifact produced by the `test` job.
- Extracts the version string from the tag in `GITHUB_REF` (e.g. `refs/tags/v1.2.0` → `v1.2.0`).
- Uses [`softprops/action-gh-release@v2`](https://github.com/softprops/action-gh-release) to create a GitHub Release for that tag, attaching `artifacts/model.onnx` as a downloadable release asset.
- The release body is generated from the version, a summary of what's included, the current benchmark numbers, and a ready-to-run `docker run` command — see [Home](index.md) for the latest benchmark figures.

### 3. `docs` (needs: `release`, permissions: `contents: write`, `pages: write`)

- Installs `mkdocs`, `mkdocs-material`, and `mkdocstrings[python]`.
- Runs `mkdocs gh-deploy --force`, which builds the site from `mkdocs.yml`/`docs/` and force-pushes the result to the `gh-pages` branch, publishing it via GitHub Pages.

Because `docs` depends on `release`, documentation is only ever redeployed alongside a successful, tagged release — not on every commit to `main`.
