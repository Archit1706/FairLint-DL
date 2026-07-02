# REQUIREMENTS

## Architecture
- x86-64 (also runs on Apple Silicon / ARM64 via native Python wheels). No specific CPU model required.

## Hardware
- CPU only; **no GPU required**. PyTorch runs on CPU.
- ~2 GB free disk (mostly the PyTorch install).
- ~4 GB RAM is sufficient.
- Reproduction of Table 2 takes roughly 2–4 minutes on a typical laptop CPU.

## Software
- **Python 3.9+** (tested with 3.11) for the backend and reproduction.
- Backend Python dependencies (see `python_backend/requirements.txt`): `torch`, `fastapi`, `uvicorn`, `pandas`, `numpy`, `scikit-learn`, `scipy`, `lime`, `shap`.
- Test dependencies (see `python_backend/requirements-test.txt`): `pytest`, `httpx`.
- Optional, only to build or run the extension UI: **Node.js 18+** and **VS Code 1.78+** (or any VS Code-compatible editor such as Cursor or Antigravity).

## Operating systems
- Verified on Windows 11. The backend and reproduction are OS-independent and also run on Linux and macOS.

## Network
- **None required for evaluation.** All datasets are bundled in `python_backend/datasets/`; the reproduction and tests run fully offline.

## Machine-readable dependency files
- `python_backend/requirements.txt`, `python_backend/requirements-test.txt`
- `package.json` (extension, Node dependencies)
- `Dockerfile` (optional containerized reproduction)

## Optional: Docker
A `Dockerfile` is provided for a containerized reproduction of Table 2:
```bash
docker build -t fairlint-dl .
docker run --rm fairlint-dl                 # runs reproduce.py
docker run --rm fairlint-dl python tests/test_backend.py   # runs the smoke test
```
