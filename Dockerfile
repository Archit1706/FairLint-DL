# Optional containerized reproduction of the FairLint-DL paper results.
#   docker build -t fairlint-dl .
#   docker run --rm fairlint-dl                              # reproduce Table 2
#   docker run --rm fairlint-dl python tests/test_backend.py # smoke test (19 tests)
FROM python:3.11-slim

WORKDIR /app/python_backend

# Install backend + test dependencies (CPU-only torch).
COPY python_backend/requirements.txt python_backend/requirements-test.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-test.txt

# Copy the backend (code, bundled datasets, tests, reproduction script).
COPY python_backend/ ./

# Default: reproduce Table 2 from the bundled datasets (no network needed).
CMD ["python", "reproduce.py"]
