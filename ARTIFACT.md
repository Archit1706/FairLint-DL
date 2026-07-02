# FairLint-DL — Artifact README

Artifact for the paper **"FairLint-DL: An IDE-Native Tool for Fairness Debugging of Deep Learning Software"** (ASE 2026, Tools and Datasets track).

FairLint-DL is a VS Code / editor extension (TypeScript) plus a Python (FastAPI + PyTorch) backend that performs pre-training fairness analysis of tabular datasets: it trains a proxy DNN, computes information-theoretic QID metrics, searches for discriminatory instances, localizes bias to layers/neurons, computes group-fairness metrics, and produces SHAP/LIME explanations.

This artifact lets a reviewer (1) install the backend and run the automated test suite, and (2) reproduce the quantitative results in **Table 2** of the paper from the bundled datasets, with no network access.

---

## Getting Started (about 15 minutes)

### Prerequisites
- Python 3.9+ (CPU only; no GPU required)
- ~2 GB free disk (mostly PyTorch)
- Optional, only to build/run the extension UI: Node.js 18+ and VS Code (or Cursor/Antigravity)

### Install (backend)
```bash
cd python_backend
python -m venv venv
# Windows:  venv\Scripts\activate
# Linux/mac: source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-test.txt      # pytest + httpx (for the smoke test)
```

### Smoke test (confirms the install works, < 1 minute)
```bash
python tests/test_backend.py
```
Expected output ends with:
```
19/19 tests passed
```
This exercises QID computation, discriminatory-instance search, causal debugging, group fairness, SHAP/LIME, data preprocessing, and the REST endpoints.

---

## Step-by-Step: Reproducing the Paper

### Reproduce Table 2 (about 2–4 minutes on CPU)
```bash
cd python_backend
python reproduce.py            # all three datasets
python reproduce.py --quick    # Adult + German only (faster)
```
This reads the bundled datasets in `python_backend/datasets/` (Adult Census, German Credit, Bank Marketing), runs the exact backend pipeline the extension uses, and prints a table plus writes `reproduce_results.json`. Compare the printed table against **Table 2** in the paper.

Expected (seeded) output:

| Metric | Adult | German | Bank |
|---|---|---|---|
| Mean QID (bits) | 0.601 | 0.935 | 0.284 |
| Mean min-entropy QID | 0.195 | 0.428 | 0.108 |
| Mean disparate impact | 0.636 | 0.939 | 0.815 |
| Discriminatory (%) | 98.6 | 100.0 | 51.6 |
| Violating 80% (%) | 72.6 | 3.5 | 42.2 |
| Demographic parity diff | 0.091 | 0.095 | 0.000 |
| Equalized odds diff | 0.102 | 0.114 | 0.000 |
| Model accuracy (%) | 84.1 | 68.5 | 88.3 |
| Composite score | 43 | 46 | 75 |

### Claims supported by this artifact
- **Individual discrimination varies widely across datasets** (Table 2): German Credit most severe (mean QID 0.935, ~100% discriminatory), Bank Marketing mildest (0.284, 51.6%). Reproduced exactly by `reproduce.py`.
- **Individual vs. group divergence on German Credit**: nearly all instances individually discriminatory while the aggregate disparate-impact ratio (0.939) satisfies the four-fifths rule. Reproduced.
- **The full six-step pipeline runs end-to-end** (QID, search, debugging, group fairness, SHAP/LIME). Exercised by the test suite and `reproduce.py`.

### Claims NOT fully supported (and why)
- **The paper's exact Adult headline numbers** (mean QID 0.619, 96.0%, DI 0.581) and **Figures 3–5** come from an earlier run in which DNN training was not seeded, so they are not bit-for-bit reproducible. The seeded run here yields close values (mean QID ~0.601, 98.6%) and reproduces the qualitative claim robustly.
- **Timing numbers** (12 s cached, 77 s first-run training) are hardware-dependent and will vary by machine.

---

## Using the Extension (optional, requires a GUI editor)
Install "FairLint-DL" from the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=ArchitRathod.fairlint-dl) or [Open VSX](https://open-vsx.org/extension/ArchitRathod/fairlint-dl) (works in VS Code, Cursor, Antigravity). Then right-click any `.csv` in the Explorer → **"FairLint-DL: Analyze This Dataset"**, choose the label and protected columns, and view the dashboard. A demo video is at https://youtu.be/AmfpjK24uwY.

To build from source: `npm install && npm run compile`, then press F5 in VS Code to launch the Extension Development Host.

---

## Layout
```
python_backend/
  reproduce.py            # reproduces Table 2 (this artifact's main entry point)
  tests/test_backend.py   # 19-test smoke/verification suite
  datasets/               # bundled Adult, German Credit, Bank Marketing CSVs
  bias_server.py          # FastAPI backend (REST endpoints)
  analyzers/              # QID, search, causal debugger, group fairness, SHAP/LIME, PCA
  models/fairness_dnn.py  # proxy DNN + trainer
  utils/                  # data loader, model cache
src/                      # TypeScript VS Code extension
Dockerfile                # optional containerized reproduction
```

See `REQUIREMENTS.md` for environment details, `STATUS.md` for the requested badges, and `LICENSE` (MIT).
