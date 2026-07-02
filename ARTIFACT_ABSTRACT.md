# Artifact Abstract — FairLint-DL

## Paper title
FairLint-DL: An IDE-Native Tool for Fairness Debugging of Deep Learning Software (ASE 2026, Tools and Datasets track).

## Link to the accepted paper
The accepted paper (PDF) is included in the archived artifact deposit and available at: **<INSERT ZENODO DOI / PAPER LINK HERE>**

## Purpose
FairLint-DL is an editor-native extension (VS Code / Cursor / Antigravity) with a Python FastAPI + PyTorch backend that performs pre-training fairness analysis of tabular datasets. It trains a proxy deep neural network, computes information-theoretic Quantitative Individual Discrimination (QID) metrics, performs a two-phase gradient-guided search for discriminatory instances, localizes bias to specific layers and neurons via causal debugging, computes group-fairness metrics (demographic parity, equalized odds, equal opportunity), and produces SHAP and LIME explanations. The artifact lets reviewers install the backend, run an automated test suite, and reproduce the paper's Table 2 offline from bundled datasets.

## Badge
We apply for **Artifacts Available** and **Artifacts Reusable**.
- *Available*: the artifact is archived on Zenodo with a permanent DOI under the MIT license.
- *Reusable*: it is documented (Getting Started + step-by-step reproduction), verified by a 19-test suite, structured by concern, self-contained (bundled datasets, offline reproduction), and the backend is reusable independently of the editor. Justification is detailed in `STATUS.md`.

## Technology skills and hardware
Basic command-line and Python familiarity. **CPU only; no GPU required.** ~2 GB disk, ~4 GB RAM. Python 3.9+. Optional Node.js 18+ and a VS Code-compatible editor to exercise the UI. No network access needed for evaluation.

## Provenance
- Archival deposit (DOI): **<INSERT ZENODO DOI HERE>**
- Source repository: https://github.com/Archit1706/FairLint-DL
- Published extension: https://marketplace.visualstudio.com/items?itemName=ArchitRathod.fairlint-dl and https://open-vsx.org/extension/ArchitRathod/fairlint-dl
- Demo video: https://youtu.be/AmfpjK24uwY

## Instructions
Tools required: Python 3.9+ (and `pip`). No special OS or GPU.

1. **Install** (≈10 min): `cd python_backend`, create a venv, `pip install -r requirements.txt -r requirements-test.txt`.
2. **Smoke test** (<1 min): `python tests/test_backend.py` → expect `19/19 tests passed`.
3. **Reproduce Table 2** (≈2–4 min): `python reproduce.py` → prints the QID, group-fairness, accuracy, and composite-score table; compare against Table 2 in the paper.
4. Optional containerized run: `docker build -t fairlint-dl . && docker run --rm fairlint-dl`.

Full details, the mapping of supported/unsupported paper claims, and the dataset schema are in `ARTIFACT.md` and `REQUIREMENTS.md` in the deposit. Reviewers should be able to complete installation and the smoke test in well under 30 minutes.
