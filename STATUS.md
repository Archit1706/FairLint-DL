# STATUS

We are applying for the following ACM artifact badges:

## Artifacts Available
The artifact is archived on Zenodo with a permanent DOI and released under the MIT license (see `LICENSE`), ensuring public, long-term availability.

- Zenodo DOI: **https://doi.org/10.5281/zenodo.21142733**
- Source repository: https://github.com/Archit1706/FairLint-DL
- Published extension: https://marketplace.visualstudio.com/items?itemName=ArchitRathod.fairlint-dl and https://open-vsx.org/extension/ArchitRathod/fairlint-dl

## Artifacts Reusable
We believe the artifact also merits the Reusable badge:
- **Documented.** `ARTIFACT.md` provides a Getting Started guide with a <1-minute smoke test and step-by-step instructions to reproduce Table 2; `REQUIREMENTS.md` specifies the environment; the code is commented and organized by concern (`analyzers/`, `models/`, `utils/`).
- **Consistent and complete.** The bundled datasets and the `reproduce.py` script regenerate the paper's Table 2 offline; the analysis code is the same code the published extension runs.
- **Exercisable with evidence of verification.** A 19-test suite (`python_backend/tests/test_backend.py`) exercises QID, discriminatory-instance search, causal debugging, group fairness, SHAP/LIME, preprocessing, and the REST endpoints, and passes cleanly.
- **Structured for reuse and repurposing.** The backend is a self-contained REST service usable independently of the editor; `reproduce.py` accepts additional datasets by editing a single config dictionary; the extension installs from two marketplaces and runs in multiple IDEs.

## Reproducibility notes
`reproduce.py` regenerates Table 2 from the bundled datasets. Runs are seeded and deterministic on a given platform; values may differ by up to ~0.01 across platforms (e.g., Linux container vs. Windows host) due to CPU/BLAS floating-point nondeterminism in DNN training, without changing the conclusions. The Adult QID rows and Figures 3–5 additionally come from an earlier unseeded run and are close but not bit-for-bit identical (mean QID ~0.60 seeded vs. 0.619 reported). This is stated in `ARTIFACT.md`. Timing figures are hardware-dependent.
