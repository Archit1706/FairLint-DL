# Change Log

All notable changes to the "FairLint-DL" extension will be documented in this file.

## [1.0.5] - 2026-10-02

### Fixed
- The backend could be started with a Python interpreter that lacked its dependencies. On a machine with more than one Python, the fallback picked `python3`, which on Windows is often the Microsoft Store build, while the packages had been installed under a different interpreter. `uvicorn` then never started, and the resulting connection failure surfaced to the user as **"The CSV file appears to be empty or has no columns"**, pointing at the data rather than the real cause. Interpreters are now verified to import every required module before one is chosen, `python` is preferred over `python3`, and if none qualify the error names each interpreter tried, what it was missing, and the exact `pip install` command.
- A failed `/columns` request was reported as an empty dataset. The request failure already raises its own error, so the misleading second message is gone.

### Added
- `fairlint-dl.python.interpreterPath` setting, to point the extension at a specific interpreter when automatic detection picks the wrong one.

## [1.0.4] - 2026-10-02

### Fixed
- `/columns` returned a bare HTTP 500 for any CSV with a missing value in its first three rows. The endpoint's row preview passed `NaN` straight to the JSON encoder, which rejects it, and because the failure happened during response serialisation rather than inside the handler it bypassed the error handling and produced no diagnostic. `NaN` and infinities are now encoded as `null`. Affected standard benchmarks such as COMPAS.
- Semicolon-separated CSVs were silently parsed as a single column, with no error, leaving an unusable column list. The delimiter is now sniffed from the file header (`utils.data_loader.detect_delimiter`) and applied in both `/columns` and `load_and_preprocess`, so the two always agree. Affected the UCI Bank Marketing files as distributed.
- The pandas 3.0 string-dtype handling from the artifact release is now in the published extension; it landed after the 1.0.3 version bump and so missed that release.

### Changed
- README install command pins `pandas<3.0` and `numpy<2.3`, matching `python_backend/requirements.txt`, and explains why.

## [1.0.3] - 2026-07-02

### Changed
- Optimized `architecture.png` (11255x5735, 36 MB to 2000x1019, 0.27 MB), shrinking the packaged extension from ~34 MB to ~1 MB with no visible quality loss.

## [1.0.2] - 2026-07-02

### Added
- Automated test suite for the Python backend: 19 unit tests covering QID (Shannon and min-entropy / disparate impact), discriminatory-instance search, causal layer/neuron localization, group-fairness metrics, internal-space PCA, SHAP and LIME explainers, model-cache key determinism, data preprocessing, and the REST endpoints. Runnable with `pytest` or standalone (`python tests/test_backend.py`).
- TypeScript unit tests for the composite fairness-score logic (`npm test`, Node built-in test runner).
- `python_backend/requirements-test.txt` (pytest, httpx) and `npm test` / `npm run test:backend` scripts.
- Evaluation on three tabular benchmarks (Adult Census, German Credit, Bank Marketing); results documented in the README.

### Changed
- Documented multi-IDE support: the extension is published on Open VSX and runs unmodified in any VS Code-compatible IDE, including Cursor and Antigravity.

### Removed
- Dead code inherited from the Microsoft `vscode-python-tools-extension-template`: the LSP language-server scaffolding (`src/common/`, `bundled/tool/lsp_*.py`), the leftover LSP test harness (`src/test/python_tests/`), and `noxfile.py`. These were unused by the live extension; removing them has no effect on features.

## [1.0.0] - 2026-03-10

### Added
- 6-step fairness analysis pipeline: Train, Activations, QID Analysis, Search, Debug, Explain
- Deep Neural Network training with configurable architecture (hidden layers, epochs, batch size)
- Model caching with SHA-256 content-based cache keys for fast re-analysis
- Force retrain command to bypass cache when needed
- QID (Quantitative Input Influence) individual discrimination metrics
- Group fairness metrics: Demographic Parity, Equalized Odds, Equal Opportunity
- Per-group confusion matrices and comparison tables
- Global discriminatory instance search (gradient-guided)
- Causal debugging to identify biased layers and neurons
- SHAP global feature importance analysis
- LIME local explanation aggregation
- Interactive LIME explorer for per-instance explanations (test set or custom input)
- Composite fairness score (0-100) combining individual and group metrics
- Interactive results dashboard with Plotly.js charts
- Activation space visualization (t-SNE/PCA)
- Auto-detection of protected/sensitive attributes
- JSON export of full analysis results
- Right-click context menu for CSV files
- Configurable analysis parameters via VS Code settings
- Dark theme dashboard matching VS Code aesthetics
