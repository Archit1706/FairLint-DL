# Change Log

All notable changes to the "FairLint-DL" extension will be documented in this file.

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
