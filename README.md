# FairLint-DL: Deep Learning-Based Fairness Debugger for VS Code

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)

**FairLint-DL** is a VS Code extension designed to detect, analyze, and localize fairness defects in machine learning datasets using deep neural networks. Based on research methodologies like DICE (Distribution-Aware Input Causal Explanation) and NeuFair, this tool integrates advanced fairness testing directly into the development workflow.

---

## Key Features

### Information-Theoretic Bias Detection

-   **Quantitative Individual Discrimination (QID)**: Metrics computed using Shannon and Min entropy to quantify the protected information leaked into model decisions.
-   **Disparate Impact Analysis**: Automatically detects violations of the 80% Rule (e.g., legal thresholds for hiring practices).

### Group Fairness Metrics

-   **Demographic Parity**: Checks whether both demographic groups receive positive predictions at equal rates.
-   **Equalized Odds**: Compares True Positive Rate and False Positive Rate across groups to detect differential error rates.
-   **Equal Opportunity**: Ensures qualified members of both groups are identified at equal rates.
-   Per-group confusion matrices with interactive attribute switching and comparison charts.

### Deep Neural Network Analysis

-   **Proxy Model**: Trains a configurable PyTorch DNN (Deep Neural Network) on your dataset to serve as a fairness oracle.
-   **Model Caching**: SHA-256 content-based caching saves trained models, preprocessor state, and data splits to disk. Subsequent analyses with identical configuration skip training entirely (77s → <1s).
-   **Causal Debugging**: Utilizes gradient-based sensitivity analysis to identify specific network layers and neurons responsible for encoding bias.

### Gradient-Guided Discriminatory Search

-   **Two-Phase Search Algorithm**:
    1.  **Global Search**: Uses gradient ascent to find regions of the input space with high discrimination (maximum QID).
    2.  **Local Search**: Performs perturbation around high-risk instances to generate concrete discriminatory test cases.

### Explainability (SHAP & LIME)

-   **SHAP**: Global feature importance using KernelExplainer on log-odds output space, with beeswarm plots, scatter plots, and heatmaps.
-   **LIME**: Local explanations via perturbation-based linear approximation.
-   **Interactive LIME Explorer**: Generate per-instance LIME explanations on demand — select test instances by index or enter custom feature values for "what-if" scenario analysis.

### Interactive Visualization

-   Real-time charts showing QID distribution, disparate impact, layer sensitivity, and group fairness comparisons.
-   Deep integration with VS Code's Webview API for a seamless dashboard experience.
-   Composite fairness score (0–100) incorporating individual-level QID and group-level metrics.

---

## Technical Architecture

The system operates as a client-server architecture:

1.  **VS Code Extension (Client)**: Written in TypeScript. It handles UI interactions, file system access, and manages the lifecycle of the Python backend.
2.  **Analysis Server (Backend)**: A FastAPI server running locally. It hosts the PyTorch models and analysis algorithms.

### Communication Flow

1.  User triggers analysis on a CSV file.
2.  Extension spawns the Python server (`python -m uvicorn bias_server:app`).
3.  Extension sends dataset path and configuration to the server via REST API.
4.  Server performs training and analysis, returning JSON results.
5.  Extension renders results in an interactive Webview.

---

## Detailed Technical Implementation

### 1. Fairness Detector DNN

The core component is a feedforward neural network (`FairnessDetectorDNN`) implemented in PyTorch. By default, it uses a 6-layer architecture designed to capture complex, non-linear dependencies between features:

-   **Input Layer**: Matches dataset feature dimension.
-   **Hidden Layers**: Configurable sizes (default: 64 -> 32 -> 16 -> 8 -> 4).
-   **Regularization**: Uses BatchNorm, ReLU activation, and Dropout (0.2) to prevent overfitting.
-   **Output**: Binary classification logits.

### 2. QID Computation (`qid_analyzer.py`)

QID measures the causal influence of protected attributes on the model's prediction.

#### Shannon Entropy QID

Quantifies the uncertainty in prediction introduced by varying protected attributes $A$ while keeping other attributes $X$ constant.

$$ QID(x) = H(Y | X) $$

Computed by:

1.  Generating counterfactuals $x'$ by modifying protected attributes of $x$.
2.  Computing predictions $P(Y|x')$ for all counterfactuals.
3.  Calculating the entropy of the _average_ prediction distribution.
4.  **Interpretation**: 0 bits implies perfect fairness; >1 bit indicates significant bias.

#### Min Entropy QID

Used for worst-case analysis (connected to Extreme Value Theory). It focuses on the maximum probability of the most likely outcome across counterfactuals.

### 3. Discriminatory Instance Search (`search.py`)

Finding individual instances of discrimination is treated as an optimization problem.

#### Phase 1: Global Search (Gradient Ascent)

The algorithm maximizes the QID score directly.

-   **Objective**: Maximize variance of predictions across counterfactuals.
-   **Method**: Backpropagates the gradient of the QID loss with respect to the input features $x$.
-   **Result**: Finds an input $x_{global}$ that maximizes discrimination potential.

#### Phase 2: Local Search (Perturbation)

Explores the neighborhood of $x_{global}$ to find concrete discriminatory instances.

-   Adds Gaussian noise to non-protected features.
-   Filters neighbors that exceed the QID threshold (>0.1 bits).

### 4. Causal Debugging (`causal_debugger.py`)

Once bias is detected, the tool localizes it to specific model components.

#### Layer Localization

Determines which layer is most sensitive to discriminatory inputs.

-   Computes the gradient of layer activations $L_i(x)$ with respect to the input $x$.
-   Aggregates the magnitude of these gradients for known discriminatory instances.
-   **Sensitivity Score**: High gradient magnitude indicates the layer significantly amplifies bias.

#### Neuron Localization

Identifies individual neurons encoding protected information.

-   Analyzes activation magnitudes of neurons in the biased layer.
-   Neurons with consistently high activations on discriminatory inputs are flagged as "biased neurons."

### 5. Group Fairness Metrics (`group_fairness.py`)

Complements individual-level QID with standard group-level fairness metrics:

-   **Demographic Parity**: Compares positive prediction rates across demographic groups. Difference close to 0 = fair.
-   **Equalized Odds**: Compares TPR and FPR across groups. Ensures error rates are balanced.
-   **Equal Opportunity**: Focuses on TPR equality — qualified individuals should be identified at equal rates regardless of group.
-   Groups are split by the standardized protected attribute value (<=0 vs >0). Per-group confusion matrices are computed for each protected attribute.

### 6. Model Caching (`model_cache.py`)

Deterministic caching system that avoids redundant retraining:

-   **Cache Key**: SHA-256 hash of the CSV file contents + label column + sorted sensitive features + hidden layers + epochs + batch size. Any change invalidates the cache.
-   **Stored Artifacts**: Model weights (`model.pt`), preprocessor state (`preprocessor.pkl`), train/val/test data splits (`data_tensors.pt`), and human-readable metadata (`metadata.json`).
-   **Cache Location**: `.fairlint_cache/` directory next to the CSV file (gitignored).
-   **Force Retrain**: Right-click menu option bypasses the cache for fresh training.

### 7. Interactive LIME Explorer

On-demand per-instance LIME explanations via the `/explain-instance` endpoint:

-   **Test Instance Mode**: Select any test set instance by index.
-   **Custom Values Mode**: Enter feature values manually for "what-if" scenario analysis.
-   Results appear inline in the dashboard with prediction probabilities and a feature importance bar chart.

---

## Installation

### Prerequisites

-   **VS Code** 1.78.0 or higher
-   **Python** 3.9+ with `pip`
-   **Node.js** 16.x+ (for building from source)

### Setup Steps

1.  **Install Python Dependencies**:

    ```bash
    cd python_backend
    pip install -r requirements.txt
    ```

    _Dependencies: PyTorch, FastAPI, Uvicorn, Pandas, Scikit-learn, SciPy, SHAP, LIME._

2.  **Install Node Dependencies**:

    ```bash
    npm install
    ```

3.  **Build Extension**:

    ```bash
    npm run compile
    ```

4.  **Run in Debug Mode**:
    Press `F5` in VS Code to launch the Extension Development Host.

---

## Usage Guide

1.  **Open Dataset**: Open a CSV file in VS Code.
2.  **Start Analysis**: Right-click the file and select **"FairLint-DL: Analyze This Dataset"** (uses cached model if available) or **"FairLint-DL: Analyze Dataset (Force Retrain)"** to train a fresh model.
3.  **Configure**:
    -   Select the **Label Column** (target variable).
    -   Select **Protected Attributes** (e.g., gender, race — often auto-detected).
    -   Choose **Model Architecture** (Default, Wide, Deep, or Custom).
4.  **Review Results**:
    -   **Fairness Score**: Composite 0–100 score combining individual and group metrics.
    -   **QID Metrics**: Mean/Max QID, disparate impact, and per-instance discrimination analysis.
    -   **Group Fairness**: Demographic Parity, Equalized Odds, and Equal Opportunity per protected attribute.
    -   **Causal Debugging**: Layer and neuron sensitivity analysis.
    -   **SHAP & LIME**: Global and local feature importance, with interactive per-instance LIME exploration.
    -   **Export**: Download all results as structured JSON.

---

## Project Structure

```
fairlint-dl/
├── .vscode/               # VS Code launch configurations
├── python_backend/        # Analysis Server
│   ├── analyzers/         # Core Algorithmic Logic
│   │   ├── causal_debugger.py    # Layer/Neuron attribution
│   │   ├── qid_analyzer.py       # Entropy-based metrics
│   │   ├── search.py             # Gradient-guided search
│   │   ├── explainability.py     # SHAP and LIME explanations
│   │   ├── group_fairness.py     # Demographic Parity, Equalized Odds, Equal Opportunity
│   │   └── internal_space.py     # PCA/t-SNE activation visualization
│   ├── models/            # PyTorch Model Definitions
│   │   └── fairness_dnn.py
│   ├── utils/             # Data Processing & Caching
│   │   ├── data_loader.py        # CSV loading, encoding, scaling, splitting
│   │   └── model_cache.py        # SHA-256 model caching system
│   ├── bias_server.py     # FastAPI Entry Point
│   └── requirements.txt   # Python Dependencies
├── src/                   # VS Code Extension Source
│   ├── extension.ts       # Main Extension Entry Point
│   ├── analysis/
│   │   ├── columns.ts     # CSV column fetching
│   │   └── pipeline.ts    # 6-step analysis pipeline orchestration
│   └── webview/
│       ├── results.ts     # WebviewPanel creation, message handling
│       ├── htmlBuilder.ts # HTML section builders with dynamic interpretations
│       ├── charts.ts      # Plotly.js chart rendering
│       ├── scoring.ts     # Composite fairness score calculation
│       ├── styles.ts      # CSS styles
│       └── types.ts       # Shared TypeScript interfaces
├── package.json           # Extension Manifest
└── README.md              # Documentation
```

---

## API Reference (Local Server)

When the extension runs, it starts a local server at `http://localhost:8765`.

-   `GET /`: Health check endpoint.
-   `POST /columns`: Returns column names, sample data, and auto-detected sensitive features from a CSV.
-   `POST /train`: Trains the proxy model (or loads from cache if available). Supports `force_retrain` flag.
-   `POST /activations`: Returns PCA/t-SNE reduced layer activations for internal space visualization.
-   `POST /analyze`: Computes bulk QID metrics and group fairness metrics (Demographic Parity, Equalized Odds, Equal Opportunity).
-   `POST /search`: Runs the global/local search for discriminatory instances.
-   `POST /debug`: Performs layer and neuron sensitivity analysis.
-   `POST /explain`: Generates batch SHAP/LIME explanations.
-   `POST /explain-instance`: Generates a single-instance LIME explanation (by test index or custom feature values).

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

If you use this tool for research, please cite the original DICE paper:

> _Information-Theoretic Testing and Debugging of Fairness Defects in Deep Neural Networks_
> Saeid Tizpaz-Niari et al.

---

**FairLint-DL** - Ensuring Fairness in AI, One Line of Code at a Time.
