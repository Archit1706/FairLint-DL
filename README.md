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

### Step 1: Install the Extension

Install **FairLint-DL** directly from the VS Code Marketplace:

1.  Open VS Code
2.  Go to the **Extensions** panel (`Ctrl+Shift+X`)
3.  Search for **"FairLint-DL"**
4.  Click **Install**

Or install via the command line:

```bash
code --install-extension ArchitRathod.fairlint-dl
```

### Step 2: Install Python Dependencies

FairLint-DL requires **Python 3.9+** installed on your system. The extension bundles the backend code, but you need to install the Python packages:

```bash
pip install torch fastapi uvicorn pandas numpy scikit-learn scipy lime shap
```

> **Tip:** If you use a virtual environment, make sure VS Code is configured to use the correct Python interpreter (`Ctrl+Shift+P` → `Python: Select Interpreter`).

### Step 3: Verify Setup

1.  Open any `.csv` file in VS Code
2.  Right-click the file in the Explorer
3.  You should see **"FairLint-DL: Analyze This Dataset"** in the context menu

If the backend server fails to start, check that:
-   Python 3.9+ is installed and accessible from your terminal
-   All required Python packages are installed
-   Port 8765 is not in use by another application

---

## Quick Start Guide

### 1. Prepare Your Dataset

FairLint-DL works with **CSV files** containing tabular data. Your dataset should have:
-   A **label/target column** (binary classification — e.g., `income`, `hired`, `approved`)
-   One or more **protected/sensitive attributes** (e.g., `gender`, `race`, `age`)

### 2. Launch the Analysis

There are two ways to start:

| Method | How | When to Use |
|--------|-----|-------------|
| **Analyze Dataset** | Right-click CSV → **"FairLint-DL: Analyze This Dataset"** | First analysis or when using cached results |
| **Force Retrain** | Right-click CSV → **"FairLint-DL: Analyze Dataset (Force Retrain)"** | When you want to retrain the model from scratch |

### 3. Configure the Analysis

After launching, you'll be prompted to configure three things:

1.  **Label Column** — Select the target variable your model predicts (e.g., `income >50K`)
2.  **Protected Attributes** — Select one or more sensitive features to test for bias (e.g., `sex`, `race`). Common sensitive attributes are auto-detected.
3.  **Model Architecture** — Choose the DNN architecture:
    -   🏗️ **Default** (64→32→16→8→4) — Balanced, works well for most datasets
    -   📐 **Wide** (128→64→32) — Better for datasets with many features
    -   📏 **Deep** (64→48→32→24→16→8→4) — Better for complex relationships
    -   ⚙️ **Custom** — Define your own layer sizes

### 4. Wait for Training

The extension will:
1.  Start the Python backend server automatically
2.  Send your dataset for preprocessing
3.  Train a proxy DNN model (or load from cache if previously trained)
4.  Run the full 6-step fairness analysis pipeline

> **Note:** First-time training may take 30–90 seconds depending on dataset size. Subsequent runs with the same configuration use cached models and complete in under 1 second.

### 5. Review the Results Dashboard

Once analysis completes, an interactive dashboard opens with the following sections:

#### 🎯 Fairness Score (0–100)
A composite score combining individual and group fairness metrics. Higher is fairer.
-   **90–100**: Low bias detected
-   **70–89**: Moderate bias — review recommended
-   **Below 70**: Significant bias — action needed

#### 📊 QID Analysis (Individual Fairness)
-   **Mean/Max QID**: How much the protected attribute influences predictions on average and at worst
-   **Disparate Impact Ratio**: Whether the 80% rule is satisfied (legal threshold in hiring)
-   **QID Distribution Chart**: Histogram showing per-instance discrimination levels

#### 👥 Group Fairness Metrics
-   **Demographic Parity**: Are positive predictions given at equal rates across groups?
-   **Equalized Odds**: Are error rates (TPR/FPR) balanced across groups?
-   **Equal Opportunity**: Are qualified individuals identified equally regardless of group?
-   Interactive charts with per-attribute switching

#### 🔍 Causal Debugging
-   **Layer Sensitivity**: Which network layers amplify bias the most
-   **Neuron Analysis**: Specific neurons encoding protected information
-   Helps understand *where* in the model bias is introduced

#### 🧪 Discriminatory Instance Search
-   Concrete examples of inputs where the model discriminates
-   Found via gradient-guided search (global) + perturbation (local)

#### 📈 Explainability (SHAP & LIME)
-   **SHAP**: Global feature importance — which features drive predictions overall
-   **LIME**: Local explanations — why the model made a specific decision
-   **Interactive LIME Explorer**: Select any test instance or enter custom feature values for "what-if" analysis

### 6. Export Results

Click the **"Export Results"** button at the bottom of the dashboard to download all analysis results as a structured JSON file.

---

## Extension Settings

Configure FairLint-DL via VS Code Settings (`Ctrl+,` → search "fairlint"):

| Setting | Default | Description |
|---------|---------|-------------|
| `fairlint-dl.pythonPath` | `python` | Path to Python interpreter |
| `fairlint-dl.serverPort` | `8765` | Port for the analysis backend server |
| `fairlint-dl.defaultEpochs` | `50` | Default training epochs |
| `fairlint-dl.defaultBatchSize` | `32` | Default training batch size |
| `fairlint-dl.autoDetectSensitiveFeatures` | `true` | Auto-detect protected attributes |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **Server Startup Failed** | Ensure Python 3.9+ is installed and packages are installed via `pip install torch fastapi uvicorn pandas numpy scikit-learn scipy lime shap` |
| **Cannot connect to analysis server** | Restart VS Code, or check if port 8765 is already in use (`lsof -i :8765` on Mac/Linux, `netstat -ano | findstr 8765` on Windows) |
| **Training is slow** | Reduce epochs in settings, or use a smaller dataset. Cached models load instantly on repeat runs. |
| **Charts not rendering** | Ensure you're not blocking CDN resources — Plotly.js is loaded from CDN in the webview |

---

## How It Works

FairLint-DL uses a **6-step analysis pipeline**:

1.  **Train** — A proxy DNN model learns the patterns in your dataset
2.  **Activations** — Internal layer activations are extracted and visualized via PCA/t-SNE
3.  **QID Analysis** — Information-theoretic metrics quantify how much protected attributes influence predictions
4.  **Search** — Gradient-guided search finds concrete discriminatory instances
5.  **Debug** — Causal analysis localizes bias to specific layers and neurons
6.  **Explain** — SHAP and LIME provide global and local feature importance explanations

The extension runs a local FastAPI server (`localhost:8765`) that handles all computation. The server starts automatically when you trigger an analysis and shuts down when VS Code closes.

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
