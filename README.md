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

### Deep Neural Network Analysis

-   **Proxy Model**: Trains a configurable PyTorch DNN (Deep Neural Network) on your dataset to serve as a fairness oracle.
-   **Causal Debugging**: Utilizes gradient-based sensitivity analysis to identify specific network layers and neurons responsible for encoding bias.

### Gradient-Guided Discriminatory Search

-   **Two-Phase Search Algorithm**:
    1.  **Global Search**: Uses gradient ascent to find regions of the input space with high discrimination (maximum QID).
    2.  **Local Search**: Performs perturbation around high-risk instances to generate concrete discriminatory test cases.

### Interactive Visualization

-   Real-time charts showing QID distribution, disparate impact, and layer sensitivity.
-   Deep integration with VS Code's Webview API for a seamless dashboard experience.

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

-   analyzes activation magnitudes of neurons in the biased layer.
-   Neurons with consistently high activations on discriminatory inputs are flagged as "biased neurons."

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

    _Dependencies: PyTorch, FastAPI, Uvicorn, Pandas, Scikit-learn, SciPy._

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
2.  **Start Analysis**: Right-click the file and select **"Fairness: Analyze This Dataset"**.
3.  **Configure**:
    -   Select the **Label Column** (target variable).
    -   Select **Protected Attributes** (e.g., gender, race - often auto-detected).
    -   Choose **Model Architecture** (Default, Wide, Deep).
4.  **Review Results**:
    -   **Metrics**: View Mean QID and Disparate Impact.
    -   **Visualization**: Explore the 3D interaction charts and heatmaps.
    -   **Updates**: Tracking progress via the status bar.

---

## Project Structure

```
fairlint-dl/
├── .vscode/               # VS Code launch configurations
├── python_backend/        # Analysis Server
│   ├── analyzers/         # Core Algorithmic Logic
│   │   ├── causal_debugger.py    # Layer/Neuron attribution
│   │   ├── qid_analyzer.py       # Entropy-based metrics
│   │   └── search.py             # Gradient-guided search
│   ├── models/            # PyTorch Model Definitions
│   │   └── fairness_dnn.py
│   ├── utils/             # Data Processing
│   │   └── data_loader.py
│   ├── bias_server.py     # FastAPI Entry Point
│   └── requirements.txt   # Python Dependencies
├── src/                   # VS Code Extension Source
│   └── extension.ts       # Main Extension Entry Point
├── package.json           # Extension Manifest
└── README.md              # Documentation
```

---

## API Reference (Local Server)

When the extension runs, it starts a local server at `http://localhost:8765`.

-   `POST /train`: Trains the proxy model on the provided CSV.
-   `POST /analyze`: Computes bulk QID metrics for the dataset.
-   `POST /search`: Runs the global/local search for discriminatory instances.
-   `POST /debug`: Performs layer and neuron sensitivity analysis.
-   `POST /activations`: Returns data for internal representation visualization.
-   `POST /explain`: Generates SHAP/LIME explanations (if enabled).

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
