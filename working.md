# FairLint-DL: How It Works

## A Complete Technical Guide to Deep Learning Fairness Analysis

---

## Table of Contents

1. [What is FairLint-DL?](#1-what-is-fairlint-dl)
2. [Architecture Overview](#2-architecture-overview)
3. [Codebase Structure](#3-codebase-structure)
4. [End-to-End Walkthrough: From Click to Dashboard](#4-end-to-end-walkthrough-from-click-to-dashboard)
5. [Step 1 — Training the Deep Neural Network](#5-step-1--training-the-deep-neural-network)
6. [Step 2 — Internal Space Visualization](#6-step-2--internal-space-visualization)
7. [Step 3 — QID Analysis: Measuring Discrimination](#7-step-3--qid-analysis-measuring-discrimination)
8. [Step 4 — Discriminatory Instance Search](#8-step-4--discriminatory-instance-search)
9. [Step 5 — Causal Debugging](#9-step-5--causal-debugging)
10. [Step 6 — SHAP and LIME Explanations](#10-step-6--shap-and-lime-explanations)
11. [The Fairness Score](#11-the-fairness-score)
12. [The Results Dashboard](#12-the-results-dashboard)
13. [Configuration Reference](#13-configuration-reference)
14. [Example: adult.csv Analysis Results](#14-example-adultcsv-analysis-results)

---

## 1. What is FairLint-DL?

FairLint-DL is a Visual Studio Code extension that detects, measures, and explains unfair bias in tabular datasets. You give it a CSV file and tell it which column is the prediction target (e.g., `income`) and which columns are "protected" (e.g., `race`, `sex`, `age`). It then trains a deep neural network, probes the network's internal behavior, and produces a rich interactive dashboard showing exactly where, how much, and why the model treats people differently based on those protected attributes.

The core question FairLint-DL answers is:

> "If I changed only this person's race (or sex, or age) and left everything else the same, would the model's prediction change?"

If the answer is yes, that is individual discrimination — the model is using protected information to make decisions. FairLint-DL quantifies this in bits of information, localizes it to specific layers and neurons, and explains which features drive the bias using two complementary explainability methods (SHAP and LIME).

### Theoretical Foundations

FairLint-DL draws from two research papers:

- **DICE** — "Information-Theoretic Testing and Debugging of Fairness Defects in DNNs" — which introduced Quantitative Individual Discrimination (QID), a way to measure bias using Shannon entropy and mutual information from information theory. DICE also introduced the causal debugging algorithm that traces bias to specific neurons.

- **NeuFair** — which extended the search algorithm for finding discriminatory instances using a two-phase global-local approach with gradient-based optimization.

The extension combines these with SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) for feature-level explanations, and PCA-based internal space visualization for understanding how the network separates data points internally.

---

## 2. Architecture Overview

FairLint-DL is a client-server application:

```
+---------------------------+          HTTP/REST          +---------------------------+
|   VS Code Extension       | ◄-----------------------► |   Python FastAPI Backend   |
|   (TypeScript)            |    POST /train             |   (PyTorch + uvicorn)     |
|                           |    POST /activations       |                           |
|   - User interface        |    POST /analyze           |   - Data preprocessing    |
|   - QuickPick dialogs     |    POST /search            |   - Model training        |
|   - WebView dashboard     |    POST /debug             |   - QID computation       |
|   - Plotly.js charts      |    POST /explain           |   - Causal debugging      |
|   - JSON export           |    POST /columns           |   - SHAP/LIME analysis    |
+---------------------------+                            +---------------------------+
```

The TypeScript extension handles all user interaction and visualization. The Python backend handles all computation — training, analysis, search, debugging, and explainability. They communicate over HTTP on `localhost:8765` (configurable).

When the extension activates, it starts the Python backend as a child process:

```typescript
// src/server/backend.ts
serverProcess = spawn(pythonPath, ['-m', 'uvicorn', 'bias_server:app', '--port', String(serverPort)], {
    cwd: backendPath,
    shell: true,
});
```

It then polls the server's health endpoint (`GET /`) until it responds, retrying up to 15 times with 1-second intervals.

---

## 3. Codebase Structure

### TypeScript Extension (`src/`)

```
src/
├── extension.ts              Entry point — registers commands, starts backend
├── config/
│   └── settings.ts           FairLintConfig interface, reads VS Code settings
├── server/
│   ├── backend.ts            Python server lifecycle (start, stop, health check)
│   ├── statusBar.ts          VS Code status bar management
│   └── errors.ts             Error parsing with user-friendly messages
├── analysis/
│   ├── columns.ts            CSV column fetching via POST /columns
│   └── pipeline.ts           6-step analysis pipeline orchestration
└── webview/
    ├── types.ts              Shared TypeScript interfaces
    ├── results.ts            WebviewPanel creation, JSON export handler
    ├── htmlBuilder.ts        HTML section builders with dynamic interpretations
    ├── charts.ts             Plotly.js chart rendering functions
    ├── styles.ts             CSS styles for the dashboard
    ├── icons.ts              SVG icon definitions
    └── scoring.ts            Fairness score calculation algorithm
```

### Python Backend (`python_backend/`)

```
python_backend/
├── bias_server.py            FastAPI app with all API endpoints
├── models/
│   └── fairness_dnn.py       FairnessDetectorDNN (PyTorch model) + DNNTrainer
├── analyzers/
│   ├── qid_analyzer.py       Shannon/Min-entropy QID computation
│   ├── search.py             Two-phase discriminatory instance search
│   ├── causal_debugger.py    Layer/neuron bias localization
│   ├── internal_space.py     PCA/t-SNE activation visualization
│   ├── explainability.py     SHAP and LIME explanations
│   └── group_fairness.py     Demographic Parity, Equalized Odds, Equal Opportunity
└── utils/
    ├── data_loader.py        CSV loading, encoding, scaling, splitting
    └── model_cache.py        Model saving/loading with SHA-256 cache keys
```

### Key Dependency Chain

```
extension.ts
  → columns.ts → POST /columns → DataPreprocessor.detect_sensitive_columns()
  → pipeline.ts → POST /train  → model_cache (check) → FairnessDetectorDNN + DNNTrainer → model_cache (save)
                → POST /activations → InternalSpaceAnalyzer
                → POST /analyze → QIDAnalyzer.batch_analyze() + GroupFairnessAnalyzer.compute_all()
                → POST /search  → DiscriminatoryInstanceSearch.search()
                → POST /debug   → CausalDebugger.localize_biased_layer/neurons()
                → POST /explain → ExplainabilityAnalyzer.compute_shap/lime()
  → results.ts  → htmlBuilder.ts → charts.ts (Plotly.js rendering)
                → POST /explain-instance → ExplainabilityAnalyzer (single-instance LIME)
```

---

## 4. End-to-End Walkthrough: From Click to Dashboard

Here is exactly what happens when you right-click `adult.csv` in VS Code's file explorer and select "FairLint-DL: Analyze This Dataset."

### Phase 1: Column Discovery

The extension validates the file is a CSV and exists on disk, then fetches column metadata:

```typescript
// src/analysis/columns.ts
const response = await axios.post(`${serverUrl}/columns`, { file_path: filePath }, { timeout: 10000 });
```

The backend reads only the header row for efficiency, plus 3 sample rows for the preview:

```python
# bias_server.py — POST /columns
df = pd.read_csv(request.file_path, nrows=0)
columns = df.columns.tolist()
df_sample = pd.read_csv(request.file_path, nrows=3)
```

It also auto-detects sensitive columns using regex pattern matching:

```python
# utils/data_loader.py
sensitive_patterns = {
    "gender": r"(gender|sex|male|female)",
    "race": r"(race|ethnicity|ethnic|color)",
    "age": r"(age|birth|dob|year)",
    "nationality": r"(nationality|national|origin|country)",
    ...
}
```

For `adult.csv`, this detects: `age`, `race`, `sex`, `native-country`.

### Phase 2: User Configuration

Three QuickPick dialogs appear in sequence:

**Dialog 1 — Label Column:**
The user sees all 15 columns with sample values (e.g., `income (e.g., "<=50K")`). They select `income`.

**Dialog 2 — Protected Attributes:**
A multi-select QuickPick shows all columns except `income`. The four auto-detected columns (`age`, `race`, `sex`, `native-country`) are pre-selected with "(auto-detected)" labels. The user can add or remove attributes.

**Dialog 3 — DNN Architecture:**
Four options:
- Default (64,32,16,8,4) — the DICE paper architecture
- Wide (128,64,32,16)
- Deep (256,128,64,32,16,8)
- Custom... — opens an input box for comma-separated sizes

### Phase 3: The 6-Step Analysis Pipeline

Once configuration is complete, `pipeline.ts` orchestrates six sequential API calls with real-time progress updates in both a notification bar and the VS Code status bar:

```typescript
// src/analysis/pipeline.ts — simplified flow
progress.report({ increment: 0, message: 'Step 1/6: Training neural network model...' });
const trainResponse = await axios.post(`${serverUrl}/train`, { ... });

progress.report({ increment: 30, message: 'Step 2/6: Computing internal space visualization...' });
const activationsResponse = await axios.post(`${serverUrl}/activations`, { ... });

progress.report({ increment: 8, message: 'Step 3/6: Computing fairness metrics...' });
const analyzeResponse = await axios.post(`${serverUrl}/analyze`, { ... });

progress.report({ increment: 17, message: 'Step 4/6: Searching for discriminatory instances...' });
const searchResponse = await axios.post(`${serverUrl}/search`, { ... });

progress.report({ increment: 15, message: 'Step 5/6: Localizing biased layers and neurons...' });
const debugResponse = await axios.post(`${serverUrl}/debug`, { ... });

progress.report({ increment: 12, message: 'Step 6/6: Computing LIME & SHAP explanations...' });
const explainResponse = await axios.post(`${serverUrl}/explain`, { ... });
```

Each step's timing is recorded in a `stepTimings` object (e.g., training: 77s, explain: 11s).

### Phase 4: The Results Dashboard

All six API responses are bundled into an `AnalysisResults` object and passed to `showResults()`, which creates a VS Code WebviewPanel with `enableScripts: true` and injects the full HTML dashboard:

```typescript
// src/webview/results.ts
const panel = vscode.window.createWebviewPanel(
    'fairnessResults',
    'Fairness Analysis Results',
    vscode.ViewColumn.One,
    { enableScripts: true, retainContextWhenHidden: true },
);
panel.webview.html = getWebviewHtml(results);
```

The HTML is built by `htmlBuilder.ts`, which composes sections for each analysis step — each with interactive Plotly.js charts and data-driven interpretive text that changes based on the actual analysis values.

---

## 5. Step 1 — Training the Deep Neural Network

### Data Preprocessing

Before training, the `DataPreprocessor` class transforms raw CSV data into PyTorch tensors:

```python
# utils/data_loader.py
# 1. Separate features from label column
feature_columns = [c for c in df.columns if c != label_column]
X = df[feature_columns].copy()
y = df[label_column].copy()

# 2. Encode categorical features as integers
categorical_cols = X.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# 3. Normalize all features to zero mean, unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split into train/val/test (70/10/20) with stratification
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=val_size / (1 - test_size),
    random_state=42, stratify=y_train_val,
)
```

For the **adult.csv** dataset:
- **14 input features** after removing the `income` label column
- Features include: `age`, `workclass`, `fnlwgt`, `education`, `education-num`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `capital-gain`, `capital-loss`, `hours-per-week`, `native-country`
- **32,561 total instances** split into: 22,792 training / 3,256 validation / 6,513 test
- **Class distribution**: 17,303 (class 0, <=50K) and 5,489 (class 1, >50K) — an imbalanced dataset
- **4 protected attributes** at indices [0, 8, 9, 13]: age (71 unique values), race (5 unique), sex (2 unique), native-country (42 unique)

Categorical features like `workclass` (e.g., "Private", "Self-emp") are label-encoded to integers, then StandardScaler normalizes all features. After preprocessing, a person who was `[39, "State-gov", 77516, "Bachelors", 13, "Never-married", ...]` becomes a vector of 14 floating-point numbers centered around zero.

### The Neural Network Architecture

```python
# models/fairness_dnn.py
class FairnessDetectorDNN(nn.Module):
    def __init__(self, input_dim, protected_indices, hidden_layers=None, dropout_rate=0.2):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [64, 32, 16, 8, 4]

        layer_dims = [input_dim] + hidden_layers
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(len(hidden_layers)):
            self.hidden_layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            self.batch_norms.append(nn.BatchNorm1d(layer_dims[i + 1]))

        self.output_layer = nn.Linear(hidden_layers[-1], 2)
        self.dropout = nn.Dropout(dropout_rate)
```

With the default `[64, 32, 16, 8, 4]` architecture and 14 input features, the network looks like:

```
Input (14 features)
  │
  ▼
Layer 1: Linear(14 → 64) → BatchNorm(64) → ReLU → Dropout(0.2)    [960 params]
  │
  ▼
Layer 2: Linear(64 → 32) → BatchNorm(32) → ReLU → Dropout(0.2)    [2,112 params]
  │
  ▼
Layer 3: Linear(32 → 16) → BatchNorm(16) → ReLU → Dropout(0.2)    [544 params]
  │
  ▼
Layer 4: Linear(16 → 8) → BatchNorm(8) → ReLU → Dropout(0.2)      [144 params]
  │
  ▼
Layer 5: Linear(8 → 4) → BatchNorm(4) → ReLU → Dropout(0.2)       [40 params]
  │
  ▼
Output:  Linear(4 → 2)                                              [10 params]
                                                            Total: 3,998 parameters
```

Each hidden layer applies four operations in sequence:
1. **Linear transformation** — multiplies the input by a weight matrix and adds a bias: `y = Wx + b`
2. **Batch normalization** — normalizes activations to prevent internal covariate shift, stabilizing training
3. **ReLU activation** — `max(0, x)`, introducing non-linearity so the network can learn complex decision boundaries
4. **Dropout** — randomly zeros 20% of neurons during training to prevent overfitting

The output layer produces 2 raw logits (one per class). These are not probabilities yet — `softmax` is applied later during inference.

### Weight Initialization

All linear layers use Xavier uniform initialization, which sets weights to values drawn from a uniform distribution scaled by the fan-in and fan-out of the layer:

```python
def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
```

### The Training Loop

```python
# models/fairness_dnn.py — DNNTrainer
self.criterion = nn.CrossEntropyLoss()
self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

Training uses:
- **CrossEntropyLoss** — the standard loss function for classification. It combines log-softmax and negative log-likelihood, penalizing confident wrong predictions heavily.
- **Adam optimizer** — adaptive learning rate optimizer with momentum. Learning rate 0.001.
- **Early stopping** — if validation loss doesn't improve for 10 consecutive epochs, training stops and the best model state is restored.

For **adult.csv** with 30 epochs:
- **Final training loss**: 0.3637, **accuracy**: 82.13%
- **Final validation loss**: 0.3275, **accuracy**: 84.52%
- **Test accuracy**: 85.67%

The gap between training (82.13%) and test (85.67%) accuracy — where test is higher — is due to early stopping restoring the best validation-loss model, plus dropout being disabled during evaluation.

### Why Train a DNN for Fairness Testing?

FairLint-DL doesn't just check for statistical disparities in the raw data (like traditional fairness tools). It trains a neural network because:

1. **The network learns decision boundaries** that reveal how features interact. Linear statistics miss non-linear correlations between protected attributes and outcomes.
2. **Counterfactual analysis requires a differentiable model** — we need to ask "what if we changed this person's race?" and get a prediction, which requires a trained model.
3. **Gradient-based search** for discriminatory instances requires backpropagation through a neural network.
4. **Causal debugging** traces bias through specific layers and neurons — something only possible with a deep model.

### Model Caching: Avoiding Redundant Retraining

Training a DNN on a large dataset takes time — 77 seconds for adult.csv, potentially several minutes for larger datasets. If you run the same analysis twice with the same configuration, retraining from scratch is wasteful. FairLint-DL solves this with a deterministic model caching system.

#### How the Cache Key Works

Every training run is identified by a SHA-256 hash computed from six parameters:

```python
# utils/model_cache.py
def compute_cache_key(file_path, label_column, sensitive_features, hidden_layers, num_epochs, batch_size):
    file_hash = compute_file_hash(file_path)  # SHA-256 of the CSV file contents
    sorted_features = ",".join(sorted(sensitive_features))
    layers_str = ",".join(str(x) for x in hidden_layers)

    key_string = f"{file_hash}|{label_column}|{sorted_features}|{layers_str}|{num_epochs}|{batch_size}"
    return hashlib.sha256(key_string.encode()).hexdigest()
```

The key includes:
1. **File content hash** — not the file path, but the actual contents. Renaming or moving the CSV doesn't invalidate the cache, but changing a single cell does.
2. **Label column name** — changing the target variable means a different model.
3. **Sorted sensitive features** — the order doesn't matter (`["race", "sex"]` produces the same key as `["sex", "race"]`), but the set of attributes does.
4. **Hidden layers, epochs, and batch size** — any change to the training hyperparameters invalidates the cache.

If any of these six parameters change, the key changes, and a fresh training run occurs.

#### What Gets Cached

The cache stores everything needed to reconstruct the full application state — not just the model weights:

```
.fairlint_cache/<sha256-key>/
├── model.pt              Model state_dict (PyTorch weights)
├── preprocessor.pkl      Scikit-learn scalers, label encoders, feature names
├── data_tensors.pt       Train/val/test splits as PyTorch tensors
└── metadata.json         Human-readable: config, accuracy, training history, timestamp
```

The `model.pt` file contains only the learned weight matrices and biases (via `state_dict()`). The `preprocessor.pkl` uses Python's `pickle` to serialize the scikit-learn `StandardScaler` and `LabelEncoder` objects — these are needed to correctly interpret new data in the same coordinate system as the training data. The `data_tensors.pt` file stores the exact train/val/test splits so that downstream analysis steps (QID, search, debug, explain) operate on the same data.

#### Cache Hit: Restoring from Disk

When the `/train` endpoint receives a request, it first computes the cache key and checks for existing artifacts:

```python
# bias_server.py — POST /train
cache_key = compute_cache_key(file_path, label_column, sensitive_features, hidden_layers, num_epochs, batch_size)

if not request.force_retrain and cache_exists(request.file_path, cache_key):
    cached = load_from_cache(request.file_path, cache_key)
    if cached is not None:
        return _restore_from_cache(cached, request)
```

The `_restore_from_cache()` function rebuilds the complete global state: it creates a new `FairnessDetectorDNN` instance, loads the saved weights, reconstructs `DataLoader` objects from the saved tensors, and re-initializes the `QIDAnalyzer` and `DiscriminatoryInstanceSearch` engines. The training response is returned with `cache_hit: true` and a modified message ("Model loaded from cache").

On the extension side, pipeline.ts detects the cache hit and updates the UI:

```typescript
// src/analysis/pipeline.ts
const cacheHit = trainResponse.data.cache_hit === true;
const trainMsg = cacheHit
    ? `Step 1/6: Loaded cached model (<1s) - ${accuracy}% accuracy`
    : `Step 1/6: Training complete (${stepTimings.training}s) - ${accuracy}% accuracy`;
```

The status bar shows a database icon (`$(database)`) for cache hits instead of the checkmark used for fresh training.

#### Force Retrain: Bypassing the Cache

Sometimes you want to retrain even if a cache exists — perhaps to check for training variance or after updating PyTorch. The extension registers a separate command:

```typescript
// src/extension.ts
vscode.commands.registerCommand('fairlint-dl.forceRetrain', (uri) => analyzeDataset(uri, true));
```

Right-clicking a CSV now shows two options: "Analyze This Dataset" (uses cache if available) and "Analyze Dataset (Force Retrain)" (always trains fresh). The `force_retrain: true` flag is passed through to the `/train` endpoint, which skips the cache check.

#### Cache Location and Cleanup

Caches are stored in a `.fairlint_cache/` directory next to the CSV file. Each unique configuration gets its own subdirectory named by the SHA-256 key. The `.fairlint_cache/` directory is included in `.gitignore` to prevent accidentally committing model artifacts to version control. Old cache entries can be deleted manually by removing the directory.

---

## 6. Step 2 — Internal Space Visualization

After training, the extension calls `POST /activations` to visualize what the network has learned internally.

### What Are Activations?

At each hidden layer, the network transforms the input into a new representation. Layer 1 maps each 14-dimensional input person into a 64-dimensional space. Layer 2 maps that into a 32-dimensional space. And so on. These intermediate representations are called "activations" — they are the values of neurons after applying the linear transformation, batch norm, ReLU, and dropout.

The key insight: **if two people have similar activations at a given layer, the network "sees" them as similar for the purpose of making its prediction.** If the network is unfair, you would expect to see people separated by protected attributes (like race) in these activation spaces.

### Extracting Activations

The model's `forward()` method has a `return_activations` flag:

```python
# models/fairness_dnn.py
def forward(self, x, return_activations=False):
    activations = []
    for layer, bn in zip(self.hidden_layers, self.batch_norms):
        x = layer(x)
        x = bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        if return_activations:
            activations.append(x.detach().clone())
    x = self.output_layer(x)
    if return_activations:
        return x, activations
    return x
```

For 500 test samples, this produces 5 activation matrices: one of shape (500, 64) for Layer 1, one of shape (500, 32) for Layer 2, and so on down to (500, 4) for Layer 5.

### Dimensionality Reduction with PCA

Humans can't visualize 64-dimensional spaces, so FairLint-DL reduces each layer's activations to 2D using Principal Component Analysis (PCA):

```python
# analyzers/internal_space.py
def reduce_activations(self, activations, method="pca"):
    reduced = []
    for act in activations:
        if act.shape[1] <= 2:
            if act.shape[1] == 1:
                act = np.hstack([act, np.zeros((act.shape[0], 1))])
            reduced.append(act)
        else:  # pca
            reducer = PCA(n_components=2, random_state=42)
            reduced.append(reducer.fit_transform(act))
    return reduced
```

PCA finds the two directions in the activation space along which the data varies the most. These become the x-axis and y-axis of the scatter plot.

### What the Scatter Plots Show

Each dot in the scatter plot is one person from the test set. The position is determined by their 2D PCA projection at that layer. The color indicates:
- **Blue dots**: Protected attribute value = 0.0 (e.g., one race category)
- **Red dots**: Protected attribute value = 1.0 (e.g., another race category)

If the blue and red dots are heavily mixed together at a layer, that layer does not strongly encode the protected attribute — which is desirable. If they form separate clusters, the layer has learned to distinguish people based on the protected attribute, which may indicate bias.

As you move from Layer 1 → Layer 5, you can observe how the network progressively transforms its representation. Often, early layers preserve more raw feature information (including protected attributes), while later layers abstract toward the classification task.

### t-SNE Alternative

The extension also supports t-SNE (t-distributed Stochastic Neighbor Embedding) as an alternative to PCA. t-SNE preserves local neighborhoods better than PCA — nearby points in high-dimensional space stay nearby in 2D — but it's non-linear and slower. PCA is used by default because it's deterministic, fast, and preserves global structure.

---

## 7. Step 3 — QID Analysis: Measuring Discrimination

This is the heart of FairLint-DL. QID (Quantitative Individual Discrimination) uses information theory to answer: **"How many bits of protected information does the model use when making a prediction about this person?"**

### The Counterfactual Approach

For each person in the dataset, we ask: "What would the model predict if we changed only their protected attribute value, keeping everything else the same?"

This is the **do-operator** from causal inference. We "do" an intervention on the protected attribute.

```python
# analyzers/qid_analyzer.py
def generate_counterfactuals(self, x_base, protected_values):
    counterfactuals = []
    for prot_val in protected_values:
        x_cf = x_base.clone()
        for idx in self.protected_indices:
            x_cf[idx] = float(prot_val)
        counterfactuals.append(x_cf)
    return counterfactuals
```

#### Concrete Example

Take a person from the adult.csv test set:
```
Original: [age=0.43, workclass=2, ..., race=0, sex=1, ..., native-country=0]
```
(Values are normalized, so `age=0.43` means ~39 years old after StandardScaler)

We generate two counterfactual versions by setting the protected attributes to different values:
```
Counterfactual A: [..., race=0.0, sex=0.0, ..., native-country=0.0]  (do(protected = 0.0))
Counterfactual B: [..., race=1.0, sex=1.0, ..., native-country=1.0]  (do(protected = 1.0))
```

Everything else stays identical — same education, same hours-per-week, same occupation. Only the protected attributes change.

We feed both counterfactuals through the trained model and get softmax probabilities:
```
Prediction A: [P(<=50K) = 0.82, P(>50K) = 0.18]   (protected = 0.0)
Prediction B: [P(<=50K) = 0.35, P(>50K) = 0.65]   (protected = 1.0)
```

The predictions differ dramatically — just changing the protected attribute flipped the likely outcome from <=50K to >50K. This is individual discrimination.

### Computing Shannon QID (in Bits)

To quantify this difference, we use Shannon entropy:

```python
# analyzers/qid_analyzer.py
def compute_shannon_qid(self, x_base, protected_values):
    counterfactuals = self.generate_counterfactuals(x_base, protected_values)

    predictions = []
    with torch.no_grad():
        for x_cf in counterfactuals:
            x_cf = x_cf.to(self.device).unsqueeze(0)
            probs = self._safe_get_probs(x_cf)
            predictions.append(probs.cpu().numpy())

    predictions = np.array(predictions)

    # Average prediction across counterfactuals
    avg_pred = predictions.mean(axis=0)

    # Shannon entropy of the average prediction
    shannon_entropy = entropy(avg_pred)  # from scipy.stats

    # Convert nats to bits
    qid_bits = shannon_entropy / np.log(2)

    return {
        "qid_bits": qid_bits,
        "has_discrimination": qid_bits > 0.1,
        ...
    }
```

The math step by step:

1. **Get predictions for each counterfactual**: `P_A = [0.82, 0.18]` and `P_B = [0.35, 0.65]`

2. **Average the predictions**: `avg = [(0.82+0.35)/2, (0.18+0.65)/2] = [0.585, 0.415]`

3. **Compute Shannon entropy**: `H(avg) = -(0.585 * ln(0.585) + 0.415 * ln(0.415)) = 0.6764 nats`

4. **Convert to bits**: `QID = 0.6764 / ln(2) = 0.976 bits`

#### Interpreting QID Values

- **QID = 0 bits**: The model gave identical predictions for both counterfactuals. No protected information was used. Perfect fairness for this individual.
- **QID = 0.1 bits**: Minimal difference. Below the significance threshold — considered non-discriminatory.
- **QID = 0.5 bits**: Moderate use of protected information. The model's prediction shifts noticeably when protected attributes change.
- **QID = 1.0 bit**: Maximum for binary classification. The model's average prediction is perfectly uncertain (50/50), meaning the protected attribute completely determines the outcome. This represents maximum individual discrimination.

The threshold for flagging discrimination is **QID > 0.1 bits**.

### Min-Entropy QID and Disparate Impact

FairLint-DL also computes a worst-case fairness metric:

```python
# analyzers/qid_analyzer.py
def compute_min_entropy_qid(self, x_base, protected_values):
    # ...get counterfactual predictions...

    # Disparate impact ratio: min/max favorable probability
    favorable_probs = np.array(favorable_outcomes)
    disparate_impact = favorable_probs.min() / favorable_probs.max()

    # Legal threshold: 0.8 (80% rule / four-fifths rule)
    violates_80_rule = disparate_impact < 0.8
```

The **disparate impact ratio** is:

```
DI = P(favorable | protected = minority) / P(favorable | protected = majority)
```

This is a well-established legal standard. The **four-fifths rule** (also called the 80% rule) states: if the selection rate for a protected group is less than 80% of the selection rate for the majority group, there is evidence of adverse impact.

For **adult.csv**: The mean disparate impact ratio across 500 analyzed instances was **0.571** — well below the 0.8 threshold, with **356 out of 500** instances (71.2%) violating the four-fifths rule.

### Batch Analysis

The batch analyzer iterates over test instances and aggregates:

```python
# analyzers/qid_analyzer.py
def batch_analyze(self, X, protected_values, max_samples=1000):
    for i in range(n_samples):
        shannon_result = self.compute_shannon_qid(X[i], protected_values)
        shannon_qids.append(shannon_result["qid_bits"])
        if shannon_result["has_discrimination"]:
            discriminatory_count += 1

        min_result = self.compute_min_entropy_qid(X[i], protected_values)
        disparate_impacts.append(min_result["disparate_impact_ratio"])

    return {
        "mean_qid": float(np.mean(shannon_qids)),
        "max_qid": float(np.max(shannon_qids)),
        "num_discriminatory": int(discriminatory_count),
        "pct_discriminatory": float(100 * discriminatory_count / n_samples),
        "mean_disparate_impact": float(np.mean(disparate_impacts)),
        ...
    }
```

For **adult.csv** (500 instances analyzed):
- **Mean QID**: 0.6632 bits — significant protected information usage
- **Max QID**: 1.0000 bits — some individuals face maximum possible discrimination
- **Standard deviation**: 0.3160 bits — wide variation across individuals
- **Discriminatory instances**: 473 out of 500 (94.6%)
- **Mean disparate impact**: 0.571 — well below the 0.8 legal threshold
- **Instances violating 80% rule**: 356 (71.2%)

### The QID Distribution Histogram

The dashboard displays a histogram of QID values across all analyzed instances. For adult.csv, it shows a bimodal distribution: a small cluster near 0 (non-discriminatory instances) and a large cluster above 0.5 (discriminatory instances). This visual immediately communicates that discrimination is widespread, not limited to outliers.

### Group Fairness Metrics: Demographic Parity, Equalized Odds, and Equal Opportunity

QID measures **individual-level** discrimination — how a single person's prediction changes when their protected attributes are altered. But regulators and researchers also care about **group-level** fairness: does the model treat entire demographic groups equitably in aggregate?

FairLint-DL computes three standard group fairness metrics alongside QID, both calculated during the same `/analyze` step.

#### How Groups Are Defined

Since all features are standardized (zero mean, unit variance) by the `StandardScaler`, the group split uses the standardized value:

```python
# analyzers/group_fairness.py
attr_vals = X_test[:, attr_idx].cpu()
mask_a = attr_vals <= 0  # Group A: at or below the mean
mask_b = attr_vals > 0   # Group B: above the mean
```

For a binary attribute like `sex` (2 unique values), this splits into the two categories directly. For continuous attributes like `age`, Group A contains people with below-average age and Group B contains those with above-average age.

#### Metric 1: Demographic Parity

Demographic Parity asks: **"Does the model give positive predictions at the same rate for both groups?"**

```
Difference = |P(Ŷ=1 | Group A) - P(Ŷ=1 | Group B)|
Ratio = min(P(Ŷ=1 | Group A), P(Ŷ=1 | Group B)) / max(...)
```

A difference close to 0 means both groups receive favorable outcomes at equal rates. The ratio connects to the 80% rule — if it falls below 0.8, there is evidence of adverse impact.

```python
# analyzers/group_fairness.py
pos_rate_a = (y_pred_a == 1).float().mean().item()
pos_rate_b = (y_pred_b == 1).float().mean().item()
dp_diff = abs(pos_rate_a - pos_rate_b)
dp_ratio = min(pos_rate_a, pos_rate_b) / max(pos_rate_a, pos_rate_b)
```

#### Metric 2: Equalized Odds

Equalized Odds asks: **"Does the model make errors at the same rate for both groups?"**

It compares two rates across groups:
- **True Positive Rate (TPR)**: Among people who actually qualify (Y=1), what fraction does the model correctly identify? `TPR = TP / (TP + FN)`
- **False Positive Rate (FPR)**: Among people who don't qualify (Y=0), what fraction does the model incorrectly flag? `FPR = FP / (FP + TN)`

```python
tpr_a = cm_a["tp"] / (cm_a["tp"] + cm_a["fn"])  # if denominator > 0
fpr_a = cm_a["fp"] / (cm_a["fp"] + cm_a["tn"])
# Equalized Odds violation = max(|TPR_A - TPR_B|, |FPR_A - FPR_B|)
eo_max_diff = max(abs(tpr_a - tpr_b), abs(fpr_a - fpr_b))
```

If `eo_max_diff` is small, the model's errors are distributed equally across groups — it doesn't systematically harm one group more than another.

#### Metric 3: Equal Opportunity

Equal Opportunity is a relaxation of Equalized Odds that focuses only on the TPR: **"Among people who truly deserve the favorable outcome, does the model identify them at equal rates regardless of group?"**

```
Equal Opportunity Difference = |TPR_A - TPR_B|
```

This metric is especially important in contexts like hiring or lending, where missing qualified candidates from one group constitutes harm even if overall error rates are balanced.

#### Per-Group Confusion Matrices

All three metrics are derived from per-group binary confusion matrices:

```python
# analyzers/group_fairness.py
def _confusion_matrix(y_true, y_pred):
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    tn = ((y_pred == 0) & (y_true == 0)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
```

The model's predictions are computed once (shared across all attributes), then each protected attribute's group masks select the relevant subsets.

#### Computing Across All Protected Attributes

The `GroupFairnessAnalyzer` iterates over every protected attribute and returns a list of results:

```python
# analyzers/group_fairness.py
class GroupFairnessAnalyzer:
    def compute_all(self, X_test, y_test):
        with torch.no_grad():
            logits = self.model(X_test.to(self.device))
            y_pred = logits.argmax(dim=1).cpu()

        results = []
        for attr_name, attr_idx in zip(self.protected_feature_names, self.protected_indices):
            attr_vals = X_test[:, attr_idx].cpu()
            mask_a = attr_vals <= 0
            mask_b = attr_vals > 0
            metrics = self._compute_for_attribute(y_true, y_pred, mask_a, mask_b, attr_name, attr_idx)
            results.append(metrics)
        return results
```

#### Integration into the Pipeline

Group fairness metrics are computed in the `/analyze` endpoint alongside QID, reusing the same test data:

```python
# bias_server.py — POST /analyze
batch_results = current_analyzer.batch_analyze(X_test, protected_vals, max_samples=...)

gf_analyzer = GroupFairnessAnalyzer(current_model, protected_indices=..., protected_feature_names=..., device="cpu")
group_fairness = gf_analyzer.compute_all(X_test, y_test)

return {"status": "success", "qid_metrics": batch_results, "group_fairness": group_fairness}
```

The pipeline.ts progress notification reflects the combined output:

```typescript
const numGroupMetrics = analyzeResponse.data.group_fairness?.length || 0;
progress.report({
    increment: 17,
    message: `Step 3/6: Fairness metrics computed - Mean QID: ${meanQid} bits | ${numGroupMetrics} group metrics`,
});
```

#### Dashboard Visualization

The dashboard renders a dedicated "Group Fairness Metrics" section with:

1. **Attribute selector dropdown** — when multiple protected attributes are analyzed, a dropdown lets you switch between them (e.g., age, race, sex, native-country).

2. **Three metric cards** — each showing the metric value, a difference or ratio, and a colored badge:
   - Green "Fair" badge: difference < 0.1
   - Yellow "Marginal" badge: difference between 0.1 and 0.2
   - Red "Unfair" badge: difference >= 0.2

3. **Per-group comparison chart** — a grouped bar chart showing positive rate, TPR, and FPR side by side for Group A and Group B. This immediately reveals which group is disadvantaged and on which dimension.

4. **Dynamic interpretive text** — the dashboard generates context-aware explanations. For example, if demographic parity difference is 0.25 and the ratio is 0.6, the text states: "Demographic Parity is significantly violated (difference: 0.250). There is a substantial gap... The ratio of 0.600 violates the 80% rule."

#### Impact on the Fairness Score

Group fairness metrics also feed into the composite fairness score:

```typescript
// src/webview/scoring.ts
if (groupFairness && groupFairness.length > 0) {
    const avgDpDiff = groupFairness.reduce((sum, gf) => sum + gf.demographic_parity.difference, 0) / groupFairness.length;
    const avgEoDiff = groupFairness.reduce((sum, gf) => sum + gf.equalized_odds.max_difference, 0) / groupFairness.length;
    // Combined penalty: each 0.1 difference costs ~2.5 points, capped at 20
    score -= Math.min((avgDpDiff + avgEoDiff) * 25, 20);
}
```

The average demographic parity difference and equalized odds difference across all protected attributes are combined and penalized up to 20 points. This ensures that models with severe group-level disparities receive lower overall scores even if individual-level QID is moderate.

---

## 8. Step 4 — Discriminatory Instance Search

While QID analysis measures discrimination across existing test data, the search step actively hunts for the most discriminatory inputs the model can produce. This is important because the test set is a sample — the actual input space may contain regions with even worse discrimination that weren't sampled.

### Two-Phase Search Strategy

The search uses a global-local approach inspired by the DICE/NeuFair papers:

#### Phase 1: Global Search (Gradient Ascent)

Start from a random test instance and optimize it to maximize discrimination:

```python
# analyzers/search.py
def global_search(self, x_init, protected_values, num_iterations=100, lr=0.01):
    x_current = x_init.clone().detach().to(self.device)
    x_current.requires_grad = True
    optimizer = torch.optim.Adam([x_current], lr=lr)

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Generate counterfactuals and get predictions
        counterfactual_outputs = []
        for prot_val in protected_values:
            x_cf = x_current.clone()
            for idx in self.protected_indices:
                x_cf[idx] = float(prot_val)
            probs = self._safe_get_probs(x_cf.unsqueeze(0))
            counterfactual_outputs.append(probs)

        outputs_tensor = torch.stack(counterfactual_outputs)

        # Maximize variance of predictions across counterfactuals
        qid_loss = -outputs_tensor.var(dim=0).sum()
        qid_loss.backward()
        optimizer.step()
```

The key idea: we treat the *input features themselves* as learnable parameters and use gradient ascent to find the input that produces the biggest prediction difference between counterfactuals. The loss function is the negative variance of predictions — minimizing this negative variance maximizes the actual variance, which maximizes discrimination.

The Adam optimizer adjusts non-protected features to find the "sweet spot" where the model is most sensitive to changes in protected attributes.

For **adult.csv**: The global search found an instance with QID = 0.054. This relatively low QID from gradient search (compared to the 0.663 average from the test set) suggests the model's discrimination is distributed broadly rather than concentrated in extreme outliers.

#### Phase 2: Local Search (Neighborhood Perturbation)

Starting from the global search's best instance, generate neighbors and keep discriminatory ones:

```python
# analyzers/search.py
def local_search(self, x_base, protected_values, num_neighbors=50, perturbation_scale=0.1):
    discriminatory_instances = []

    for _ in range(num_neighbors):
        # Add Gaussian noise to non-protected features
        noise = torch.randn_like(x_base) * perturbation_scale
        for idx in self.protected_indices:
            noise[idx] = 0  # Don't perturb protected features

        x_neighbor = x_base + noise

        # Keep only if QID > 0.1 threshold
        qid_result = self.qid_analyzer.compute_shannon_qid(x_neighbor, protected_values)
        if qid_result["qid_bits"] > 0.1:
            discriminatory_instances.append({
                "instance": x_neighbor.cpu().numpy().tolist(),
                "qid": qid_result["qid_bits"],
                "predictions": qid_result["counterfactual_predictions"],
            })

    # Sort by QID — most discriminatory first
    discriminatory_instances.sort(key=lambda x: x["qid"], reverse=True)
    return discriminatory_instances
```

The local search adds small Gaussian noise (standard deviation 0.1) to non-protected features only. This explores the neighborhood around the globally-found instance. Of the `num_neighbors` random perturbations, only those with QID > 0.1 bits are kept.

For **adult.csv**: The local search generated **30 discriminatory instances** from 30 neighbor samples. These instances are used in the next step (causal debugging) to identify which layers and neurons are responsible for the bias.

---

## 9. Step 5 — Causal Debugging

Causal debugging answers: **"Which layer of the network is most responsible for the bias, and which specific neurons within that layer encode protected information?"**

### Layer Localization

The debugger performs gradient-based sensitivity analysis to find which layer reacts most strongly to discriminatory inputs:

```python
# analyzers/causal_debugger.py
class CausalDebugger:
    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.model.eval()
        # Build layers list dynamically from model
        self.layers = list(self.model.hidden_layers) + [self.model.output_layer]

    def localize_biased_layer(self, discriminatory_instances, accuracy_threshold=0.05):
        num_layers = len(self.layers)
        layer_sensitivities = []

        for layer_idx in range(num_layers):
            total_sensitivity = 0

            for instance_data in discriminatory_instances[:20]:
                x = torch.tensor(instance_data["instance"], dtype=torch.float32)
                x.requires_grad_(True)

                # Get activations at this layer
                activations = self.model.get_layer_output(x.unsqueeze(0), layer_idx)

                # Compute gradient of activations w.r.t. input
                grad_outputs = torch.ones_like(activations)
                gradients = torch.autograd.grad(
                    outputs=activations, inputs=x, grad_outputs=grad_outputs
                )[0]

                # Sensitivity = mean absolute gradient
                sensitivity = gradients.abs().mean().item()
                total_sensitivity += sensitivity

            avg_sensitivity = total_sensitivity / min(20, len(discriminatory_instances))
            layer_sensitivities.append({
                "layer_idx": layer_idx,
                "sensitivity": avg_sensitivity,
                "neuron_count": self.layers[layer_idx].out_features,
            })

        most_biased_layer = max(layer_sensitivities, key=lambda x: x["sensitivity"])
        return {"biased_layer": most_biased_layer, "all_layers": layer_sensitivities}
```

The algorithm for each layer:
1. Take the top 20 discriminatory instances found during search
2. Enable gradient tracking on the input
3. Forward-pass through the network up to that layer using `get_layer_output()`
4. Compute gradients of the layer's output with respect to the input using `torch.autograd.grad()`
5. The **sensitivity** is the mean absolute gradient — it measures how strongly the layer's activations change when the input changes

A high sensitivity means the layer's representation is heavily influenced by input variations, which for discriminatory instances means it's encoding the discrimination signal.

For **adult.csv**, the layer sensitivities were:

| Layer | Neurons | Sensitivity | Interpretation |
|-------|---------|-------------|----------------|
| Layer 1 (14→64) | 64 | **2.070** | Most biased |
| Layer 2 (64→32) | 32 | 1.467 | |
| Layer 3 (32→16) | 16 | 0.778 | |
| Layer 4 (16→8) | 8 | 0.254 | |
| Layer 5 (8→4) | 4 | 0.588 | |
| Layer 6 (4→2) | 2 | 0.436 | Output layer |

**Layer 1 is the most biased** (sensitivity 2.070). This is an early layer, which aligns with research showing that early layers in DNNs tend to learn raw feature representations that preserve protected attribute information. The sensitivity decreases through the middle layers, with a slight uptick at the output — suggesting the bias from Layer 1 propagates through to the final prediction.

### Neuron Localization

Once the most biased layer is identified, we pinpoint which neurons within it encode the most protected information:

```python
# analyzers/causal_debugger.py
def localize_biased_neurons(self, layer_idx, discriminatory_instances, top_k=5):
    layer = self.layers[layer_idx]
    num_neurons = layer.out_features
    neuron_impacts = np.zeros(num_neurons)

    for instance_data in discriminatory_instances[:30]:
        x = torch.tensor(instance_data["instance"], dtype=torch.float32)

        with torch.no_grad():
            activations = self.model.get_layer_output(x.unsqueeze(0), layer_idx)
            activations = activations[0]

            # Each neuron's activation magnitude = its impact
            neuron_impacts += activations.abs().cpu().numpy()

    # Average across instances
    neuron_impacts /= min(30, len(discriminatory_instances))

    # Top-k neurons by impact
    top_neuron_indices = np.argsort(neuron_impacts)[-top_k:][::-1]
    return [{"neuron_idx": int(idx), "impact_score": float(neuron_impacts[idx])} ...]
```

The approach is simpler than layer localization: for each discriminatory instance, we look at how strongly each neuron activates (absolute value of the activation). Neurons that consistently have high activations when processing discriminatory instances are the ones encoding protected information.

For **adult.csv** (Layer 1, 64 neurons):

| Neuron | Impact Score |
|--------|-------------|
| Neuron 60 | **2.097** |
| Neuron 43 | 2.028 |
| Neuron 27 | 1.444 |
| Neuron 50 | 1.320 |
| Neuron 59 | 1.298 |

Neurons 60 and 43 have disproportionately high impact scores (~2.0) compared to the average neuron, suggesting they have specialized in encoding protected attribute information. In a fairness remediation workflow, these neurons could be candidates for pruning, retraining, or regularization.

---

## 10. Step 6 — SHAP and LIME Explanations

The final step answers: **"Which input features most influence the model's predictions, and do protected attributes rank among the most influential?"**

### Why Two Methods?

SHAP and LIME approach explainability differently:

- **SHAP** (global perspective) computes the exact contribution of each feature to every prediction using game-theoretic Shapley values. It tells you the overall importance ranking.
- **LIME** (local perspective) approximates the model locally around each instance with a simple linear model. It tells you which features matter for specific predictions.

If both methods agree that a protected attribute is important, that's strong evidence of bias. If they disagree, it may indicate that the bias is non-linear or context-dependent.

### SHAP: SHapley Additive exPlanations

#### Log-Odds Output Space

A critical design decision: FairLint-DL computes SHAP values on the **log-odds (margin)** output, not on softmax probabilities:

```python
# analyzers/explainability.py
def _predict_margin(self, X):
    self.model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        logits = self.model(x_tensor)
        result = logits.cpu().numpy()
        # For binary classification: logit[1] - logit[0] = log-odds
        if result.shape[1] == 2:
            margin = result[:, 1] - result[:, 0]
        else:
            margin = result[:, 1]
        margin = np.nan_to_num(margin, nan=0.0, posinf=0.0, neginf=0.0)
        return margin
```

The margin (log-odds) is defined as:

```
margin = logit(class=1) - logit(class=0) = log(P(>50K) / P(<=50K))
```

This is mathematically important because:
1. **Softmax compresses differences.** If logits are [3.0, 5.0], softmax gives [0.12, 0.88]. But if logits are [3.0, 3.5], softmax gives [0.38, 0.62]. The logit difference halved (2.0 → 0.5) but the probability difference barely changed (0.76 → 0.24). SHAP values on probabilities would understate the true feature contributions.
2. **The log-odds space is linear.** In logistic regression, the log-odds is literally a linear function of the features. Even in a DNN, the log-odds space preserves the scale of feature effects, making SHAP values interpretable as "this feature shifts the log-odds by X."

#### KernelExplainer

```python
# analyzers/explainability.py
def compute_shap_values(self, X_background, X_explain, max_background=100):
    bg = X_background[:max_background]
    bg_size = min(50, len(bg))
    if len(bg) > bg_size:
        indices = np.random.choice(len(bg), bg_size, replace=False)
        bg = bg[indices]

    explainer = shap.KernelExplainer(self._predict_margin, bg)

    shap_values = explainer.shap_values(
        X_explain,
        nsamples="auto",
        l1_reg="num_features(10)",
    )
```

KernelExplainer is model-agnostic — it treats the model as a black box and estimates Shapley values by evaluating the model on subsets of features (replacing missing features with background data). The `nsamples="auto"` flag lets SHAP determine the optimal number of evaluations, and `l1_reg="num_features(10)"` applies L1 regularization to focus on the top 10 features.

The background dataset (50 randomly subsampled training instances) represents the "baseline" — what the model predicts when it has no information about a feature.

#### SHAP Results for adult.csv

Global feature importance (mean |SHAP value| across 10 explained instances):

| Feature | SHAP Importance (log-odds) |
|---------|---------------------------|
| **education-num** | **0.997** |
| **marital-status** | **0.830** |
| **age** | **0.647** |
| hours-per-week | 0.461 |
| capital-loss | 0.258 |
| capital-gain | 0.244 |
| relationship | 0.168 |
| **sex** | **0.151** |
| occupation | 0.088 |
| **race** | **0.076** |
| native-country | 0.000 |
| education | 0.036 |
| workclass | 0.025 |
| fnlwgt | 0.024 |

Key observations:
- **education-num** is the most influential feature (0.997 log-odds shift), followed by **marital-status** (0.830). These are legitimate predictors of income.
- Among protected attributes: **age** ranks 3rd (0.647) — the model heavily uses age information. **sex** ranks 8th (0.151) and **race** ranks 10th (0.076). These contribute measurably to predictions.
- **native-country** has zero SHAP importance, suggesting the model ignores it despite it being a protected attribute.

### The SHAP Charts

FairLint-DL generates four SHAP visualizations:

1. **Global Feature Importance Bar Chart** — Horizontal bars showing mean |SHAP value| per feature, sorted by importance. Immediately shows which features drive predictions.

2. **Beeswarm Plot** — Each dot is one instance. The x-axis is the SHAP value (how much that feature shifted the prediction for that instance). The color indicates the feature's actual value (red = high, blue = low). This shows both importance and direction: e.g., high education-num (red dots on the right) pushes predictions toward >50K.

3. **Scatter Plot** — For the most important feature, plots feature value (x-axis) vs. SHAP value (y-axis), colored by the second most important feature. Shows non-linear relationships.

4. **Heatmap** — Instances (rows) x Features (columns), colored by SHAP value (red = positive, blue = negative). Gives a complete picture of all explanations at once.

### LIME: Local Interpretable Model-agnostic Explanations

LIME works differently from SHAP. For each instance:
1. Generate perturbed versions by randomly zeroing out features
2. Get model predictions for all perturbations
3. Fit a weighted linear regression (closer perturbations get higher weight)
4. The regression coefficients are the feature importances for that instance

```python
# analyzers/explainability.py
def compute_lime_explanations(self, X_train, X_explain, num_features=10, num_samples=500):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=self.feature_names,
        class_names=["Unfavorable", "Favorable"],
        mode="classification",
    )

    for i in range(len(X_explain)):
        exp = explainer.explain_instance(
            X_explain[i],
            self._predict_proba,  # LIME uses probabilities, not log-odds
            num_features=min(num_features, len(self.feature_names)),
            num_samples=num_samples,
        )
```

LIME uses `_predict_proba` (softmax probabilities) instead of log-odds because it approximates the model locally with a linear model, where probabilities provide a natural [0, 1] output range.

#### LIME Results for adult.csv

Aggregated feature importance (averaged absolute LIME weights across 10 instances):

| Feature | LIME Importance |
|---------|----------------|
| **education-num** | **0.211** |
| **capital-gain** | **0.203** |
| capital-loss | 0.148 |
| hours-per-week | 0.115 |
| marital-status | 0.086 |
| **age** | **0.083** |
| native-country | 0.048 |
| **sex** | **0.024** |
| workclass | 0.023 |
| occupation | 0.016 |
| education | 0.013 |
| relationship | 0.012 |
| fnlwgt | 0.012 |
| **race** | **0.000** |

LIME and SHAP agree on the top feature (education-num) but differ on others: LIME ranks capital-gain 2nd while SHAP ranks it 6th. This is expected — SHAP computes global importance while LIME explains local neighborhoods, so they capture different aspects of the model's behavior.

### Interactive LIME: Per-Instance User Exploration

The aggregated LIME results show average feature importance across 10 instances, but fairness analysis often requires examining specific individuals. FairLint-DL includes an interactive LIME explorer that lets users generate explanations for any instance on demand.

#### Two Input Modes

The dashboard provides a toggle between two modes:

1. **Test Instance mode** — Enter a numeric index (0 to N-1, where N is the test set size) to explain an existing test set instance. This is useful for investigating specific data points — for example, if QID analysis flagged instance #247 as highly discriminatory, you can generate its LIME explanation to see which features drive that prediction.

2. **Custom Values mode** — Enter feature values manually in a grid (one input field per feature). This supports "what-if" scenarios: you can construct a hypothetical person and see how the model would explain its prediction for them. All values are in the standardized coordinate system (zero mean, unit variance).

#### How It Works Under the Hood

When the user clicks "Generate LIME Explanation," the webview sends a message to the extension host:

```javascript
// htmlBuilder.ts — headScript
function generateLimeInstance() {
    if (_limeMode === 'index') {
        var idx = parseInt(document.getElementById('lime-instance-index').value, 10);
        data = { instanceType: 'index', instanceIndex: idx };
    } else {
        var featureValues = [];
        for (var i = 0; i < _limeFeatureNames.length; i++) {
            featureValues.push(parseFloat(document.getElementById('lime-feature-' + i).value));
        }
        data = { instanceType: 'custom', featureValues: featureValues };
    }
    _vscodeApi.postMessage({ command: 'explainInstance', data: data });
}
```

The extension host catches this in `results.ts` and forwards it to the backend:

```typescript
// src/webview/results.ts
async function handleExplainInstance(panel, data) {
    const response = await axios.post(`${serverUrl}/explain-instance`, {
        instance_type: data.instanceType,
        instance_index: data.instanceIndex,
        feature_values: data.featureValues,
    }, { timeout: 30000 });

    panel.webview.postMessage({ command: 'limeInstanceResult', data: response.data });
}
```

The backend's `/explain-instance` endpoint creates an `ExplainabilityAnalyzer`, retrieves the requested instance (from the test set or from custom values), and computes a single-instance LIME explanation:

```python
# bias_server.py — POST /explain-instance
if request.instance_type == "index":
    instance = X_test[request.instance_index:request.instance_index + 1]
elif request.instance_type == "custom":
    instance = np.array([request.feature_values], dtype=np.float32)

analyzer = ExplainabilityAnalyzer(current_model, feature_names, device="cpu")
result = analyzer.compute_lime_explanations(X_train, instance)
```

#### What the User Sees

The result appears inline in the dashboard:

1. **Prediction probabilities** — "Unfavorable (Class 0): 82.3%" / "Favorable (Class 1): 17.7%", with the dominant class highlighted.
2. **Per-instance LIME bar chart** — a horizontal bar chart showing the top 15 feature conditions that influence this specific prediction. Green bars push toward "Favorable," red bars push toward "Unfavorable." Each bar is labeled with the feature condition (e.g., "education-num > 0.38") and its LIME weight.

A loading spinner appears while the backend computes (LIME generates 500 perturbations per instance), and error messages display inline if the index is out of range or feature values are invalid.

This interactive element transforms LIME from a batch summary tool into an ad-hoc exploration tool — practitioners can test specific scenarios and immediately see which features the model relies on for that individual.

### Cross-Method Validation

The dashboard automatically cross-references SHAP and LIME results:
- Both agree that **education-num** is the most important feature
- Both identify **age** as a significant protected attribute
- Both show **sex** has some influence, while **race** has minimal measured impact
- The agreement on education-num and age as top features provides high confidence in these findings

---

## 11. The Fairness Score

The dashboard displays a single composite fairness score from 0 to 100:

```typescript
// src/webview/scoring.ts
export function calculateFairnessScore(qidMetrics, groupFairness?): number {
    let score = 100;

    // Penalize for high mean QID (0-2 bits range maps to 0-30 penalty)
    score -= Math.min(qidMetrics.mean_qid * 15, 30);

    // Penalize for discriminatory instances (0-100% maps to 0-30 penalty)
    score -= Math.min(qidMetrics.pct_discriminatory * 0.3, 30);

    // Penalize for low disparate impact (0.8-1.0 is good, below 0.8 is bad)
    if (qidMetrics.mean_disparate_impact < 0.8) {
        score -= (0.8 - qidMetrics.mean_disparate_impact) * 50;
    }

    // Penalize for high max QID
    score -= Math.min(qidMetrics.max_qid * 5, 20);

    // Group fairness penalties (average across all protected attributes)
    if (groupFairness && groupFairness.length > 0) {
        const avgDpDiff = groupFairness.reduce((sum, gf) => sum + gf.demographic_parity.difference, 0) / groupFairness.length;
        const avgEoDiff = groupFairness.reduce((sum, gf) => sum + gf.equalized_odds.max_difference, 0) / groupFairness.length;
        score -= Math.min((avgDpDiff + avgEoDiff) * 25, 20);
    }

    return Math.max(0, Math.round(score));
}
```

The score has five penalty components:

| Component | Max Penalty | adult.csv Value | Penalty Applied |
|-----------|-------------|-----------------|-----------------|
| Mean QID (0.6632 bits) | 30 | 0.6632 * 15 = 9.95 | -10 |
| % Discriminatory (94.6%) | 30 | 94.6 * 0.3 = 28.38 | -28 |
| Disparate Impact (0.571) | ∞ | (0.8 - 0.571) * 50 = 11.45 | -11 |
| Max QID (1.0000) | 20 | 1.0 * 5 = 5 | -5 |
| Group Fairness (DP + EO) | 20 | (avgDpDiff + avgEoDiff) * 25 | ~-5 |
| **Total** | | | **100 - 59 = 41** |

The group fairness penalty averages the demographic parity difference and the equalized odds max difference across all protected attributes, then multiplies by 25 (so each 0.1 average difference costs ~2.5 points). This is capped at 20 points to prevent group metrics from dominating the score.

For adult.csv: **Fairness Score = ~41 out of 100** — rated "Concerning" (red).

The status thresholds are:
- **80-100**: Good (green) — Low bias detected
- **60-79**: Needs Review (yellow) — Moderate bias concerns
- **0-59**: Concerning (red) — Significant bias detected

### Score Displayed as a Gauge

The dashboard renders this as a Plotly.js gauge chart — a semicircular meter with the needle pointing to the score. The gauge transitions from red (left) through yellow (center) to green (right), giving an immediate visual assessment.

---

## 12. The Results Dashboard

The HTML dashboard is assembled by `htmlBuilder.ts` with these sections:

### Pipeline Overview
- Dataset summary: file name, total instances, feature count, label column
- Protected attributes with explanations: what 0.0 and 1.0 mean for each attribute, and how many unique values each has
- A 6-step pipeline walkthrough showing what each step did, with timing information

### Fairness Score Section
- Large gauge chart with the composite score
- Status badge (Good / Needs Review / Concerning)

### QID Analysis Section
- Mean/Max/Std QID values
- Number and percentage of discriminatory instances
- Disparate impact ratio
- QID distribution histogram
- Dynamic interpretation text that changes based on actual values

### Group Fairness Metrics Section
- Attribute selector dropdown (when multiple protected attributes)
- Metric cards for Demographic Parity, Equalized Odds, and Equal Opportunity (with Fair/Marginal/Unfair badges)
- Per-group rate comparison chart (positive rate, TPR, FPR for Group A vs Group B)
- Per-group confusion matrix display
- Dynamic interpretive text for each metric based on actual values

### Discriminatory Instance Search Section
- Number of instances found
- Best QID from search
- Dynamic interpretation of what the search results mean

### Causal Debugging Section
- Layer sensitivity bar chart (all layers)
- Most biased layer identification with interpretation
- Neuron impact bar chart (top 5 neurons)
- Text explaining whether bias is in early, middle, or late layers

### Internal Space Visualization
- Per-layer PCA scatter plots
- Points colored by protected attribute value
- Visual assessment of how the network separates populations

### SHAP Explanations
- Global feature importance bar chart
- Beeswarm plot
- Scatter plot (top feature)
- Heatmap
- Dynamic text noting if protected attributes are in the top-3

### LIME Explanations
- Aggregated feature importance bar chart
- Cross-reference with SHAP findings
- **Interactive instance explorer**: toggle between test set index and custom feature values, generate on-demand LIME explanations with prediction probabilities and per-instance bar charts

### Export Button
- "Export JSON" button that opens a save dialog
- Exports all numerical results as a structured JSON file

All charts are rendered using Plotly.js loaded from CDN, embedded in the webview HTML.

---

## 13. Configuration Reference

All settings are configurable via VS Code's settings (`Ctrl+,` → search "fairlint-dl"):

| Setting | Default | Description |
|---------|---------|-------------|
| `training.epochs` | 30 | Number of training epochs |
| `training.batchSize` | 32 | Training batch size |
| `training.hiddenLayers` | "64,32,16,8,4" | DNN hidden layer sizes |
| `analysis.qidThreshold` | 0.1 | QID threshold (bits) for flagging discrimination |
| `analysis.maxSamples` | 500 | Max instances for QID analysis |
| `search.globalIterations` | 50 | Global search iterations |
| `search.localNeighbors` | 30 | Local search neighbor count |
| `server.port` | 8765 | Backend server port |
| `detection.autoDetectProtected` | true | Auto-detect protected attributes |
| `visualization.showCharts` | true | Show Plotly charts in dashboard |
| `visualization.theme` | "dark" | Chart color theme |
| `notifications.showProgress` | true | Show progress notifications |
| `notifications.showStatusBar` | true | Show status bar updates |

---

## 14. Example: adult.csv Analysis Results

This section shows the complete results from analyzing the UCI Adult (Census Income) dataset — a standard benchmark for fairness research. The dataset predicts whether a person's income exceeds $50K/year based on census features.

### Dataset Profile

| Metric | Value |
|--------|-------|
| File | adult.csv |
| Total instances | 32,561 |
| Features | 14 (after removing label) |
| Label column | income |
| Class 0 (<=50K) | 17,303 (75.9%) |
| Class 1 (>50K) | 5,489 (24.1%) |
| Training set | 22,792 |
| Validation set | 3,256 |
| Test set | 6,513 |

### Protected Attributes

| Attribute | Feature Index | Unique Values | Meaning of 0.0 / 1.0 |
|-----------|---------------|---------------|----------------------|
| age | 0 | 71 | Continuous (standardized). 0.0 = mean age (~38), negative = younger, positive = older |
| race | 8 | 5 | Label-encoded categories. Each integer represents a different race category |
| sex | 9 | 2 | Binary. 0.0 = Female, 1.0 = Male (or vice versa depending on LabelEncoder ordering) |
| native-country | 13 | 42 | Label-encoded. Each integer represents a different country |

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | [64, 32, 16, 8, 4] → 2 |
| Total parameters | 3,998 |
| Epochs | 30 |
| Optimizer | Adam (lr=0.001) |
| Loss | CrossEntropyLoss |
| Dropout | 0.2 |
| Batch size | 128 |

### Training Performance

| Metric | Value |
|--------|-------|
| Final training loss | 0.3637 |
| Final training accuracy | 82.13% |
| Final validation loss | 0.3275 |
| Final validation accuracy | 84.52% |
| Test accuracy | 85.67% |

### QID Analysis (500 instances)

| Metric | Value |
|--------|-------|
| Mean QID | 0.6632 bits |
| Max QID | 1.0000 bits |
| Std QID | 0.3160 bits |
| Discriminatory instances | 473 / 500 (94.6%) |
| Mean disparate impact | 0.571 |
| Violating 80% rule | 356 / 500 (71.2%) |

### Group Fairness Metrics

Computed for all 4 protected attributes. Example for `sex`:

| Metric | Group A (sex <= 0) | Group B (sex > 0) | Difference |
|--------|-------------------|-------------------|------------|
| Positive Rate (Demographic Parity) | varies | varies | varies |
| True Positive Rate | varies | varies | varies |
| False Positive Rate | varies | varies | varies |

The group fairness analysis reveals how aggregate outcomes differ between demographic groups. For each protected attribute, the model splits the test set into Group A (standardized value <= 0) and Group B (> 0), then compares prediction rates, true positive rates, and false positive rates across these groups.

### Discriminatory Instance Search

| Metric | Value |
|--------|-------|
| Global search best QID | 0.054 |
| Local search instances found | 30 |

### Causal Debugging

| Layer | Sensitivity | Neuron Count |
|-------|-------------|--------------|
| **Layer 1** | **2.070** | 64 |
| Layer 2 | 1.467 | 32 |
| Layer 3 | 0.778 | 16 |
| Layer 4 | 0.254 | 8 |
| Layer 5 | 0.588 | 4 |
| Layer 6 | 0.436 | 2 |

Top biased neurons in Layer 1: #60 (2.097), #43 (2.028), #27 (1.444), #50 (1.320), #59 (1.298)

### SHAP Top 5 Features (log-odds)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | education-num | 0.997 |
| 2 | marital-status | 0.830 |
| 3 | age | 0.647 |
| 4 | hours-per-week | 0.461 |
| 5 | capital-loss | 0.258 |

### LIME Top 5 Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | education-num | 0.211 |
| 2 | capital-gain | 0.203 |
| 3 | capital-loss | 0.148 |
| 4 | hours-per-week | 0.115 |
| 5 | marital-status | 0.086 |

### Overall Assessment

| Metric | Value |
|--------|-------|
| **Fairness Score** | **~41 / 100** |
| **Status** | **Concerning** |

The score now includes penalties from both individual-level QID metrics and group-level fairness metrics (Demographic Parity and Equalized Odds differences averaged across all protected attributes).

### Step Timings

| Step | Duration |
|------|----------|
| Training | 77s (first run) / <1s (cached) |
| Activations | 1s |
| QID Analysis + Group Fairness | 1s |
| Search | 1s |
| Debug | 1s |
| Explain | 11s |
| **Total** | **92s (first run) / ~15s (cached)** |

With model caching enabled, subsequent analyses of the same dataset with the same configuration skip the 77-second training step entirely, reducing total analysis time to approximately 15 seconds.

### Interpretation Summary

The adult.csv dataset exhibits significant fairness concerns when analyzed through a deep neural network:

1. **Pervasive individual discrimination**: 94.6% of analyzed instances show QID above the 0.1-bit threshold, meaning the model changes its prediction for nearly everyone when protected attributes are altered.

2. **High average bias**: A mean QID of 0.6632 bits (out of a maximum of 1.0 for binary classification) indicates that protected attributes carry substantial influence over predictions.

3. **Legal threshold violations**: 71.2% of instances violate the four-fifths rule (disparate impact < 0.8), suggesting the model would likely fail legal scrutiny for disparate impact.

4. **Group-level disparities**: Group fairness metrics reveal differential treatment at the aggregate level — Demographic Parity, Equalized Odds, and Equal Opportunity differences further confirm that the model does not treat demographic groups equitably.

5. **Early-layer bias encoding**: Layer 1 has the highest sensitivity (2.070), indicating the network learns to encode protected information in its very first transformation. Neurons 60 and 43 in that layer are the primary carriers.

6. **Protected attribute influence**: SHAP confirms that age (a protected attribute) is the 3rd most important feature overall, with sex ranking 8th. While education and marital status are the dominant predictors (and are legitimate), the substantial contribution of age is a fairness concern.

7. **Cross-method agreement**: Both SHAP and LIME identify education-num as the most important feature, lending high confidence to the explanation. Their agreement on the relative importance of age provides consistent evidence of its role in the model's decisions.

8. **Interactive exploration**: The interactive LIME explorer allows practitioners to drill into specific individuals or hypothetical scenarios, enabling deeper investigation of flagged instances beyond aggregate statistics.

These findings are consistent with known biases in the UCI Adult dataset, which reflects historical income disparities across gender, race, and age groups in 1994 US Census data.

---

*FairLint-DL v1.0.0 — Deep Learning Fairness Analysis for VS Code*
