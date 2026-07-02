"""
Unit tests for the FairLint-DL analysis backend.

These exercise the core analysis components named in the paper: QID computation
(Shannon and min-entropy / disparate impact), the two-phase discriminatory
instance search, causal layer/neuron localization, group fairness metrics, data
preprocessing, the DNN proxy model, and the REST endpoints.

Runnable two ways:
    pytest python_backend/tests/
    python python_backend/tests/test_backend.py   (no pytest required)
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

# Make the backend package importable when run directly.
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import tempfile

from analyzers.qid_analyzer import QIDAnalyzer
from analyzers.group_fairness import GroupFairnessAnalyzer
from analyzers.search import DiscriminatoryInstanceSearch
from analyzers.causal_debugger import CausalDebugger
from analyzers.internal_space import InternalSpaceAnalyzer
from analyzers.explainability import ExplainabilityAnalyzer
from models.fairness_dnn import FairnessDetectorDNN, DNNTrainer
from utils.data_loader import DataPreprocessor
from utils.model_cache import compute_cache_key


# --------------------------------------------------------------------------
# Controllable dummy models for deterministic QID behaviour
# --------------------------------------------------------------------------
class ProtectedSensitiveModel(nn.Module):
    """Prediction depends only on the protected feature (maximally unfair)."""

    def __init__(self, protected_idx=0, scale=10.0):
        super().__init__()
        self.protected_idx = protected_idx
        self.scale = scale

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        p = x[:, self.protected_idx]
        zeros = torch.zeros_like(p)
        return torch.stack([zeros, p * self.scale], dim=1)


class ConfidentInvariantModel(nn.Module):
    """Always predicts class 1 confidently, ignoring every feature (fair)."""

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        n = x.shape[0]
        return torch.stack([torch.full((n,), -30.0), torch.full((n,), 30.0)], dim=1)


# --------------------------------------------------------------------------
# QID: Shannon entropy
# --------------------------------------------------------------------------
def test_shannon_qid_high_for_protected_sensitive_model():
    analyzer = QIDAnalyzer(ProtectedSensitiveModel(), protected_indices=[0])
    res = analyzer.compute_shannon_qid(torch.zeros(6), [-3.0, 3.0])
    assert res["qid_bits"] > 0.9, res["qid_bits"]
    assert bool(res["has_discrimination"]) is True


def test_shannon_qid_near_zero_for_invariant_model():
    analyzer = QIDAnalyzer(ConfidentInvariantModel(), protected_indices=[0])
    res = analyzer.compute_shannon_qid(torch.randn(6), [-3.0, 3.0])
    assert res["qid_bits"] < 0.05, res["qid_bits"]
    assert bool(res["has_discrimination"]) is False


# --------------------------------------------------------------------------
# QID: min-entropy / disparate impact
# --------------------------------------------------------------------------
def test_disparate_impact_low_for_sensitive_model():
    analyzer = QIDAnalyzer(ProtectedSensitiveModel(), protected_indices=[0])
    res = analyzer.compute_min_entropy_qid(torch.zeros(6), [-3.0, 3.0])
    assert res["disparate_impact_ratio"] < 0.1, res["disparate_impact_ratio"]
    assert res["violates_80_rule"] is True


def test_disparate_impact_high_for_invariant_model():
    analyzer = QIDAnalyzer(ConfidentInvariantModel(), protected_indices=[0])
    res = analyzer.compute_min_entropy_qid(torch.randn(6), [-3.0, 3.0])
    assert res["disparate_impact_ratio"] > 0.95, res["disparate_impact_ratio"]
    assert res["violates_80_rule"] is False


# --------------------------------------------------------------------------
# QID: batch aggregation
# --------------------------------------------------------------------------
def test_batch_analyze_reports_full_discrimination_for_sensitive_model():
    analyzer = QIDAnalyzer(ProtectedSensitiveModel(), protected_indices=[0])
    X = torch.zeros(20, 6)
    out = analyzer.batch_analyze(X, [-3.0, 3.0], max_samples=20)
    assert out["num_analyzed"] == 20
    assert out["pct_discriminatory"] == 100.0
    assert 0.0 <= out["mean_disparate_impact"] <= 1.0
    for key in ("mean_qid", "max_qid", "num_violating_80_rule"):
        assert key in out


# --------------------------------------------------------------------------
# Group fairness
# --------------------------------------------------------------------------
def test_confusion_matrix_counts():
    y_true = torch.tensor([1, 1, 0, 0])
    y_pred = torch.tensor([1, 0, 1, 0])
    cm = GroupFairnessAnalyzer._confusion_matrix(y_true, y_pred)
    assert cm == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}


def test_safe_divide_handles_zero_denominator():
    assert GroupFairnessAnalyzer._safe_divide(3, 0) == 0.0
    assert GroupFairnessAnalyzer._safe_divide(1, 4) == 0.25


def test_group_fairness_structure():
    model = ConfidentInvariantModel()
    gfa = GroupFairnessAnalyzer(model, [0], ["sex"])
    X = torch.randn(30, 5)
    y = torch.randint(0, 2, (30,))
    results = gfa.compute_all(X, y)
    assert len(results) == 1
    r = results[0]
    for key in ("demographic_parity", "equalized_odds", "equal_opportunity"):
        assert key in r
    assert 0.0 <= r["demographic_parity"]["ratio"] <= 1.0


# --------------------------------------------------------------------------
# Discriminatory instance search
# --------------------------------------------------------------------------
def test_search_returns_expected_keys():
    model = FairnessDetectorDNN(input_dim=8, protected_indices=[0])
    analyzer = QIDAnalyzer(model, protected_indices=[0])
    search = DiscriminatoryInstanceSearch(model, analyzer, protected_indices=[0])
    X = torch.randn(40, 8)
    res = search.search(X, [-3.0, 3.0], num_global_iterations=10, num_local_neighbors=15)
    for key in ("best_instance", "best_qid", "discriminatory_instances", "num_found"):
        assert key in res
    assert res["num_found"] == len(res["discriminatory_instances"])


# --------------------------------------------------------------------------
# Causal debugging
# --------------------------------------------------------------------------
def test_causal_debugger_localizes_layer_and_neurons():
    model = FairnessDetectorDNN(input_dim=8, protected_indices=[0])
    debugger = CausalDebugger(model)
    instances = [{"instance": x.tolist()} for x in torch.randn(10, 8)]

    layer_res = debugger.localize_biased_layer(instances)
    assert "biased_layer" in layer_res
    assert "all_layers" in layer_res
    biased = layer_res["biased_layer"]
    assert set(("layer_idx", "sensitivity", "neuron_count")).issubset(biased)

    neurons = debugger.localize_biased_neurons(biased["layer_idx"], instances, top_k=3)
    assert isinstance(neurons, list)
    assert len(neurons) <= 3


# --------------------------------------------------------------------------
# Data preprocessing
# --------------------------------------------------------------------------
def test_detect_sensitive_columns():
    pre = DataPreprocessor()
    cols = ["age", "sex", "race", "native-country", "education-num", "income"]
    detected = pre.detect_sensitive_columns(cols)
    for expected in ("age", "sex", "race", "native-country"):
        assert expected in detected
    assert "education-num" not in detected
    assert "income" not in detected


# --------------------------------------------------------------------------
# DNN proxy model
# --------------------------------------------------------------------------
def test_model_forward_shape_and_params():
    model = FairnessDetectorDNN(input_dim=10, protected_indices=[0])
    model.eval()
    out = model(torch.randn(4, 10))
    assert out.shape == (4, 2)
    assert model.count_parameters() > 0


# --------------------------------------------------------------------------
# REST endpoints
# --------------------------------------------------------------------------
def test_root_and_columns_endpoints():
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        print("  (skipped: fastapi test client unavailable)")
        return
    try:
        import bias_server
        client = TestClient(bias_server.app)
    except RuntimeError as e:
        # TestClient needs httpx; skip cleanly if it is not installed.
        if "httpx" in str(e):
            print("  (skipped: httpx not installed)")
            return
        raise

    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "running"

    adult = os.path.join(BACKEND_DIR, "adult.csv")
    if os.path.exists(adult):
        resp = client.post("/columns", json={"file_path": adult})
        assert resp.status_code == 200
        body = resp.json()
        assert "income" in body["columns"]
        assert body["num_columns"] > 0


# --------------------------------------------------------------------------
# Internal space (PCA of activations)
# --------------------------------------------------------------------------
def test_internal_space_pca_structure():
    model = FairnessDetectorDNN(input_dim=8, protected_indices=[0])
    analyzer = InternalSpaceAnalyzer(model)
    X = torch.randn(40, 8)
    y = np.random.randint(0, 2, size=40)
    prot = X[:, 0].numpy()
    out = analyzer.compute_visualization_data(X, y, prot, method="pca", max_samples=30)
    assert out["method"] == "pca"
    assert out["num_samples"] == 30
    assert len(out["layers"]) >= 1
    for layer in out["layers"]:
        assert len(layer["x"]) == out["num_samples"]
        assert len(layer["y"]) == out["num_samples"]


# --------------------------------------------------------------------------
# Explainability (SHAP and LIME)
# --------------------------------------------------------------------------
def test_shap_values_structure():
    names = [f"f{i}" for i in range(5)]
    model = FairnessDetectorDNN(input_dim=5, protected_indices=[0])
    analyzer = ExplainabilityAnalyzer(model, feature_names=names)
    X_bg = np.random.randn(20, 5).astype(np.float32)
    X_exp = np.random.randn(2, 5).astype(np.float32)
    res = analyzer.compute_shap_values(X_bg, X_exp, max_background=20)
    assert len(res["global_importance"]) == 5
    assert res["feature_names"] == names
    assert res["num_explained"] == 2


def test_lime_explanations_structure():
    names = [f"f{i}" for i in range(5)]
    model = FairnessDetectorDNN(input_dim=5, protected_indices=[0])
    analyzer = ExplainabilityAnalyzer(model, feature_names=names)
    X_bg = np.random.randn(30, 5).astype(np.float32)
    X_exp = np.random.randn(2, 5).astype(np.float32)
    res = analyzer.compute_lime_explanations(X_bg, X_exp)
    assert "explanations" in res
    assert len(res["explanations"]) == 2


# --------------------------------------------------------------------------
# Model cache key
# --------------------------------------------------------------------------
def test_cache_key_deterministic_and_sensitive():
    adult = os.path.join(BACKEND_DIR, "adult.csv")
    if not os.path.exists(adult):
        print("  (skipped: adult.csv not present)")
        return
    base = dict(
        file_path=adult, label_column="income", sensitive_features=["sex", "race"],
        hidden_layers=[64, 32], num_epochs=30, batch_size=128,
    )
    k1 = compute_cache_key(**base)
    # Deterministic and order-independent in sensitive features.
    k2 = compute_cache_key(**{**base, "sensitive_features": ["race", "sex"]})
    assert k1 == k2
    # Any config change invalidates the key.
    assert compute_cache_key(**{**base, "num_epochs": 31}) != k1
    assert compute_cache_key(**{**base, "hidden_layers": [64, 32, 16]}) != k1


# --------------------------------------------------------------------------
# End-to-end: preprocess -> train -> analyze on a synthetic CSV
# --------------------------------------------------------------------------
def test_end_to_end_train_and_analyze():
    csv = (
        "age,sex,score,label\n"
        + "\n".join(
            f"{20 + i % 40},{i % 2},{(i * 7) % 100},{i % 2}" for i in range(120)
        )
    )
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(csv)
        path = f.name
    try:
        pre = DataPreprocessor()
        info = pre.load_and_preprocess(path, label_column="label", sensitive_features=["age", "sex"])
        assert info["num_classes"] == 2
        assert len(info["protected_indices"]) == 2

        model = FairnessDetectorDNN(
            input_dim=info["input_dim"], protected_indices=info["protected_indices"]
        )
        trainer = DNNTrainer(model, device="cpu")
        trainer.train(info["train_loader"], info["val_loader"], num_epochs=3, verbose=False)

        X_test = torch.cat([bx for bx, _ in info["test_loader"]], dim=0)
        analyzer = QIDAnalyzer(model, protected_indices=info["protected_indices"])
        out = analyzer.batch_analyze(X_test, [0.0, 1.0], max_samples=len(X_test))
        assert out["num_analyzed"] == len(X_test)
        assert 0.0 <= out["pct_discriminatory"] <= 100.0
    finally:
        os.unlink(path)


def test_preprocessing_handles_categorical_and_missing():
    rows = ["age,city,label"]
    for i in range(60):
        age = "" if i % 15 == 0 else str(20 + i % 40)  # some missing ages
        city = "NYC" if i % 2 == 0 else "LA"
        rows.append(f"{age},{city},{i % 2}")
    csv = "\n".join(rows) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(csv)
        path = f.name
    try:
        pre = DataPreprocessor()
        info = pre.load_and_preprocess(path, label_column="label", sensitive_features=["age"])
        # City (categorical) is label-encoded into the feature matrix; age is numeric.
        assert info["input_dim"] == 2
        assert info["num_classes"] == 2
    finally:
        os.unlink(path)


# --------------------------------------------------------------------------
# Standalone runner (no pytest dependency)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
