"""
Reproduce Table 2 of the paper (FairLint-DL) from the bundled datasets.

Self-contained: reads local CSVs from ./datasets/ (no network access) and runs
the exact analysis pipeline used by the extension backend
(DataPreprocessor -> DNNTrainer -> QIDAnalyzer.batch_analyze ->
GroupFairnessAnalyzer). Runs are seeded for determinism.

Usage:
    cd python_backend
    python reproduce.py            # all three datasets (~2-4 min on CPU)
    python reproduce.py --quick    # Adult + German only (faster smoke test)

Note on Adult: the paper's headline Adult QID numbers come from an earlier,
unseeded run and are not bit-for-bit reproducible. This seeded run yields the
values reported in the accuracy/group-fairness rows of Table 2 and closely
matches the QID rows (mean QID ~0.60 vs 0.619 reported); the qualitative
claim (pervasive individual discrimination on Adult) reproduces robustly.
German Credit and Bank Marketing reproduce exactly.
"""
import os
import sys
import json
import argparse
import warnings

warnings.filterwarnings("ignore")

BACKEND = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND)
DATASETS = os.path.join(BACKEND, "datasets")

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from utils.data_loader import DataPreprocessor
from models.fairness_dnn import FairnessDetectorDNN, DNNTrainer
from analyzers.qid_analyzer import QIDAnalyzer
from analyzers.group_fairness import GroupFairnessAnalyzer

CONFIGS = {
    "Adult": {"file": "adult.csv", "label": "income",
              "sensitive": ["age", "race", "sex", "native-country"]},
    "German": {"file": "german.csv", "label": "class",
               "sensitive": ["age", "personal_status"]},
    "Bank": {"file": "bank.csv", "label": "Class",
             "sensitive": ["age", "marital"]},
}


def composite_score(qid, gf):
    """Replica of calculateFairnessScore in src/webview/scoring.ts."""
    score = 100.0
    score -= min(qid["mean_qid"] * 15, 30)
    score -= min(qid["pct_discriminatory"] * 0.3, 30)
    if qid["mean_disparate_impact"] < 0.8:
        score -= (0.8 - qid["mean_disparate_impact"]) * 50
    score -= min(qid["max_qid"] * 5, 20)
    if gf:
        avg_dp = sum(g["demographic_parity"]["difference"] for g in gf) / len(gf)
        avg_eo = sum(g["equalized_odds"]["max_difference"] for g in gf) / len(gf)
        score -= min((avg_dp + avg_eo) * 25, 20)
    return max(0, round(score))


def prepared_csv(name, cfg):
    """Return a path to a CSV whose label column is 0-based integer encoded."""
    import pandas as pd
    path = os.path.join(DATASETS, cfg["file"])
    df = pd.read_csv(path, skipinitialspace=True)
    if not pd.api.types.is_numeric_dtype(df[cfg["label"]]):
        df[cfg["label"]] = LabelEncoder().fit_transform(df[cfg["label"]].astype(str))
        out = os.path.join(DATASETS, f"_prepared_{cfg['file']}")
        df.to_csv(out, index=False)
        return out
    return path


def run(name, cfg):
    torch.manual_seed(42)
    np.random.seed(42)
    path = prepared_csv(name, cfg)
    info = DataPreprocessor().load_and_preprocess(
        path, label_column=cfg["label"], sensitive_features=cfg["sensitive"]
    )
    model = FairnessDetectorDNN(
        input_dim=info["input_dim"],
        protected_indices=info["protected_indices"],
        hidden_layers=[64, 32, 16, 8, 4],
    )
    trainer = DNNTrainer(model, device="cpu")
    trainer.train(info["train_loader"], info["val_loader"], num_epochs=30, verbose=False)
    _, acc = trainer.validate(info["test_loader"])

    X = torch.cat([bx for bx, _ in info["test_loader"]], dim=0)
    y = torch.cat([by for _, by in info["test_loader"]], dim=0)
    qid = QIDAnalyzer(model, info["protected_indices"], "cpu").batch_analyze(
        X, [0.0, 1.0], max_samples=min(500, len(X))
    )
    gf = GroupFairnessAnalyzer(
        model, info["protected_indices"], info["protected_features"], "cpu"
    ).compute_all(X, y)
    n = qid["num_analyzed"]
    return {
        "Mean QID (bits)": round(qid["mean_qid"], 3),
        "Mean min-entropy QID": round(qid["mean_min_entropy"], 3),
        "Mean disparate impact": round(qid["mean_disparate_impact"], 3),
        "Discriminatory (%)": round(qid["pct_discriminatory"], 1),
        "Violating 80% (%)": round(100 * qid["num_violating_80_rule"] / n, 1),
        "Demographic parity diff": round(
            sum(g["demographic_parity"]["difference"] for g in gf) / len(gf), 3),
        "Equalized odds diff": round(
            sum(g["equalized_odds"]["max_difference"] for g in gf) / len(gf), 3),
        "Model accuracy (%)": round(acc, 1),
        "Composite score": composite_score(qid, gf),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Adult + German only")
    args = ap.parse_args()
    names = ["Adult", "German"] if args.quick else ["Adult", "German", "Bank"]

    results = {}
    for name in names:
        print(f"[reproduce] running {name} ...", flush=True)
        results[name] = run(name, CONFIGS[name])

    metrics = list(next(iter(results.values())).keys())
    col = max(len(m) for m in metrics)
    header = "Metric".ljust(col) + "".join(f"{n:>12}" for n in names)
    print("\n" + "=" * len(header))
    print("Table 2 (reproduced)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for m in metrics:
        print(m.ljust(col) + "".join(f"{str(results[n][m]):>12}" for n in names))
    print("=" * len(header))

    with open(os.path.join(BACKEND, "reproduce_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved reproduce_results.json")


if __name__ == "__main__":
    main()
