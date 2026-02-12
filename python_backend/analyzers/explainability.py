"""
LIME and SHAP explainability for the fairness DNN.
"""

import torch
import numpy as np
import shap
import lime
import lime.lime_tabular
from typing import Dict, List


class ExplainabilityAnalyzer:
    """Compute LIME and SHAP explanations for the DNN model."""

    def __init__(self, model, feature_names: List[str], device="cpu"):
        self.model = model
        self.model.eval()
        self.device = device
        self.feature_names = feature_names

    def _predict_fn(self, X: np.ndarray) -> np.ndarray:
        """Prediction function wrapper for LIME/SHAP (numpy in, numpy out)."""
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(x_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            return probs.cpu().numpy()

    def compute_shap_values(
        self,
        X_background: np.ndarray,
        X_explain: np.ndarray,
        max_background: int = 100,
    ) -> Dict:
        """
        Compute SHAP values using KernelExplainer.

        Args:
            X_background: Background dataset for SHAP (training data subset)
            X_explain: Instances to explain
            max_background: Max background samples (for speed)

        Returns:
            Dict with shap_values, feature_names, base_value, global_importance
        """
        bg = X_background[:max_background]
        explainer = shap.KernelExplainer(self._predict_fn, bg)
        shap_values = explainer.shap_values(X_explain)

        # shap_values is a list of arrays (one per class) for binary classification
        # Use class 1 (favorable outcome)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values

        # Global feature importance: mean absolute SHAP value per feature
        global_importance = np.abs(sv).mean(axis=0).tolist()

        # Get base value
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            base_value = float(explainer.expected_value[1])
        else:
            base_value = float(explainer.expected_value)

        return {
            "shap_values": sv.tolist(),
            "global_importance": global_importance,
            "feature_names": self.feature_names,
            "base_value": base_value,
            "num_explained": len(X_explain),
        }

    def compute_lime_explanations(
        self,
        X_train: np.ndarray,
        X_explain: np.ndarray,
        num_features: int = 10,
        num_samples: int = 500,
    ) -> Dict:
        """
        Compute LIME explanations for given instances.

        Args:
            X_train: Training data (for LIME's background distribution)
            X_explain: Instances to explain
            num_features: Number of top features in explanation
            num_samples: Perturbation samples for LIME

        Returns:
            Dict with per-instance explanations and aggregated importance
        """
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=self.feature_names,
            class_names=["Unfavorable", "Favorable"],
            mode="classification",
        )

        explanations = []
        aggregated_importance = np.zeros(len(self.feature_names))

        for i in range(len(X_explain)):
            exp = explainer.explain_instance(
                X_explain[i],
                self._predict_fn,
                num_features=min(num_features, len(self.feature_names)),
                num_samples=num_samples,
            )
            feature_weights = exp.as_list()

            explanations.append(
                {
                    "instance_idx": i,
                    "feature_weights": feature_weights,
                    "prediction_proba": self._predict_fn(
                        X_explain[i : i + 1]
                    )[0].tolist(),
                }
            )

            # Aggregate absolute feature importance from the map output
            local_map = exp.as_map()
            for class_idx in local_map:
                for feat_idx, weight in local_map[class_idx]:
                    if feat_idx < len(self.feature_names):
                        aggregated_importance[feat_idx] += abs(weight)

        aggregated_importance /= max(len(X_explain), 1)

        return {
            "explanations": explanations,
            "aggregated_importance": aggregated_importance.tolist(),
            "feature_names": self.feature_names,
            "num_explained": len(X_explain),
        }
