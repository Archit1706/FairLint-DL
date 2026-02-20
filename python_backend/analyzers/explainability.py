"""
LIME and SHAP explainability for the fairness DNN.

Uses SHAP KernelExplainer on log-odds output for more meaningful
feature importance values, following SHAP documentation best practices.
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

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Prediction function returning class probabilities (for LIME)."""
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(x_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            result = probs.cpu().numpy()
            mask = ~np.isfinite(result).all(axis=1)
            if mask.any():
                result[mask] = 1.0 / result.shape[1]
            return result

    def _predict_logits(self, X: np.ndarray) -> np.ndarray:
        """
        Prediction function returning raw logits (for SHAP).

        Using logits instead of softmax probabilities produces much more
        meaningful SHAP values because the logit space is linear and
        differences are not compressed by the sigmoid/softmax.
        See: https://shap.readthedocs.io/en/latest/ (logistic regression section)
        """
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(x_tensor)
            result = logits.cpu().numpy()
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            return result

    def _predict_margin(self, X: np.ndarray) -> np.ndarray:
        """
        Prediction function returning the margin (logit difference) for class 1.

        margin = logit(class=1) - logit(class=0)
        This is the log-odds of the positive class, which gives SHAP values
        in interpretable units.
        """
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(x_tensor)
            result = logits.cpu().numpy()
            # For binary classification: return logit[1] - logit[0] (log-odds)
            if result.shape[1] == 2:
                margin = result[:, 1] - result[:, 0]
            else:
                margin = result[:, 1]
            margin = np.nan_to_num(margin, nan=0.0, posinf=0.0, neginf=0.0)
            return margin

    def compute_shap_values(
        self,
        X_background: np.ndarray,
        X_explain: np.ndarray,
        max_background: int = 100,
    ) -> Dict:
        """
        Compute SHAP values using KernelExplainer on log-odds output.

        Uses the margin (log-odds) output space rather than softmax probabilities
        to produce meaningful SHAP values, following SHAP documentation guidance
        for classification models.
        """
        # Subsample background data
        bg = X_background[:max_background]

        # Use raw background data (not kmeans) for KernelExplainer
        # kmeans can distort the background distribution too much
        # Subsample to reasonable size for speed
        bg_size = min(50, len(bg))
        if len(bg) > bg_size:
            indices = np.random.choice(len(bg), bg_size, replace=False)
            bg = bg[indices]

        # Use margin (log-odds) for more meaningful SHAP values
        explainer = shap.KernelExplainer(self._predict_margin, bg)

        # Compute SHAP values with sufficient samples for stability
        shap_values = explainer.shap_values(
            X_explain,
            nsamples="auto",  # Let SHAP determine optimal sample count
            l1_reg="num_features(10)",  # Regularization for stability
        )

        sv = np.array(shap_values)

        # Ensure 2D array
        if sv.ndim == 1:
            sv = sv.reshape(1, -1)

        # Replace NaN/Inf values with 0
        sv = np.nan_to_num(sv, nan=0.0, posinf=0.0, neginf=0.0)

        # Global feature importance: mean absolute SHAP value per feature
        global_importance = np.abs(sv).mean(axis=0).tolist()

        # Get base value (expected margin value on background)
        base_value = float(np.nan_to_num(explainer.expected_value, nan=0.0))

        # Feature values for coloring in beeswarm plot
        feature_values = X_explain.tolist()

        return {
            "shap_values": sv.tolist(),
            "global_importance": global_importance,
            "feature_names": self.feature_names,
            "base_value": base_value,
            "num_explained": len(X_explain),
            "feature_values": feature_values,
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

        LIME uses probabilities (not logits) as it locally approximates
        the decision boundary with a linear model.
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
                self._predict_proba,
                num_features=min(num_features, len(self.feature_names)),
                num_samples=num_samples,
            )
            feature_weights = exp.as_list()

            explanations.append(
                {
                    "instance_idx": i,
                    "feature_weights": feature_weights,
                    "prediction_proba": self._predict_proba(
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
