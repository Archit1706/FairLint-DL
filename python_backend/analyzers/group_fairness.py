"""
Group fairness metrics: Demographic Parity, Equalized Odds, Equal Opportunity.

Computes per-group confusion matrices and derives standard group-level
fairness metrics for each protected attribute.
"""

import torch
import numpy as np
from typing import List, Dict


class GroupFairnessAnalyzer:
    """Compute group fairness metrics for all protected attributes."""

    def __init__(
        self,
        model,
        protected_indices: List[int],
        protected_feature_names: List[str],
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.protected_indices = protected_indices
        self.protected_feature_names = protected_feature_names
        self.device = device

    def compute_all(
        self, X_test: torch.Tensor, y_test: torch.Tensor
    ) -> List[Dict]:
        """Compute group fairness metrics for every protected attribute.

        Args:
            X_test: Test features tensor (n_samples, n_features).
            y_test: Test labels tensor (n_samples,).

        Returns:
            List of dicts, one per protected attribute, each containing
            demographic_parity, equalized_odds, equal_opportunity,
            and per-group confusion matrices.
        """
        # Run model predictions once (shared across all attributes)
        with torch.no_grad():
            logits = self.model(X_test.to(self.device))
            y_pred = logits.argmax(dim=1).cpu()

        y_true = y_test.cpu()

        results = []
        for attr_name, attr_idx in zip(
            self.protected_feature_names, self.protected_indices
        ):
            attr_vals = X_test[:, attr_idx].cpu()
            # Split groups: <=0 is Group A, >0 is Group B (standardized data)
            mask_a = attr_vals <= 0
            mask_b = attr_vals > 0

            metrics = self._compute_for_attribute(
                y_true, y_pred, mask_a, mask_b, attr_name, attr_idx
            )
            results.append(metrics)

        return results

    def _compute_for_attribute(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mask_a: torch.Tensor,
        mask_b: torch.Tensor,
        attr_name: str,
        attr_idx: int,
    ) -> Dict:
        """Compute all fairness metrics for a single protected attribute."""
        y_true_a, y_pred_a = y_true[mask_a], y_pred[mask_a]
        y_true_b, y_pred_b = y_true[mask_b], y_pred[mask_b]

        # Per-group confusion matrices
        cm_a = self._confusion_matrix(y_true_a, y_pred_a)
        cm_b = self._confusion_matrix(y_true_b, y_pred_b)

        # Demographic Parity: P(Y_hat=1 | Group)
        pos_rate_a = (
            (y_pred_a == 1).float().mean().item() if len(y_pred_a) > 0 else 0.0
        )
        pos_rate_b = (
            (y_pred_b == 1).float().mean().item() if len(y_pred_b) > 0 else 0.0
        )
        dp_diff = abs(pos_rate_a - pos_rate_b)
        max_rate = max(pos_rate_a, pos_rate_b)
        dp_ratio = (
            min(pos_rate_a, pos_rate_b) / max_rate if max_rate > 0 else 1.0
        )

        # TPR and FPR per group
        tpr_a = self._safe_divide(cm_a["tp"], cm_a["tp"] + cm_a["fn"])
        tpr_b = self._safe_divide(cm_b["tp"], cm_b["tp"] + cm_b["fn"])
        fpr_a = self._safe_divide(cm_a["fp"], cm_a["fp"] + cm_a["tn"])
        fpr_b = self._safe_divide(cm_b["fp"], cm_b["fp"] + cm_b["tn"])

        # Equalized Odds: max(|TPR_A - TPR_B|, |FPR_A - FPR_B|)
        tpr_diff = abs(tpr_a - tpr_b)
        fpr_diff = abs(fpr_a - fpr_b)
        eo_max_diff = max(tpr_diff, fpr_diff)

        # Equal Opportunity: |TPR_A - TPR_B|
        eop_diff = tpr_diff

        return {
            "attribute_name": attr_name,
            "attribute_index": attr_idx,
            "group_a": {
                "label": f"{attr_name} <= 0 (Group A)",
                "size": int(mask_a.sum().item()),
            },
            "group_b": {
                "label": f"{attr_name} > 0 (Group B)",
                "size": int(mask_b.sum().item()),
            },
            "demographic_parity": {
                "positive_rate_a": round(pos_rate_a, 6),
                "positive_rate_b": round(pos_rate_b, 6),
                "difference": round(dp_diff, 6),
                "ratio": round(dp_ratio, 6),
            },
            "equalized_odds": {
                "tpr_a": round(tpr_a, 6),
                "tpr_b": round(tpr_b, 6),
                "fpr_a": round(fpr_a, 6),
                "fpr_b": round(fpr_b, 6),
                "tpr_difference": round(tpr_diff, 6),
                "fpr_difference": round(fpr_diff, 6),
                "max_difference": round(eo_max_diff, 6),
            },
            "equal_opportunity": {
                "tpr_a": round(tpr_a, 6),
                "tpr_b": round(tpr_b, 6),
                "difference": round(eop_diff, 6),
            },
            "confusion_matrix": {
                "group_a": cm_a,
                "group_b": cm_b,
            },
        }

    @staticmethod
    def _confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict:
        """Compute binary confusion matrix from tensors."""
        if len(y_true) == 0:
            return {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        tp = int(((y_pred == 1) & (y_true == 1)).sum().item())
        fp = int(((y_pred == 1) & (y_true == 0)).sum().item())
        tn = int(((y_pred == 0) & (y_true == 0)).sum().item())
        fn = int(((y_pred == 0) & (y_true == 1)).sum().item())
        return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

    @staticmethod
    def _safe_divide(numerator: int, denominator: int) -> float:
        """Safe division that returns 0.0 when denominator is zero."""
        return numerator / denominator if denominator > 0 else 0.0
