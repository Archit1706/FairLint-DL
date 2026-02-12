"""
Internal space visualization: extract and reduce layer activations for 2D plotting.
"""

import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Dict


class InternalSpaceAnalyzer:
    """Extract and reduce intermediate layer activations for visualization."""

    def __init__(self, model, device="cpu"):
        self.model = model
        self.model.eval()
        self.device = device

    def extract_activations(
        self, X: torch.Tensor, max_samples: int = 500
    ) -> List[np.ndarray]:
        """
        Extract activations from all hidden layers for a batch of inputs.

        Returns:
            List of numpy arrays, one per hidden layer, each shape (n_samples, layer_dim)
        """
        n = min(len(X), max_samples)
        X_subset = X[:n].to(self.device)

        with torch.no_grad():
            _, activations = self.model(X_subset, return_activations=True)

        return [act.cpu().numpy() for act in activations]

    def reduce_activations(
        self, activations: List[np.ndarray], method: str = "pca"
    ) -> List[np.ndarray]:
        """
        Reduce each layer's activations to 2D.

        Args:
            activations: List of (n_samples, layer_dim) arrays
            method: 'pca' or 'tsne'

        Returns:
            List of (n_samples, 2) arrays
        """
        reduced = []
        for act in activations:
            if act.shape[1] <= 2:
                if act.shape[1] == 1:
                    act = np.hstack([act, np.zeros((act.shape[0], 1))])
                reduced.append(act)
            elif method == "tsne":
                perplexity = min(30, act.shape[0] - 1)
                reducer = TSNE(n_components=2, perplexity=max(perplexity, 2), random_state=42)
                reduced.append(reducer.fit_transform(act))
            else:  # pca
                reducer = PCA(n_components=2, random_state=42)
                reduced.append(reducer.fit_transform(act))
        return reduced

    def compute_visualization_data(
        self,
        X: torch.Tensor,
        y: np.ndarray,
        protected_values: np.ndarray,
        method: str = "pca",
        max_samples: int = 500,
    ) -> Dict:
        """
        Full pipeline: extract, reduce, and format for frontend.

        Returns dict with per-layer 2D coordinates, labels, and protected attrs.
        """
        activations = self.extract_activations(X, max_samples)
        reduced = self.reduce_activations(activations, method)

        n = len(reduced[0])
        layers_data = []
        for i, coords in enumerate(reduced):
            layers_data.append(
                {
                    "layer_idx": i,
                    "layer_name": f"Layer {i + 1}",
                    "x": coords[:, 0].tolist(),
                    "y": coords[:, 1].tolist(),
                }
            )

        return {
            "layers": layers_data,
            "labels": y[:n].tolist(),
            "protected": protected_values[:n].tolist(),
            "method": method,
            "num_samples": int(n),
        }
