"""
Model caching utilities for FairLint-DL.
Saves and loads trained models + preprocessor state + data tensors
to avoid redundant retraining when the same config is used.
"""

import hashlib
import json
import os
import pickle
import torch
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of file contents (8KB chunked reads)."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_cache_key(
    file_path: str,
    label_column: str,
    sensitive_features: List[str],
    hidden_layers: List[int],
    num_epochs: int,
    batch_size: int,
) -> str:
    """
    Compute deterministic cache key from training configuration.
    Returns SHA-256 hex digest.
    """
    file_hash = compute_file_hash(file_path)
    sorted_features = ",".join(sorted(sensitive_features))
    layers_str = ",".join(str(x) for x in hidden_layers)

    key_string = (
        f"{file_hash}|{label_column}|{sorted_features}"
        f"|{layers_str}|{num_epochs}|{batch_size}"
    )
    return hashlib.sha256(key_string.encode()).hexdigest()


def get_cache_dir(file_path: str, cache_key: str) -> Path:
    """Get cache directory path: <csv_dir>/.fairlint_cache/<cache_key>/"""
    csv_dir = Path(file_path).parent
    return csv_dir / ".fairlint_cache" / cache_key


def cache_exists(file_path: str, cache_key: str) -> bool:
    """Check if a valid cache exists for the given key."""
    cache_dir = get_cache_dir(file_path, cache_key)
    required_files = ["model.pt", "preprocessor.pkl", "metadata.json", "data_tensors.pt"]
    return all((cache_dir / f).exists() for f in required_files)


def save_to_cache(
    file_path: str,
    cache_key: str,
    model: torch.nn.Module,
    model_config: Dict,
    preprocessor_state: Dict,
    data_tensors: Dict[str, torch.Tensor],
    training_response: Dict,
) -> str:
    """
    Save all artifacts needed to reconstruct global state.

    Args:
        file_path: Original CSV file path
        cache_key: The computed cache key
        model: Trained FairnessDetectorDNN
        model_config: {input_dim, hidden_layers, protected_indices, dropout_rate}
        preprocessor_state: {scalers, label_encoders, feature_names, protected_feature_names}
        data_tensors: {X_train, y_train, X_val, y_val, X_test, y_test}
        training_response: The full /train response dict (accuracy, training_history, etc.)

    Returns:
        Path to cache directory as string
    """
    cache_dir = get_cache_dir(file_path, cache_key)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Model state_dict
    torch.save(model.state_dict(), cache_dir / "model.pt")

    # 2. Preprocessor state (sklearn objects need pickle)
    with open(cache_dir / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor_state, f)

    # 3. Data tensors (train/val/test splits)
    torch.save(data_tensors, cache_dir / "data_tensors.pt")

    # 4. Human-readable metadata
    metadata = {
        "cache_key": cache_key,
        "file_path": file_path,
        "file_size": os.path.getsize(file_path),
        "model_config": model_config,
        "training_response": training_response,
        "cached_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(cache_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Model cached to {cache_dir}")
    return str(cache_dir)


def load_from_cache(file_path: str, cache_key: str) -> Optional[Dict]:
    """
    Load all cached artifacts.

    Returns:
        Dict with: model_state_dict, model_config, preprocessor_state,
                   data_tensors, training_response
        Or None if cache is invalid/missing.
    """
    cache_dir = get_cache_dir(file_path, cache_key)

    if not cache_exists(file_path, cache_key):
        return None

    try:
        model_state_dict = torch.load(
            cache_dir / "model.pt",
            map_location="cpu",
            weights_only=True,
        )

        with open(cache_dir / "preprocessor.pkl", "rb") as f:
            preprocessor_state = pickle.load(f)

        data_tensors = torch.load(
            cache_dir / "data_tensors.pt",
            map_location="cpu",
            weights_only=True,
        )

        with open(cache_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        print(f"Model loaded from cache: {cache_dir}")
        return {
            "model_state_dict": model_state_dict,
            "model_config": metadata["model_config"],
            "preprocessor_state": preprocessor_state,
            "data_tensors": data_tensors,
            "training_response": metadata["training_response"],
        }
    except Exception as e:
        print(f"Cache load failed ({e}), will retrain")
        return None
