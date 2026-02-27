"""
FastAPI server for fairness analysis.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import os
from typing import List, Dict, Optional
import json

from models.fairness_dnn import FairnessDetectorDNN, DNNTrainer
from analyzers.qid_analyzer import QIDAnalyzer
from analyzers.search import DiscriminatoryInstanceSearch
from utils.data_loader import DataPreprocessor
from analyzers.causal_debugger import CausalDebugger


app = FastAPI(title="FairLint-DL Analysis API")

# Enable CORS for VS Code extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
current_model = None
current_analyzer = None
current_search_engine = None
current_data_info = None


class TrainRequest(BaseModel):
    file_path: str
    label_column: str
    sensitive_features: Optional[List[str]] = None
    num_epochs: int = 50
    batch_size: int = 128
    hidden_layers: Optional[List[int]] = None
    force_retrain: bool = False


class AnalyzeRequest(BaseModel):
    file_path: str
    label_column: str
    sensitive_features: List[str]
    protected_values: Dict[str, List[float]]
    max_samples: int = 1000


class SearchRequest(BaseModel):
    protected_values: Dict[str, List[float]]
    num_iterations: int = 100
    num_neighbors: int = 50


class ActivationsRequest(BaseModel):
    method: str = "pca"
    max_samples: int = 500


class ExplainRequest(BaseModel):
    method: str = "both"  # 'lime', 'shap', or 'both'
    num_instances: int = 10
    max_background: int = 100


class ExplainInstanceRequest(BaseModel):
    instance_type: str  # "index" or "custom"
    instance_index: Optional[int] = None  # test set index (for "index" type)
    feature_values: Optional[List[float]] = None  # raw feature values (for "custom" type)


class ColumnsRequest(BaseModel):
    file_path: str


@app.get("/")
async def root():
    return {"message": "Fairness Analysis API", "status": "running"}


@app.post("/columns")
async def get_columns(request: ColumnsRequest):
    """Get column names from a CSV file for dropdown selection."""
    import pandas as pd

    try:
        # Validate file exists
        if not os.path.exists(request.file_path):
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {request.file_path}"
            )

        # Validate file extension
        if not request.file_path.lower().endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail="Only CSV files are supported. Please select a .csv file."
            )

        # Read only header row for efficiency
        df = pd.read_csv(request.file_path, nrows=0)
        columns = df.columns.tolist()

        if not columns:
            raise HTTPException(
                status_code=400,
                detail="CSV file appears to be empty or has no column headers."
            )

        # Also get sample data for preview (first 3 rows)
        df_sample = pd.read_csv(request.file_path, nrows=3)
        sample_data = df_sample.to_dict('records')

        # Auto-detect sensitive columns for pre-selection
        preprocessor = DataPreprocessor()
        detected_sensitive = preprocessor.detect_sensitive_columns(columns)

        return {
            "status": "success",
            "columns": columns,
            "num_columns": len(columns),
            "sample_data": sample_data,
            "detected_sensitive": detected_sensitive,
        }

    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail="CSV file is empty or cannot be parsed."
        )
    except pd.errors.ParserError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse CSV file: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading file: {str(e)}"
        )


def _extract_tensors_from_loaders(data_info: dict) -> dict:
    """Extract raw tensors from DataLoaders for caching."""
    def _concat_loader(loader):
        X_list, y_list = [], []
        for batch_X, batch_y in loader:
            X_list.append(batch_X)
            y_list.append(batch_y)
        return torch.cat(X_list, dim=0), torch.cat(y_list, dim=0)

    X_train, y_train = _concat_loader(data_info["train_loader"])
    X_val, y_val = _concat_loader(data_info["val_loader"])
    X_test, y_test = _concat_loader(data_info["test_loader"])

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


def _restore_from_cache(cached: dict, request: TrainRequest) -> dict:
    """Reconstruct all global state from cache and return the training response."""
    global current_model, current_analyzer, current_search_engine, current_data_info
    import numpy as np
    from torch.utils.data import DataLoader
    from utils.data_loader import FairnessDataset

    config = cached["model_config"]
    tensors = cached["data_tensors"]
    preproc = cached["preprocessor_state"]

    # 1. Reconstruct model
    model = FairnessDetectorDNN(
        input_dim=config["input_dim"],
        protected_indices=config["protected_indices"],
        hidden_layers=config["hidden_layers"],
        dropout_rate=config.get("dropout_rate", 0.2),
    )
    model.load_state_dict(cached["model_state_dict"])
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    current_model = model.to(device)

    # 2. Reconstruct DataLoaders from saved tensors
    train_dataset = FairnessDataset(tensors["X_train"].numpy(), tensors["y_train"].numpy())
    val_dataset = FairnessDataset(tensors["X_val"].numpy(), tensors["y_val"].numpy())
    test_dataset = FairnessDataset(tensors["X_test"].numpy(), tensors["y_test"].numpy())

    batch_size = request.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 3. Rebuild current_data_info
    current_data_info = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "input_dim": config["input_dim"],
        "num_classes": len(torch.unique(tensors["y_train"])),
        "protected_indices": config["protected_indices"],
        "protected_features": preproc["protected_feature_names"],
        "feature_names": preproc["feature_names"],
        "scaler": preproc["scalers"].get("features"),
        "label_encoders": preproc["label_encoders"],
    }

    # 4. Reconstruct protected_attr_info
    protected_attr_info = {}
    for feat_name, feat_idx in zip(
        preproc["protected_feature_names"], config["protected_indices"]
    ):
        feat_vals = []
        for batch_X, _ in train_loader:
            feat_vals.extend(batch_X[:, feat_idx].numpy().tolist())
        unique_vals = np.unique(np.round(feat_vals, 2))
        protected_attr_info[feat_name] = {
            "index": feat_idx,
            "num_unique_values": len(unique_vals),
        }
    current_data_info["protected_attr_info"] = protected_attr_info

    # 5. Reconstruct analyzers
    current_analyzer = QIDAnalyzer(
        current_model,
        protected_indices=config["protected_indices"],
        device=device,
    )
    current_search_engine = DiscriminatoryInstanceSearch(
        current_model, current_analyzer,
        protected_indices=config["protected_indices"],
        device=device,
    )

    # 6. Return the cached training response with cache_hit flag
    response = dict(cached["training_response"])
    response["cache_hit"] = True
    response["message"] = "Model loaded from cache (skipped training)"
    return response


@app.post("/train")
async def train_model(request: TrainRequest):
    """Train fairness detector DNN on dataset, or load from cache if available."""
    global current_model, current_analyzer, current_search_engine, current_data_info

    try:
        from utils.model_cache import (
            compute_cache_key, cache_exists, load_from_cache, save_to_cache,
        )

        # Resolve defaults
        hidden_layers = request.hidden_layers or [64, 32, 16, 8, 4]
        sensitive_features = request.sensitive_features or []

        # Compute cache key
        cache_key = compute_cache_key(
            file_path=request.file_path,
            label_column=request.label_column,
            sensitive_features=sensitive_features,
            hidden_layers=hidden_layers,
            num_epochs=request.num_epochs,
            batch_size=request.batch_size,
        )

        # --- Cache hit path ---
        if not request.force_retrain and cache_exists(request.file_path, cache_key):
            cached = load_from_cache(request.file_path, cache_key)
            if cached is not None:
                return _restore_from_cache(cached, request)

        # --- Cache miss path: train normally ---
        preprocessor = DataPreprocessor()
        data_info = preprocessor.load_and_preprocess(
            file_path=request.file_path,
            label_column=request.label_column,
            sensitive_features=request.sensitive_features,
        )

        current_data_info = data_info

        model = FairnessDetectorDNN(
            input_dim=data_info["input_dim"],
            protected_indices=data_info["protected_indices"],
            hidden_layers=request.hidden_layers,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = DNNTrainer(model, device=device)

        history = trainer.train(
            train_loader=data_info["train_loader"],
            val_loader=data_info["val_loader"],
            num_epochs=request.num_epochs,
            verbose=True,
        )

        current_model = trainer.model

        current_analyzer = QIDAnalyzer(
            current_model,
            protected_indices=data_info["protected_indices"],
            device=device,
        )

        current_search_engine = DiscriminatoryInstanceSearch(
            current_model,
            current_analyzer,
            protected_indices=data_info["protected_indices"],
            device=device,
        )

        _, test_acc = trainer.validate(data_info["test_loader"])

        import numpy as np
        num_train = sum(len(batch_y) for _, batch_y in data_info["train_loader"])
        num_val = sum(len(batch_y) for _, batch_y in data_info["val_loader"])
        num_test = sum(len(batch_y) for _, batch_y in data_info["test_loader"])

        all_labels = []
        for _, batch_y in data_info["train_loader"]:
            all_labels.extend(batch_y.numpy().tolist())
        unique, counts = np.unique(all_labels, return_counts=True)
        class_distribution = {int(k): int(v) for k, v in zip(unique, counts)}

        protected_attr_info = {}
        for feat_name, feat_idx in zip(
            data_info["protected_features"], data_info["protected_indices"]
        ):
            feat_vals = []
            for batch_X, _ in data_info["train_loader"]:
                feat_vals.extend(batch_X[:, feat_idx].numpy().tolist())
            unique_vals = np.unique(np.round(feat_vals, 2))
            protected_attr_info[feat_name] = {
                "index": feat_idx,
                "num_unique_values": len(unique_vals),
            }
        current_data_info["protected_attr_info"] = protected_attr_info

        # Build response
        training_response = {
            "status": "success",
            "message": "Model trained successfully",
            "cache_hit": False,
            "accuracy": test_acc,
            "num_parameters": model.count_parameters(),
            "protected_features": data_info["protected_features"],
            "hidden_layers": hidden_layers,
            "training_history": {
                "final_train_loss": history["train_losses"][-1],
                "final_val_loss": history["val_losses"][-1],
                "final_train_acc": history["train_accuracies"][-1],
                "final_val_acc": history["val_accuracies"][-1],
                "epochs_trained": len(history["train_losses"]),
            },
            "dataset_info": {
                "num_features": data_info["input_dim"],
                "num_train": num_train,
                "num_val": num_val,
                "num_test": num_test,
                "num_total": num_train + num_val + num_test,
                "class_distribution": class_distribution,
                "feature_names": data_info["feature_names"],
                "protected_attr_info": protected_attr_info,
            },
        }

        # Save to cache for future runs
        try:
            data_tensors = _extract_tensors_from_loaders(data_info)
            model_config = {
                "input_dim": data_info["input_dim"],
                "hidden_layers": hidden_layers,
                "protected_indices": data_info["protected_indices"],
                "dropout_rate": 0.2,
            }
            preprocessor_state = {
                "scalers": preprocessor.scalers,
                "label_encoders": preprocessor.label_encoders,
                "feature_names": preprocessor.feature_names,
                "protected_feature_names": preprocessor.protected_feature_names,
            }
            save_to_cache(
                request.file_path, cache_key, current_model,
                model_config, preprocessor_state, data_tensors,
                training_response,
            )
        except Exception as cache_err:
            print(f"Warning: Failed to save model cache: {cache_err}")

        return training_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
async def analyze_bias(request: AnalyzeRequest):
    """Analyze dataset for bias including group fairness metrics."""
    global current_analyzer, current_data_info, current_model

    if current_analyzer is None:
        raise HTTPException(status_code=400, detail="Train model first")

    try:
        # Get test set from current_data_info (already loaded during training)
        X_test, y_test = [], []
        for batch_X, batch_y in current_data_info["test_loader"]:
            X_test.append(batch_X)
            y_test.append(batch_y)
        X_test = torch.cat(X_test, dim=0)
        y_test = torch.cat(y_test, dim=0)

        # Get protected values - convert to simple Python floats
        protected_vals_dict = request.protected_values

        # Get the first protected feature's values
        first_key = list(protected_vals_dict.keys())[0]
        protected_vals = protected_vals_dict[first_key]

        # Convert to Python floats (not tensors!)
        protected_vals = [float(v) for v in protected_vals]

        print(f"Analyzing with protected values: {protected_vals}")

        # Batch analysis (QID metrics)
        batch_results = current_analyzer.batch_analyze(
            X_test, protected_vals, max_samples=min(request.max_samples, len(X_test))
        )

        # Group fairness metrics (Demographic Parity, Equalized Odds, Equal Opportunity)
        from analyzers.group_fairness import GroupFairnessAnalyzer

        gf_analyzer = GroupFairnessAnalyzer(
            current_model,
            protected_indices=current_data_info["protected_indices"],
            protected_feature_names=current_data_info["protected_features"],
            device="cpu",
        )
        group_fairness = gf_analyzer.compute_all(X_test, y_test)

        return {
            "status": "success",
            "qid_metrics": batch_results,
            "group_fairness": group_fairness,
        }

    except Exception as e:
        import traceback

        traceback.print_exc()  # Print full error to console
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_discriminatory(request: SearchRequest):
    """Search for discriminatory instances."""
    global current_search_engine, current_data_info

    if current_search_engine is None:
        raise HTTPException(status_code=400, detail="Train model first")

    try:
        # Get test data
        X_test = []
        for batch_X, _ in current_data_info["test_loader"]:
            X_test.append(batch_X)
        X_test = torch.cat(X_test, dim=0)

        # Get protected values
        protected_vals = list(request.protected_values.values())[0]
        protected_vals = [torch.tensor(v) for v in protected_vals]

        # Run search
        results = current_search_engine.search(
            X_test,
            protected_vals,
            num_global_iterations=request.num_iterations,
            num_local_neighbors=request.num_neighbors,
        )

        return {"status": "success", "search_results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug")
async def debug_bias(request: SearchRequest):
    """Perform causal debugging to localize biased layers/neurons."""
    global current_model, current_search_engine, current_data_info

    if current_model is None:
        raise HTTPException(status_code=400, detail="Train model first")

    try:
        # Get discriminatory instances from search
        X_test = []
        for batch_X, _ in current_data_info["test_loader"]:
            X_test.append(batch_X)
        X_test = torch.cat(X_test, dim=0)

        protected_vals = list(request.protected_values.values())[0]
        protected_vals = [torch.tensor(v) for v in protected_vals]

        # Search for discriminatory instances
        search_results = current_search_engine.search(
            X_test, protected_vals, num_global_iterations=50, num_local_neighbors=30
        )

        # Causal debugging
        debugger = CausalDebugger(current_model, device="cpu")

        layer_results = debugger.localize_biased_layer(
            search_results["discriminatory_instances"]
        )

        biased_layer_idx = layer_results["biased_layer"]["layer_idx"]

        neuron_results = debugger.localize_biased_neurons(
            biased_layer_idx, search_results["discriminatory_instances"]
        )

        return {
            "status": "success",
            "layer_analysis": layer_results,
            "neuron_analysis": neuron_results,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/activations")
async def get_activations(request: ActivationsRequest):
    """Compute internal space visualization data (layer activations reduced to 2D)."""
    global current_model, current_data_info

    if current_model is None:
        raise HTTPException(status_code=400, detail="Train model first")

    try:
        from analyzers.internal_space import InternalSpaceAnalyzer
        import numpy as np

        analyzer = InternalSpaceAnalyzer(current_model, device="cpu")

        # Get test data
        X_test, y_test = [], []
        for batch_X, batch_y in current_data_info["test_loader"]:
            X_test.append(batch_X)
            y_test.append(batch_y)
        X_test = torch.cat(X_test, dim=0)
        y_test = torch.cat(y_test, dim=0).numpy()

        # Get protected attribute values for coloring
        protected_idx = (
            current_data_info["protected_indices"][0]
            if current_data_info["protected_indices"]
            else 0
        )
        protected_vals = X_test[:, protected_idx].numpy()

        result = analyzer.compute_visualization_data(
            X_test,
            y_test,
            protected_vals,
            method=request.method,
            max_samples=request.max_samples,
        )

        return {"status": "success", "activations": result}

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
async def explain_model(request: ExplainRequest):
    """Compute LIME and/or SHAP explanations."""
    global current_model, current_data_info

    if current_model is None:
        raise HTTPException(status_code=400, detail="Train model first")

    try:
        from analyzers.explainability import ExplainabilityAnalyzer

        analyzer = ExplainabilityAnalyzer(
            current_model,
            feature_names=current_data_info["feature_names"],
            device="cpu",
        )

        # Get train and test data as numpy
        X_train_list, X_test_list = [], []
        for batch_X, _ in current_data_info["train_loader"]:
            X_train_list.append(batch_X)
        for batch_X, _ in current_data_info["test_loader"]:
            X_test_list.append(batch_X)

        X_train = torch.cat(X_train_list, dim=0).numpy()
        X_test = torch.cat(X_test_list, dim=0).numpy()

        # Select instances to explain (first N from test set)
        X_explain = X_test[: request.num_instances]

        result = {}

        if request.method in ("shap", "both"):
            result["shap"] = analyzer.compute_shap_values(
                X_train,
                X_explain,
                max_background=request.max_background,
            )

        if request.method in ("lime", "both"):
            result["lime"] = analyzer.compute_lime_explanations(
                X_train,
                X_explain,
            )

        return {"status": "success", "explanations": result}

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain-instance")
async def explain_single_instance(request: ExplainInstanceRequest):
    """Compute LIME explanation for a single instance (by test index or custom feature values)."""
    global current_model, current_data_info

    if current_model is None:
        raise HTTPException(status_code=400, detail="Train model first")

    try:
        from analyzers.explainability import ExplainabilityAnalyzer
        import numpy as np

        feature_names = current_data_info["feature_names"]

        # Get X_train for LIME background
        X_train_list = []
        for batch_X, _ in current_data_info["train_loader"]:
            X_train_list.append(batch_X)
        X_train = torch.cat(X_train_list, dim=0).numpy()

        # Get the instance to explain
        if request.instance_type == "index":
            if request.instance_index is None:
                raise HTTPException(status_code=400, detail="instance_index is required for type 'index'")

            X_test_list = []
            for batch_X, _ in current_data_info["test_loader"]:
                X_test_list.append(batch_X)
            X_test = torch.cat(X_test_list, dim=0).numpy()

            if request.instance_index < 0 or request.instance_index >= len(X_test):
                raise HTTPException(
                    status_code=400,
                    detail=f"instance_index {request.instance_index} out of range [0, {len(X_test) - 1}]"
                )

            instance = X_test[request.instance_index:request.instance_index + 1]
        elif request.instance_type == "custom":
            if request.feature_values is None:
                raise HTTPException(status_code=400, detail="feature_values is required for type 'custom'")

            expected_len = len(feature_names)
            if len(request.feature_values) != expected_len:
                raise HTTPException(
                    status_code=400,
                    detail=f"Expected {expected_len} feature values, got {len(request.feature_values)}"
                )

            instance = np.array([request.feature_values], dtype=np.float32)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown instance_type: {request.instance_type}")

        # Compute LIME for this single instance
        analyzer = ExplainabilityAnalyzer(current_model, feature_names, device="cpu")
        result = analyzer.compute_lime_explanations(X_train, instance)

        return {
            "status": "success",
            "explanation": result["explanations"][0],
            "feature_names": result["feature_names"],
            "feature_values": instance[0].tolist(),
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8765)
