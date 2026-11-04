"""
07_train_baselines.py

Purpose:
    - Load artifacts produced by 06_preprocess_and_audit.py.
    - Train clean baseline LR, RF, and DNN models.
    - Evaluate validation and test partitions.
    - Save metrics, predictions, checkpoints, configuration, and hashes.
    - Create a thesis baseline run-record Excel workbook.

This script does not perform adversarial training, distillation, or attacks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(r"E:\Thesis")

PREPROCESSING_DIR = (
    PROJECT_ROOT
    / "reports"
    / "preprocessing"
    / "preprocessing_20261104T040755_364187Z"
)

PREPROCESSING_REPORT = PREPROCESSING_DIR / "preprocessing_report.json"

OUTPUT_ROOT = PROJECT_ROOT / "reports" / "baselines"

RANDOM_STATE = 42
MAX_ITER = 2000
RF_TREES = 300
RF_JOBS = -1

DNN_EPOCHS = 100
DNN_BATCH_SIZE = 1024
DNN_LEARNING_RATE = 0.001
DNN_WEIGHT_DECAY = 1e-5
DNN_PATIENCE = 10
DNN_HIDDEN_DIMS = (256, 128, 64)

DATASETS = ["NSL-KDD", "CICIDS2017"]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


def log(message: str) -> None:
    print(f"[INFO] {message}")


def load_json(path: Path) -> dict:
    if not path.exists():
        fail(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def format_metrics(metrics: dict) -> str:
    return (
        f"Accuracy:    {metrics['accuracy']:.8f}\n"
        f"Precision:   {metrics['precision']:.8f}\n"
        f"Recall:      {metrics['recall']:.8f}\n"
        f"Specificity: {metrics['specificity']:.8f}\n"
        f"FPR:         {metrics['fpr']:.8f}\n"
        f"FNR:         {metrics['fnr']:.8f}\n"
        f"F1-Score:    {metrics['f1']:.8f}\n"
        f"ROC-AUC:     {metrics['roc_auc']:.8f}\n"
        f"TP/FP/TN/FN: {metrics['tp']}/{metrics['fp']}/{metrics['tn']}/{metrics['fn']}"
    )


def write_results_workbook(
    output_dir: Path,
    all_results: list[dict],
    artifact_hashes: dict[str, str],
    preprocessing_hash: str | None,
) -> Path:
    """Create the thesis baseline run-record workbook."""

    excel_path = output_dir / "Thesis_Baseline_Run_Record.xlsx"

    run_rows = []
    metric_rows = []
    configuration_rows = []

    for result in all_results:
        run_id = f"{result['dataset']}_{result['model']}"

        run_rows.append(
            {
                "Run ID": run_id,
                "Dataset": result["dataset"],
                "Model": result["model"],
                "Run Type": "Clean baseline",
                "Script": "07_train_baselines.py",
                "Random Seed": result["random_state"],
                "Train Records": result["train_records"],
                "Validation Records": result["validation_records"],
                "Test Records": result["test_records"],
                "Input Features": result["input_features"],
                "Checkpoint": result["checkpoint"],
                "Preprocessing Report SHA-256": preprocessing_hash,
                "Status": "Completed",
                "Notes": "",
            }
        )

        for split in ["validation_metrics", "test_metrics"]:
            metrics = result[split]

            metric_rows.append(
                {
                    "Run ID": run_id,
                    "Dataset": result["dataset"],
                    "Model": result["model"],
                    "Split": split.replace("_metrics", ""),
                    "Accuracy": metrics["accuracy"],
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "Specificity": metrics["specificity"],
                    "FPR": metrics["fpr"],
                    "FNR": metrics["fnr"],
                    "F1-Score": metrics["f1"],
                    "ROC-AUC": metrics["roc_auc"],
                    "TP": metrics["tp"],
                    "FP": metrics["fp"],
                    "TN": metrics["tn"],
                    "FN": metrics["fn"],
                }
            )

        configuration_rows.append(
            {
                "Run ID": run_id,
                "Dataset": result["dataset"],
                "Model": result["model"],
                "DNN Best Epoch": (
                    result["dnn_history"]["best_epoch"]
                    if result["dnn_history"]
                    else None
                ),
                "DNN Best Validation AUC": (
                    result["dnn_history"]["best_validation_auc"]
                    if result["dnn_history"]
                    else None
                ),
            }
        )

    hash_rows = [
        {
            "Artifact": artifact,
            "SHA-256": artifact_hash,
        }
        for artifact, artifact_hash in sorted(artifact_hashes.items())
    ]

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.DataFrame(run_rows).to_excel(
            writer,
            sheet_name="Run Index",
            index=False,
        )
        pd.DataFrame(metric_rows).to_excel(
            writer,
            sheet_name="Metrics",
            index=False,
        )
        pd.DataFrame(configuration_rows).to_excel(
            writer,
            sheet_name="Configuration",
            index=False,
        )
        pd.DataFrame(hash_rows).to_excel(
            writer,
            sheet_name="Artifact Hashes",
            index=False,
        )

    return excel_path


# ---------------------------------------------------------------------------
# DNN
# ---------------------------------------------------------------------------

class BaselineDNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()

        layers = []
        previous = input_dim

        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            previous = hidden

        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def train_dnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    input_dim: int,
    device: torch.device,
    checkpoint_path: Path,
) -> tuple[BaselineDNN, dict]:
    set_seed(RANDOM_STATE)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_validation_tensor = torch.tensor(X_validation, dtype=torch.float32)
    y_validation_tensor = torch.tensor(y_validation, dtype=torch.float32)

    train_dataset = torch.utils.data.TensorDataset(
        X_train_tensor,
        y_train_tensor,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=DNN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    model = BaselineDNN(input_dim, DNN_HIDDEN_DIMS).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=DNN_LEARNING_RATE,
        weight_decay=DNN_WEIGHT_DECAY,
    )

    best_validation_auc = -np.inf
    best_epoch = 0
    patience_counter = 0
    best_state = None

    for epoch in range(1, DNN_EPOCHS + 1):
        model.train()
        training_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            training_loss += loss.item() * len(X_batch)

        training_loss /= len(train_dataset)

        model.eval()
        with torch.no_grad():
            validation_logits = model(X_validation_tensor.to(device))
            validation_scores = torch.sigmoid(validation_logits).cpu().numpy()

        validation_auc = roc_auc_score(y_validation, validation_scores)

        if validation_auc > best_validation_auc:
            best_validation_auc = validation_auc
            best_epoch = epoch
            patience_counter = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            patience_counter += 1

        print(
            f"  Epoch {epoch:03d} | "
            f"Train loss: {training_loss:.8f} | "
            f"Validation AUC: {validation_auc:.8f}"
        )

        if patience_counter >= DNN_PATIENCE:
            print(f"  Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        fail("DNN training did not produce a valid best model state.")

    model.load_state_dict(best_state)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dims": DNN_HIDDEN_DIMS,
            "best_epoch": best_epoch,
            "best_validation_auc": best_validation_auc,
            "random_state": RANDOM_STATE,
        },
        checkpoint_path,
    )

    history = {
        "best_epoch": best_epoch,
        "best_validation_auc": float(best_validation_auc),
        "epochs_requested": DNN_EPOCHS,
        "early_stopping_patience": DNN_PATIENCE,
    }

    return model, history


def dnn_predict(
    model: BaselineDNN,
    X: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        scores = torch.sigmoid(logits).cpu().numpy()

    predictions = (scores >= 0.5).astype(np.int8)
    return predictions, scores


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script 07: clean baseline training from 06_preprocess_and_audit artifacts"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        choices=DATASETS,
        help="Datasets to train",
    )
    args = parser.parse_args()

    started_utc = utc_now()
    log("Starting 07_train_baselines")

    if not PREPROCESSING_DIR.exists():
        fail(
            f"06_preprocess_and_audit output folder not found: {PREPROCESSING_DIR}"
        )

    preprocessing_report = load_json(PREPROCESSING_REPORT)

    if not preprocessing_report.get("preprocessing_completed", False):
        fail(
            "06_preprocess_and_audit report is not marked preprocessing_completed."
        )

    if not preprocessing_report.get("ready_for_model_training", False):
        fail(
            "06_preprocess_and_audit report is not marked ready_for_model_training."
        )

    preprocessing_hash = preprocessing_report.get("preprocessing_report_sha256")

    if preprocessing_hash:
        log(
            "06_preprocess_and_audit report self-hash recorded; skipped direct "
            "file-hash verification because the report contains its own hash."
        )
    else:
        fail(
            "06_preprocess_and_audit report does not contain "
            "preprocessing_report_sha256."
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = OUTPUT_ROOT / f"baselines_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    log(f"Using device: {device}")

    all_results = []
    artifact_hashes = {}

    for dataset in args.datasets:
        log(f"Training baselines for {dataset}")

        dataset_dir = PREPROCESSING_DIR / dataset.lower().replace("-", "_")
        dataset_output_dir = output_dir / dataset.lower().replace("-", "_")
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        required_files = [
            "X_train.npy",
            "X_validation.npy",
            "X_test.npy",
            "y_train.npy",
            "y_validation.npy",
            "y_test.npy",
            "feature_names.json",
        ]

        for filename in required_files:
            path = dataset_dir / filename
            if not path.exists():
                fail(f"Required preprocessing artifact not found: {path}")

        X_train = np.load(dataset_dir / "X_train.npy")
        X_validation = np.load(dataset_dir / "X_validation.npy")
        X_test = np.load(dataset_dir / "X_test.npy")

        y_train = np.load(dataset_dir / "y_train.npy").astype(np.int8)
        y_validation = np.load(dataset_dir / "y_validation.npy").astype(np.int8)
        y_test = np.load(dataset_dir / "y_test.npy").astype(np.int8)

        feature_schema = load_json(dataset_dir / "feature_names.json")
        feature_names = feature_schema["ordered_model_input_features"]

        if X_train.shape[1] != len(feature_names):
            fail(
                f"{dataset}: X_train has {X_train.shape[1]} features, "
                f"but feature schema has {len(feature_names)}."
            )

        log(
            f"{dataset}: train={len(X_train)}, "
            f"validation={len(X_validation)}, test={len(X_test)}, "
            f"features={X_train.shape[1]}"
        )

        models = {}

        log(f"{dataset}: training LogisticRegression")
        lr = LogisticRegression(
            max_iter=MAX_ITER,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )
        lr.fit(X_train, y_train)
        models["logistic_regression"] = lr

        log(f"{dataset}: training RandomForestClassifier")
        rf = RandomForestClassifier(
            n_estimators=RF_TREES,
            random_state=RANDOM_STATE,
            n_jobs=RF_JOBS,
        )
        rf.fit(X_train, y_train)
        models["random_forest"] = rf

        log(f"{dataset}: training BaselineDNN")
        dnn_checkpoint = dataset_output_dir / "dnn_baseline.pt"
        dnn, dnn_history = train_dnn(
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_train.shape[1],
            device,
            dnn_checkpoint,
        )
        models["dnn"] = dnn

        for model_name, model in models.items():
            log(f"{dataset}: evaluating {model_name}")

            if model_name == "dnn":
                validation_pred, validation_score = dnn_predict(
                    model,
                    X_validation,
                    device,
                )
                test_pred, test_score = dnn_predict(
                    model,
                    X_test,
                    device,
                )
            else:
                validation_pred = model.predict(X_validation).astype(np.int8)
                validation_score = model.predict_proba(X_validation)[:, 1]
                test_pred = model.predict(X_test).astype(np.int8)
                test_score = model.predict_proba(X_test)[:, 1]

            validation_metrics = compute_metrics(
                y_validation,
                validation_pred,
                validation_score,
            )
            test_metrics = compute_metrics(
                y_test,
                test_pred,
                test_score,
            )

            model_output_dir = dataset_output_dir / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(
                model_output_dir / "predictions.npz",
                validation_predictions=validation_pred,
                validation_scores=validation_score,
                test_predictions=test_pred,
                test_scores=test_score,
            )

            if model_name != "dnn":
                model_path = model_output_dir / "model.joblib"
                joblib.dump(model, model_path)

            result = {
                "dataset": dataset,
                "model": model_name,
                "random_state": RANDOM_STATE,
                "train_records": int(len(X_train)),
                "validation_records": int(len(X_validation)),
                "test_records": int(len(X_test)),
                "input_features": int(X_train.shape[1]),
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "dnn_history": dnn_history if model_name == "dnn" else None,
                "checkpoint": (
                    str(dnn_checkpoint.relative_to(output_dir))
                    if model_name == "dnn"
                    else str(
                        (model_output_dir / "model.joblib").relative_to(output_dir)
                    )
                ),
            }

            all_results.append(result)

            print(f"\n{dataset} | {model_name}")
            print("Validation:")
            print(format_metrics(validation_metrics))
            print("Test:")
            print(format_metrics(test_metrics))

        configuration = {
            "dataset": dataset,
            "preprocessing_dir": str(PREPROCESSING_DIR),
            "preprocessing_report_sha256": preprocessing_hash,
            "feature_count": int(X_train.shape[1]),
            "feature_names": feature_names,
            "random_state": RANDOM_STATE,
            "logistic_regression": {
                "max_iter": MAX_ITER,
                "solver": "lbfgs",
            },
            "random_forest": {
                "n_estimators": RF_TREES,
                "n_jobs": RF_JOBS,
            },
            "dnn": {
                "hidden_dims": list(DNN_HIDDEN_DIMS),
                "epochs": DNN_EPOCHS,
                "batch_size": DNN_BATCH_SIZE,
                "learning_rate": DNN_LEARNING_RATE,
                "weight_decay": DNN_WEIGHT_DECAY,
                "patience": DNN_PATIENCE,
            },
        }

        with (dataset_output_dir / "configuration.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(configuration, f, indent=2)

    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            artifact_hashes[str(path.relative_to(output_dir))] = sha256_file(path)

    results_path = output_dir / "baseline_results.json"
    metrics_csv_path = output_dir / "baseline_metrics.csv"

    metrics_rows = []

    for result in all_results:
        for split in ["validation_metrics", "test_metrics"]:
            row = {
                "dataset": result["dataset"],
                "model": result["model"],
                "split": split.replace("_metrics", ""),
                **result[split],
            }
            metrics_rows.append(row)

    pd.DataFrame(metrics_rows).to_csv(metrics_csv_path, index=False)

    excel_path = write_results_workbook(
        output_dir=output_dir,
        all_results=all_results,
        artifact_hashes=artifact_hashes,
        preprocessing_hash=preprocessing_hash,
    )
    log(f"Wrote baseline run-record workbook: {excel_path}")

    final_report = {
        "script": "07_train_baselines",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "preprocessing_dir": str(PREPROCESSING_DIR),
        "preprocessing_report_sha256": preprocessing_hash,
        "random_state": RANDOM_STATE,
        "datasets": args.datasets,
        "results": all_results,
        "artifact_hashes": artifact_hashes,
        "excel_workbook": str(excel_path),
        "baselines_completed": True,
        "ready_for_adversarial_training": True,
    }

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    report_hash = sha256_file(results_path)
    final_report["baseline_results_sha256"] = report_hash

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print("\n================ 07 BASELINE SUMMARY ================")

    for result in all_results:
        print(
            f"\n{result['dataset']} | {result['model']} | "
            f"Test accuracy: {result['test_metrics']['accuracy']:.8f} | "
            f"Test F1: {result['test_metrics']['f1']:.8f} | "
            f"Test FPR: {result['test_metrics']['fpr']:.8f} | "
            f"Test FNR: {result['test_metrics']['fnr']:.8f}"
        )

    print("\n=====================================================")
    print("[PASS] 07_train_baselines completed successfully.")
    print(f"Results: {results_path}")
    print(f"Metrics CSV: {metrics_csv_path}")
    print(f"Excel run record: {excel_path}")
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()