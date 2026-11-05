"""
08_adversarial_training.py

Purpose:
    - Load clean baseline DNN checkpoints from 07_train_baselines.py.
    - Load the completed 06_preprocess_and_audit model-input arrays.
    - Train PGD-adversarially trained DNNs for NSL-KDD and CICIDS2017.
    - Evaluate clean validation and test performance.
    - Save defended checkpoints, metrics, predictions, configuration, and hashes.
    - Create an adversarial-training run-record Excel workbook.

This script does not perform distillation or final attack evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

PROJECT_ROOT = Path(r"E:\Thesis")

PREPROCESSING_ROOT = PROJECT_ROOT / "reports" / "preprocessing"
BASELINE_ROOT = PROJECT_ROOT / "reports" / "baselines"
OUTPUT_ROOT = PROJECT_ROOT / "reports" / "adversarial_training"

RANDOM_STATE = 42

PGD_EPSILON = 0.10
PGD_ALPHA = 0.02
PGD_STEPS = 10
PGD_BATCH_SIZE = 1024

ADVERSARIAL_EPOCHS = 30
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-5
PATIENCE = 5

DNN_HIDDEN_DIMS = (256, 128, 64)

DATASETS = ["NSL-KDD", "CICIDS2017"]
REQUIRED_ANNOTATION_EXCLUSIONS = {"difficulty", "Attempted Category"}


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


def resolve_preprocessing_dir(explicit_dir: str | None) -> Path:
    if explicit_dir:
        selected = Path(explicit_dir)
        if not selected.is_absolute():
            selected = PROJECT_ROOT / selected
    else:
        candidates = sorted(
            [p for p in PREPROCESSING_ROOT.glob("preprocessing_*") if p.is_dir()],
            key=lambda p: p.name,
        )
        if not candidates:
            fail(f"No preprocessing folders found under: {PREPROCESSING_ROOT}")
        selected = candidates[-1]

    report = load_json(selected / "preprocessing_report.json")

    if not report.get("preprocessing_completed", False):
        fail(f"Selected preprocessing folder is not completed: {selected}")

    if not report.get("ready_for_model_training", False):
        fail(f"Selected preprocessing folder is not ready for training: {selected}")

    excluded = set(
        report.get("configuration", {}).get("annotation_excluded_columns", [])
    )

    if excluded != REQUIRED_ANNOTATION_EXCLUSIONS:
        fail(
            "Selected preprocessing output does not exclude required annotation "
            f"columns.\nFolder: {selected}\nExcluded: {sorted(excluded)}"
        )

    log(f"Using preprocessing output: {selected}")
    return selected


def resolve_baseline_dir(
    explicit_dir: str | None,
    preprocessing_dir: Path,
    preprocessing_hash: str | None,
) -> Path:
    if explicit_dir:
        selected = Path(explicit_dir)
        if not selected.is_absolute():
            selected = PROJECT_ROOT / selected
    else:
        candidates = sorted(
            [p for p in BASELINE_ROOT.glob("baselines_*") if p.is_dir()],
            key=lambda p: p.name,
        )
        if not candidates:
            fail(f"No baseline folders found under: {BASELINE_ROOT}")
        selected = candidates[-1]

    report_path = selected / "baseline_results.json"
    if not report_path.exists():
        fail(f"Selected baseline folder has no results report: {report_path}")

    report = load_json(report_path)

    if not report.get("baselines_completed", False):
        fail(f"Selected baseline folder is not completed: {selected}")

    if report.get("preprocessing_dir") != str(preprocessing_dir):
        fail(
            "Selected baseline folder was not trained on the selected "
            "preprocessing output.\n"
            f"Baseline preprocessing: {report.get('preprocessing_dir')}\n"
            f"Selected preprocessing:  {preprocessing_dir}"
        )

    if preprocessing_hash and report.get("preprocessing_report_sha256") != preprocessing_hash:
        fail(
            "Baseline preprocessing hash does not match the selected "
            "preprocessing report hash."
        )

    log(f"Using baseline output: {selected}")
    return selected


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


def load_clean_dnn(
    checkpoint_path: Path,
    expected_input_dim: int,
    device: torch.device,
) -> BaselineDNN:
    if not checkpoint_path.exists():
        fail(f"Clean baseline DNN checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if int(checkpoint["input_dim"]) != expected_input_dim:
        fail(
            "Baseline checkpoint input dimension does not match preprocessing "
            "features.\n"
            f"Checkpoint input dim: {checkpoint['input_dim']}\n"
            f"Expected input dim:   {expected_input_dim}\n"
            f"Checkpoint: {checkpoint_path}"
        )

    model = BaselineDNN(
        int(checkpoint["input_dim"]),
        tuple(checkpoint["hidden_dims"]),
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    return model


def pgd_attack(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
) -> torch.Tensor:
    """Untargeted PGD attack in the standardized feature space."""

    was_training = model.training
    model.eval()

    X_adv = X.clone().detach()
    X_adv = X_adv + torch.empty_like(X_adv).uniform_(-epsilon, epsilon)
    X_adv = torch.clamp(X_adv, X - epsilon, X + epsilon)

    criterion = nn.BCEWithLogitsLoss()

    for _ in range(steps):
        X_adv.requires_grad_(True)

        logits = model(X_adv)
        loss = criterion(logits, y)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            gradient = X_adv.grad
            X_adv = X_adv + alpha * gradient.sign()
            X_adv = torch.clamp(X_adv, X - epsilon, X + epsilon)

    if was_training:
        model.train()

    return X_adv.detach()


def train_adversarial_dnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    clean_model: BaselineDNN,
    device: torch.device,
    checkpoint_path: Path,
) -> tuple[BaselineDNN, dict]:
    set_seed(RANDOM_STATE)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_validation_tensor = torch.tensor(X_validation, dtype=torch.float32)
    y_validation_tensor = torch.tensor(y_validation, dtype=torch.float32)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=PGD_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    model = BaselineDNN(
        X_train.shape[1],
        DNN_HIDDEN_DIMS,
    ).to(device)

    model.load_state_dict(clean_model.state_dict())

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_validation_auc = -np.inf
    best_epoch = 0
    patience_counter = 0
    best_state = None

    for epoch in range(1, ADVERSARIAL_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            X_adv_batch = pgd_attack(
                model,
                X_batch,
                y_batch,
                PGD_EPSILON,
                PGD_ALPHA,
                PGD_STEPS,
            )

            optimizer.zero_grad()

            clean_logits = model(X_batch)
            adversarial_logits = model(X_adv_batch)

            clean_loss = criterion(clean_logits, y_batch)
            adversarial_loss = criterion(adversarial_logits, y_batch)
            loss = 0.5 * clean_loss + 0.5 * adversarial_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(X_batch)

        epoch_loss /= len(train_dataset)

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
            f"Adversarial training loss: {epoch_loss:.8f} | "
            f"Clean validation AUC: {validation_auc:.8f}"
        )

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        fail("Adversarial training did not produce a valid best model state.")

    model.load_state_dict(best_state)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": X_train.shape[1],
            "hidden_dims": DNN_HIDDEN_DIMS,
            "best_epoch": best_epoch,
            "best_clean_validation_auc": best_validation_auc,
            "pgd_epsilon": PGD_EPSILON,
            "pgd_alpha": PGD_ALPHA,
            "pgd_steps": PGD_STEPS,
            "random_state": RANDOM_STATE,
        },
        checkpoint_path,
    )

    history = {
        "best_epoch": best_epoch,
        "best_clean_validation_auc": float(best_validation_auc),
        "epochs_requested": ADVERSARIAL_EPOCHS,
        "early_stopping_patience": PATIENCE,
    }

    return model, history


def predict(
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


def write_adversarial_workbook(
    output_dir: Path,
    all_results: list[dict],
    artifact_hashes: dict[str, str],
    preprocessing_hash: str | None,
) -> Path:
    excel_path = output_dir / "Thesis_Adversarial_Training_Run_Record.xlsx"

    run_rows = []
    metric_rows = []
    hash_rows = []

    for result in all_results:
        run_id = f"{result['dataset']}_adversarial_DNN"

        run_rows.append(
            {
                "Run ID": run_id,
                "Dataset": result["dataset"],
                "Model": "Adversarially trained DNN",
                "Run Type": "PGD adversarial training",
                "Script": "08_adversarial_training.py",
                "Random Seed": result["random_state"],
                "Train Records": result["train_records"],
                "Validation Records": result["validation_records"],
                "Test Records": result["test_records"],
                "Input Features": result["input_features"],
                "Clean Baseline Checkpoint": result["clean_checkpoint"],
                "Adversarial Checkpoint": result["checkpoint"],
                "PGD Epsilon": result["pgd_epsilon"],
                "PGD Alpha": result["pgd_alpha"],
                "PGD Steps": result["pgd_steps"],
                "Preprocessing Report SHA-256": preprocessing_hash,
                "Status": "Completed",
                "Notes": "",
            }
        )

        for split in ["validation_metrics", "test_metrics"]:
            metric_rows.append(
                {
                    "Run ID": run_id,
                    "Dataset": result["dataset"],
                    "Model": "Adversarially trained DNN",
                    "Split": split.replace("_metrics", ""),
                    **result[split],
                }
            )

    for artifact, artifact_hash in sorted(artifact_hashes.items()):
        hash_rows.append({"Artifact": artifact, "SHA-256": artifact_hash})

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.DataFrame(run_rows).to_excel(writer, sheet_name="Run Index", index=False)
        pd.DataFrame(metric_rows).to_excel(
            writer, sheet_name="Clean Metrics", index=False
        )
        pd.DataFrame(hash_rows).to_excel(
            writer, sheet_name="Artifact Hashes", index=False
        )

    return excel_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script 08: PGD adversarial training from clean baseline DNNs"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        choices=DATASETS,
        help="Datasets to train",
    )
    parser.add_argument(
        "--preprocessing-dir",
        type=str,
        default=None,
        help="Optional specific 06 preprocessing output folder",
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default=None,
        help="Optional specific 07 baseline output folder",
    )
    args = parser.parse_args()

    started_utc = utc_now()
    log("Starting 08_adversarial_training")

    preprocessing_dir = resolve_preprocessing_dir(args.preprocessing_dir)
    preprocessing_report = load_json(
        preprocessing_dir / "preprocessing_report.json"
    )
    preprocessing_hash = preprocessing_report.get("preprocessing_report_sha256")

    if not preprocessing_hash:
        fail("Preprocessing report does not contain preprocessing_report_sha256.")

    baseline_dir = resolve_baseline_dir(
        args.baseline_dir,
        preprocessing_dir,
        preprocessing_hash,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = OUTPUT_ROOT / f"adversarial_training_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    log(f"Using device: {device}")

    all_results = []
    artifact_hashes = {}

    for dataset in args.datasets:
        log(f"Training adversarial DNN for {dataset}")

        dataset_key = dataset.lower().replace("-", "_")
        preprocessing_dataset_dir = preprocessing_dir / dataset_key
        baseline_dataset_dir = baseline_dir / dataset_key
        dataset_output_dir = output_dir / dataset_key
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
            path = preprocessing_dataset_dir / filename
            if not path.exists():
                fail(f"Required preprocessing artifact not found: {path}")

        X_train = np.load(preprocessing_dataset_dir / "X_train.npy")
        X_validation = np.load(preprocessing_dataset_dir / "X_validation.npy")
        X_test = np.load(preprocessing_dataset_dir / "X_test.npy")

        y_train = np.load(preprocessing_dataset_dir / "y_train.npy").astype(np.int8)
        y_validation = np.load(
            preprocessing_dataset_dir / "y_validation.npy"
        ).astype(np.int8)
        y_test = np.load(preprocessing_dataset_dir / "y_test.npy").astype(np.int8)

        feature_schema = load_json(
            preprocessing_dataset_dir / "feature_names.json"
        )
        feature_names = feature_schema["ordered_model_input_features"]

        excluded_annotation_columns = set(
            feature_schema.get("excluded_annotation_columns", [])
        )

        if excluded_annotation_columns != REQUIRED_ANNOTATION_EXCLUSIONS:
            fail(
                f"{dataset}: preprocessing schema does not confirm required "
                "annotation-column exclusions."
            )

        leaked_columns = sorted(
            set(feature_names).intersection(REQUIRED_ANNOTATION_EXCLUSIONS)
        )

        if leaked_columns:
            fail(
                f"{dataset}: leaked annotation columns found in model inputs: "
                f"{leaked_columns}"
            )

        if X_train.shape[1] != len(feature_names):
            fail(
                f"{dataset}: X_train has {X_train.shape[1]} features, "
                f"but feature schema has {len(feature_names)}."
            )

        clean_checkpoint = baseline_dataset_dir / "dnn_baseline.pt"
        clean_model = load_clean_dnn(
            clean_checkpoint,
            X_train.shape[1],
            device,
        )

        adversarial_checkpoint = dataset_output_dir / "dnn_adversarial.pt"

        model, history = train_adversarial_dnn(
            X_train,
            y_train,
            X_validation,
            y_validation,
            clean_model,
            device,
            adversarial_checkpoint,
        )

        validation_pred, validation_score = predict(
            model, X_validation, device
        )
        test_pred, test_score = predict(model, X_test, device)

        validation_metrics = compute_metrics(
            y_validation, validation_pred, validation_score
        )
        test_metrics = compute_metrics(y_test, test_pred, test_score)

        np.savez_compressed(
            dataset_output_dir / "predictions.npz",
            validation_predictions=validation_pred,
            validation_scores=validation_score,
            test_predictions=test_pred,
            test_scores=test_score,
        )

        result = {
            "dataset": dataset,
            "model": "adversarial_dnn",
            "random_state": RANDOM_STATE,
            "train_records": int(len(X_train)),
            "validation_records": int(len(X_validation)),
            "test_records": int(len(X_test)),
            "input_features": int(X_train.shape[1]),
            "clean_checkpoint": str(clean_checkpoint.relative_to(baseline_dir)),
            "checkpoint": str(adversarial_checkpoint.relative_to(output_dir)),
            "pgd_epsilon": PGD_EPSILON,
            "pgd_alpha": PGD_ALPHA,
            "pgd_steps": PGD_STEPS,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "training_history": history,
        }

        all_results.append(result)

        print(f"\n{dataset} | Adversarially trained DNN")
        print("Clean validation:")
        print(format_metrics(validation_metrics))
        print("Clean test:")
        print(format_metrics(test_metrics))

    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            artifact_hashes[str(path.relative_to(output_dir))] = sha256_file(path)

    results_path = output_dir / "adversarial_training_results.json"
    metrics_csv_path = output_dir / "adversarial_training_metrics.csv"

    metrics_rows = []

    for result in all_results:
        for split in ["validation_metrics", "test_metrics"]:
            metrics_rows.append(
                {
                    "dataset": result["dataset"],
                    "model": result["model"],
                    "split": split.replace("_metrics", ""),
                    **result[split],
                }
            )

    pd.DataFrame(metrics_rows).to_csv(metrics_csv_path, index=False)

    excel_path = write_adversarial_workbook(
        output_dir=output_dir,
        all_results=all_results,
        artifact_hashes=artifact_hashes,
        preprocessing_hash=preprocessing_hash,
    )
    log(f"Wrote adversarial-training run-record workbook: {excel_path}")

    final_report = {
        "script": "08_adversarial_training",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "preprocessing_dir": str(preprocessing_dir),
        "preprocessing_report_sha256": preprocessing_hash,
        "baseline_dir": str(baseline_dir),
        "random_state": RANDOM_STATE,
        "datasets": args.datasets,
        "pgd_epsilon": PGD_EPSILON,
        "pgd_alpha": PGD_ALPHA,
        "pgd_steps": PGD_STEPS,
        "results": all_results,
        "artifact_hashes": artifact_hashes,
        "excel_workbook": str(excel_path),
        "adversarial_training_completed": True,
        "ready_for_distillation": True,
    }

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    report_hash = sha256_file(results_path)
    final_report["adversarial_training_results_sha256"] = report_hash

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print("\n============ 08 ADVERSARIAL TRAINING SUMMARY ============")

    for result in all_results:
        print(
            f"\n{result['dataset']} | Adversarially trained DNN | "
            f"Clean test accuracy: {result['test_metrics']['accuracy']:.8f} | "
            f"Clean test F1: {result['test_metrics']['f1']:.8f} | "
            f"Clean test FPR: {result['test_metrics']['fpr']:.8f} | "
            f"Clean test FNR: {result['test_metrics']['fnr']:.8f}"
        )

    print("\n=========================================================")
    print("[PASS] 08_adversarial_training completed successfully.")
    print(f"Results: {results_path}")
    print(f"Metrics CSV: {metrics_csv_path}")
    print(f"Excel run record: {excel_path}")
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()