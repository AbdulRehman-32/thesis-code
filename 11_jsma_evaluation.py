r"""
11_jsma_evaluation.py

JSMA adversarial evaluation for the thesis IDS pipeline.

Evaluates:
  1. Clean baseline DNN
  2. PGD adversarially trained DNN
  3. Defensive-distillation DNN

JSMA is computationally expensive, so this script uses a fixed random
sample of 1,000 test records per dataset. The same sample indices are
used for all three DNN variants.

Usage:
  python 11_jsma_evaluation.py
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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(r"E:\Thesis")

PREPROCESSING_ROOT = PROJECT_ROOT / "reports" / "preprocessing"
BASELINE_ROOT = PROJECT_ROOT / "reports" / "baselines"
ADVERSARIAL_ROOT = PROJECT_ROOT / "reports" / "adversarial_training"
DISTILLATION_ROOT = PROJECT_ROOT / "reports" / "distillation"
OUTPUT_ROOT = PROJECT_ROOT / "reports" / "jsma_evaluation"

SEED = 42
BATCH_SIZE = 512
DROPOUT = 0.2

JSMA_SAMPLES = 1000
MAX_MODIFIED_FEATURES = 30
MAX_ITERATIONS = 100
THETA = 0.1

DATASETS = ["NSL-KDD", "CICIDS2017"]
REQUIRED_ANNOTATION_EXCLUSIONS = {"difficulty", "Attempted Category"}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def timestamp() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%S_%fZ")


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


def set_seed(seed: int = SEED) -> None:
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


def resolve_latest_folder(
    root: Path,
    prefix: str,
    report_filename: str,
    completed_key: str,
) -> Path:
    candidates = sorted(
        [p for p in root.glob(f"{prefix}_*") if p.is_dir()],
        key=lambda p: p.name,
    )

    if not candidates:
        fail(f"No {prefix} folders found under: {root}")

    selected = candidates[-1]
    report = load_json(selected / report_filename)

    if not report.get(completed_key, False):
        fail(f"Selected {prefix} folder is not completed: {selected}")

    log(f"Using {prefix} output: {selected}")
    return selected


def verify_chain(
    preprocessing_dir: Path,
    preprocessing_hash: str | None,
    baseline_dir: Path,
    adversarial_dir: Path,
    distillation_dir: Path,
) -> None:
    baseline_report = load_json(baseline_dir / "baseline_results.json")
    adversarial_report = load_json(
        adversarial_dir / "adversarial_training_results.json"
    )
    distillation_report = load_json(
        distillation_dir / "distillation_results.json"
    )

    if baseline_report.get("preprocessing_dir") != str(preprocessing_dir):
        fail("Baseline folder was not trained on the selected preprocessing folder.")

    if adversarial_report.get("preprocessing_dir") != str(preprocessing_dir):
        fail(
            "Adversarial-training folder was not trained on the selected "
            "preprocessing folder."
        )

    if distillation_report.get("config", {}).get(
        "preprocessing_dir"
    ) != str(preprocessing_dir):
        fail(
            "Distillation folder was not trained on the selected preprocessing "
            "folder."
        )

    if adversarial_report.get("baseline_dir") != str(baseline_dir):
        fail("Adversarial-training folder does not match the selected baseline folder.")

    if distillation_report.get("config", {}).get(
        "baseline_dir"
    ) != str(baseline_dir):
        fail("Distillation folder does not match the selected baseline folder.")

    if preprocessing_hash:
        if baseline_report.get("preprocessing_report_sha256") != preprocessing_hash:
            fail("Baseline preprocessing hash does not match selected preprocessing.")

        if adversarial_report.get(
            "preprocessing_report_sha256"
        ) != preprocessing_hash:
            fail(
                "Adversarial-training preprocessing hash does not match selected "
                "preprocessing."
            )

        if distillation_report.get("config", {}).get(
            "preprocessing_report_sha256"
        ) != preprocessing_hash:
            fail(
                "Distillation preprocessing hash does not match selected "
                "preprocessing."
            )


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
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def format_metrics(metrics: dict) -> str:
    return (
        f"Accuracy: {metrics['accuracy']:.8f} | "
        f"F1: {metrics['f1']:.8f} | "
        f"FPR: {metrics['false_positive_rate']:.8f} | "
        f"FNR: {metrics['false_negative_rate']:.8f} | "
        f"ROC-AUC: {metrics['roc_auc']:.8f}"
    )


class EvaluationDNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()

        layers = []
        previous = input_dim

        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    nn.ReLU(),
                    nn.Dropout(DROPOUT),
                ]
            )
            previous = hidden

        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def load_model(
    checkpoint_path: Path,
    expected_input_dim: int,
    device: torch.device,
) -> EvaluationDNN:
    if not checkpoint_path.exists():
        fail(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if int(checkpoint["input_dim"]) != expected_input_dim:
        fail(
            "Checkpoint input dimension does not match preprocessing features.\n"
            f"Checkpoint input dim: {checkpoint['input_dim']}\n"
            f"Expected input dim:   {expected_input_dim}\n"
            f"Checkpoint: {checkpoint_path}"
        )

    model = EvaluationDNN(
        int(checkpoint["input_dim"]),
        tuple(checkpoint["hidden_dims"]),
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def make_loader(X: np.ndarray, y: np.ndarray) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


def evaluate(
    model: EvaluationDNN,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    loader = make_loader(X, y)

    probs = []
    preds = []
    labels = []

    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            scores = torch.sigmoid(logits).cpu().numpy()
            predictions = (scores >= 0.5).astype(np.int8)

            probs.extend(scores)
            preds.extend(predictions)
            labels.extend(yb.numpy())

    probs = np.asarray(probs)
    preds = np.asarray(preds)
    labels = np.asarray(labels).astype(np.int8)

    metrics = compute_metrics(labels, preds, probs)
    return metrics, probs, preds


def jsma_attack_single(
    model: EvaluationDNN,
    x_original: np.ndarray,
    y_true: int,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    """Apply JSMA to one standardized test sample."""

    x_original_tensor = torch.tensor(
        x_original,
        dtype=torch.float32,
        device=device,
    )

    x_adv = x_original_tensor.clone().detach()

    with torch.no_grad():
        original_logit = model(x_adv.unsqueeze(0)).item()
        original_prediction = int(original_logit >= 0.0)

    # Already misclassified: no perturbation is required.
    if original_prediction != y_true:
        return x_adv.cpu().numpy(), 0

    modified = np.zeros(x_original.shape[0], dtype=bool)
    lower_bound = x_original_tensor - THETA
    upper_bound = x_original_tensor + THETA

    for _ in range(MAX_ITERATIONS):
        if modified.sum() >= MAX_MODIFIED_FEATURES:
            break

        x_input = x_adv.clone().detach().unsqueeze(0)
        x_input.requires_grad_(True)

        logit = model(x_input).squeeze()

        with torch.no_grad():
            current_prediction = int(logit.item() >= 0.0)

        if current_prediction != y_true:
            break

        # For a single-output binary model, ascend the positive-class logit
        # to push the sample toward the attack class.
        gradient = torch.autograd.grad(logit, x_input)[0].detach().cpu().numpy()[0]

        available = np.where(~modified)[0]
        if available.size == 0:
            break

        saliency = np.abs(gradient[available])
        best_feature = int(available[np.argmax(saliency)])

        with torch.no_grad():
            if gradient[best_feature] >= 0:
                x_adv[best_feature] += THETA
            else:
                x_adv[best_feature] -= THETA

            x_adv = torch.clamp(x_adv, lower_bound, upper_bound)

        modified[best_feature] = True

        with torch.no_grad():
            adversarial_logit = model(x_adv.unsqueeze(0)).item()
            adversarial_prediction = int(adversarial_logit >= 0.0)

        if adversarial_prediction != y_true:
            break

    return x_adv.cpu().numpy(), int(modified.sum())


def jsma_attack(
    model: EvaluationDNN,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    X_adv = np.copy(X)
    modified_counts = []

    for index in range(len(X)):
        X_adv[index], modified_count = jsma_attack_single(
            model,
            X[index],
            int(y[index]),
            device,
        )
        modified_counts.append(modified_count)

        if (index + 1) % 100 == 0:
            print(f"    JSMA progress: {index + 1}/{len(X)} samples")

    return X_adv, np.asarray(modified_counts)


def evaluate_model(
    model_name: str,
    model: EvaluationDNN,
    X_sample: np.ndarray,
    y_sample: np.ndarray,
    sample_indices: np.ndarray,
    dataset_name: str,
    output_dir: Path,
    device: torch.device,
) -> list[dict]:
    print(f"\nEvaluating {dataset_name} | {model_name}")

    clean_metrics, clean_probs, clean_preds = evaluate(
        model, X_sample, y_sample, device
    )

    print(f"  Clean | {format_metrics(clean_metrics)}")

    np.savez_compressed(
        output_dir / f"{dataset_name}_{model_name}_jsma_clean_predictions.npz",
        y_true=y_sample,
        y_prob=clean_probs,
        y_pred=clean_preds,
        sample_indices=sample_indices,
    )

    print("  Running JSMA attack...")

    X_adv, modified_counts = jsma_attack(
        model,
        X_sample,
        y_sample,
        device,
    )

    adv_metrics, adv_probs, adv_preds = evaluate(
        model, X_adv, y_sample, device
    )

    print(f"  JSMA  | {format_metrics(adv_metrics)}")
    print(
        f"  Mean modified features: {modified_counts.mean():.4f} | "
        f"Successful attack rate: "
        f"{float(np.mean(adv_preds != y_sample)):.8f}"
    )

    np.savez_compressed(
        output_dir
        / f"{dataset_name}_{model_name}_jsma_adversarial_predictions.npz",
        y_true=y_sample,
        y_prob=adv_probs,
        y_pred=adv_preds,
        modified_features=modified_counts,
        sample_indices=sample_indices,
    )

    return [
        {
            "dataset": dataset_name,
            "model": model_name,
            "attack": "none",
            "sample_size": len(X_sample),
            **clean_metrics,
        },
        {
            "dataset": dataset_name,
            "model": model_name,
            "attack": "JSMA",
            "sample_size": len(X_sample),
            "mean_modified_features": float(modified_counts.mean()),
            **adv_metrics,
        },
    ]


def write_workbook(
    output_dir: Path,
    results: list[dict],
    checkpoint_paths: dict,
    config: dict,
    artifact_hashes: dict[str, str],
) -> Path:
    excel_path = output_dir / "Thesis_JSMA_Evaluation_Run_Record.xlsx"

    results_df = pd.DataFrame(results)
    checkpoint_rows = [
        {"dataset": dataset, **paths}
        for dataset, paths in checkpoint_paths.items()
    ]
    hash_rows = [
        {"Artifact": artifact, "SHA-256": artifact_hash}
        for artifact, artifact_hash in sorted(artifact_hashes.items())
    ]

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="JSMA Results", index=False)
        pd.DataFrame(checkpoint_rows).to_excel(
            writer, sheet_name="Model Checkpoints", index=False
        )
        pd.DataFrame([config]).to_excel(
            writer, sheet_name="Run Configuration", index=False
        )
        pd.DataFrame(hash_rows).to_excel(
            writer, sheet_name="Artifact Hashes", index=False
        )

    return excel_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script 11: JSMA adversarial evaluation"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        choices=DATASETS,
        help="Datasets to evaluate",
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
    parser.add_argument(
        "--adversarial-dir",
        type=str,
        default=None,
        help="Optional specific 08 adversarial-training output folder",
    )
    parser.add_argument(
        "--distillation-dir",
        type=str,
        default=None,
        help="Optional specific 09 distillation output folder",
    )
    args = parser.parse_args()

    started_utc = utc_now().isoformat()
    log("Starting 11_jsma_evaluation")

    preprocessing_dir = resolve_preprocessing_dir(args.preprocessing_dir)
    preprocessing_report = load_json(
        preprocessing_dir / "preprocessing_report.json"
    )
    preprocessing_hash = preprocessing_report.get("preprocessing_report_sha256")

    if not preprocessing_hash:
        fail("Preprocessing report does not contain preprocessing_report_sha256.")

    baseline_dir = (
        Path(args.baseline_dir)
        if args.baseline_dir
        else resolve_latest_folder(
            BASELINE_ROOT,
            "baselines",
            "baseline_results.json",
            "baselines_completed",
        )
    )

    adversarial_dir = (
        Path(args.adversarial_dir)
        if args.adversarial_dir
        else resolve_latest_folder(
            ADVERSARIAL_ROOT,
            "adversarial_training",
            "adversarial_training_results.json",
            "adversarial_training_completed",
        )
    )

    distillation_dir = (
        Path(args.distillation_dir)
        if args.distillation_dir
        else resolve_latest_folder(
            DISTILLATION_ROOT,
            "distillation",
            "distillation_results.json",
            "distillation_completed",
        )
    )

    verify_chain(
        preprocessing_dir,
        preprocessing_hash,
        baseline_dir,
        adversarial_dir,
        distillation_dir,
    )

    run_id = timestamp()
    output_dir = OUTPUT_ROOT / f"jsma_evaluation_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    log(f"Using device: {device}")

    config = {
        "script": "11_jsma_evaluation.py",
        "started_utc": started_utc,
        "device": str(device),
        "preprocessing_dir": str(preprocessing_dir),
        "preprocessing_report_sha256": preprocessing_hash,
        "baseline_dir": str(baseline_dir),
        "adversarial_dir": str(adversarial_dir),
        "distillation_dir": str(distillation_dir),
        "jsma_sample_size": JSMA_SAMPLES,
        "max_modified_features": MAX_MODIFIED_FEATURES,
        "max_iterations": MAX_ITERATIONS,
        "theta": THETA,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "attack": "JSMA",
        "perturbation_constraint": (
            "Each modified feature remains within plus/minus theta of its "
            "original standardized value"
        ),
    }

    with (output_dir / "jsma_evaluation_config.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(config, f, indent=2)

    all_rows = []
    checkpoint_paths = {}

    for dataset in args.datasets:
        dataset_key = dataset.lower().replace("-", "_")
        preprocessing_dataset_dir = preprocessing_dir / dataset_key

        required_files = [
            "X_test.npy",
            "y_test.npy",
            "feature_names.json",
        ]

        for filename in required_files:
            path = preprocessing_dataset_dir / filename
            if not path.exists():
                fail(f"Required preprocessing artifact not found: {path}")

        X_test = np.load(preprocessing_dataset_dir / "X_test.npy")
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

        if X_test.shape[1] != len(feature_names):
            fail(
                f"{dataset}: X_test has {X_test.shape[1]} features, "
                f"but feature schema has {len(feature_names)}."
            )

        rng = np.random.default_rng(SEED)

        if len(X_test) > JSMA_SAMPLES:
            sample_indices = rng.choice(
                len(X_test),
                size=JSMA_SAMPLES,
                replace=False,
            )
        else:
            sample_indices = np.arange(len(X_test))

        X_sample = X_test[sample_indices]
        y_sample = y_test[sample_indices]

        baseline_checkpoint = baseline_dir / dataset_key / "dnn_baseline.pt"
        adversarial_checkpoint = (
            adversarial_dir / dataset_key / "dnn_adversarial.pt"
        )
        distilled_checkpoint = (
            distillation_dir / dataset_key / "dnn_distilled.pt"
        )

        checkpoint_paths[dataset] = {
            "baseline": str(baseline_checkpoint),
            "adversarial": str(adversarial_checkpoint),
            "distilled": str(distilled_checkpoint),
        }

        models = {
            "baseline_dnn": baseline_checkpoint,
            "adversarially_trained_dnn": adversarial_checkpoint,
            "distilled_dnn": distilled_checkpoint,
        }

        for model_name, checkpoint_path in models.items():
            model = load_model(
                checkpoint_path,
                X_sample.shape[1],
                device,
            )

            rows = evaluate_model(
                model_name,
                model,
                X_sample,
                y_sample,
                sample_indices,
                dataset,
                output_dir,
                device,
            )

            all_rows.extend(rows)

    artifact_hashes = {}

    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            artifact_hashes[str(path.relative_to(output_dir))] = sha256_file(path)

    results_path = output_dir / "jsma_evaluation_results.json"
    metrics_csv_path = output_dir / "jsma_evaluation_metrics.csv"

    pd.DataFrame(all_rows).to_csv(metrics_csv_path, index=False)

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config,
                "checkpoint_paths": checkpoint_paths,
                "results": all_rows,
            },
            f,
            indent=2,
        )

    report_hash = sha256_file(results_path)

    final_report = {
        "script": "11_jsma_evaluation",
        "started_utc": started_utc,
        "finished_utc": utc_now().isoformat(),
        "config": config,
        "checkpoint_paths": checkpoint_paths,
        "results": all_rows,
        "artifact_hashes": artifact_hashes,
        "excel_workbook": str(
            output_dir / "Thesis_JSMA_Evaluation_Run_Record.xlsx"
        ),
        "jsma_evaluation_completed": True,
        "jsma_evaluation_results_sha256": report_hash,
    }

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    excel_path = write_workbook(
        output_dir=output_dir,
        results=all_rows,
        checkpoint_paths=checkpoint_paths,
        config=config,
        artifact_hashes=artifact_hashes,
    )
    log(f"Wrote JSMA evaluation run-record workbook: {excel_path}")

    print("\n============ 11 JSMA EVALUATION SUMMARY ============")

    for dataset in args.datasets:
        dataset_rows = [row for row in all_rows if row["dataset"] == dataset]

        for model_name in [
            "baseline_dnn",
            "adversarially_trained_dnn",
            "distilled_dnn",
        ]:
            model_rows = [
                row for row in dataset_rows if row["model"] == model_name
            ]

            clean_row = next(
                row for row in model_rows if row["attack"] == "none"
            )
            jsma_row = next(
                row for row in model_rows if row["attack"] == "JSMA"
            )

            print(
                f"\n{dataset} | {model_name}\n"
                f"  Clean accuracy: {clean_row['accuracy']:.8f}\n"
                f"  JSMA accuracy: {jsma_row['accuracy']:.8f}\n"
                f"  Mean modified features: "
                f"{jsma_row['mean_modified_features']:.4f}"
            )

    print("\n====================================================")
    print("[PASS] 11_jsma_evaluation completed successfully.")
    print(f"Results: {results_path}")
    print(f"Metrics CSV: {metrics_csv_path}")
    print(f"Excel run record: {excel_path}")
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()