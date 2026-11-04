r"""
10_fgsm_evaluation.py

FGSM adversarial evaluation for the thesis IDS pipeline.

Evaluates:
  1. Clean baseline DNN
  2. PGD adversarially trained DNN
  3. Defensive-distillation DNN

Usage:
  python 10_fgsm_evaluation.py
"""

import argparse
import hashlib
import json
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

SEED = 42
BATCH_SIZE = 512
DROPOUT = 0.3

EPSILONS = [0.01, 0.02, 0.05, 0.10, 0.20]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS = {
    "NSL-KDD": "nsl_kdd",
    "CICIDS2017": "cicids2017",
}


def utc_now():
    return datetime.now(timezone.utc)


def timestamp():
    return utc_now().strftime("%Y%m%dT%H%M%S_%f")[:-3] + "Z"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_checkpoint(root, dataset_folder, filename):
    root = Path(root)
    matches = list(root.rglob(filename))

    dataset_matches = [
        path for path in matches
        if dataset_folder.lower() in str(path).lower()
    ]

    if dataset_matches:
        return dataset_matches[0]

    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find {filename} for {dataset_folder} under {root}"
    )


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if any(key.startswith("network.") for key in state_dict):
        prefix = "network"
    elif any(key.startswith("net.") for key in state_dict):
        prefix = "net"
    else:
        raise ValueError(
            f"Could not detect model layer prefix in {checkpoint_path}"
        )

    fixed_state_dict = {
        key.replace(f"{prefix}.", ""): value
        for key, value in state_dict.items()
    }

    linear_keys = sorted(
        [key for key in fixed_state_dict if key.endswith(".weight")],
        key=lambda key: int(key.split(".")[0]),
    )

    if not linear_keys:
        raise ValueError(f"No Linear layers found in {checkpoint_path}")

    input_dim = fixed_state_dict[linear_keys[0]].shape[1]
    output_dim = fixed_state_dict[linear_keys[-1]].shape[0]

    hidden_dims = [
        fixed_state_dict[key].shape[0]
        for key in linear_keys[:-1]
    ]

    layers = []
    current_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(DROPOUT))
        current_dim = hidden_dim

    layers.append(nn.Linear(current_dim, output_dim))

    model = nn.Sequential(*layers).to(DEVICE)
    model.load_state_dict(fixed_state_dict)

    return model


def load_test_data(preprocessing_dir, dataset_folder):
    dataset_dir = Path(preprocessing_dir) / dataset_folder

    X_test = np.load(dataset_dir / "X_test.npy")
    y_test = np.load(dataset_dir / "y_test.npy")

    return X_test, y_test


def make_loader(X, y):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )

    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


def get_probabilities(logits):
    if logits.shape[1] == 2:
        return torch.softmax(logits, dim=1)[:, 1]
    return torch.sigmoid(logits).squeeze(1)


def get_predictions(logits):
    if logits.shape[1] == 2:
        return torch.argmax(logits, dim=1)
    return (torch.sigmoid(logits).squeeze(1) >= 0.5).long()


def get_loss(logits, y):
    if logits.shape[1] == 2:
        return nn.CrossEntropyLoss()(
            logits,
            y.long(),
        )

    return nn.BCEWithLogitsLoss()(
        logits.squeeze(1),
        y.float(),
    )


def evaluate(model, X, y):
    model.eval()
    loader = make_loader(X, y)

    probs, preds, labels = [], [], []

    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(DEVICE))
            prob = get_probabilities(logits)
            pred = get_predictions(logits)

            probs.extend(prob.cpu().numpy())
            preds.extend(pred.cpu().numpy())
            labels.extend(yb.numpy())

    probs = np.asarray(probs)
    preds = np.asarray(preds)
    labels = np.asarray(labels).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "false_negative_rate": float(fn / (tp + fn)) if (tp + fn) else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "roc_auc": float(roc_auc_score(labels, probs)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

    return metrics, probs, preds


def fgsm_attack(model, X, y, epsilon):
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    adversarial_batches = []

    for xb, yb in loader:
        xb = xb.clone().detach().to(DEVICE)
        yb = yb.to(DEVICE)
        xb.requires_grad = True

        logits = model(xb)
        loss = get_loss(logits, yb)

        model.zero_grad()
        loss.backward()

        gradient = xb.grad.data
        perturbation = epsilon * gradient.sign()
        X_adv = xb + perturbation
        X_adv = torch.clamp(X_adv, 0.0, 1.0)

        adversarial_batches.append(X_adv.detach().cpu())

    return torch.cat(adversarial_batches).numpy()


def evaluate_model(model_name, model, X_test, y_test, dataset_name, output_dir):
    print(f"\nEvaluating {dataset_name} - {model_name}")

    clean_metrics, clean_probs, clean_preds = evaluate(model, X_test, y_test)

    print(
        f"  Clean accuracy: {clean_metrics['accuracy']:.8f} | "
        f"ROC-AUC: {clean_metrics['roc_auc']:.8f}"
    )

    np.savez_compressed(
        output_dir / f"{dataset_name}_{model_name}_clean_predictions.npz",
        y_true=y_test,
        y_prob=clean_probs,
        y_pred=clean_preds,
    )

    rows = [
        {
            "dataset": dataset_name,
            "model": model_name,
            "attack": "none",
            "epsilon": 0.0,
            **clean_metrics,
        }
    ]

    for epsilon in EPSILONS:
        X_adv = fgsm_attack(model, X_test, y_test, epsilon)

        adv_metrics, adv_probs, adv_preds = evaluate(model, X_adv, y_test)

        print(
            f"  FGSM epsilon {epsilon:.2f} | "
            f"accuracy {adv_metrics['accuracy']:.8f} | "
            f"ROC-AUC {adv_metrics['roc_auc']:.8f}"
        )

        np.savez_compressed(
            output_dir / f"{dataset_name}_{model_name}_fgsm_{epsilon:.2f}_predictions.npz",
            y_true=y_test,
            y_prob=adv_probs,
            y_pred=adv_preds,
            x_adv=X_adv,
        )

        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "attack": "FGSM",
                "epsilon": epsilon,
                **adv_metrics,
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocessing-dir",
        default=r"E:\Thesis\reports\preprocessing\preprocessing_20261104T040755_364187Z",
    )
    parser.add_argument(
        "--baseline-dir",
        default=r"E:\Thesis\reports\baselines\baselines_20261104T043514_850585Z",
    )
    parser.add_argument(
        "--adversarial-dir",
        default=r"E:\Thesis\reports\adversarial_training\adversarial_training_20261104T051559_260516Z",
    )
    parser.add_argument(
        "--distillation-dir",
        default=r"E:\Thesis\reports\distillation\distillation_20261104T070817_076Z",
    )
    parser.add_argument(
        "--output-root",
        default=r"E:\Thesis\reports\fgsm_evaluation",
    )
    args = parser.parse_args()

    set_seed()

    preprocessing_dir = Path(args.preprocessing_dir)
    baseline_dir = Path(args.baseline_dir)
    adversarial_dir = Path(args.adversarial_dir)
    distillation_dir = Path(args.distillation_dir)

    run_id = timestamp()
    output_dir = Path(args.output_root) / f"fgsm_evaluation_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SCRIPT 10: FGSM ADVERSARIAL EVALUATION")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Preprocessing folder: {preprocessing_dir}")
    print(f"Baseline folder: {baseline_dir}")
    print(f"Adversarial folder: {adversarial_dir}")
    print(f"Distillation folder: {distillation_dir}")
    print(f"Output folder: {output_dir}")
    print(f"FGSM epsilon values: {EPSILONS}")

    config = {
        "script": "10_fgsm_evaluation.py",
        "run_id": run_id,
        "created_utc": utc_now().isoformat(),
        "device": str(DEVICE),
        "preprocessing_dir": str(preprocessing_dir),
        "baseline_dir": str(baseline_dir),
        "adversarial_dir": str(adversarial_dir),
        "distillation_dir": str(distillation_dir),
        "epsilon_values": EPSILONS,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "attack": "FGSM",
        "clamp_range": [0.0, 1.0],
    }

    with open(output_dir / "fgsm_evaluation_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    all_rows = []
    checkpoint_paths = {}

    for dataset_name, dataset_folder in DATASETS.items():
        X_test, y_test = load_test_data(preprocessing_dir, dataset_folder)

        baseline_path = find_checkpoint(
            baseline_dir,
            dataset_folder,
            "dnn_baseline.pt",
        )

        adversarial_path = find_checkpoint(
            adversarial_dir,
            dataset_folder,
            "dnn_adversarial.pt",
        )

        distilled_path = find_checkpoint(
            distillation_dir,
            dataset_folder,
            f"{dataset_folder}_distilled_dnn.pt",
        )

        checkpoint_paths[dataset_name] = {
            "baseline": str(baseline_path),
            "adversarial": str(adversarial_path),
            "distilled": str(distilled_path),
        }

        models = {
            "baseline_dnn": baseline_path,
            "adversarially_trained_dnn": adversarial_path,
            "distilled_dnn": distilled_path,
        }

        for model_name, checkpoint_path in models.items():
            model = load_model(checkpoint_path)

            rows = evaluate_model(
                model_name,
                model,
                X_test,
                y_test,
                dataset_name,
                output_dir,
            )

            all_rows.extend(rows)

    results_df = pd.DataFrame(all_rows)

    float_cols = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "false_negative_rate",
        "false_positive_rate",
        "roc_auc",
    ]

    for col in float_cols:
        results_df[col] = results_df[col].map(lambda x: f"{x:.8f}")

    results_df.to_csv(
        output_dir / "fgsm_evaluation_metrics.csv",
        index=False,
    )

    with open(output_dir / "fgsm_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config,
                "checkpoint_paths": checkpoint_paths,
                "results": all_rows,
            },
            f,
            indent=2,
        )

    artifact_hashes = {}

    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            artifact_hashes[str(path.relative_to(output_dir))] = sha256_file(path)

    with open(output_dir / "artifact_hashes.json", "w", encoding="utf-8") as f:
        json.dump(artifact_hashes, f, indent=2)

    excel_path = output_dir / "Thesis_FGSM_Evaluation_Run_Record.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        results_df.to_excel(
            writer,
            sheet_name="FGSM Results",
            index=False,
        )

        pd.DataFrame([config]).to_excel(
            writer,
            sheet_name="Run Configuration",
            index=False,
        )

        pd.DataFrame(
            [
                {"dataset": dataset, **paths}
                for dataset, paths in checkpoint_paths.items()
            ]
        ).to_excel(
            writer,
            sheet_name="Model Checkpoints",
            index=False,
        )

        pd.DataFrame(
            [
                {"artifact": artifact, "sha256": digest}
                for artifact, digest in artifact_hashes.items()
            ]
        ).to_excel(
            writer,
            sheet_name="Artifact Hashes",
            index=False,
        )

    print("\n" + "=" * 70)
    print("FGSM EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Saved outputs: {output_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()