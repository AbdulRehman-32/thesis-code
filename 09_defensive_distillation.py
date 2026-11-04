r"""
09_defensive_distillation.py

Defensive distillation for the thesis IDS pipeline.

Uses:
  E:\Thesis\reports\preprocessing\preprocessing_20261104T040755_364187Z
  E:\Thesis\reports\baselines\baselines_20261104T043514_850585Z

Outputs:
  E:\Thesis\reports\distillation\distillation_<timestamp>\
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
import torch.optim as optim
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
TEMPERATURE = 20.0
EPOCHS_TEACHER = 40
EPOCHS_STUDENT = 40
BATCH_SIZE = 512
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
PATIENCE = 5
HIDDEN_DIM = 128
DROPOUT = 0.3

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


class DNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, 2),
        )

    def forward(self, x):
        return self.net(x)


def load_split(dataset_dir, split):
    X = np.load(dataset_dir / f"X_{split}.npy")
    y = np.load(dataset_dir / f"y_{split}.npy")
    return X, y


def make_loader(X, y, shuffle=False):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )

    g = torch.Generator()
    g.manual_seed(SEED)

    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=g if shuffle else None,
    )


def train_teacher(X_train, y_train, X_val, y_val, input_dim):
    model = DNN(input_dim).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val, shuffle=False)

    best_auc = -np.inf
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, EPOCHS_TEACHER + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)

        train_loss = total_loss / len(train_loader.dataset)

        val_metrics, _, _ = evaluate(model, X_val, y_val)

        print(
            f"Teacher epoch {epoch:02d} | "
            f"train loss {train_loss:.8f} | "
            f"validation ROC-AUC {val_metrics['roc_auc']:.8f}"
        )

        if val_metrics["roc_auc"] > best_auc:
            best_auc = val_metrics["roc_auc"]
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"Teacher early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)

    return model, {
        "best_epoch": best_epoch,
        "best_validation_roc_auc": float(best_auc),
    }


def generate_soft_labels(model, X):
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    soft_labels = []

    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(DEVICE))

            # Correct: soft targets must be probabilities.
            probs = torch.softmax(logits / TEMPERATURE, dim=1)
            soft_labels.append(probs.cpu())

    return torch.cat(soft_labels)


def train_student(X_train, y_train, soft_labels, X_val, y_val, input_dim):
    model = DNN(input_dim).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        soft_labels,
    )
    val_loader = make_loader(X_val, y_val, shuffle=False)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g,
    )

    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    best_auc = -np.inf
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, EPOCHS_STUDENT + 1):
        model.train()
        total_loss = 0.0

        for xb, yb, soft in train_loader:
            xb, yb, soft = xb.to(DEVICE), yb.to(DEVICE), soft.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)

            hard = ce_loss(logits, yb)
            soft_loss = kl_loss(
                nn.functional.log_softmax(logits / TEMPERATURE, dim=1),
                soft,
            ) * (TEMPERATURE ** 2)

            loss = 0.5 * hard + 0.5 * soft_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)

        train_loss = total_loss / len(train_loader.dataset)

        val_metrics, _, _ = evaluate(model, X_val, y_val)

        print(
            f"Student epoch {epoch:02d} | "
            f"train loss {train_loss:.8f} | "
            f"validation ROC-AUC {val_metrics['roc_auc']:.8f}"
        )

        if val_metrics["roc_auc"] > best_auc:
            best_auc = val_metrics["roc_auc"]
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"Student early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)

    return model, {
        "best_epoch": best_epoch,
        "best_validation_roc_auc": float(best_auc),
    }


def evaluate(model, X, y):
    model.eval()
    loader = make_loader(X, y, shuffle=False)

    probs, preds, labels = [], [], []

    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(DEVICE))
            prob = torch.softmax(logits, dim=1)[:, 1]
            pred = torch.argmax(logits, dim=1)

            probs.extend(prob.cpu().numpy())
            preds.extend(pred.cpu().numpy())
            labels.extend(yb.numpy())

    probs = np.asarray(probs)
    preds = np.asarray(preds)
    labels = np.asarray(labels)

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


def run_dataset(dataset_name, dataset_dir, output_dir):
    print("\n" + "=" * 70)
    print(f"DEFENSIVE DISTILLATION: {dataset_name}")
    print("=" * 70)

    X_train, y_train = load_split(dataset_dir, "train")
    X_val, y_val = load_split(dataset_dir, "validation")
    X_test, y_test = load_split(dataset_dir, "test")

    print(f"Train: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")

    input_dim = X_train.shape[1]

    teacher, teacher_info = train_teacher(
        X_train,
        y_train,
        X_val,
        y_val,
        input_dim,
    )

    teacher_path = output_dir / f"{dataset_dir.name}_teacher_dnn.pt"
    torch.save(teacher.state_dict(), teacher_path)

    soft_labels = generate_soft_labels(teacher, X_train)

    student, student_info = train_student(
        X_train,
        y_train,
        soft_labels,
        X_val,
        y_val,
        input_dim,
    )

    student_path = output_dir / f"{dataset_dir.name}_distilled_dnn.pt"
    torch.save(student.state_dict(), student_path)

    val_metrics, val_probs, val_preds = evaluate(student, X_val, y_val)
    test_metrics, test_probs, test_preds = evaluate(student, X_test, y_test)

    np.savez_compressed(
        output_dir / f"{dataset_dir.name}_distilled_validation_predictions.npz",
        y_true=y_val,
        y_prob=val_probs,
        y_pred=val_preds,
    )

    np.savez_compressed(
        output_dir / f"{dataset_dir.name}_distilled_test_predictions.npz",
        y_true=y_test,
        y_prob=test_probs,
        y_pred=test_preds,
    )

    metric_rows = []

    for split, metrics in [
        ("validation", val_metrics),
        ("test", test_metrics),
    ]:
        metric_rows.append(
            {
                "dataset": dataset_name,
                "model": "distilled_dnn",
                "split": split,
                **metrics,
            }
        )

    return {
        "dataset": dataset_name,
        "teacher": teacher_info,
        "student": student_info,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "teacher_checkpoint": teacher_path.name,
        "student_checkpoint": student_path.name,
        "metric_rows": metric_rows,
    }


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
        "--output-root",
        default=r"E:\Thesis\reports\distillation",
    )
    args = parser.parse_args()

    set_seed()

    preprocessing_dir = Path(args.preprocessing_dir)
    baseline_dir = Path(args.baseline_dir)

    run_id = timestamp()
    output_dir = Path(args.output_root) / f"distillation_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SCRIPT 09: DEFENSIVE DISTILLATION")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Preprocessing folder: {preprocessing_dir}")
    print(f"Baseline folder: {baseline_dir}")
    print(f"Output folder: {output_dir}")

    config = {
        "script": "09_defensive_distillation.py",
        "run_id": run_id,
        "created_utc": utc_now().isoformat(),
        "device": str(DEVICE),
        "preprocessing_dir": str(preprocessing_dir),
        "baseline_dir": str(baseline_dir),
        "temperature": TEMPERATURE,
        "teacher_epochs": EPOCHS_TEACHER,
        "student_epochs": EPOCHS_STUDENT,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "seed": SEED,
        "loss": "0.5 * hard cross-entropy + 0.5 * temperature-scaled KL divergence",
    }

    with open(output_dir / "distillation_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    results = {}
    all_metric_rows = []

    for dataset_name, folder_name in DATASETS.items():
        dataset_dir = preprocessing_dir / folder_name

        result = run_dataset(
            dataset_name,
            dataset_dir,
            output_dir,
        )

        results[dataset_name] = result
        all_metric_rows.extend(result["metric_rows"])

    metrics_df = pd.DataFrame(all_metric_rows)

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
        metrics_df[col] = metrics_df[col].map(lambda x: f"{x:.8f}")

    metrics_df.to_csv(
        output_dir / "distillation_metrics.csv",
        index=False,
    )

    with open(output_dir / "distillation_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config,
                "results": results,
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

    excel_path = output_dir / "Thesis_Distillation_Run_Record.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        metrics_df.to_excel(
            writer,
            sheet_name="Distilled Metrics",
            index=False,
        )

        pd.DataFrame([config]).to_excel(
            writer,
            sheet_name="Run Configuration",
            index=False,
        )

        pd.DataFrame(
            [
                {"artifact": k, "sha256": v}
                for k, v in artifact_hashes.items()
            ]
        ).to_excel(
            writer,
            sheet_name="Artifact Hashes",
            index=False,
        )

    print("\n" + "=" * 70)
    print("DEFENSIVE DISTILLATION COMPLETE")
    print("=" * 70)

    for dataset_name in DATASETS:
        test_metrics = results[dataset_name]["test_metrics"]

        print(f"\n{dataset_name} distilled DNN test metrics:")
        for col in float_cols:
            print(f"  {col}: {test_metrics[col]:.8f}")

    print("\nSaved outputs:")
    print(f"  {output_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()