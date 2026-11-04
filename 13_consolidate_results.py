r"""
13_consolidate_results.py

Consolidates all final thesis pipeline results into:
  - One combined results table
  - Chapter 4 result tables
  - Figures 4.1-4.6
  - A final Excel workbook
  - A provenance record

Uses only the final artifact folders from scripts 06-12.

Usage:
  python 13_consolidate_results.py
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 42

DATASETS = {
    "NSL-KDD": "nsl_kdd",
    "CICIDS2017": "cicids2017",
}

MODEL_LABELS = {
    "baseline_dnn": "Baseline DNN",
    "adversarially_trained_dnn": "Adversarially Trained DNN",
    "distilled_dnn": "Defensive Distillation DNN",
}

ATTACK_LABELS = {
    "none": "No Attack",
    "FGSM": "FGSM",
    "JSMA": "JSMA",
    "CW-L2": "CW-L2",
}

METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "false_negative_rate",
    "false_positive_rate",
    "roc_auc",
]


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


def latest_folder(root, prefix):
    root = Path(root)
    folders = [
        path for path in root.glob(f"{prefix}_*")
        if path.is_dir()
    ]

    if not folders:
        raise FileNotFoundError(
            f"No folder starting with {prefix}_ found under {root}"
        )

    return max(folders, key=lambda path: path.name)


def load_metrics_csv(folder, filename):
    path = Path(folder) / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    return pd.read_csv(path)


def standardize_attack_results(df, attack_name):
    df = df.copy()

    required = [
        "dataset",
        "model",
        "attack",
        "accuracy",
        "roc_auc",
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"{attack_name} metrics file is missing columns: {missing}"
        )

    # Clean rows have attack = "none".
    # They must be labelled as "none", not as FGSM/JSMA/CW-L2.
    df.loc[
        df["attack"].astype(str).str.lower() == "none",
        "attack_type",
    ] = "none"

    # Attacked rows keep their actual attack name.
    attacked = df["attack"].astype(str).str.lower() != "none"

    df.loc[
        attacked,
        "attack_type",
    ] = df.loc[
        attacked,
        "attack",
    ]

    if "epsilon" not in df.columns:
        df["epsilon"] = np.nan

    if "sample_size" not in df.columns:
        df["sample_size"] = np.nan

    if "mean_l2_distance" not in df.columns:
        df["mean_l2_distance"] = np.nan

    if "mean_modified_features" not in df.columns:
        df["mean_modified_features"] = np.nan

    columns = [
        "dataset",
        "model",
        "attack_type",
        "attack",
        "epsilon",
        "sample_size",
        "mean_l2_distance",
        "mean_modified_features",
    ] + METRIC_COLUMNS

    for column in columns:
        if column not in df.columns:
            df[column] = np.nan

    return df[columns]


def load_clean_baselines(baseline_dir):
    path = Path(baseline_dir) / "baseline_metrics.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing baseline metrics file: {path}")

    df = pd.read_csv(path)

    if "split" in df.columns:
        df = df[df["split"].str.lower() == "test"]

    df["attack_type"] = "none"
    df["attack"] = "none"
    df["epsilon"] = 0.0
    df["sample_size"] = np.nan
    df["mean_l2_distance"] = np.nan
    df["mean_modified_features"] = np.nan

    columns = [
        "dataset",
        "model",
        "attack_type",
        "attack",
        "epsilon",
        "sample_size",
        "mean_l2_distance",
        "mean_modified_features",
    ] + METRIC_COLUMNS

    for column in columns:
        if column not in df.columns:
            df[column] = np.nan

    return df[columns]


def make_combined_results(
    baseline_df,
    fgsm_df,
    jsma_df,
    cw_df,
):
    combined = pd.concat(
        [baseline_df, fgsm_df, jsma_df, cw_df],
        ignore_index=True,
    )

    combined["dataset_label"] = combined["dataset"].map(
        lambda value: next(
            (label for label, folder in DATASETS.items() if folder == value),
            value,
        )
    )

    combined["model_label"] = combined["model"].map(
        lambda value: MODEL_LABELS.get(value, value)
    )

    combined["attack_label"] = combined["attack_type"].map(
        lambda value: ATTACK_LABELS.get(value, value)
    )

    return combined


def save_figure(df, metric, title, ylabel, output_path):
    plt.figure(figsize=(10, 6))

    for dataset_label, dataset_group in df.groupby("dataset_label"):
        for model_label, model_group in dataset_group.groupby("model_label"):
            model_group = model_group.sort_values("epsilon")

            plt.plot(
                model_group["epsilon"],
                model_group[metric],
                marker="o",
                label=f"{dataset_label} - {model_label}",
            )

    plt.title(title)
    plt.xlabel("FGSM Epsilon")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_attack_bar_figure(
    combined,
    attack_type,
    metric,
    figure_number,
    title,
    ylabel,
    output_dir,
):
    attack_df = combined[
        combined["attack_type"].isin(["none", attack_type])
    ].copy()

    attack_df = attack_df[
        attack_df["model"].isin(
            [
                "baseline_dnn",
                "adversarially_trained_dnn",
                "distilled_dnn",
            ]
        )
    ]

    if attack_df.empty:
        return

    pivot = attack_df.pivot_table(
        index=["dataset_label", "model_label"],
        columns="attack_type",
        values=metric,
        aggfunc="mean",
    ).reset_index()

    if "none" not in pivot.columns:
        pivot["none"] = np.nan

    if attack_type not in pivot.columns:
        pivot[attack_type] = np.nan

    plt.figure(figsize=(12, 6))

    x = np.arange(len(pivot))
    width = 0.35

    plt.bar(
        x - width / 2,
        pivot["none"],
        width,
        label="No Attack",
    )

    plt.bar(
        x + width / 2,
        pivot[attack_type],
        width,
        label=ATTACK_LABELS[attack_type],
    )

    plt.xticks(
        x,
        [
            f"{dataset}\n{model}"
            for dataset, model in zip(
                pivot["dataset_label"],
                pivot["model_label"],
            )
        ],
        rotation=30,
        ha="right",
        fontsize=8,
    )

    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / f"figure_{figure_number.replace('.', '_')}.png",
        dpi=300,
    )
    plt.close()


def create_figures(combined, output_dir):
    fgsm = combined[
        (combined["attack_type"] == "FGSM")
        & combined["epsilon"].notna()
    ].copy()

    if not fgsm.empty:
        save_figure(
            fgsm,
            "accuracy",
            "Figure 4.1: FGSM Accuracy vs Epsilon",
            "Accuracy",
            output_dir / "figure_4_1_fgsm_accuracy.png",
        )

        save_figure(
            fgsm,
            "roc_auc",
            "Figure 4.2: FGSM ROC-AUC vs Epsilon",
            "ROC-AUC",
            output_dir / "figure_4_2_fgsm_roc_auc.png",
        )

    create_attack_bar_figure(
        combined,
        "JSMA",
        "accuracy",
        "4.3",
        "Figure 4.3: JSMA Accuracy, Clean vs Attack",
        "Accuracy",
        output_dir,
    )

    create_attack_bar_figure(
        combined,
        "JSMA",
        "roc_auc",
        "4.4",
        "Figure 4.4: JSMA ROC-AUC, Clean vs Attack",
        "ROC-AUC",
        output_dir,
    )

    create_attack_bar_figure(
        combined,
        "CW-L2",
        "accuracy",
        "4.5",
        "Figure 4.5: CW-L2 Accuracy, Clean vs Attack",
        "Accuracy",
        output_dir,
    )

    create_attack_bar_figure(
        combined,
        "CW-L2",
        "roc_auc",
        "4.6",
        "Figure 4.6: CW-L2 ROC-AUC, Clean vs Attack",
        "ROC-AUC",
        output_dir,
    )


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
        "--fgsm-dir",
        default=None,
        help="Final FGSM evaluation folder",
    )
    parser.add_argument(
        "--jsma-dir",
        default=None,
        help="Final JSMA evaluation folder",
    )
    parser.add_argument(
        "--cw-dir",
        default=None,
        help="Final CW evaluation folder",
    )
    parser.add_argument(
        "--output-root",
        default=r"E:\Thesis\reports\final_results",
    )
    args = parser.parse_args()

    preprocessing_dir = Path(args.preprocessing_dir)
    baseline_dir = Path(args.baseline_dir)
    adversarial_dir = Path(args.adversarial_dir)
    distillation_dir = Path(args.distillation_dir)

    fgsm_dir = (
        Path(args.fgsm_dir)
        if args.fgsm_dir
        else latest_folder(
            r"E:\Thesis\reports\fgsm_evaluation",
            "fgsm_evaluation",
        )
    )

    jsma_dir = (
        Path(args.jsma_dir)
        if args.jsma_dir
        else latest_folder(
            r"E:\Thesis\reports\jsma_evaluation",
            "jsma_evaluation",
        )
    )

    cw_dir = (
        Path(args.cw_dir)
        if args.cw_dir
        else latest_folder(
            r"E:\Thesis\reports\cw_evaluation",
            "cw_evaluation",
        )
    )

    run_id = timestamp()
    output_dir = Path(args.output_root) / f"final_results_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SCRIPT 13: FINAL RESULTS CONSOLIDATION")
    print("=" * 70)
    print(f"Preprocessing folder: {preprocessing_dir}")
    print(f"Baseline folder: {baseline_dir}")
    print(f"Adversarial-training folder: {adversarial_dir}")
    print(f"Distillation folder: {distillation_dir}")
    print(f"FGSM folder: {fgsm_dir}")
    print(f"JSMA folder: {jsma_dir}")
    print(f"CW folder: {cw_dir}")
    print(f"Output folder: {output_dir}")

    baseline_df = load_clean_baselines(baseline_dir)

    fgsm_df = standardize_attack_results(
        load_metrics_csv(fgsm_dir, "fgsm_evaluation_metrics.csv"),
        "FGSM",
    )

    jsma_df = standardize_attack_results(
        load_metrics_csv(jsma_dir, "jsma_evaluation_metrics.csv"),
        "JSMA",
    )

    cw_df = standardize_attack_results(
        load_metrics_csv(cw_dir, "cw_evaluation_metrics.csv"),
        "CW-L2",
    )

    combined = make_combined_results(
        baseline_df,
        fgsm_df,
        jsma_df,
        cw_df,
    )

    combined.to_csv(
        output_dir / "combined_thesis_results.csv",
        index=False,
    )

    clean_results = combined[
        combined["attack_type"] == "none"
    ].copy()

    clean_results.to_csv(
        output_dir / "table_4_1_clean_results.csv",
        index=False,
    )

    fgsm_results = combined[
        combined["attack_type"] == "FGSM"
    ].copy()

    fgsm_results.to_csv(
        output_dir / "table_4_2_fgsm_results.csv",
        index=False,
    )

    jsma_results = combined[
        combined["attack_type"] == "JSMA"
    ].copy()

    jsma_results.to_csv(
        output_dir / "table_4_3_jsma_results.csv",
        index=False,
    )

    cw_results = combined[
        combined["attack_type"] == "CW-L2"
    ].copy()

    cw_results.to_csv(
        output_dir / "table_4_4_cw_results.csv",
        index=False,
    )

    create_figures(combined, output_dir)

    source_folders = {
        "preprocessing": str(preprocessing_dir),
        "baselines": str(baseline_dir),
        "adversarial_training": str(adversarial_dir),
        "distillation": str(distillation_dir),
        "fgsm_evaluation": str(fgsm_dir),
        "jsma_evaluation": str(jsma_dir),
        "cw_evaluation": str(cw_dir),
    }

    provenance = {
        "run_id": run_id,
        "created_utc": utc_now().isoformat(),
        "source_folders": source_folders,
        "seed": SEED,
    }

    with open(output_dir / "provenance.json", "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    artifact_hashes = {}

    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            artifact_hashes[str(path.relative_to(output_dir))] = sha256_file(path)

    with open(output_dir / "artifact_hashes.json", "w", encoding="utf-8") as f:
        json.dump(artifact_hashes, f, indent=2)

    excel_path = output_dir / "Thesis_Final_Results.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        combined.to_excel(
            writer,
            sheet_name="Combined Results",
            index=False,
        )

        clean_results.to_excel(
            writer,
            sheet_name="Clean Results",
            index=False,
        )

        fgsm_results.to_excel(
            writer,
            sheet_name="FGSM Results",
            index=False,
        )

        jsma_results.to_excel(
            writer,
            sheet_name="JSMA Results",
            index=False,
        )

        cw_results.to_excel(
            writer,
            sheet_name="CW Results",
            index=False,
        )

        pd.DataFrame(
            [
                {"source": source, "folder": folder}
                for source, folder in source_folders.items()
            ]
        ).to_excel(
            writer,
            sheet_name="Provenance",
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
    print("FINAL RESULTS CONSOLIDATION COMPLETE")
    print("=" * 70)
    print(f"Combined results rows: {len(combined)}")
    print(f"Saved outputs: {output_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()