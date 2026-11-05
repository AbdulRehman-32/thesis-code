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

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"E:\Thesis")

PREPROCESSING_ROOT = PROJECT_ROOT / "reports" / "preprocessing"
BASELINE_ROOT = PROJECT_ROOT / "reports" / "baselines"
ADVERSARIAL_ROOT = PROJECT_ROOT / "reports" / "adversarial_training"
DISTILLATION_ROOT = PROJECT_ROOT / "reports" / "distillation"
FGSM_ROOT = PROJECT_ROOT / "reports" / "fgsm_evaluation"
JSMA_ROOT = PROJECT_ROOT / "reports" / "jsma_evaluation"
CW_ROOT = PROJECT_ROOT / "reports" / "cw_evaluation"
OUTPUT_ROOT = PROJECT_ROOT / "reports" / "final_results"

SEED = 42

DATASETS = {
    "NSL-KDD": "nsl_kdd",
    "CICIDS2017": "cicids2017",
}

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
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

BASELINE_ONLY_MODELS = {"logistic_regression", "random_forest"}
DNN_MODELS = {
    "baseline_dnn",
    "adversarially_trained_dnn",
    "distilled_dnn",
}


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

    if excluded != {"difficulty", "Attempted Category"}:
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
    fgsm_dir: Path,
    jsma_dir: Path,
    cw_dir: Path,
) -> None:
    baseline_report = load_json(baseline_dir / "baseline_results.json")
    adversarial_report = load_json(
        adversarial_dir / "adversarial_training_results.json"
    )
    distillation_report = load_json(
        distillation_dir / "distillation_results.json"
    )
    fgsm_report = load_json(fgsm_dir / "fgsm_evaluation_results.json")
    jsma_report = load_json(jsma_dir / "jsma_evaluation_results.json")
    cw_report = load_json(cw_dir / "cw_evaluation_results.json")

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

    for label, report in [
        ("FGSM", fgsm_report),
        ("JSMA", jsma_report),
        ("CW-L2", cw_report),
    ]:
        if report.get("config", {}).get(
            "preprocessing_dir"
        ) != str(preprocessing_dir):
            fail(
                f"{label} evaluation folder was not evaluated on the selected "
                "preprocessing folder."
            )

    if adversarial_report.get("baseline_dir") != str(baseline_dir):
        fail("Adversarial-training folder does not match the selected baseline folder.")

    if distillation_report.get("config", {}).get(
        "baseline_dir"
    ) != str(baseline_dir):
        fail("Distillation folder does not match the selected baseline folder.")

    for label, report in [
        ("FGSM", fgsm_report),
        ("JSMA", jsma_report),
        ("CW-L2", cw_report),
    ]:
        if report.get("config", {}).get("baseline_dir") != str(baseline_dir):
            fail(f"{label} evaluation folder does not match the selected baseline folder.")

    if preprocessing_hash:
        expected_hash = preprocessing_hash

        if baseline_report.get("preprocessing_report_sha256") != expected_hash:
            fail("Baseline preprocessing hash does not match selected preprocessing.")

        if adversarial_report.get(
            "preprocessing_report_sha256"
        ) != expected_hash:
            fail(
                "Adversarial-training preprocessing hash does not match selected "
                "preprocessing."
            )

        if distillation_report.get("config", {}).get(
            "preprocessing_report_sha256"
        ) != expected_hash:
            fail(
                "Distillation preprocessing hash does not match selected "
                "preprocessing."
            )

        for label, report in [
            ("FGSM", fgsm_report),
            ("JSMA", jsma_report),
            ("CW-L2", cw_report),
        ]:
            if report.get("config", {}).get(
                "preprocessing_report_sha256"
            ) != expected_hash:
                fail(
                    f"{label} evaluation preprocessing hash does not match selected "
                    "preprocessing."
                )


def load_metrics_csv(folder: Path, filename: str) -> pd.DataFrame:
    path = folder / filename

    if not path.exists():
        fail(f"Missing required file: {path}")

    return pd.read_csv(path)


def standardize_attack_results(df: pd.DataFrame, attack_name: str) -> pd.DataFrame:
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
        fail(f"{attack_name} metrics file is missing columns: {missing}")

    df["attack_type"] = np.where(
        df["attack"].astype(str).str.lower() == "none",
        "none",
        attack_name,
    )

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


def load_classical_baselines(baseline_dir: Path) -> pd.DataFrame:
    path = baseline_dir / "baseline_metrics.csv"

    if not path.exists():
        fail(f"Missing baseline metrics file: {path}")

    df = pd.read_csv(path)

    if "split" in df.columns:
        df = df[df["split"].str.lower() == "test"]

    df = df[df["model"].isin(BASELINE_ONLY_MODELS)].copy()

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
    classical_baseline_df: pd.DataFrame,
    fgsm_df: pd.DataFrame,
    jsma_df: pd.DataFrame,
    cw_df: pd.DataFrame,
) -> pd.DataFrame:
    combined = pd.concat(
        [classical_baseline_df, fgsm_df, jsma_df, cw_df],
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


def save_fgsm_figure(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
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
    combined: pd.DataFrame,
    attack_type: str,
    metric: str,
    figure_number: str,
    title: str,
    ylabel: str,
    output_dir: Path,
) -> None:
    attack_df = combined[
        (combined["attack_type"].isin(["none", attack_type]))
        & (combined["model"].isin(DNN_MODELS))
    ].copy()

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


def create_figures(combined: pd.DataFrame, output_dir: Path) -> None:
    fgsm = combined[
        (combined["attack_type"] == "FGSM")
        & combined["epsilon"].notna()
    ].copy()

    if not fgsm.empty:
        save_fgsm_figure(
            fgsm,
            "accuracy",
            "Figure 4.1: FGSM Accuracy vs Epsilon",
            "Accuracy",
            output_dir / "figure_4_1_fgsm_accuracy.png",
        )

        save_fgsm_figure(
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


def write_workbook(
    output_dir: Path,
    combined: pd.DataFrame,
    clean_results: pd.DataFrame,
    fgsm_results: pd.DataFrame,
    jsma_results: pd.DataFrame,
    cw_results: pd.DataFrame,
    source_folders: dict,
    artifact_hashes: dict[str, str],
) -> Path:
    excel_path = output_dir / "Thesis_Final_Results.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name="Combined Results", index=False)
        clean_results.to_excel(writer, sheet_name="Clean Results", index=False)
        fgsm_results.to_excel(writer, sheet_name="FGSM Results", index=False)
        jsma_results.to_excel(writer, sheet_name="JSMA Results", index=False)
        cw_results.to_excel(writer, sheet_name="CW Results", index=False)

        pd.DataFrame(
            [
                {"source": source, "folder": folder}
                for source, folder in source_folders.items()
            ]
        ).to_excel(writer, sheet_name="Provenance", index=False)

        pd.DataFrame(
            [
                {"artifact": artifact, "sha256": digest}
                for artifact, digest in artifact_hashes.items()
            ]
        ).to_excel(writer, sheet_name="Artifact Hashes", index=False)

    return excel_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script 13: final thesis results consolidation"
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
    parser.add_argument(
        "--fgsm-dir",
        type=str,
        default=None,
        help="Optional specific 10 FGSM evaluation output folder",
    )
    parser.add_argument(
        "--jsma-dir",
        type=str,
        default=None,
        help="Optional specific 11 JSMA evaluation output folder",
    )
    parser.add_argument(
        "--cw-dir",
        type=str,
        default=None,
        help="Optional specific 12 CW evaluation output folder",
    )
    args = parser.parse_args()

    started_utc = utc_now().isoformat()
    log("Starting 13_consolidate_results")

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

    fgsm_dir = (
        Path(args.fgsm_dir)
        if args.fgsm_dir
        else resolve_latest_folder(
            FGSM_ROOT,
            "fgsm_evaluation",
            "fgsm_evaluation_results.json",
            "fgsm_evaluation_completed",
        )
    )

    jsma_dir = (
        Path(args.jsma_dir)
        if args.jsma_dir
        else resolve_latest_folder(
            JSMA_ROOT,
            "jsma_evaluation",
            "jsma_evaluation_results.json",
            "jsma_evaluation_completed",
        )
    )

    cw_dir = (
        Path(args.cw_dir)
        if args.cw_dir
        else resolve_latest_folder(
            CW_ROOT,
            "cw_evaluation",
            "cw_evaluation_results.json",
            "cw_evaluation_completed",
        )
    )

    verify_chain(
        preprocessing_dir,
        preprocessing_hash,
        baseline_dir,
        adversarial_dir,
        distillation_dir,
        fgsm_dir,
        jsma_dir,
        cw_dir,
    )

    run_id = timestamp()
    output_dir = OUTPUT_ROOT / f"final_results_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    classical_baseline_df = load_classical_baselines(baseline_dir)

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
        classical_baseline_df,
        fgsm_df,
        jsma_df,
        cw_df,
    )

    combined.to_csv(output_dir / "combined_thesis_results.csv", index=False)

    clean_results = combined[combined["attack_type"] == "none"].copy()
    clean_results.to_csv(output_dir / "table_4_1_clean_results.csv", index=False)

    fgsm_results = combined[combined["attack_type"] == "FGSM"].copy()
    fgsm_results.to_csv(output_dir / "table_4_2_fgsm_results.csv", index=False)

    jsma_results = combined[combined["attack_type"] == "JSMA"].copy()
    jsma_results.to_csv(output_dir / "table_4_3_jsma_results.csv", index=False)

    cw_results = combined[combined["attack_type"] == "CW-L2"].copy()
    cw_results.to_csv(output_dir / "table_4_4_cw_results.csv", index=False)

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
        "script": "13_consolidate_results",
        "run_id": run_id,
        "started_utc": started_utc,
        "created_utc": utc_now().isoformat(),
        "source_folders": source_folders,
        "seed": SEED,
    }

    with (output_dir / "provenance.json").open("w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    artifact_hashes = {}

    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            artifact_hashes[str(path.relative_to(output_dir))] = sha256_file(path)

    with (output_dir / "artifact_hashes.json").open("w", encoding="utf-8") as f:
        json.dump(artifact_hashes, f, indent=2)

    excel_path = write_workbook(
        output_dir=output_dir,
        combined=combined,
        clean_results=clean_results,
        fgsm_results=fgsm_results,
        jsma_results=jsma_results,
        cw_results=cw_results,
        source_folders=source_folders,
        artifact_hashes=artifact_hashes,
    )
    log(f"Wrote final results workbook: {excel_path}")

    final_report = {
        "script": "13_consolidate_results",
        "started_utc": started_utc,
        "finished_utc": utc_now().isoformat(),
        "source_folders": source_folders,
        "combined_result_rows": int(len(combined)),
        "artifact_hashes": artifact_hashes,
        "excel_workbook": str(excel_path),
        "consolidation_completed": True,
    }

    with (output_dir / "final_results_report.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(final_report, f, indent=2)

    report_hash = sha256_file(output_dir / "final_results_report.json")
    final_report["final_results_report_sha256"] = report_hash

    with (output_dir / "final_results_report.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(final_report, f, indent=2)

    print("\n============ 13 FINAL RESULTS CONSOLIDATION ============")

    for dataset in DATASETS:
        dataset_rows = combined[combined["dataset"] == dataset]

        print(f"\n{dataset}")
        print(f"  Combined rows: {len(dataset_rows)}")

        for attack_type in ["none", "FGSM", "JSMA", "CW-L2"]:
            attack_rows = dataset_rows[
                dataset_rows["attack_type"] == attack_type
            ]

            if not attack_rows.empty:
                print(f"  {attack_type}: {len(attack_rows)} rows")

    print("\n========================================================")
    print("[PASS] 13_consolidate_results completed successfully.")
    print(f"Combined results: {output_dir / 'combined_thesis_results.csv'}")
    print(f"Excel workbook: {excel_path}")
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()