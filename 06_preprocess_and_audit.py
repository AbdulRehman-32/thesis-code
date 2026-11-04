"""
06_preprocess_and_audit.py

Purpose:
    - Verify outputs from 05_deduplicate_and_validate.py.
    - Load only curated train/validation/test records.
    - Apply documented value-quality rules.
    - Fit imputation, categorical encoding, and scaling on TRAINING ONLY.
    - Transform validation/test using the fitted training objects.
    - Save fitted preprocessing artifacts, ordered feature schema, and hashes.
    - Audit actual model-input duplicates and cross-partition collisions.

This script does not train models.
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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(r"E:\Thesis")

SCRIPT05_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "deduplication_20261104T025915_705097Z"
    / "deduplication_report.json"
)

RECORD_INDEX_BASE = 1

NON_FEATURE_COLUMNS = {
    "record_id",
    "source_path",
    "source_record_number",
    "source_partition",
}

CICIDS_EXCLUDED_COLUMNS = {
    "id",
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp",
}

LABEL_COLUMNS = {
    "NSL-KDD": "label",
    "CICIDS2017": "Label",
}

BENIGN_LABELS = {
    "NSL-KDD": {"normal"},
    "CICIDS2017": {"BENIGN"},
}
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]
MAX_ABS_NUMERIC_VALUE = 1e12
NEGATIVE_HEADER_LENGTH_POLICY = "quarantine"
INFINITE_VALUE_POLICY = "quarantine"
UNKNOWN_CATEGORY_POLICY = "ignore"

OUTPUT_ROOT = PROJECT_ROOT / "reports" / "preprocessing"


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
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        fail(f"Could not read JSON file {path}: {exc}")


def verify_sha256(path: Path, expected: str, label: str) -> None:
    if not path.exists():
        fail(f"{label} file not found: {path}")
    actual = sha256_file(path)
    if actual != expected:
        fail(
            f"{label} hash mismatch.\n"
            f"  Expected: {expected}\n"
            f"  Actual:   {actual}\n"
            f"  File:     {path}"
        )
    log(f"Verified {label}: {path.name}")


def read_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        fail(f"Manifest not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        fail(f"Manifest is empty: {path}")
    return df


def parse_manifest_references(manifest: pd.DataFrame, path: Path) -> pd.DataFrame:
    if "record_id" in manifest.columns:
        id_column = "record_id"
    else:
        id_column = manifest.columns[0]
        log(
            f"Manifest has no record_id column; using first column "
            f"'{id_column}' as the ID column: {path.name}"
        )

    ids = manifest[id_column].astype(str)

    if ids.duplicated().any():
        fail(f"Duplicate record IDs found in manifest: {path.name}")

    parsed = ids.str.extract(r"^(?P<source_path>.+):record:(?P<record_number>\d+)$")
    parsed["record_id"] = ids

    if parsed.isna().any().any():
        fail(
            "Manifest IDs do not match the expected format "
            f"'<source_path>:record:<number>': {path.name}"
        )

    parsed["record_number"] = parsed["record_number"].astype(np.int64)
    parsed["record_index"] = parsed["record_number"] - RECORD_INDEX_BASE

    if (parsed["record_index"] < 0).any():
        fail(
            f"Manifest contains record numbers below RECORD_INDEX_BASE="
            f"{RECORD_INDEX_BASE}: {path.name}"
        )

    return parsed


def normalize_labels(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    label_col = LABEL_COLUMNS[dataset]

    if label_col not in df.columns:
        fail(f"Label column '{label_col}' not found for {dataset}")

    df[label_col] = df[label_col].astype(str).str.strip()

    if dataset == "CICIDS2017":
        is_benign = (df[label_col] == "BENIGN") | (
            df[label_col].str.endswith("- Attempted")
        )
        df["binary_target"] = (~is_benign).astype(np.int8)
    else:
        df["binary_target"] = (
            ~df[label_col].isin(BENIGN_LABELS[dataset])
        ).astype(np.int8)

    return df


def identify_columns(train: pd.DataFrame, dataset: str):
    label_col = LABEL_COLUMNS[dataset]
    exclude = NON_FEATURE_COLUMNS | {label_col, "binary_target"}

    if dataset == "CICIDS2017":
        exclude = exclude | CICIDS_EXCLUDED_COLUMNS

    feature_columns = [c for c in train.columns if c not in exclude]
    numeric_columns = []
    categorical_columns = []

    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(train[col]):
            numeric_columns.append(col)
        else:
            categorical_columns.append(col)

    if not numeric_columns and not categorical_columns:
        fail(f"No usable predictor columns found for {dataset}")

    return feature_columns, numeric_columns, categorical_columns


def apply_value_quality_rules(
    df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    dataset: str,
    partition: str,
):
    exclusions = []
    keep_mask = np.ones(len(df), dtype=bool)

    for col in numeric_columns:
        values = pd.to_numeric(df[col], errors="coerce")

        infinite = np.isinf(values.to_numpy(dtype=np.float64, na_value=np.nan))
        if infinite.any() and INFINITE_VALUE_POLICY == "quarantine":
            for idx in df.index[infinite]:
                exclusions.append(
                    {
                        "dataset": dataset,
                        "partition": partition,
                        "record_id": str(df.loc[idx, "record_id"]),
                        "reason": "infinite_numeric_value",
                        "column": col,
                        "value": str(df.loc[idx, col]),
                    }
                )
            keep_mask &= ~infinite

        finite_abs = values.abs().replace([np.inf, -np.inf], np.nan)
        extreme = (finite_abs > MAX_ABS_NUMERIC_VALUE).fillna(False).to_numpy()
        if extreme.any():
            for idx in df.index[extreme]:
                exclusions.append(
                    {
                        "dataset": dataset,
                        "partition": partition,
                        "record_id": str(df.loc[idx, "record_id"]),
                        "reason": "numeric_value_outside_configured_range",
                        "column": col,
                        "value": str(df.loc[idx, col]),
                    }
                )
            keep_mask &= ~extreme

    header_candidates = [
        c
        for c in numeric_columns
        if any(term in c.lower() for term in ["header", "hdr", "length"])
    ]

    for col in header_candidates:
        values = pd.to_numeric(df[col], errors="coerce")
        negative = (values < 0).fillna(False).to_numpy()

        if not negative.any():
            continue

        if NEGATIVE_HEADER_LENGTH_POLICY == "quarantine":
            for idx in df.index[negative]:
                exclusions.append(
                    {
                        "dataset": dataset,
                        "partition": partition,
                        "record_id": str(df.loc[idx, "record_id"]),
                        "reason": "negative_header_length",
                        "column": col,
                        "value": str(df.loc[idx, col]),
                    }
                )
            keep_mask &= ~negative
        elif NEGATIVE_HEADER_LENGTH_POLICY == "clip_to_zero":
            df.loc[negative, col] = 0
            log(
                f"{dataset}/{partition}: clipped {int(negative.sum())} negative "
                f"values in '{col}' to zero."
            )

    before = len(df)
    df = df.loc[keep_mask].copy()
    removed = before - len(df)

    if removed:
        log(
            f"{dataset}/{partition}: quarantined {removed} records "
            f"during value-quality checks."
        )

    return df, exclusions


def hash_feature_matrix(X: np.ndarray) -> np.ndarray:
    hashes = []
    for row in X:
        hashes.append(
            hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        )
    return np.asarray(hashes)


def audit_collisions(
    dataset: str,
    partitions: dict[str, np.ndarray],
    hashes: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
) -> dict:
    result = {
        "within_partition_duplicate_groups": {},
        "cross_partition_duplicate_groups": 0,
        "cross_partition_conflicting_label_groups": 0,
        "cross_partition_same_label_duplicate_records": 0,
    }

    all_hashes = []
    all_partitions = []
    all_labels = []

    for partition, values in hashes.items():
        unique, counts = np.unique(values, return_counts=True)
        duplicate_groups = int(np.sum(counts > 1))
        duplicate_records = int(np.sum(counts[counts > 1]))

        result["within_partition_duplicate_groups"][partition] = {
            "groups": duplicate_groups,
            "records_in_duplicate_groups": duplicate_records,
        }

        all_hashes.extend(values.tolist())
        all_partitions.extend([partition] * len(values))
        all_labels.extend(labels[partition].tolist())

    df = pd.DataFrame(
        {
            "hash": all_hashes,
            "partition": all_partitions,
            "label": all_labels,
        }
    )

    cross_groups = 0
    conflict_groups = 0
    same_label_records = 0

    for _, group in df.groupby("hash"):
        if group["partition"].nunique() > 1:
            cross_groups += 1
            if group["label"].nunique() > 1:
                conflict_groups += 1
            else:
                same_label_records += len(group)

    result["cross_partition_duplicate_groups"] = cross_groups
    result["cross_partition_conflicting_label_groups"] = conflict_groups
    result["cross_partition_same_label_duplicate_records"] = same_label_records
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script 06: leakage-free preprocessing and model-input audit"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify inputs and transformations without writing output artifacts",
    )
    args = parser.parse_args()

    started_utc = utc_now()
    log("Starting 06_preprocess_and_audit")

    report05 = load_json(SCRIPT05_REPORT)

    if not report05.get("completed", False):
        fail("Script-05 report is not marked completed.")

    if not report05.get("ready_for_preprocessing_review", False):
        fail("Script-05 report is not marked ready_for_preprocessing_review.")

    report05_hash = report05.get("deduplication_report_sha256")
    if report05_hash:
        verify_sha256(SCRIPT05_REPORT, report05_hash, "script-05 deduplication report")

    partition_results = {
        (item["dataset"], item["partition"]): item
        for item in report05.get("partitions", [])
    }

    if not partition_results:
        fail("No partition results found in script-05 report.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = OUTPUT_ROOT / f"preprocessing_{timestamp}"

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    all_exclusions = []
    partition_outputs = []
    transformed_partitions = {}
    transformed_hashes = {}
    transformed_labels = {}

    datasets = sorted({key[0] for key in partition_results})

    for dataset in datasets:
        log(f"Processing dataset: {dataset}")

        dataset_partitions = {
            partition: result
            for (ds, partition), result in partition_results.items()
            if ds == dataset
        }

        required_partitions = {"train", "validation", "test"}
        if not required_partitions.issubset(dataset_partitions.keys()):
            fail(f"{dataset} is missing required partitions: {required_partitions}")

        curated = {}

        for partition in ["train", "validation", "test"]:
            result = dataset_partitions[partition]
            manifest_path = PROJECT_ROOT / result["manifest"]
            expected_hash = result["manifest_sha256"]

            verify_sha256(
                manifest_path,
                expected_hash,
                f"{dataset} {partition} curated manifest",
            )

            manifest = read_manifest(manifest_path)
            references = parse_manifest_references(manifest, manifest_path)

            expected_records = int(result["retained_records"])
            if len(references) != expected_records:
                fail(
                    f"{dataset} {partition}: manifest has {len(references)} IDs, "
                    f"but script-05 report expects {expected_records}."
                )

            selected_parts = []

            for source_relative_path, group in references.groupby(
                "source_path", sort=True
            ):
                source_path = PROJECT_ROOT / source_relative_path

                if not source_path.exists():
                    fail(
                        f"Source file referenced by manifest not found: {source_path}"
                    )

                if dataset == "NSL-KDD":
                    source_df = pd.read_csv(
                        source_path, header=None, low_memory=False
                    )

                    if len(source_df.columns) != len(NSL_KDD_COLUMNS):
                        fail(
                            f"NSL-KDD source file has {len(source_df.columns)} columns, "
                            f"but expected {len(NSL_KDD_COLUMNS)}: {source_path}"
                        )

                    source_df.columns = NSL_KDD_COLUMNS
                else:
                    source_df = pd.read_csv(source_path, low_memory=False)
                    source_df.columns = [
                        str(c).strip() for c in source_df.columns
                    ]

                max_index = len(source_df) - 1
                if group["record_index"].max() > max_index:
                    fail(
                        f"Record index {group['record_index'].max()} exceeds "
                        f"source-file range 0..{max_index}: {source_path}"
                    )

                part = source_df.iloc[group["record_index"].to_numpy()].copy()
                part["record_id"] = group["record_id"].to_numpy()
                part["source_path"] = source_relative_path
                part["source_record_number"] = group["record_number"].to_numpy()
                selected_parts.append(part)

            selected = pd.concat(selected_parts, ignore_index=True)
            selected["source_partition"] = partition
            selected = normalize_labels(selected, dataset)

            expected_binary_counts = result["binary_class_counts"]
            actual_counts = (
                selected["binary_target"].value_counts().sort_index().to_dict()
            )
            actual_counts = {str(k): int(v) for k, v in actual_counts.items()}

            if actual_counts != expected_binary_counts:
                fail(
                    f"{dataset} {partition}: binary class counts do not match "
                    f"script-05.\n"
                    f"  Expected: {expected_binary_counts}\n"
                    f"  Actual:   {actual_counts}"
                )

            curated[partition] = selected
            log(
                f"{dataset} {partition}: loaded {len(selected)} curated records "
                f"from {len(references['source_path'].unique())} source file(s)."
            )

        feature_columns, numeric_columns, categorical_columns = identify_columns(
            curated["train"], dataset
        )

        log(
            f"{dataset}: {len(numeric_columns)} numeric and "
            f"{len(categorical_columns)} categorical predictors."
        )

        cleaned = {}
        for partition in ["train", "validation", "test"]:
            cleaned[partition], exclusions = apply_value_quality_rules(
                curated[partition].copy(),
                numeric_columns,
                categorical_columns,
                dataset,
                partition,
            )
            all_exclusions.extend(exclusions)

        X_train = cleaned["train"][feature_columns].copy()
        X_validation = cleaned["validation"][feature_columns].copy()
        X_test = cleaned["test"][feature_columns].copy()

        numeric_imputer = None
        categorical_imputer = None
        encoder = None
        scaler = None

        if numeric_columns:
            numeric_imputer = SimpleImputer(
                strategy="median", keep_empty_features=True
            )
            X_train_numeric = numeric_imputer.fit_transform(X_train[numeric_columns])
            X_validation_numeric = numeric_imputer.transform(
                X_validation[numeric_columns]
            )
            X_test_numeric = numeric_imputer.transform(X_test[numeric_columns])

            scaler = StandardScaler()
            X_train_numeric = scaler.fit_transform(X_train_numeric)
            X_validation_numeric = scaler.transform(X_validation_numeric)
            X_test_numeric = scaler.transform(X_test_numeric)
        else:
            X_train_numeric = np.empty((len(X_train), 0), dtype=np.float64)
            X_validation_numeric = np.empty((len(X_validation), 0), dtype=np.float64)
            X_test_numeric = np.empty((len(X_test), 0), dtype=np.float64)

        if categorical_columns:
            categorical_imputer = SimpleImputer(
                strategy="most_frequent", keep_empty_features=True
            )
            X_train_categorical = categorical_imputer.fit_transform(
                X_train[categorical_columns].astype(str)
            )
            X_validation_categorical = categorical_imputer.transform(
                X_validation[categorical_columns].astype(str)
            )
            X_test_categorical = categorical_imputer.transform(
                X_test[categorical_columns].astype(str)
            )

            encoder = OneHotEncoder(
                handle_unknown=UNKNOWN_CATEGORY_POLICY,
                sparse_output=False,
                dtype=np.float64,
            )
            X_train_categorical = encoder.fit_transform(X_train_categorical)
            X_validation_categorical = encoder.transform(X_validation_categorical)
            X_test_categorical = encoder.transform(X_test_categorical)
        else:
            X_train_categorical = np.empty((len(X_train), 0), dtype=np.float64)
            X_validation_categorical = np.empty(
                (len(X_validation), 0), dtype=np.float64
            )
            X_test_categorical = np.empty((len(X_test), 0), dtype=np.float64)

        X_train_transformed = np.hstack(
            [X_train_numeric, X_train_categorical]
        ).astype(np.float64)
        X_validation_transformed = np.hstack(
            [X_validation_numeric, X_validation_categorical]
        ).astype(np.float64)
        X_test_transformed = np.hstack(
            [X_test_numeric, X_test_categorical]
        ).astype(np.float64)

        feature_names = list(numeric_columns)

        if encoder is not None:
            feature_names.extend(
                [
                    f"categorical__{name}"
                    for name in encoder.get_feature_names_out(categorical_columns)
                ]
            )

        if len(feature_names) != X_train_transformed.shape[1]:
            fail(
                f"{dataset}: feature-name count {len(feature_names)} does not match "
                f"transformed feature count {X_train_transformed.shape[1]}."
            )

        transformed_partitions[dataset] = {
            "train": X_train_transformed,
            "validation": X_validation_transformed,
            "test": X_test_transformed,
        }

        transformed_hashes[dataset] = {
            partition: hash_feature_matrix(X)
            for partition, X in transformed_partitions[dataset].items()
        }

        transformed_labels[dataset] = {
            partition: cleaned[partition]["binary_target"].to_numpy(dtype=np.int8)
            for partition in ["train", "validation", "test"]
        }

        dataset_output_dir = output_dir / dataset.lower().replace("-", "_")

        if not args.dry_run:
            dataset_output_dir.mkdir(parents=True, exist_ok=True)

            joblib.dump(
                numeric_imputer, dataset_output_dir / "numeric_imputer.joblib"
            )
            joblib.dump(scaler, dataset_output_dir / "scaler.joblib")
            joblib.dump(
                categorical_imputer,
                dataset_output_dir / "categorical_imputer.joblib",
            )
            joblib.dump(encoder, dataset_output_dir / "encoder.joblib")

            np.save(dataset_output_dir / "X_train.npy", X_train_transformed)
            np.save(dataset_output_dir / "X_validation.npy", X_validation_transformed)
            np.save(dataset_output_dir / "X_test.npy", X_test_transformed)

            np.save(
                dataset_output_dir / "y_train.npy",
                transformed_labels[dataset]["train"],
            )
            np.save(
                dataset_output_dir / "y_validation.npy",
                transformed_labels[dataset]["validation"],
            )
            np.save(
                dataset_output_dir / "y_test.npy",
                transformed_labels[dataset]["test"],
            )

            with (dataset_output_dir / "feature_names.json").open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(
                    {
                        "feature_columns": feature_columns,
                        "numeric_columns": numeric_columns,
                        "categorical_columns": categorical_columns,
                        "ordered_model_input_features": feature_names,
                    },
                    f,
                    indent=2,
                )

        partition_outputs.append(
            {
                "dataset": dataset,
                "feature_columns": feature_columns,
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "model_input_feature_count": len(feature_names),
                "records": {
                    partition: {
                        "curated_records": int(len(curated[partition])),
                        "model_input_records": int(X.shape[0]),
                        "preprocessing_excluded_records": int(
                            len(curated[partition]) - X.shape[0]
                        ),
                        "binary_class_counts": {
                            str(k): int(v)
                            for k, v in pd.Series(
                                transformed_labels[dataset][partition]
                            )
                            .value_counts()
                            .sort_index()
                            .items()
                        },
                    }
                    for partition, X in transformed_partitions[dataset].items()
                },
            }
        )

    collision_audit = {}
    for dataset in datasets:
        collision_audit[dataset] = audit_collisions(
            dataset,
            transformed_partitions[dataset],
            transformed_hashes[dataset],
            transformed_labels[dataset],
        )

        unexpected_collisions = False

    for dataset, audit in collision_audit.items():
        for partition, counts in audit["within_partition_duplicate_groups"].items():
            if counts["groups"] > 0:
                log(
                    f"[NOTE] {dataset}/{partition}: {counts['groups']} within-partition "
                    f"duplicate model-input groups remain; retained and documented."
                )

        if audit["cross_partition_duplicate_groups"] > 0:
            unexpected_collisions = True
            log(
                f"[FAIL] {dataset}: {audit['cross_partition_duplicate_groups']} "
                f"cross-partition duplicate model-input groups found."
            )

    artifact_hashes = {}
    if not args.dry_run:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                artifact_hashes[str(path.relative_to(output_dir))] = sha256_file(path)

    final_report = {
        "script": "06_preprocess_and_audit",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "input_script05_report": str(SCRIPT05_REPORT),
        "input_script05_report_sha256": report05.get("deduplication_report_sha256"),
        "configuration": {
            "record_index_base": RECORD_INDEX_BASE,
            "negative_header_length_policy": NEGATIVE_HEADER_LENGTH_POLICY,
            "infinite_value_policy": INFINITE_VALUE_POLICY,
            "unknown_category_policy": UNKNOWN_CATEGORY_POLICY,
            "max_abs_numeric_value": MAX_ABS_NUMERIC_VALUE,
            "cicids_excluded_columns": sorted(CICIDS_EXCLUDED_COLUMNS),
        },
        "datasets": partition_outputs,
        "preprocessing_exclusions": all_exclusions,
        "model_input_collision_audit": collision_audit,
        "unexpected_model_input_collisions": unexpected_collisions,
        "preprocessing_completed": not unexpected_collisions,
        "ready_for_model_training": not unexpected_collisions,
        "artifact_hashes": artifact_hashes,
    }

    report_path = output_dir / "preprocessing_report.json"

    if not args.dry_run:
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2)

        report_hash = sha256_file(report_path)
        final_report["preprocessing_report_sha256"] = report_hash

        with report_path.open("w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2)

    print("\n================ 06 PREPROCESSING SUMMARY ================")

    for item in partition_outputs:
        dataset = item["dataset"]
        print(f"\n{dataset}")
        print(f"  Model-input features: {item['model_input_feature_count']}")

        for partition, stats in item["records"].items():
            print(
                f"  {partition}: {stats['model_input_records']} records "
                f"({stats['preprocessing_excluded_records']} excluded by preprocessing)"
            )

    print("\nModel-input collision audit:")

    for dataset, audit in collision_audit.items():
        print(f"\n{dataset}")
        for partition, counts in audit["within_partition_duplicate_groups"].items():
            print(
                f"  {partition}: {counts['groups']} duplicate groups, "
                f"{counts['records_in_duplicate_groups']} records"
            )
        print(
            f"  Cross-partition duplicate groups: "
            f"{audit['cross_partition_duplicate_groups']}"
        )
        print(
            f"  Cross-partition conflicting-label groups: "
            f"{audit['cross_partition_conflicting_label_groups']}"
        )

    print("\n==========================================================")

    if args.dry_run:
        print("\nDry run complete. No outputs were written.")
    elif unexpected_collisions:
        print(
            f"\n[FAIL] Unexpected model-input collisions remain. Report: {report_path}"
        )
        sys.exit(2)
    else:
        print("\n[PASS] 06_preprocess_and_audit completed successfully.")
        print(f"Report: {report_path}")
        print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()
