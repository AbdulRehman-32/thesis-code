from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import csv
import hashlib
import json
import math
import platform
import sys

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
EXPECTED = ["KDDTrain+.txt", "KDDTest+.txt", "monday.csv", "tuesday.csv",
            "wednesday.csv", "thursday.csv", "friday.csv"]
MISSING = {"", "na", "n/a", "nan", "null", "none", "?"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def new_stats(name):
    return {"column": name, "missing_tokens": 0, "finite_numeric": 0,
            "positive_infinity": 0, "negative_infinity": 0,
            "nonnumeric": 0, "nonnumeric_examples": [],
            "finite_min": None, "finite_max": None}


def update_stats(stats, raw):
    value = raw.strip()
    if value.casefold() in MISSING:
        stats["missing_tokens"] += 1
        return
    try:
        number = float(value)
    except (ValueError, OverflowError):
        stats["nonnumeric"] += 1
        examples = stats["nonnumeric_examples"]
        if len(examples) < 5 and value not in examples:
            examples.append(value[:160])
        return
    if math.isnan(number):
        stats["missing_tokens"] += 1
    elif math.isinf(number):
        key = "positive_infinity" if number > 0 else "negative_infinity"
        stats[key] += 1
    else:
        stats["finite_numeric"] += 1
        if stats["finite_min"] is None or number < stats["finite_min"]:
            stats["finite_min"] = number
        if stats["finite_max"] is None or number > stats["finite_max"]:
            stats["finite_max"] = number


def audit_file(path):
    print(f"\nAuditing {path.name} ...", flush=True)
    before = path.stat()
    digest = sha256(path)
    is_cic = path.suffix.lower() == ".csv"
    labels = Counter()
    attempted = Counter()
    joint = Counter()
    widths = Counter()
    malformed = []
    accepted = 0
    records = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        if is_cic:
            header = next(reader, [])
            names = [name.strip() for name in header]
            duplicates = [name for name, n in Counter(names).items() if n > 1]
            if duplicates:
                raise ValueError(f"Duplicate normalized column names: {duplicates}")
            if "Label" not in names or "Attempted Category" not in names:
                raise ValueError("Expected Label and Attempted Category columns are missing.")
            label_index = names.index("Label")
            attempted_index = names.index("Attempted Category")
        else:
            header = None
            names = [f"field_{i:02d}" for i in range(1, 42)] + ["label", "difficulty"]
            label_index = 41
            attempted_index = None
        columns = [new_stats(name) for name in names]
        for row in reader:
            records += 1
            widths[len(row)] += 1
            if len(row) != len(names):
                if len(malformed) < 10:
                    malformed.append({"data_record_number": records,
                                      "csv_physical_end_line": reader.line_num,
                                      "field_count": len(row)})
                continue
            accepted += 1
            label = row[label_index]
            labels[label] += 1
            if attempted_index is not None:
                category = row[attempted_index]
                attempted[category] += 1
                joint[(label, category)] += 1
            for stats, value in zip(columns, row):
                update_stats(stats, value)
            if records % 100000 == 0:
                print(f"  {records:,} records inspected", flush=True)
    after = path.stat()
    stable = before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
    result = {
        "path": path.relative_to(ROOT).as_posix(), "sha256": digest,
        "size_bytes": before.st_size, "stable_during_audit": stable,
        "encoding": "utf-8-sig", "delimiter": ",", "has_header": is_cic,
        "raw_header": header, "audit_column_names": names,
        "nsl_schema_note": None if is_cic else (
            "Headerless 43-field layout expected: 41 predictors, label, difficulty. "
            "Predictor names remain positional pending explicit schema documentation."
        ),
        "data_records_seen": records, "accepted_width_records": accepted,
        "wrong_width_records": records - accepted,
        "field_count_histogram": dict(widths),
        "wrong_width_examples": malformed,
        "raw_label_counts_accepted_records": dict(labels),
        "raw_attempted_category_counts_accepted_records": dict(attempted),
        "label_attempted_pairs": [
            {"label": label, "attempted_category": category, "count": count}
            for (label, category), count in sorted(joint.items())
        ],
        "column_statistics_accepted_records": columns,
    }
    print(f"  Records: {records:,}; wrong-width: {records - accepted:,}")
    print(f"  Original labels: {dict(labels)}")
    if is_cic:
        print(f"  Attempted Category: {dict(attempted)}")
    return result


def main():
    paths = []
    for filename in EXPECTED:
        matches = list(DATA.rglob(filename)) if DATA.is_dir() else []
        if len(matches) != 1:
            raise SystemExit(f"Expected exactly one {filename} under data; found {len(matches)}.")
        paths.append(matches[0])

    timestamp = datetime.now(timezone.utc)
    output = ROOT / "reports" / timestamp.strftime("audit_%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "started_utc": timestamp.isoformat(), "script_sha256": sha256(Path(__file__)),
        "python_version": sys.version, "platform": platform.platform(),
        "missing_token_policy": sorted(MISSING),
        "scope": {
            "raw_data_modified": False, "preprocessing_fitted": False,
            "binary_label_mapping_applied": False, "attempted_filter_applied": False,
            "duplicates_audited": False, "train_test_overlap_audited": False,
            "dataset_release_verified": False,
            "note": "Descriptive raw-data audit only. No imputation, encoding, scaling or training."
        }, "files": [], "errors": [],
    }
    for path in paths:
        try:
            report["files"].append(audit_file(path))
        except (OSError, UnicodeError, csv.Error, ValueError) as exc:
            report["errors"].append({"file": path.name, "error": str(exc)})
            print(f"  ERROR: {path.name}: {exc}", flush=True)
        (output / "data_audit.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )

    headers = [f["raw_header"] for f in report["files"] if f["has_header"]]
    report["cic_headers_identical_including_order"] = (
        len(headers) == 5 and all(h == headers[0] for h in headers)
    )
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["audit_completed_without_record_or_file_errors"] = (
        len(report["files"]) == len(EXPECTED) and not report["errors"]
        and all(f["stable_during_audit"] and f["wrong_width_records"] == 0
                for f in report["files"])
    )
    (output / "data_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    rows = []
    for item in report["files"]:
        for stats in item["column_statistics_accepted_records"]:
            row = {"file": item["path"], **stats}
            row["nonnumeric_examples"] = json.dumps(row["nonnumeric_examples"], ensure_ascii=False)
            rows.append(row)
    if rows:
        with (output / "column_quality.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    note = ["# Raw-data audit", "", f"Started (UTC): {timestamp.isoformat()}", "",
            "No rows or columns were removed from the source files. Wrong-width records, if any,",
            "are counted but excluded from label and column summaries; they must be resolved.",
            "No Attempted-category exclusion or binary label mapping was applied.",
            "No duplicate, split-overlap, release or preprocessing-validity claim is made.", "",
            "Whole-dataset descriptive ranges must not be reused as learned preprocessing or",
            "attack bounds; those decisions require a documented training-only procedure.", ""]
    for item in report["files"]:
        note.append(f"- {item['path']}: {item['data_records_seen']:,} records; "
                    f"{item['wrong_width_records']:,} wrong-width records.")
    (output / "audit_notes.md").write_text("\n".join(note) + "\n", encoding="utf-8")
    print(f"\nReports saved in: {output}")
    print(f"Audit completed without record/file errors: {report['audit_completed_without_record_or_file_errors']}")
    print("Upload data_audit.json for review. Do not train models yet.")


if __name__ == "__main__":
    main()