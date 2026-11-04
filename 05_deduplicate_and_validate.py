from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from itertools import zip_longest
import argparse
import csv
import gzip
import hashlib
import json
import math
import sqlite3
import sys

ROOT = Path(__file__).resolve().parent
PARTITIONS = ("train", "validation", "test")
PRIORITY = {"train": 0, "validation": 1, "test": 2, "excluded": -1}
CIC_EXCLUDED = {"id", "Flow ID", "Src IP", "Dst IP", "Timestamp", "Label", "Attempted Category"}


def project_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path is outside the project: {value}")
    return path


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def digest(values):
    text = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).digest()


def numeric_token(raw):
    text = raw.strip()
    if text.casefold() in {"", "na", "n/a", "null", "none", "?"}:
        return "missing:" + text.casefold()
    value = float(text)
    if math.isnan(value):
        return "numeric:nan"
    if math.isinf(value):
        if text.casefold() not in {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
            raise ValueError(f"Numeric overflow to infinity: {text[:80]}")
        return "numeric:+inf" if value > 0 else "numeric:-inf"
    if value == 0:
        value = 0.0
    return "float64:" + value.hex()


def canonical_key(row, indexes, categorical):
    return digest(["category:" + row[i].strip() if i in categorical
                   else numeric_token(row[i]) for i in indexes])


def save_report(output, report):
    (output / "deduplication_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def import_manifests(db, partition_report):
    seen = set()
    references = []
    for item in partition_report["partitions"]:
        dataset, split = item["dataset"], item["partition"]
        if dataset not in {"NSL-KDD", "CICIDS2017"} or split not in PRIORITY:
            raise ValueError(f"Unexpected dataset/partition: {dataset}, {split}")
        if (dataset, split) in seen:
            raise ValueError("Duplicate dataset/partition entry.")
        seen.add((dataset, split))
        path = project_path(item["manifest"])
        actual_hash = file_hash(path)
        if actual_hash != item["manifest_sha256"]:
            raise ValueError(f"Manifest hash mismatch: {path.name}")
        count = 0
        classes = Counter()
        batch = []
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"record_id", "source", "data_record_number", "group_id", "feature_sha256", "original_label", "binary_target", "reason"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing manifest columns: {path.name}")
            for row in reader:
                count += 1
                number = int(row["data_record_number"])
                target = int(row["binary_target"])
                if number < 1 or target not in (0, 1):
                    raise ValueError("Invalid manifest record number or binary target.")
                if row["record_id"] != f"{row['source']}:record:{number}":
                    raise ValueError("Inconsistent record ID.")
                project_path(row["source"])
                classes[str(target)] += 1
                status = "input_excluded" if split == "excluded" else "candidate"
                batch.append((dataset, split, PRIORITY[split], row["source"], number,
                              row["record_id"], row["group_id"], row["original_label"], target,
                              row["feature_sha256"], row["reason"], status))
                if len(batch) >= 10000:
                    db.executemany("INSERT INTO records(dataset,partition,priority,source,record_no,record_id,original_group_id,label,target,old_key,input_reason,status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", batch)
                    db.commit()
                    batch.clear()
            if batch:
                db.executemany("INSERT INTO records(dataset,partition,priority,source,record_no,record_id,original_group_id,label,target,old_key,input_reason,status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", batch)
                db.commit()
        if count != item["records"] or dict(classes) != item["binary_class_counts"]:
            raise ValueError(f"Manifest counts differ from partition report: {path.name}")
        references.append({"dataset": dataset, "partition": split,
                           "manifest": item["manifest"], "sha256": actual_hash, "records": count})
    expected = {(d, p) for d in ("NSL-KDD", "CICIDS2017") for p in PRIORITY}
    if seen != expected:
        raise ValueError("Expected all eight input partition manifests, including exclusions.")
    return references


def index_raw_source(db, item, schema, attempted_policy):
    source = item["path"]
    path = project_path(source)
    before = path.stat()
    if file_hash(path) != item["sha256"]:
        raise ValueError(f"Raw input hash mismatch: {source}")
    cic = path.suffix.lower() == ".csv"
    dataset = "CICIDS2017" if cic else "NSL-KDD"
    print(f"\nCanonicalizing {source} ...", flush=True)
    metadata = db.execute("SELECT id,record_no,label,target,old_key,dataset FROM records WHERE source=? ORDER BY record_no", [source])
    batch = []
    count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        if cic:
            header = [v.strip() for v in next(reader)]
            if len(header) != len(set(header)):
                raise ValueError("Duplicate normalized headers.")
            expected_columns = [name for name in header if name not in CIC_EXCLUDED]
            if expected_columns != schema:
                raise ValueError(f"CIC schema changed since partition creation: {source}")
            pos = {name: i for i, name in enumerate(header)}
            required = CIC_EXCLUDED | {"Src Port", "Dst Port", "Protocol"}
            if not required.issubset(header):
                raise ValueError(f"Missing CIC metadata columns: {source}")
            indexes = [pos[name] for name in schema]
            categorical = set()
            label_pos = pos["Label"]
        else:
            header = [f"field_{i:02d}" for i in range(1, 42)] + ["label", "difficulty"]
            if header[:41] != schema:
                raise ValueError(f"NSL positional schema changed: {source}")
            indexes = list(range(41))
            categorical = {1, 2, 3}
            label_pos = 41
        for number, pair in enumerate(zip_longest(reader, metadata), 1):
            raw, meta = pair
            if raw is None or meta is None:
                raise ValueError(f"Raw/manifest record-count disagreement: {source}")
            if len(raw) != len(header):
                raise ValueError(f"Wrong field count: {source}:record:{number}")
            row = [value.strip() for value in raw]
            record_id, manifest_number, label, target, old_key, manifest_dataset = meta
            if manifest_number != number or manifest_dataset != dataset or row[label_pos] != label:
                raise ValueError(f"Record identity/label mismatch: {source}:record:{number}")
            lexical = digest([row[i] for i in indexes]).hex()
            if lexical != old_key:
                raise ValueError(f"Old feature fingerprint mismatch: {source}:record:{number}")
            expected_target = int(label.casefold() not in {"normal", "benign"})
            endpoint = None
            if cic:
                attempted = label.endswith(" - Attempted")
                if attempted != (row[pos["Attempted Category"]] != "-1"):
                    raise ValueError("Attempted annotation disagreement.")
                if attempted:
                    expected_target = 0 if attempted_policy == "benign" else 1
                ends = sorted([[row[pos["Src IP"]], row[pos["Src Port"]]],
                               [row[pos["Dst IP"]], row[pos["Dst Port"]]]])
                endpoint = digest([ends, row[pos["Protocol"]]])
            if target != expected_target:
                raise ValueError(f"Binary mapping mismatch: {source}:record:{number}")
            key = canonical_key(row, indexes, categorical)
            batch.append((key, endpoint, record_id))
            count = number
            if len(batch) >= 10000:
                db.executemany("UPDATE records SET canonical_key=?,endpoint_key=? WHERE id=?", batch)
                db.commit()
                batch.clear()
            if number % 100000 == 0:
                print(f"  {number:,} records checked", flush=True)
        if batch:
            db.executemany("UPDATE records SET canonical_key=?,endpoint_key=? WHERE id=?", batch)
            db.commit()
    after = path.stat()
    if count != item["records"] or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Raw input count changed or file changed during reading: {source}")
    return {"source": source, "sha256": item["sha256"], "records_verified": count}


def check_remaining(db, key_column, dataset=None):
    condition = "status='keep'"
    params = []
    if dataset:
        condition += " AND dataset=?"
        params.append(dataset)
    groups = "dataset," + key_column
    sql = f"SELECT COUNT(*) FROM (SELECT {groups} FROM records WHERE {condition} GROUP BY {groups} HAVING COUNT(DISTINCT partition)>1)"
    return db.execute(sql, params).fetchone()[0]


def main():
    parser = argparse.ArgumentParser(description="05: deduplicate/validate manifests only; no fitted preprocessing.")
    parser.add_argument("--partition-report", help="Project-relative path; defaults to latest partition_report.json under reports.")
    args = parser.parse_args()
    if args.partition_report:
        path = project_path(args.partition_report)
        previous = json.loads(path.read_text(encoding="utf-8"))
    else:
        choices = [(json.loads(p.read_text(encoding="utf-8")), p) for p in (ROOT / "reports").rglob("partition_report.json")]
        if not choices:
            raise SystemExit("No partition_report.json found. Keep the script-04 report under reports.")
        previous, path = max(choices, key=lambda pair: pair[0].get("started_utc", ""))
    if not previous.get("completed"):
        raise SystemExit("Input partition report did not complete.")
    integrity_path = project_path(previous["integrity_audit"])
    if file_hash(integrity_path) != previous["integrity_audit_sha256"]:
        raise SystemExit("Integrity-report hash does not match the partition report.")
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if not integrity.get("completed"):
        raise SystemExit("Referenced integrity audit did not complete.")
    attempted_policy = previous["attempted_policy"]
    if attempted_policy not in {"benign", "attack", "exclude"}:
        raise SystemExit("Unknown Attempted policy.")
    stamp = datetime.now(timezone.utc)
    output = ROOT / "reports" / stamp.strftime("deduplication_%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "started_utc": stamp.isoformat(), "completed": False,
        "script_sha256": file_hash(Path(__file__)), "python_version": sys.version,
        "input_partition_report": path.relative_to(ROOT).as_posix(),
        "input_partition_report_sha256": file_hash(path), "attempted_policy": attempted_policy,
        "policy": {
            "raw_data_modified": False, "input_manifests_modified": False,
            "preprocessing_fitted": False, "models_trained": False,
            "comparison_representation": "Finite numeric predictors parsed as float64 and compared via float.hex(); signed zero unified. NSL fields 2,3,4 remain stripped categorical strings. CIC Protocol/ports compare numerically. No encoding, rounding, imputation or scaling is performed.",
            "missing_and_infinite_policy": "Missing markers and numeric NaN/infinity receive explicit comparison tokens, not fitted replacements. Infinite values and negative lengths are NOT repaired here.",
            "cross_partition_priority": "test > validation > train. Lower-priority feature-equal records are excluded based on predictor values only, not labels. Previously excluded records never return.",
            "conflicting_targets": "After cross-partition precedence, quarantine ALL records of each canonical-feature group with both binary targets in its remaining partition. This is label-aware curation of that partition, not pooled majority-vote relabelling. Held-out exclusions change the evaluation population.",
            "same_target_duplicates": "Keep the first remaining representative by source path then 1-based data-record number; preserve removed member IDs and original labels in exclusion logs.",
            "test_set_status": "All output tests are CURATED. The original official NSL test file and script-04 test manifest remain unchanged for a separate reference evaluation. Do not claim benchmark-identical evaluation.",
            "schema_status": "Use the explicitly saved script-04 predictor schema. Any later predictor removal, float32 conversion, clipping, encoding or imputation requires new model-input collision checks.",
            "near_duplicates_checked": False, "true_sessions_identified": False,
            "hash_collisions_independently_checked": False,
        }, "input_manifests": [], "raw_inputs_verified": [],
    }
    db = sqlite3.connect(output / "deduplication_index.sqlite")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("""CREATE TABLE records (
        id INTEGER PRIMARY KEY, dataset TEXT, partition TEXT, priority INTEGER,
        source TEXT, record_no INTEGER, record_id TEXT UNIQUE, original_group_id TEXT,
        label TEXT, target INTEGER, old_key TEXT, input_reason TEXT, status TEXT,
        canonical_key BLOB, endpoint_key BLOB, UNIQUE(source,record_no))""")
    try:
        print("Checking and importing all script-04 manifests ...", flush=True)
        report["input_manifests"] = import_manifests(db, previous)
        db.execute("CREATE INDEX source_records ON records(source,record_no)")
        source_counts = dict(db.execute("SELECT source,COUNT(*) FROM records GROUP BY source"))
        expected = {item["path"]: item["records"] for item in integrity["files"]}
        if source_counts != expected:
            raise ValueError("Manifest source counts do not cover the exact audited raw files.")
        schemas = previous["scope"]["fingerprint_schema"]
        for item in integrity["files"]:
            schema = schemas[item["path"]]
            if schema != item["fingerprint_columns_in_order"]:
                raise ValueError("Integrity and partition predictor schemas differ.")
            report["raw_inputs_verified"].append(index_raw_source(db, item, schema, attempted_policy))
        report["predictor_schemas"] = schemas
        if db.execute("SELECT COUNT(*) FROM records WHERE canonical_key IS NULL").fetchone()[0]:
            raise ValueError("Some records lack canonical fingerprints.")
        db.execute("CREATE INDEX canonical_lookup ON records(dataset,canonical_key,partition)")
        print("Resolving cross-partition numeric-equivalent patterns without consulting labels ...", flush=True)
        db.execute("""CREATE TABLE preferred AS SELECT dataset,canonical_key,MAX(priority) AS priority
            FROM records WHERE status='candidate' GROUP BY dataset,canonical_key""")
        db.execute("CREATE UNIQUE INDEX preferred_key ON preferred(dataset,canonical_key)")
        db.execute("""UPDATE records SET status='cross_partition_lower_priority'
            WHERE status='candidate' AND priority < (
            SELECT priority FROM preferred p WHERE p.dataset=records.dataset AND p.canonical_key=records.canonical_key)""")
        print("Quarantining binary-target conflicts within the remaining partition ...", flush=True)
        db.execute("""CREATE TABLE conflicts AS SELECT dataset,partition,canonical_key,COUNT(*) AS records
            FROM records WHERE status='candidate' GROUP BY dataset,partition,canonical_key
            HAVING COUNT(DISTINCT target)>1""")
        db.execute("CREATE UNIQUE INDEX conflict_key ON conflicts(dataset,partition,canonical_key)")
        db.execute("""UPDATE records SET status='ambiguous_binary_target'
            WHERE status='candidate' AND EXISTS (SELECT 1 FROM conflicts c
            WHERE c.dataset=records.dataset AND c.partition=records.partition AND c.canonical_key=records.canonical_key)""")
        print("Selecting deterministic representatives for same-target duplicates ...", flush=True)
        db.execute("""CREATE TABLE ranked AS SELECT id,
            ROW_NUMBER() OVER (PARTITION BY dataset,partition,canonical_key ORDER BY source,record_no) AS position
            FROM records WHERE status='candidate'""")
        db.execute("CREATE UNIQUE INDEX ranked_id ON ranked(id)")
        db.execute("UPDATE records SET status='same_target_duplicate' WHERE id IN (SELECT id FROM ranked WHERE position>1)")
        db.execute("UPDATE records SET status='keep' WHERE status='candidate'")
        db.execute("CREATE TABLE representatives AS SELECT dataset,canonical_key,record_id AS representative FROM records WHERE status='keep'")
        db.execute("CREATE UNIQUE INDEX representative_key ON representatives(dataset,canonical_key)")
        db.commit()
        cross = check_remaining(db, "canonical_key")
        endpoints = check_remaining(db, "endpoint_key", "CICIDS2017")
        duplicates = db.execute("""SELECT COUNT(*) FROM (SELECT dataset,partition,canonical_key
            FROM records WHERE status='keep' GROUP BY dataset,partition,canonical_key HAVING COUNT(*)>1)""").fetchone()[0]
        if cross or endpoints or duplicates:
            raise ValueError(f"Integrity checks failed: cross_features={cross}, cross_endpoints={endpoints}, within_duplicates={duplicates}")
        report["checks"] = {"canonical_feature_groups_crossing_partitions": cross,
                            "cic_endpoint_groups_crossing_partitions": endpoints,
                            "retained_duplicate_groups_within_partitions": duplicates}
        print("Writing curated manifests and complete exclusion logs ...", flush=True)
        report["partitions"] = []
        fieldnames = ["record_id", "source", "data_record_number", "original_group_id",
                      "original_label", "binary_target", "lexical_feature_sha256", "canonical_feature_sha256"]
        for dataset in ("NSL-KDD", "CICIDS2017"):
            prefix = dataset.lower().replace("-", "_")
            for split in PARTITIONS:
                destination = output / f"{prefix}_{split}_curated_ids.csv.gz"
                count = 0
                classes, labels = Counter(), Counter()
                with gzip.open(destination, "wt", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(fieldnames)
                    query = """SELECT record_id,source,record_no,original_group_id,label,target,old_key,hex(canonical_key)
                        FROM records WHERE dataset=? AND partition=? AND status='keep' ORDER BY source,record_no"""
                    for row in db.execute(query, [dataset, split]):
                        writer.writerow(list(row[:-1]) + [row[-1].lower()])
                        count += 1
                        labels[row[4]] += 1
                        classes[str(row[5])] += 1
                before = db.execute("SELECT COUNT(*) FROM records WHERE dataset=? AND partition=?", [dataset, split]).fetchone()[0]
                reasons = dict(db.execute("SELECT status,COUNT(*) FROM records WHERE dataset=? AND partition=? AND status!='keep' GROUP BY status", [dataset, split]))
                conflict_groups = db.execute("SELECT COUNT(*) FROM conflicts WHERE dataset=? AND partition=?", [dataset, split]).fetchone()[0]
                if count + sum(reasons.values()) != before:
                    raise ValueError("Partition accounting failed.")
                report["partitions"].append({"dataset": dataset, "partition": split,
                    "evaluation_status": "curated_not_official_unchanged" if split=="test" else "curated",
                    "input_records": before, "retained_records": count, "excluded_records": before-count,
                    "exclusion_reasons": reasons, "quarantined_binary_conflict_groups": conflict_groups,
                    "binary_class_counts": dict(classes), "original_label_counts": dict(labels),
                    "manifest": destination.relative_to(ROOT).as_posix(), "manifest_sha256": file_hash(destination)})
            exclusions = output / f"{prefix}_all_exclusions.csv.gz"
            with gzip.open(exclusions, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["record_id", "input_partition", "source", "data_record_number", "original_label", "binary_target", "exclusion_reason", "script04_reason", "canonical_feature_sha256", "retained_representative_record_id"])
                query = """SELECT r.record_id,r.partition,r.source,r.record_no,r.label,r.target,r.status,r.input_reason,
                    hex(r.canonical_key),COALESCE(p.representative,'') FROM records r LEFT JOIN representatives p
                    ON p.dataset=r.dataset AND p.canonical_key=r.canonical_key
                    WHERE r.dataset=? AND r.status!='keep' ORDER BY r.source,r.record_no"""
                for row in db.execute(query, [dataset]):
                    writer.writerow(list(row[:8]) + [row[8].lower(), row[9]])
            report.setdefault("exclusion_logs", []).append({"dataset": dataset,
                "path": exclusions.relative_to(ROOT).as_posix(), "sha256": file_hash(exclusions)})
        report["status_counts_including_prior_exclusions"] = [
            {"dataset": d, "input_partition": p, "status": s, "records": n}
            for d,p,s,n in db.execute("SELECT dataset,partition,status,COUNT(*) FROM records GROUP BY dataset,partition,status ORDER BY dataset,partition,status")]
        input_total = db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        kept_total = db.execute("SELECT COUNT(*) FROM records WHERE status='keep'").fetchone()[0]
        report["accounting"] = {"all_input_records_including_script04_exclusions": input_total,
                                "retained_records": kept_total, "excluded_records_including_prior": input_total-kept_total}
        warnings = [
            "Output tests are curated: held-out label-conflict quarantine is label-aware evaluation curation. Report every exclusion and retain original-test reference evaluation.",
            "Numeric comparison is float64 before fitting transforms. Float32 conversion, encoding, clipping or imputation may create additional collisions; script 06 must check them.",
            "Negative header lengths and infinite rates were not corrected; near-duplicate and true-session checks remain open.",
            "Deduplication changes class balance and sample weighting; do not substitute old results or old partition counts.",
        ]
        for item in report["partitions"]:
            if item["retained_records"]==0 or len(item["binary_class_counts"])<2:
                warnings.append(f"STOP: {item['dataset']} {item['partition']} is empty or lacks a binary class.")
        report["warnings"] = warnings
        report["ready_for_preprocessing_review"] = not any(w.startswith("STOP:") for w in warnings)
        report["completed"] = True
        for item in report["partitions"]:
            print(f"  {item['dataset']} {item['partition']}: {item['input_records']:,} -> {item['retained_records']:,}; reasons={item['exclusion_reasons']}")
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        db.close()
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save_report(output, report)
        notes = ["# Script 05: deduplication and validation", "", f"Completed: {report['completed']}", "",
                 "Source files and script-04 manifests remain unchanged.",
                 "This script creates curated manifests only, not transformed features or models.",
                 "Cross-partition numeric-equal patterns: keep the higher-priority partition (test > validation > train), without using labels for this priority.",
                 "Within the remaining partition, quarantine every conflicting-binary-target group; do not vote or relabel.",
                 "For same-target duplicates, keep the earliest source-path/record-number representative and log the removed members.",
                 "All output test sets are curated. Preserve an original NSL official-test reference evaluation and disclose label-aware held-out exclusions.",
                 "Negative header lengths, infinities, near-duplicates, final-transform collisions and release provenance remain open.",
                 "Script 06 has not run: no imputer, encoder or scaler has been fitted.", "",
                 "Inspect deduplication_report.json. Keep deduplication_index.sqlite locally."]
        (output / "deduplication_notes.md").write_text("\n".join(notes)+"\n", encoding="utf-8")
        print(f"\nReports saved in: {output}")
    print("deduplication_report.json needs review. Do not run preprocessing or training yet.")


if __name__ == "__main__":
    main()