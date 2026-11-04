from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import csv
import hashlib
import json
import math
import sqlite3
import sys

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
EXPECTED = ["KDDTrain+.txt", "KDDTest+.txt", "monday.csv", "tuesday.csv",
            "wednesday.csv", "thursday.csv", "friday.csv"]
CIC_EXCLUDED = {"id", "Flow ID", "Src IP", "Dst IP", "Timestamp", "Label", "Attempted Category"}
ENDPOINT_FIELDS = ["Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol"]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(values):
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).digest()


def duplicate_summary(connection, where, parameters):
    groups = f"""SELECT feature_key, COUNT(*) AS n,
        COUNT(DISTINCT label) AS raw_classes,
        COUNT(DISTINCT target) AS binary_classes,
        COUNT(DISTINCT source) AS source_count
        FROM records WHERE {where} GROUP BY feature_key"""
    row = connection.execute(f"""SELECT COUNT(*), COALESCE(SUM(n),0),
        COALESCE(SUM(CASE WHEN n>1 THEN 1 ELSE 0 END),0),
        COALESCE(SUM(CASE WHEN n>1 THEN n ELSE 0 END),0),
        COALESCE(SUM(n-1),0),
        COALESCE(SUM(CASE WHEN raw_classes>1 THEN 1 ELSE 0 END),0),
        COALESCE(SUM(CASE WHEN binary_classes>1 THEN 1 ELSE 0 END),0),
        COALESCE(SUM(CASE WHEN source_count>1 THEN 1 ELSE 0 END),0)
        FROM ({groups})""", parameters).fetchone()
    keys = ["unique_feature_fingerprints", "records", "duplicate_groups",
            "records_in_duplicate_groups", "extra_records_beyond_one_per_group",
            "raw_label_conflicting_groups", "binary_target_conflicting_groups",
            "feature_groups_shared_between_source_files"]
    return dict(zip(keys, row))


def main():
    candidates = list((ROOT / "reports").rglob("data_audit.json"))
    if not candidates:
        raise SystemExit("No data_audit.json found under reports. Run 02_data_audit.py first.")
    audits = [(json.loads(p.read_text(encoding="utf-8")), p) for p in candidates]
    audit, audit_path = max(audits, key=lambda pair: pair[0].get("started_utc", ""))
    if not audit.get("audit_completed_without_record_or_file_errors"):
        raise SystemExit("Latest data audit did not pass its record/file checks. Resolve it first.")
    expected_hashes = {item["path"]: item["sha256"] for item in audit["files"]}
    paths = []
    for name in EXPECTED:
        matches = list(DATA.rglob(name))
        if len(matches) != 1:
            raise SystemExit(f"Expected exactly one {name}; found {len(matches)}.")
        paths.append(matches[0])

    stamp = datetime.now(timezone.utc)
    output = ROOT / "reports" / stamp.strftime("integrity_%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "started_utc": stamp.isoformat(), "script_sha256": sha256_file(Path(__file__)),
        "input_audit": audit_path.relative_to(ROOT).as_posix(),
        "input_audit_sha256": sha256_file(audit_path),
        "python_version": sys.version, "completed": False,
        "scope": {
            "raw_data_modified": False, "rows_removed": False,
            "splits_created": False, "preprocessing_fitted": False,
            "final_predictors_selected": False, "near_duplicates_checked": False,
            "same_session_overlap_proven_or_excluded": False,
            "fingerprint_definition": (
                "SHA-256 of an ordered JSON list of whitespace-stripped raw field strings. "
                "This is lexical equality, not numeric-equivalence or near-duplicate detection. "
                "For example, 1 and 1.0 remain different. Hash collisions are not independently checked."
            ),
            "cic_feature_view": (
                "Provisional audit-only view excludes id, Flow ID, Src IP, Dst IP, "
                "Timestamp, Label, Attempted Category; ports and protocol remain included. "
                "This does not decide the final model schema."
            ),
            "nsl_feature_view": "First 41 fields only; label and difficulty excluded.",
            "endpoint_view": (
                "Bidirectional endpoint tuple: sorted (IP, port) endpoints plus protocol. "
                "Reuse across days is NOT proof of the same session. No time-window sessionization is performed."
            ),
            "record_id_definition": "Relative source path plus :record: and 1-based CSV data-record number.",
            "binary_label_view": "normal/BENIGN = 0, all other nonempty labels = 1; audit view only.",
            "attempted_view": "Label ends in ' - Attempted'; checked against Attempted Category != -1.",
        }, "files": [],
    }
    db_path = output / "integrity_index.sqlite"
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("""CREATE TABLE records (
        dataset TEXT, source TEXT, record_no INTEGER, feature_key BLOB,
        label TEXT, target INTEGER, attempted INTEGER, endpoint_key BLOB)""")
    try:
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            print(f"\nChecking {relative} ...", flush=True)
            before = path.stat()
            digest = sha256_file(path)
            if digest != expected_hashes.get(relative):
                raise ValueError(f"File differs from the audited input: {relative}")
            cic = path.suffix.lower() == ".csv"
            dataset = "CICIDS2017" if cic else "NSL-KDD"
            anomalies = Counter()
            examples = {}
            batch = []
            count = 0
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle, strict=True)
                if cic:
                    raw_header = next(reader)
                    header = [name.strip() for name in raw_header]
                    if len(set(header)) != len(header):
                        raise ValueError("Duplicate normalized header names.")
                    required = CIC_EXCLUDED | set(ENDPOINT_FIELDS) | {"Fwd Header Length", "Bwd Header Length", "Flow Bytes/s", "Flow Packets/s"}
                    if not required.issubset(header):
                        raise ValueError(f"Missing fields: {sorted(required - set(header))}")
                    indexes = [i for i, name in enumerate(header) if name not in CIC_EXCLUDED]
                    label_index = header.index("Label")
                    attempted_index = header.index("Attempted Category")
                    feature_names = [header[i] for i in indexes]
                else:
                    header = [f"field_{i:02d}" for i in range(1,42)] + ["label", "difficulty"]
                    indexes = list(range(41))
                    label_index = 41
                    feature_names = header[:41]
                for count, raw in enumerate(reader, 1):
                    if len(raw) != len(header):
                        raise ValueError(f"Wrong record width in {relative}, data record {count}.")
                    row = [value.strip() for value in raw]
                    label = row[label_index]
                    if not label:
                        raise ValueError(f"Empty label in {relative}, data record {count}.")
                    attempted = cic and label.endswith(" - Attempted")
                    endpoint = None
                    if cic:
                        if attempted != (row[attempted_index] != "-1"):
                            raise ValueError(f"Attempted indicator disagreement in {relative}, record {count}.")
                        ends = sorted([[row[header.index("Src IP")], row[header.index("Src Port")]],
                                       [row[header.index("Dst IP")], row[header.index("Dst Port")]]])
                        endpoint = fingerprint([ends, row[header.index("Protocol")]])
                        flags = []
                        for name in ["Fwd Header Length", "Bwd Header Length"]:
                            if float(row[header.index(name)]) < 0:
                                flags.append("negative_" + name.replace(" ", "_"))
                        if any(not math.isfinite(float(row[header.index(name)]))
                               for name in ["Flow Bytes/s", "Flow Packets/s"]):
                            flags.append("nonfinite_rate_record")
                        if flags:
                            anomalies["records_with_any_flag"] += 1
                        for flag in flags:
                            anomalies[flag] += 1
                            sample = examples.setdefault(flag, [])
                            if len(sample) < 10:
                                sample.append(f"{relative}:record:{count}")
                    key = fingerprint([row[i] for i in indexes])
                    batch.append((dataset, relative, count, key, label,
                                  int(label.casefold() not in {"normal", "benign"}),
                                  int(attempted), endpoint))
                    if len(batch) == 10000:
                        connection.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?,?)", batch)
                        connection.commit()
                        batch.clear()
                    if count % 100000 == 0:
                        print(f"  {count:,} records indexed", flush=True)
                if batch:
                    connection.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?,?)", batch)
                    connection.commit()
            after = path.stat()
            if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
                raise ValueError(f"File changed while being inspected: {relative}")
            report["files"].append({"path": relative, "sha256": digest, "records": count,
                                    "fingerprint_columns_in_order": feature_names,
                                    "anomaly_record_counts": dict(anomalies),
                                    "anomaly_record_id_examples": examples})
        print("\nBuilding indexes and comparing fingerprint groups ...", flush=True)
        connection.execute("CREATE INDEX feature_lookup ON records(dataset, feature_key)")
        connection.execute("CREATE INDEX source_lookup ON records(source)")
        connection.commit()
        summaries = {}
        for dataset in ["NSL-KDD", "CICIDS2017"]:
            summaries[dataset + "_all_records"] = duplicate_summary(connection, "dataset=?", [dataset])
        summaries["CICIDS2017_excluding_attempted_audit_view"] = duplicate_summary(
            connection, "dataset=? AND attempted=0", ["CICIDS2017"])
        summaries["per_source"] = {
            item["path"]: duplicate_summary(connection, "source=?", [item["path"]])
            for item in report["files"]
        }
        report["feature_duplicate_summaries"] = summaries
        overlap = connection.execute("""SELECT COUNT(*), COALESCE(SUM(n),0) FROM (
            SELECT feature_key, COUNT(*) AS n FROM records WHERE dataset='NSL-KDD'
            GROUP BY feature_key HAVING COUNT(DISTINCT source)>1)""").fetchone()
        report["nsl_train_test_lexical_feature_overlap"] = {
            "shared_unique_feature_fingerprints": overlap[0],
            "total_train_and_test_records_in_shared_groups": overlap[1],
        }
        endpoint = connection.execute("""SELECT COUNT(*), COALESCE(SUM(n),0) FROM (
            SELECT endpoint_key, COUNT(*) AS n FROM records WHERE dataset='CICIDS2017'
            GROUP BY endpoint_key HAVING COUNT(DISTINCT source)>1)""").fetchone()
        report["cic_cross_file_endpoint_reuse"] = {
            "reused_endpoint_tuples": endpoint[0], "records_in_reused_tuple_groups": endpoint[1],
            "interpretation": "Endpoint reuse only, not a count of overlapping sessions.",
        }
        example_rows = connection.execute("""SELECT dataset, hex(feature_key), COUNT(*),
            COUNT(DISTINCT label), COUNT(DISTINCT target)
            FROM records GROUP BY dataset, feature_key
            HAVING COUNT(DISTINCT label)>1 ORDER BY dataset, feature_key LIMIT 20""").fetchall()
        conflict_examples = []
        for dataset, key, n, raw_classes, binary_classes in example_rows:
            members = connection.execute("""SELECT source, record_no, label FROM records
                WHERE dataset=? AND feature_key=? ORDER BY source, record_no LIMIT 10""",
                [dataset, bytes.fromhex(key)]).fetchall()
            conflict_examples.append({"dataset": dataset, "feature_sha256": key.lower(),
                "group_size": n, "raw_label_count": raw_classes, "binary_target_count": binary_classes,
                "record_examples": [{"record_id": f"{s}:record:{r}", "label": label}
                                    for s, r, label in members]})
        report["conflicting_label_group_examples_up_to_20"] = conflict_examples
        report["completed"] = True
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        connection.close()
        (output / "integrity_audit.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        notes = ["# Integrity audit", "", f"Completed: {report['completed']}", "",
                 "The raw files were not modified; no rows were removed or split.",
                 "Fingerprints compare stripped raw strings, not numerical equivalence or near-duplicates.",
                 "CIC predictor columns are provisional for this audit, not a final training schema.",
                 "The excluding-Attempted summary is a query view only, not a filtered dataset.",
                 "Endpoint reuse is not proof of shared sessions; time-based session checks remain open.",
                 "If the final predictor set or normalization changes, repeat content-overlap checks.",
                 "Do not claim R10 or R12 fully resolved from this audit.", "",
                 "Keep integrity_index.sqlite locally: it links fingerprint groups to source record IDs.",
                 "Upload integrity_audit.json, not the SQLite database."]
        (output / "integrity_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
        print(f"\nReports saved in: {output}")
    print("integrity_audit.json needs to be reviewed. Do not train models yet.")


if __name__ == "__main__":
    main()