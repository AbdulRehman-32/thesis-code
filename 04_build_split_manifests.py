from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
from array import array
import argparse
import csv
import gzip
import hashlib
import json
import sqlite3
import sys

ROOT = Path(__file__).resolve().parent
SPLITS = {0: "train", 1: "validation", 2: "test"}
SOURCES = {
    "release_guidance": "https://intrusion-detection.distrinet-research.be/CNS2022/Dataset_Download.html",
    "improved_release_publication": "https://intrusion-detection.distrinet-research.be/CNS2022/index.html",
    "download_source_reported_by_user": "https://www.kaggle.com/datasets/ernie55ernie/improved-cicids2017-and-csecicids2018?select=CICIDS2017_improved",
    "retrieval_date": None,
    "upstream_byte_for_byte_release_match_verified": False,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def find(parent, value):
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = parent[value]
    return value


def union(parent, left, right):
    a, b = find(parent, left), find(parent, right)
    if a != b:
        parent[max(a, b)] = min(a, b)


def score(seed, group_id):
    raw = hashlib.sha256(f"{seed}|{group_id}".encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big") / 18446744073709551616


def link_equal_keys(db, parent, column):
    previous = None
    anchor = None
    linked = 0
    query = f"SELECT rowid, {column} FROM records WHERE dataset='CICIDS2017' ORDER BY {column}, rowid"
    for row_id, key in db.execute(query):
        if key is None:
            raise ValueError(f"Missing CICIDS {column} at SQLite row {row_id}.")
        if key == previous:
            union(parent, anchor, row_id)
            linked += 1
        else:
            previous, anchor = key, row_id
    return linked


def verify_key_isolation(db, assignments, dataset, column):
    previous = None
    observed = set()
    groups = 0
    for row_id, key in db.execute(
        f"SELECT rowid, {column} FROM records WHERE dataset=? ORDER BY {column}, rowid", [dataset]
    ):
        split = assignments[row_id]
        if key != previous:
            previous = key
            observed = set()
            groups += 1
        if split >= 0:
            observed.add(split)
            if len(observed) > 1:
                raise ValueError(f"Cross-partition {column} overlap detected for {dataset}.")
    return {"groups_inspected": groups, "cross_partition_groups": 0}


def main():
    parser = argparse.ArgumentParser(description="Build prospective split manifests; no preprocessing or training.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attempted-policy", choices=["benign", "exclude", "attack"], default="benign")
    args = parser.parse_args()
    candidates = list((ROOT / "reports").rglob("integrity_audit.json"))
    if not candidates:
        raise SystemExit("No integrity_audit.json found under reports. Run 03_integrity_audit.py first.")
    audits = [(json.loads(p.read_text(encoding="utf-8")), p) for p in candidates]
    audit, audit_path = max(audits, key=lambda item: item[0].get("started_utc", ""))
    if not audit.get("completed"):
        raise SystemExit("Latest integrity audit did not complete.")
    db_path = audit_path.parent / "integrity_index.sqlite"
    if not db_path.is_file():
        raise SystemExit(f"Keep the integrity database next to its report: {db_path}")

    print("Checking raw input hashes ...", flush=True)
    for item in audit["files"]:
        path = ROOT / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise SystemExit(f"Input differs from the integrity audit: {item['path']}")

    stamp = datetime.now(timezone.utc)
    output = ROOT / "reports" / stamp.strftime("partitions_%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "started_utc": stamp.isoformat(), "completed": False,
        "script_sha256": sha256_file(Path(__file__)), "python_version": sys.version,
        "seed": args.seed, "attempted_policy": args.attempted_policy,
        "integrity_audit": audit_path.relative_to(ROOT).as_posix(),
        "integrity_audit_sha256": sha256_file(audit_path),
        "integrity_database_sha256": sha256_file(db_path),
        "dataset_sources": SOURCES,
        "scope": {
            "raw_data_modified": False, "preprocessing_fitted": False,
            "models_trained": False, "numeric_equivalence_checked": False,
            "near_duplicates_checked": False,
            "true_sessions_identified": False,
            "fingerprint_schema": {item["path"]: item["fingerprint_columns_in_order"] for item in audit["files"]},
            "feature_schema_status": "Inherited audit schema, still subject to final model-schema checks.",
            "nsl_policy": (
                "Preserve every official test record. Exclude official training records whose "
                "lexical feature fingerprint occurs in test, without using test labels. "
                "Use a deterministic feature-group hash for an approximate 80/20 split of remaining "
                "official training records. No majority-vote relabelling or automatic conflict removal."
            ),
            "cic_policy": (
                "Conservatively group connected components of equal lexical feature fingerprints OR "
                "equal bidirectional endpoint tuples across the full week. Assign each complete component "
                "to train/validation/test using seeded hash thresholds 0.70/0.85. This is a group-random, "
                "not chronological or stratified, split. Endpoint reuse is not proof of a true session."
            ),
            "duplicate_policy": "Retain repeated records but keep feature-equal records in the same partition. No deduplication or label-conflict removal in this step.",
            "split_selection_policy": "One prospectively specified seed; do not search seeds using test metrics. Inspect support before freezing the protocol.",
            "anomaly_policy": "This step makes no value corrections; negative header lengths and infinite rates remain unresolved for later preprocessing.",
            "future_check": "Repeat content-overlap checks on final numeric/encoded model inputs before training; these manifests alone do not close R10/R12.",
        }, "files": [],
    }
    db = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    handles = []
    try:
        actual_counts = dict(db.execute("SELECT source, COUNT(*) FROM records GROUP BY source"))
        expected_counts = {item["path"]: item["records"] for item in audit["files"]}
        if actual_counts != expected_counts:
            raise ValueError("SQLite source counts do not match the integrity report.")
        total, maximum = db.execute("SELECT COUNT(*), MAX(rowid) FROM records").fetchone()
        if maximum is None or maximum >= 4294967295:
            raise ValueError("Unsupported SQLite row-id range.")
        parent = array("I", range(maximum + 1))
        assigned = array("b", [-1]) * (maximum + 1)
        print("Grouping CICIDS duplicate features ...", flush=True)
        report["cic_feature_links"] = link_equal_keys(db, parent, "feature_key")
        print("Grouping CICIDS endpoint tuples ...", flush=True)
        report["cic_endpoint_links"] = link_equal_keys(db, parent, "endpoint_key")
        group_sizes = Counter()
        for row_id, in db.execute("SELECT rowid FROM records WHERE dataset='CICIDS2017'"):
            group_sizes[find(parent, row_id)] += 1
        cic_total = sum(group_sizes.values())
        report["cic_grouping"] = {
            "connected_components": len(group_sizes),
            "largest_component_records": max(group_sizes.values(), default=0),
            "largest_component_fraction": max(group_sizes.values(), default=0) / max(cic_total, 1),
            "note": "Counts precede any optional Attempted exclusion; retained endpoints remain conservatively linked.",
        }
        del group_sizes
        print("Finding label-independent NSL train/test feature overlap ...", flush=True)
        test_keys = {row[0] for row in db.execute(
            "SELECT DISTINCT feature_key FROM records WHERE dataset='NSL-KDD' AND source LIKE '%KDDTest+.txt'"
        )}
        if not test_keys:
            raise ValueError("Official NSL-KDD test records were not found.")
        writers = {}
        for dataset in ["NSL-KDD", "CICIDS2017"]:
            for split in ["train", "validation", "test", "excluded"]:
                path = output / f"{dataset.lower().replace('-', '_')}_{split}_ids.csv.gz"
                handle = gzip.open(path, "wt", encoding="utf-8", newline="")
                handles.append(handle)
                writer = csv.writer(handle)
                writer.writerow(["record_id", "source", "data_record_number", "group_id", "feature_sha256", "original_label", "binary_target", "reason"])
                writers[(dataset, split)] = (writer, path)
        counts = Counter()
        supports = defaultdict(Counter)
        targets = defaultdict(Counter)
        component_splits = {}
        for n, values in enumerate(db.execute("SELECT rowid, dataset, source, record_no, feature_key, label, target, attempted FROM records ORDER BY rowid"), 1):
            row_id, dataset, source, number, key, label, target, attempted = values
            reason = "retained"
            if dataset == "NSL-KDD":
                group_id = "nsl:" + key.hex()
                if source.endswith("KDDTest+.txt"):
                    split = 2
                elif source.endswith("KDDTrain+.txt"):
                    if key in test_keys:
                        split, reason = -2, "official_train_feature_present_in_official_test"
                    else:
                        split = 0 if score(args.seed, group_id) < 0.80 else 1
                else:
                    raise ValueError(f"Unexpected NSL source: {source}")
            elif dataset == "CICIDS2017":
                root = find(parent, row_id)
                group_id = f"cic:component_min_index_row:{root}"
                if root not in component_splits:
                    value = score(args.seed, group_id)
                    component_splits[root] = 0 if value < 0.70 else (1 if value < 0.85 else 2)
                split = component_splits[root]
                if attempted:
                    if args.attempted_policy == "exclude":
                        split, reason = -2, "attempted_policy_exclude"
                    elif args.attempted_policy == "benign":
                        target, reason = 0, "attempted_mapped_to_benign"
                    else:
                        target, reason = 1, "attempted_mapped_to_attack"
            else:
                raise ValueError(f"Unexpected dataset: {dataset}")
            assigned[row_id] = split
            name = SPLITS.get(split, "excluded")
            writers[(dataset, name)][0].writerow([
                f"{source}:record:{number}", source, number, group_id, key.hex(), label, target, reason
            ])
            counts[(dataset, name)] += 1
            supports[(dataset, name)][label] += 1
            targets[(dataset, name)][str(target)] += 1
            if n % 200000 == 0:
                print(f"  {n:,}/{total:,} records assigned", flush=True)
        del component_splits, test_keys, parent
        for handle in handles:
            handle.close()
        print("Verifying feature and endpoint group isolation ...", flush=True)
        report["partition_checks"] = {
            "nsl_lexical_features": verify_key_isolation(db, assigned, "NSL-KDD", "feature_key"),
            "cic_lexical_features": verify_key_isolation(db, assigned, "CICIDS2017", "feature_key"),
            "cic_endpoint_tuples": verify_key_isolation(db, assigned, "CICIDS2017", "endpoint_key"),
        }
        report["partitions"] = []
        warnings = []
        for (dataset, name), (writer, path) in writers.items():
            count = counts[(dataset, name)]
            report["partitions"].append({
                "dataset": dataset, "partition": name, "records": count,
                "binary_class_counts": dict(targets[(dataset, name)]),
                "original_label_counts": dict(supports[(dataset, name)]),
                "manifest": path.relative_to(ROOT).as_posix(), "manifest_sha256": sha256_file(path)
            })
            if name != "excluded" and (count == 0 or len(targets[(dataset, name)]) < 2):
                warnings.append(f"{dataset} {name} is empty or lacks one binary class; do not train.")
        if report["cic_grouping"]["largest_component_fraction"] > 0.20:
            warnings.append("A CICIDS component exceeds 20% of all raw CICIDS records. Inspect group imbalance before accepting this split.")
        warnings.append("Binary-label conflicts and duplicate weighting remain to be evaluated under the chosen Attempted policy; no labels were repaired by majority vote.")
        warnings.append("Numeric-equivalence, near-duplicate, final-transformation collisions and release-provenance checks remain open.")
        report["warnings"] = warnings
        report["completed"] = True
        print("\nPartition counts:")
        for item in report["partitions"]:
            print(f"  {item['dataset']} {item['partition']}: {item['records']:,}; classes={item['binary_class_counts']}")
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        for handle in handles:
            handle.close()
        db.close()
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (output / "partition_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        notes = ["# Prospective partition manifests", "", f"Completed: {report['completed']}",
                 f"Seed: {args.seed}", f"Attempted policy: {args.attempted_policy}", "",
                 "Raw data and the integrity database were opened read-only for this step.",
                 "These are candidate partition manifests, not fitted preprocessing or model results.",
                 "NSL official test records are preserved. Matching official training fingerprints are excluded without using test labels.",
                 "CIC grouping conservatively links duplicate lexical features and entire endpoint tuples across days.",
                 "Hash thresholds give approximate proportions, not stratified or chronological evaluation.",
                 "Do not reroll seeds to select favourable test outcomes.",
                 "Original labels are retained alongside the explicit binary mapping.",
                 "Anomalous values, conflicts, near-duplicates and final-input collisions are not automatically repaired.",
                 "Inspect partition_report.json and freeze/document the protocol before fitting preprocessing."]
        (output / "partition_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
        print(f"\nReports saved in: {output}")
    print("partition_report.json needs review. Do not train yet.")


if __name__ == "__main__":
    main()
