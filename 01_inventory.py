from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import platform
import sys

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_first_record(path):
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                record = next(csv.reader(handle), [])
            result = {
                "inspection_status": "read",
                "inspection_encoding": encoding,
                "delimiter_used": ",",
                "first_record_field_count": len(record),
            }
            if path.suffix.lower() == ".csv":
                result["possible_header_unverified"] = record
            return result
        except UnicodeDecodeError:
            continue
        except csv.Error as exc:
            return {"inspection_status": "error", "inspection_error": str(exc)}
    return {
        "inspection_status": "error",
        "inspection_error": "Could not decode the first record with UTF-8 or CP1252.",
    }


def main():
    if not DATA.is_dir():
        raise SystemExit(f"Data folder not found: {DATA}")

    paths = sorted(
        (p for p in DATA.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(DATA).as_posix(),
    )
    if not paths:
        raise SystemExit("The data folder contains no files.")

    timestamp = datetime.now(timezone.utc)
    report = {
        "inventory_created_utc": timestamp.isoformat(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "scope": {
            "raw_data_modified": False,
            "row_counts_computed": False,
            "labels_audited": False,
            "duplicates_audited": False,
            "preprocessing_performed": False,
            "release_verified": False,
            "note": (
                "Inventory time is not the download date. CSV first records are "
                "recorded as possible headers, not verified schemas. "
                "No feature count or dataset identity is inferred from filenames."
            ),
        },
        "files": [],
    }

    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        print(f"\nInspecting: {relative}", flush=True)
        record = {"path": relative}
        try:
            before = path.stat()
            record["size_bytes"] = before.st_size
            record["sha256"] = sha256_file(path)
            if path.suffix.lower() in {".csv", ".txt"}:
                record.update(inspect_first_record(path))
            after = path.stat()
            record["stable_during_inspection"] = (
                before.st_size == after.st_size
                and before.st_mtime_ns == after.st_mtime_ns
            )
            record["status"] = (
                "ok" if record["stable_during_inspection"] else "changed_during_inspection"
            )
            print(f"  Size: {record['size_bytes']:,} bytes")
            print(f"  SHA256: {record['sha256']}")
            if "first_record_field_count" in record:
                print(f"  First-record fields: {record['first_record_field_count']}")
            if record.get("inspection_status") == "error":
                print(f"  Inspection warning: {record['inspection_error']}")
            if not record["stable_during_inspection"]:
                print("  WARNING: File changed during inspection; rerun when stable.")
        except OSError as exc:
            record["status"] = "error"
            record["error"] = str(exc)
            print(f"  ERROR: {exc}")
        report["files"].append(record)

    output = ROOT / "reports" / timestamp.strftime("inventory_%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "data_inventory.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nFiles inventoried: {len(report['files'])}")
    print(f"Report saved: {report_path}")
    print("Raw data files were not modified.")
    print("This is an inventory, not a completed dataset or leakage audit.")


if __name__ == "__main__":
    main()