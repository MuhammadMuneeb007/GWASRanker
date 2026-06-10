#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


def clean_text(x):
    if x is None:
        return ""
    return str(x).strip()


def read_csv_rows(path):
    rows = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: clean_text(v) for k, v in row.items()})

    return rows


def resolve_feature8_path(job):
    output_feature_file = Path(clean_text(job.get("output_feature_file", "")))
    file_path = Path(clean_text(job.get("file_path", "")))

    candidates = [
        output_feature_file.parent / "Feature8.csv",
        output_feature_file.with_name("Feature8.csv"),
        file_path.parent / "Feature8.csv",
    ]

    seen = set()
    unique_candidates = []

    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    return unique_candidates[0]


def build_standard_gwas_path(job, row):
    phenotype = clean_text(job.get("phenotype", ""))
    accession = clean_text(job.get("accessionId", ""))
    file_name = clean_text(row.get("file_name", ""))

    if not file_name:
        file_name = Path(clean_text(job.get("file_path", ""))).name

    return "/".join(part for part in [phenotype, "allgwas", accession, file_name] if part)


def combine_feature8(jobs_file, output_file):
    jobs_file = Path(jobs_file)
    output_file = Path(output_file)

    if not jobs_file.exists():
        raise FileNotFoundError(f"Missing jobs file: {jobs_file}")

    jobs = read_csv_rows(jobs_file)
    required = ["job_index", "phenotype", "accessionId", "file_path", "output_feature_file"]

    if not jobs:
        raise ValueError(f"No rows found in jobs file: {jobs_file}")

    missing = [col for col in required if col not in jobs[0]]
    if missing:
        raise ValueError(f"Missing columns in {jobs_file}: {missing}")

    merged_rows = []
    all_columns = []
    all_columns_seen = set()
    missing_files = 0
    read_failures = 0

    for job in jobs:
        feature_path = resolve_feature8_path(job)

        if not feature_path.exists():
            missing_files += 1
            continue

        try:
            rows = read_csv_rows(feature_path)
            if not rows:
                read_failures += 1
                continue

            row = rows[0]
            row["feature8_file"] = str(feature_path)
            row["gwas_standard_path"] = build_standard_gwas_path(job, row)
            merged_rows.append(row)

            for column in row:
                if column not in all_columns_seen:
                    all_columns_seen.add(column)
                    all_columns.append(column)
        except Exception:
            read_failures += 1

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
        if all_columns:
            writer.writeheader()
            for row in merged_rows:
                writer.writerow({column: row.get(column, "") for column in all_columns})

    print("=" * 100)
    print("[COMBINED Feature8.csv FILES]")
    print(f"[INPUT JOBS] {jobs_file}")
    print(f"[TOTAL JOBS] {len(jobs)}")
    print(f"[FILES COMBINED] {len(merged_rows)}")
    print(f"[MISSING FEATURE8 FILES] {missing_files}")
    print(f"[READ FAILURES OR EMPTY FILES] {read_failures}")
    print(f"[OUTPUT] {output_file}")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Combine all per-GWAS Feature8.csv files into one Feature8_All.csv file."
    )
    parser.add_argument("--jobs-file", default="GWAS_jobs.csv")
    parser.add_argument("--output-file", default="Feature8_All.csv")
    args = parser.parse_args()
    combine_feature8(args.jobs_file, args.output_file)


if __name__ == "__main__":
    main()
