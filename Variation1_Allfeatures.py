#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def resolve_feature1_path(job):
    output_feature_file = Path(clean_text(job["output_feature_file"]))

    candidates = [
        output_feature_file,
        output_feature_file.parent / "Feature1.csv",
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


def build_standard_gwas_path(job, feature_df):
    phenotype = clean_text(job.get("phenotype", ""))
    accession = clean_text(job.get("accessionId", ""))

    file_name = ""
    if "file_name" in feature_df.columns and not feature_df.empty:
        file_name = clean_text(feature_df.iloc[0].get("file_name", ""))

    if not file_name:
        file_name = Path(clean_text(job.get("file_path", ""))).name

    parts = [phenotype, "allgwas", accession, file_name]
    parts = [p for p in parts if p]
    return "/".join(parts)


def combine_feature1(jobs_file, output_file):
    jobs_file = Path(jobs_file)
    output_file = Path(output_file)

    if not jobs_file.exists():
        raise FileNotFoundError(f"Missing jobs file: {jobs_file}")

    jobs = pd.read_csv(jobs_file)

    required = ["job_index", "phenotype", "accessionId", "file_path", "output_feature_file"]
    missing = [c for c in required if c not in jobs.columns]

    if missing:
        raise ValueError(f"Missing columns in {jobs_file}: {missing}")

    rows = []
    missing_files = 0
    read_failures = 0

    for _, job in jobs.iterrows():
        feature1_path = resolve_feature1_path(job)

        if not feature1_path.exists():
            missing_files += 1
            continue

        try:
            df = pd.read_csv(feature1_path)

            if df.empty:
                read_failures += 1
                continue

            df = df.copy()
            df["feature1_file"] = str(feature1_path)
            df["gwas_standard_path"] = build_standard_gwas_path(job, df)
            rows.append(df)

        except Exception:
            read_failures += 1

    if rows:
        out_df = pd.concat(rows, ignore_index=True, sort=False)
    else:
        out_df = pd.DataFrame()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_file, index=False)

    print("=" * 100)
    print("[COMBINED Feature1.csv FILES]")
    print(f"[INPUT JOBS] {jobs_file}")
    print(f"[TOTAL JOBS] {len(jobs)}")
    print(f"[FILES COMBINED] {len(rows)}")
    print(f"[MISSING FEATURE1 FILES] {missing_files}")
    print(f"[READ FAILURES OR EMPTY FILES] {read_failures}")
    print(f"[OUTPUT] {output_file}")
    print("=" * 100)

    if not out_df.empty:
        print()
        print("[SUMMARY]")
        summary_cols = [
            "compression_type",
            "delimiter",
            "header_detected",
            "read_status",
            "compression_detection_source",
            "compression_suffix_magic_mismatch",
            "reader_used",
        ]

        for col in summary_cols:
            if col in out_df.columns:
                print()
                print(col)
                print(out_df[col].value_counts(dropna=False).to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Combine all per-GWAS Feature1.csv files into one Feature1_All.csv file."
    )

    parser.add_argument(
        "--jobs-file",
        default="GWAS_jobs.csv",
        help="GWAS job manifest file. Default: GWAS_jobs.csv"
    )

    parser.add_argument(
        "--output-file",
        default="Feature1_All.csv",
        help="Combined output CSV. Default: Feature1_All.csv"
    )

    args = parser.parse_args()

    combine_feature1(
        jobs_file=args.jobs_file,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
