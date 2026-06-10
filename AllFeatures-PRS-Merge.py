#!/usr/bin/env python3

"""
AllFeatures-PRS-Merge.py

Create one GWAS-level table by merging:
    - Feature1_All.csv
    - Feature2_All.csv
    - Feature3_All.csv
    - Feature4_All.csv
    - Feature5_All.csv
    - Feature6_All.csv
    - Feature7_All.csv
    - Feature8_All.csv
    - Feature9_All.csv
    - PRS2_All_GWAS_Performance_Comparison.csv

The output keeps one row per job from GWAS_jobs.csv and records whether each
feature/PRS source was available, missing as a combined file, or missing for
that specific job.
"""

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_JOBS_FILE = "GWAS_jobs.csv"
DEFAULT_OUTPUT_FILE = "AllFeatures_PRS_Merged.csv"
DEFAULT_PRS_FILE = "PRS2_All_GWAS_Performance_Comparison.csv"

FEATURE_SPECS = [
    {"number": 1, "candidates": ["Feature1_All.csv"]},
    {"number": 2, "candidates": ["Feature2_All.csv"]},
    {"number": 3, "candidates": ["Feature3_All.csv"]},
    {"number": 4, "candidates": ["Feature4_All.csv"]},
    {"number": 5, "candidates": ["Feature5_All.csv"]},
    {"number": 6, "candidates": ["Feature6_All.csv"]},
    {"number": 7, "candidates": ["Feature7_All.csv"]},
    {"number": 8, "candidates": ["Feature8_All.csv"]},
    {"number": 9, "candidates": ["Feature9_All.csv"]},
]

BASE_KEY_COLUMNS = ["job_index", "phenotype", "accessionId", "file_path", "gwas_output_dir"]


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def load_jobs(jobs_file):
    jobs_file = Path(jobs_file)
    if not jobs_file.exists():
        raise FileNotFoundError(f"Missing jobs file: {jobs_file}")

    jobs = pd.read_csv(jobs_file, low_memory=False)
    required = ["job_index", "file_path"]
    missing = [c for c in required if c not in jobs.columns]
    if missing:
        raise ValueError(f"Missing required columns in {jobs_file}: {missing}")

    jobs["job_index"] = pd.to_numeric(jobs["job_index"], errors="coerce").astype("Int64")
    jobs = jobs.dropna(subset=["job_index"]).copy()
    jobs["job_index"] = jobs["job_index"].astype(int)

    if "phenotype" not in jobs.columns:
        jobs["phenotype"] = ""
    if "accessionId" not in jobs.columns:
        jobs["accessionId"] = ""
    if "output_feature_file" not in jobs.columns:
        jobs["output_feature_file"] = ""

    jobs["phenotype"] = jobs["phenotype"].map(clean_text)
    jobs["accessionId"] = jobs["accessionId"].map(clean_text)
    jobs["file_path"] = jobs["file_path"].map(clean_text)
    jobs["output_feature_file"] = jobs["output_feature_file"].map(clean_text)

    jobs["gwas_output_dir"] = jobs.apply(get_gwas_output_dir, axis=1)
    return jobs


def get_gwas_output_dir(job):
    output_feature_file = clean_text(job.get("output_feature_file", ""))
    if output_feature_file:
        return str(Path(output_feature_file).parent)
    return str(Path(clean_text(job.get("file_path", ""))).parent)


def resolve_existing_file(candidates, search_dir):
    for candidate in candidates:
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            candidate_path = Path(search_dir) / candidate_path
        if candidate_path.exists():
            return candidate_path
    return None


def normalise_job_index(df, source_name):
    if "job_index" not in df.columns:
        raise ValueError(f"{source_name} is missing required column: job_index")

    out = df.copy()
    out["job_index"] = pd.to_numeric(out["job_index"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["job_index"]).copy()
    out["job_index"] = out["job_index"].astype(int)
    return out


def deduplicate_on_job_index(df, source_name):
    if df.empty:
        return df, 0

    duplicate_count = int(df.duplicated(subset=["job_index"], keep="first").sum())
    if duplicate_count > 0:
        df = df.drop_duplicates(subset=["job_index"], keep="first").copy()
    return df, duplicate_count


def prefix_non_key_columns(df, prefix, keep_columns=None):
    keep_columns = set(keep_columns or [])
    rename_map = {}
    for column in df.columns:
        if column == "job_index":
            continue
        if column in keep_columns:
            continue
        rename_map[column] = f"{prefix}{column}"
    return df.rename(columns=rename_map)


def merge_feature_table(base_df, feature_spec, search_dir):
    number = feature_spec["number"]
    prefix = f"feature{number}_"
    status_col = f"{prefix}status"
    path_col = f"{prefix}combined_file"
    duplicate_col = f"{prefix}duplicate_job_rows_removed"

    feature_path = resolve_existing_file(feature_spec["candidates"], search_dir)
    info = {
        "name": f"Feature{number}",
        "path": str(feature_path) if feature_path else "",
        "rows_loaded": 0,
        "duplicates_removed": 0,
        "status": "missing_combined_file" if feature_path is None else "loaded",
    }

    if feature_path is None:
        out = base_df.copy()
        out[status_col] = "missing_combined_file"
        out[path_col] = ""
        out[duplicate_col] = 0
        return out, info

    feature_df = pd.read_csv(feature_path, low_memory=False)
    feature_df = normalise_job_index(feature_df, feature_path.name)
    feature_df, duplicates_removed = deduplicate_on_job_index(feature_df, feature_path.name)

    info["rows_loaded"] = int(len(feature_df))
    info["duplicates_removed"] = int(duplicates_removed)

    merge_df = feature_df.copy()
    merge_df[status_col] = "available"
    merge_df[path_col] = str(feature_path)
    merge_df[duplicate_col] = int(duplicates_removed)
    merge_df = prefix_non_key_columns(
        merge_df,
        prefix=prefix,
        keep_columns={status_col, path_col, duplicate_col},
    )

    out = base_df.merge(merge_df, on="job_index", how="left")
    out[status_col] = out[status_col].fillna("missing_job_row")
    out[path_col] = out[path_col].fillna(str(feature_path))
    out[duplicate_col] = pd.to_numeric(out[duplicate_col], errors="coerce").fillna(int(duplicates_removed)).astype(int)
    return out, info


def merge_prs_table(base_df, prs_file, search_dir):
    status_col = "prs_performance_status"
    path_col = "prs_performance_file"
    duplicate_col = "prs_duplicate_job_rows_removed"

    prs_path = resolve_existing_file([prs_file], search_dir)
    info = {
        "name": "PRS",
        "path": str(prs_path) if prs_path else "",
        "rows_loaded": 0,
        "duplicates_removed": 0,
        "status": "missing_combined_file" if prs_path is None else "loaded",
    }

    if prs_path is None:
        out = base_df.copy()
        out[status_col] = "missing_combined_file"
        out[path_col] = ""
        out[duplicate_col] = 0
        return out, info

    prs_df = pd.read_csv(prs_path, low_memory=False)
    prs_df = normalise_job_index(prs_df, prs_path.name)
    prs_df, duplicates_removed = deduplicate_on_job_index(prs_df, prs_path.name)

    info["rows_loaded"] = int(len(prs_df))
    info["duplicates_removed"] = int(duplicates_removed)

    drop_meta = [c for c in ["phenotype", "accessionId", "file_path", "gwas_output_dir"] if c in prs_df.columns]
    merge_df = prs_df.drop(columns=drop_meta, errors="ignore").copy()
    merge_df[status_col] = "available"
    merge_df[path_col] = str(prs_path)
    merge_df[duplicate_col] = int(duplicates_removed)

    out = base_df.merge(merge_df, on="job_index", how="left")
    out[status_col] = out[status_col].fillna("missing_job_row")
    out[path_col] = out[path_col].fillna(str(prs_path))
    out[duplicate_col] = pd.to_numeric(out[duplicate_col], errors="coerce").fillna(int(duplicates_removed)).astype(int)
    return out, info


def reorder_columns(df):
    preferred = BASE_KEY_COLUMNS + [
        f"feature{i}_status" for i in range(1, 10)
    ] + [
        "prs_performance_status",
        "plink_best_result_status",
        "prsice2_best_result_status",
        "plink_likely_reason",
        "prsice2_likely_reason",
    ]
    first = [c for c in preferred if c in df.columns]
    rest = [c for c in df.columns if c not in first]
    return df[first + rest]


def build_base_dataframe(jobs):
    base = jobs.copy()
    for column in BASE_KEY_COLUMNS:
        if column not in base.columns:
            base[column] = ""
    base = base[BASE_KEY_COLUMNS].copy()
    return base.sort_values("job_index").reset_index(drop=True)


def merge_all_features_and_prs(jobs_file, output_file, prs_file):
    jobs = load_jobs(jobs_file)
    base_df = build_base_dataframe(jobs)
    search_dir = Path(output_file).parent if Path(output_file).parent != Path("") else Path(".")

    merged = base_df.copy()
    merge_infos = []

    for feature_spec in FEATURE_SPECS:
        merged, info = merge_feature_table(merged, feature_spec, search_dir)
        merge_infos.append(info)

    merged, prs_info = merge_prs_table(merged, prs_file, search_dir)
    merge_infos.append(prs_info)

    merged = reorder_columns(merged)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_file, index=False)

    print("=" * 100)
    print("[ALL FEATURES + PRS MERGE]")
    print(f"[INPUT JOBS] {jobs_file}")
    print(f"[TOTAL JOBS] {len(base_df)}")
    for info in merge_infos:
        print(
            f"[{info['name']}] status={info['status']} "
            f"rows_loaded={info['rows_loaded']} duplicates_removed={info['duplicates_removed']} "
            f"path={info['path'] or '<missing>'}"
        )
    print(f"[OUTPUT] {output_file}")
    print("=" * 100)

    status_cols = [f"feature{i}_status" for i in range(1, 10)] + ["prs_performance_status"]
    for status_col in status_cols:
        if status_col in merged.columns:
            print()
            print(f"[{status_col}]")
            print(merged[status_col].value_counts(dropna=False).to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Merge Feature1-Feature9 combined tables and PRS performance into one GWAS-level CSV."
    )
    parser.add_argument(
        "--jobs-file",
        default=DEFAULT_JOBS_FILE,
        help="GWAS job manifest file. Default: GWAS_jobs.csv",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help="Final merged output CSV. Default: AllFeatures_PRS_Merged.csv",
    )
    parser.add_argument(
        "--prs-file",
        default=DEFAULT_PRS_FILE,
        help="PRS comparison CSV from PRS2-Calculation-Merge.py. Default: PRS2_All_GWAS_Performance_Comparison.csv",
    )

    args = parser.parse_args()
    merge_all_features_and_prs(
        jobs_file=args.jobs_file,
        output_file=args.output_file,
        prs_file=args.prs_file,
    )


if __name__ == "__main__":
    main()
