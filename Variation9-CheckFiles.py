#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

# ============================================================
# Variation9-CheckFiles.py
# Check Feature1.csv to Feature8.csv for all GWAS folders
# using GWAS_jobs.csv
# ============================================================

JOBS_FILE = "GWAS_jobs.csv"
OUTPUT_FILE = "Variation9_Feature_File_Check_All_Phenotypes.csv"

feature_files = [f"Feature{i}.csv" for i in range(1, 9)]

jobs_path = Path(JOBS_FILE)

if not jobs_path.exists():
    print(f"[ERROR] Cannot find {JOBS_FILE} in current directory")
    raise SystemExit(1)

jobs = pd.read_csv(jobs_path)

required_cols = ["job_index", "phenotype", "accessionId", "file_path"]

missing_cols = [c for c in required_cols if c not in jobs.columns]
if missing_cols:
    print(f"[ERROR] Missing columns in {JOBS_FILE}: {missing_cols}")
    raise SystemExit(1)

rows = []

for _, row in jobs.iterrows():
    job_index = row["job_index"]
    phenotype = row["phenotype"]
    accession = row["accessionId"]
    file_path = Path(str(row["file_path"]))

    # Folder like:
    # migraine/allgwas/GCST90000016/
    gwas_folder = file_path.parent

    out_row = {
        "job_index": job_index,
        "phenotype": phenotype,
        "accessionId": accession,
        "path": str(gwas_folder) + "/"
    }

    total_found = 0

    for feature_file in feature_files:
        exists = (gwas_folder / feature_file).exists()
        out_row[feature_file] = "YES" if exists else "NO"

        if exists:
            total_found += 1

    out_row["features_found"] = total_found
    out_row["features_missing"] = len(feature_files) - total_found

    rows.append(out_row)

df = pd.DataFrame(rows)

df.to_csv(OUTPUT_FILE, index=False)

print("=" * 100)
print("FEATURE FILE CHECK COMPLETE")
print("=" * 100)
print(f"Jobs checked: {len(df)}")
print(f"Output file: {OUTPUT_FILE}")
print("=" * 100)

print("\nSUMMARY BY PHENOTYPE")
print("-" * 100)

summary = (
    df.groupby("phenotype")
    .agg(
        total_gwas=("path", "count"),
        complete_gwas=("features_missing", lambda x: (x == 0).sum()),
        incomplete_gwas=("features_missing", lambda x: (x > 0).sum()),
    )
    .reset_index()
)

print(summary.to_string(index=False))

print("\nFEATURE-WISE COUNTS")
print("-" * 100)

feature_summary_rows = []

for phenotype, sub in df.groupby("phenotype"):
    feature_row = {"phenotype": phenotype, "total_gwas": len(sub)}

    for feature_file in feature_files:
        feature_row[feature_file] = (sub[feature_file] == "YES").sum()

    feature_summary_rows.append(feature_row)

feature_summary = pd.DataFrame(feature_summary_rows)
print(feature_summary.to_string(index=False))

summary_file = "Variation9_Feature_File_Check_Summary_By_Phenotype.csv"
feature_summary.to_csv(summary_file, index=False)

print("=" * 100)
print(f"Detailed output saved: {OUTPUT_FILE}")
print(f"Summary output saved:  {summary_file}")
print("=" * 100)