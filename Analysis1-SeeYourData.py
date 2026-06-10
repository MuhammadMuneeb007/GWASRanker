#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_FILE = "AllFeatures_PRS_Merged.csv"


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalise_series_for_signature(series):
    values = []
    for value in series:
        if pd.isna(value):
            values.append("<NA>")
        else:
            values.append(str(value).strip())
    return tuple(values)


def infer_column_family(column):
    lower = column.lower()

    if lower.startswith("feature1_"):
        return "Feature1"
    if lower.startswith("feature2_"):
        return "Feature2"
    if lower.startswith("feature3_"):
        return "Feature3"
    if lower.startswith("feature4_"):
        return "Feature4"
    if lower.startswith("feature5_"):
        return "Feature5"
    if lower.startswith("feature6_"):
        return "Feature6"
    if lower.startswith("feature7_"):
        return "Feature7"
    if lower.startswith("feature8_"):
        return "Feature8"
    if lower.startswith("feature9_"):
        return "Feature9"
    if lower.startswith("plink_"):
        return "PRS_Plink"
    if lower.startswith("prsice2_"):
        return "PRS_PRSice-2"
    if lower.startswith("prs_"):
        return "PRS_Other"
    if lower in {
        "job_index",
        "phenotype",
        "accessionid",
        "file_path",
        "gwas_output_dir",
    }:
        return "BaseMetadata"
    if lower.endswith("_status"):
        return "Status"
    if lower.endswith("_file") or lower.endswith("_dir") or lower.endswith("_path"):
        return "PathOrFile"
    return "Other"


def infer_dtype_family(series):
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "string_or_mixed"


def format_column_list(columns, indent="  ", per_line=6):
    columns = list(columns)
    if not columns:
        return f"{indent}<none>"

    lines = []
    for i in range(0, len(columns), per_line):
        chunk = columns[i:i + per_line]
        lines.append(indent + " | ".join(chunk))
    return "\n".join(lines)


def summarise_constant_columns(df):
    constant = []
    all_missing = []

    for column in df.columns:
        series = df[column]
        if series.isna().all():
            constant.append(column)
            all_missing.append(column)
            continue

        if series.nunique(dropna=False) <= 1:
            constant.append(column)

    return constant, all_missing


def summarise_duplicate_content_columns(df):
    signatures = {}
    for column in df.columns:
        signature = normalise_series_for_signature(df[column])
        signatures.setdefault(signature, []).append(column)

    groups = [cols for cols in signatures.values() if len(cols) > 1]
    groups.sort(key=lambda cols: (-len(cols), cols))
    return groups


def summarise_by_family(columns):
    counts = {}
    members = {}

    for column in columns:
        family = infer_column_family(column)
        counts[family] = counts.get(family, 0) + 1
        members.setdefault(family, []).append(column)

    return counts, members


def summarise_by_dtype(df):
    counts = {}
    members = {}

    for column in df.columns:
        family = infer_dtype_family(df[column])
        counts[family] = counts.get(family, 0) + 1
        members.setdefault(family, []).append(column)

    return counts, members


def print_dataset_summary(df, input_file):
    print("=" * 100)
    print("ANALYSIS 1: SEE YOUR DATA")
    print("=" * 100)
    print(f"[INPUT FILE] {input_file}")
    print(f"[ROWS]       {len(df)}")
    print(f"[COLUMNS]    {len(df.columns)}")
    print("=" * 100)

    print()
    print("[ALL COLUMNS]")
    for idx, column in enumerate(df.columns, start=1):
        print(f"{idx:4d}. {column}")

    family_counts, family_members = summarise_by_family(df.columns)
    print()
    print("=" * 100)
    print("[COLUMN FAMILY COUNTS]")
    for family in sorted(family_counts):
        print(f"{family:16s} {family_counts[family]}")
    print("=" * 100)

    print()
    print("[FEATURE FAMILY DETAILS]")
    for family in [
        "Feature1", "Feature2", "Feature3", "Feature4", "Feature5",
        "Feature6", "Feature7", "Feature8", "Feature9",
        "PRS_Plink", "PRS_PRSice-2", "PRS_Other",
        "BaseMetadata", "Status", "PathOrFile", "Other",
    ]:
        columns = family_members.get(family, [])
        print(f"{family} ({len(columns)})")
        print(format_column_list(columns))
        print()

    dtype_counts, dtype_members = summarise_by_dtype(df)
    print("=" * 100)
    print("[DTYPE FAMILY COUNTS]")
    for family in ["numeric", "bool", "string_or_mixed"]:
        print(f"{family:16s} {dtype_counts.get(family, 0)}")
    print("=" * 100)

    print()
    print("[NUMERIC COLUMNS]")
    print(format_column_list(dtype_members.get("numeric", [])))
    print()
    print("[BOOLEAN COLUMNS]")
    print(format_column_list(dtype_members.get("bool", [])))
    print()
    print("[STRING OR MIXED COLUMNS]")
    print(format_column_list(dtype_members.get("string_or_mixed", [])))

    constant_columns, all_missing_columns = summarise_constant_columns(df)
    print()
    print("=" * 100)
    print("[CONSTANT COLUMN SUMMARY]")
    print(f"constant_columns_count: {len(constant_columns)}")
    print(f"all_missing_columns_count: {len(all_missing_columns)}")
    print("=" * 100)

    print()
    print("[CONSTANT COLUMNS]")
    print(format_column_list(constant_columns))
    print()
    print("[ALL-MISSING COLUMNS]")
    print(format_column_list(all_missing_columns))

    duplicate_groups = summarise_duplicate_content_columns(df)
    duplicate_columns_count = sum(len(group) for group in duplicate_groups)
    print()
    print("=" * 100)
    print("[SIMILAR / DUPLICATE-CONTENT COLUMN SUMMARY]")
    print(f"duplicate_content_groups_count: {len(duplicate_groups)}")
    print(f"columns_in_duplicate_content_groups: {duplicate_columns_count}")
    print("=" * 100)

    print()
    print("[SIMILAR / DUPLICATE-CONTENT COLUMN GROUPS]")
    if not duplicate_groups:
        print("  <none>")
    else:
        for idx, group in enumerate(duplicate_groups, start=1):
            print(f"{idx:3d}. " + " | ".join(group))

    print()
    print("=" * 100)
    print("[NON-CONSTANT COLUMNS THAT MAY BE MORE USEFUL]")
    non_constant = [c for c in df.columns if c not in set(constant_columns)]
    print(f"non_constant_columns_count: {len(non_constant)}")
    print(format_column_list(non_constant))
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Print a screen-only summary of the merged GWAS feature/PRS dataset."
    )
    parser.add_argument(
        "--input-file",
        default=DEFAULT_INPUT_FILE,
        help="Input merged CSV file. Default: AllFeatures_PRS_Merged.csv",
    )

    args = parser.parse_args()
    input_file = Path(args.input_file)

    if not input_file.exists():
        raise FileNotFoundError(f"Missing input file: {input_file}")

    df = pd.read_csv(input_file, low_memory=False)
    print_dataset_summary(df, input_file)


if __name__ == "__main__":
    main()
