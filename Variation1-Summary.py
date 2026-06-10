#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


ROOT = Path(".")
FEATURE_NAME = "Feature1.csv"


def print_section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def safe_read_csv(path):
    try:
        df = pd.read_csv(path)
        df["source_feature_file"] = str(path)
        return df
    except Exception as e:
        print(f"[READ FAILED] {path} | {type(e).__name__}: {e}")
        return None


def print_value_counts(df, col, top_n=30):
    if col not in df.columns:
        return

    print_section(f"{col}")

    s = df[col].fillna("").astype(str).str.strip()
    counts = s.value_counts(dropna=False)

    print(f"[UNIQUE VALUES] {s.nunique(dropna=False)}")
    print(f"[TOP {top_n}]")
    print(counts.head(top_n).to_string())


def print_numeric_summary(df, col):
    if col not in df.columns:
        return

    print_section(f"NUMERIC SUMMARY: {col}")

    x = pd.to_numeric(df[col], errors="coerce")

    print(f"[COUNT]  {x.notna().sum()}")
    print(f"[MIN]    {x.min()}")
    print(f"[MAX]    {x.max()}")
    print(f"[MEAN]   {round(x.mean(), 4)}")
    print(f"[MEDIAN] {x.median()}")

    print()
    print("[MOST COMMON VALUES]")
    print(x.value_counts(dropna=False).head(30).to_string())


def print_group_summary(df, group_cols, title, top_n=50):
    existing = [c for c in group_cols if c in df.columns]

    if not existing:
        return

    print_section(title)

    summary = (
        df.groupby(existing, dropna=False)
        .size()
        .reset_index(name="n_files")
        .sort_values("n_files", ascending=False)
    )

    print(summary.head(top_n).to_string(index=False))


def print_column_uniqueness(df):
    print_section("COLUMN UNIQUENESS SUMMARY")

    rows = []

    for col in df.columns:
        s = df[col].fillna("").astype(str).str.strip()
        vc = s.value_counts(dropna=False)

        rows.append({
            "column": col,
            "non_empty": int((s != "").sum()),
            "empty": int((s == "").sum()),
            "unique_values": int(s.nunique(dropna=False)),
            "most_common_value": vc.index[0] if len(vc) else "",
            "most_common_count": int(vc.iloc[0]) if len(vc) else 0,
        })

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))


def print_problem_files(df):
    print_section("POTENTIAL PROBLEM FILES")

    mask = pd.Series(False, index=df.index)

    if "read_status" in df.columns:
        mask |= df["read_status"].fillna("").astype(str).str.strip().ne("read_ok")

    if "header_detected" in df.columns:
        mask |= df["header_detected"].fillna("").astype(str).str.lower().ne("true")

    if "delimiter" in df.columns:
        mask |= df["delimiter"].fillna("").astype(str).str.lower().isin(["unknown", ""])

    bad = df[mask].copy()

    print(f"[POTENTIAL PROBLEM FILES] {len(bad)}")

    if bad.empty:
        return

    show_cols = [
        "phenotype",
        "accessionId",
        "file_name",
        "file_extension",
        "compression_type",
        "delimiter",
        "n_columns",
        "header_detected",
        "read_status",
        "encoding_issues",
        "file_path",
    ]

    show_cols = [c for c in show_cols if c in bad.columns]

    print(bad[show_cols].head(100).to_string(index=False))


def main():
    print_section("FINDING Feature1.csv FILES")

    feature_files = sorted(ROOT.rglob(FEATURE_NAME))

    print(f"[FOUND Feature1.csv FILES] {len(feature_files)}")

    if not feature_files:
        print("[STOP] No Feature1.csv files found.")
        return

    dfs = []

    for path in feature_files:
        df = safe_read_csv(path)
        if df is not None and not df.empty:
            dfs.append(df)

    if not dfs:
        print("[STOP] No readable Feature1.csv files.")
        return

    merged = pd.concat(dfs, ignore_index=True)

    print_section("BASIC OVERVIEW")
    print(f"Files found:      {len(feature_files)}")
    print(f"Files processed:  {len(dfs)}")
    print(f"Merged rows:      {len(merged)}")
    print(f"Merged columns:   {len(merged.columns)}")

    if "phenotype" in merged.columns:
        print_group_summary(
            merged,
            ["phenotype"],
            "FILES PROCESSED BY PHENOTYPE"
        )

    print_group_summary(
        merged,
        ["file_extension"],
        "FILE EXTENSION GROUPS"
    )

    print_group_summary(
        merged,
        ["compression_type"],
        "COMPRESSION TYPE GROUPS"
    )

    print_group_summary(
        merged,
        ["delimiter"],
        "DELIMITER GROUPS"
    )

    print_group_summary(
        merged,
        ["header_detected"],
        "HEADER DETECTION GROUPS"
    )

    print_group_summary(
        merged,
        ["read_status"],
        "READ STATUS GROUPS"
    )

    print_group_summary(
        merged,
        ["file_extension", "compression_type", "delimiter", "header_detected"],
        "STORAGE PATTERN GROUPS"
    )

    print_group_summary(
        merged,
        ["phenotype", "file_extension", "compression_type", "delimiter"],
        "PHENOTYPE-WISE STORAGE PATTERNS"
    )

    for col in [
        "file_extension",
        "compression_type",
        "delimiter",
        "header_detected",
        "read_status",
        "quote_usage",
        "encoding_used",
        "n_columns",
        "comment_or_metadata_lines_before_header",
    ]:
        print_value_counts(merged, col)

    for col in [
        "raw_file_size_mb",
        "extracted_file_size_mb",
        "n_rows_physical_lines",
        "n_columns",
        "n_sample_lines_read",
    ]:
        print_numeric_summary(merged, col)

    print_problem_files(merged)
    print_column_uniqueness(merged)

    print_section("DONE")
    print("Nothing was saved. All summaries were printed only on screen.")


if __name__ == "__main__":
    main()