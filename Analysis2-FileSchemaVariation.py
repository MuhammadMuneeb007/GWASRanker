#!/usr/bin/env python3
"""
Synthesis analysis for Feature1 and Feature2 columns.
Prints a structured synthesis report to stdout. Saves nothing.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT_FILE = "AllFeatures_PRS_Merged.csv"

MISSING_TOKENS = {"", "nan", "na", "n/a", "none", "null", ".", "<na>"}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def is_missing(x):
    if pd.isna(x):
        return True
    return str(x).strip().lower() in MISSING_TOKENS


def clean(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def non_missing(series):
    return series[~series.map(is_missing)].map(clean)


def pct(count, total):
    if total == 0:
        return 0.0
    return round(100.0 * count / total, 1)


def sep(title="", width=90, char="─"):
    if title:
        side = (width - len(title) - 2) // 2
        print(f"\n{'─' * side} {title} {'─' * (width - side - len(title) - 2)}")
    else:
        print(char * width)


# ---------------------------------------------------------------------------
# Printers
# ---------------------------------------------------------------------------

def print_categorical(series, label, total, max_show=10):
    """Print ranked frequency table for a categorical column."""
    vals = non_missing(series)
    if vals.empty:
        print(f"  {label}: all missing")
        return
    counts = vals.value_counts()
    print(f"  {label}  (n non-missing = {len(vals)} / {total})")
    for val, cnt in counts.head(max_show).items():
        bar = "█" * int(20 * cnt / counts.iloc[0])
        print(f"    {val:<45s}  {cnt:>5d}  ({pct(cnt, total):>5.1f}%)  {bar}")
    if len(counts) > max_show:
        print(f"    ... and {len(counts) - max_show} more unique values")


def print_numeric_dist(series, label, total):
    """Print distribution summary for a numeric column."""
    vals = non_missing(series)
    numeric = pd.to_numeric(vals, errors="coerce").dropna()
    if numeric.empty:
        print(f"  {label}: no numeric values")
        return
    q = numeric.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    print(f"  {label}  (n = {len(numeric)} / {total})")
    print(f"    min={numeric.min():.4g}  p5={q[0.05]:.4g}  "
          f"median={q[0.50]:.4g}  mean={numeric.mean():.4g}  "
          f"p95={q[0.95]:.4g}  max={numeric.max():.4g}  "
          f"sd={numeric.std():.4g}")


def print_boolean(series, label, total):
    """Print True/False prevalence for a boolean-like column."""
    vals = non_missing(series)
    if vals.empty:
        print(f"  {label}: all missing")
        return
    lower = vals.str.lower()
    true_count  = (lower.isin({"true",  "1", "yes"})).sum()
    false_count = (lower.isin({"false", "0", "no" })).sum()
    other_count = len(vals) - true_count - false_count
    print(f"  {label}")
    print(f"    True:  {true_count:>5d} / {total}  ({pct(true_count, total):>5.1f}%)")
    print(f"    False: {false_count:>5d} / {total}  ({pct(false_count, total):>5.1f}%)")
    if other_count:
        print(f"    Other: {other_count:>5d}")


def print_mapped_column_prevalence(df, total, prefix="feature2_has_"):
    """
    For each has_X flag in Feature2, report how many files
    successfully mapped that standard column, and what source column name
    was most commonly used.
    """
    standard_cols = ["SNP", "CHR", "BP", "A1", "A2",
                     "BETA", "OR", "Z", "SE", "P",
                     "N", "N_CASES", "N_CONTROLS", "EAF", "MAF", "INFO", "DIRECTION"]

    print(f"\n  {'Column':<14} {'Mapped N':>10} {'Mapped %':>10}  {'Top source column name(s)'}")
    print(f"  {'─'*14} {'─'*10} {'─'*10}  {'─'*40}")

    for col in standard_cols:
        has_col   = f"{prefix}{col}"
        map_col   = f"feature2_mapped_{col}"

        if has_col not in df.columns:
            continue

        has_series = non_missing(df[has_col])
        mapped_count = (has_series.str.lower() == "true").sum()

        # Most common raw source column name used for mapping
        if map_col in df.columns:
            src = non_missing(df[map_col])
            top_sources = src[src.str.lower() != "false"].value_counts().head(3)
            src_str = " | ".join(
                f"{v} ({pct(c, total):.0f}%)" for v, c in top_sources.items()
            ) if not top_sources.empty else "—"
        else:
            src_str = "—"

        print(f"  {col:<14} {mapped_count:>10d} {pct(mapped_count, total):>9.1f}%  {src_str}")


def print_file_sizes(df, total):
    """Distribution of raw file sizes in MB."""
    for col, label in [
        ("feature1_raw_file_size_mb",       "Raw file size (MB)"),
        ("feature1_extracted_file_size_mb",  "Extracted file size (MB)  [0 = not compressed]"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)


def print_row_counts(df, total):
    col = "feature1_n_rows_physical_lines"
    if col in df.columns:
        vals = pd.to_numeric(non_missing(df[col]), errors="coerce").dropna()
        q = vals.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        print(f"  Physical line counts  (n = {len(vals)} / {total})")
        print(f"    min={vals.min():.4g}  p5={q[0.05]:.4g}  "
              f"median={q[0.50]:.4g}  mean={vals.mean():.4g}  "
              f"p95={q[0.95]:.4g}  max={vals.max():.4g}")
        # bucket breakdown
        buckets = [
            ("<100K",   0,           100_000),
            ("100K–1M", 100_000,   1_000_000),
            ("1M–10M",1_000_000,  10_000_000),
            (">10M",  10_000_000, float("inf")),
        ]
        for label, lo, hi in buckets:
            cnt = ((vals >= lo) & (vals < hi)).sum()
            print(f"    {label:<12} {cnt:>5d} files  ({pct(cnt, total):.1f}%)")


def print_column_counts(df, total):
    col = "feature1_n_columns"
    if col in df.columns:
        vals = pd.to_numeric(non_missing(df[col]), errors="coerce").dropna().astype(int)
        counts = vals.value_counts().sort_index()
        print(f"  Number of columns in raw file  (n = {len(vals)} / {total})")
        for ncol, cnt in counts.items():
            print(f"    {ncol:>3d} columns:  {cnt:>4d} files  ({pct(cnt, total):.1f}%)")


def print_missingness_distributions(df, total):
    """
    For key GWAS fields, show the distribution of per-file missingness %.
    """
    fields = {
        "SNP id":       "feature2_missing_snp_percentage",
        "A1 allele":    "feature2_missing_a1_percentage",
        "A2 allele":    "feature2_missing_a2_percentage",
        "P-value":      "feature2_missing_p_percentage",
        "Effect (B/Z)": "feature2_missing_effect_beta_or_z_percentage",
        "Dup SNP":      "feature2_duplicated_snp_percentage",
        "Ambig allele": "feature2_ambiguous_allele_percentage",
        "Invalid allele":"feature2_invalid_allele_percentage",
    }
    print(f"\n  {'Field':<16} {'Files=0%':>9} {'Files<1%':>9} {'Files>10%':>10}  dist (median / max)")
    print(f"  {'─'*16} {'─'*9} {'─'*9} {'─'*10}  {'─'*30}")
    for label, col in fields.items():
        if col not in df.columns:
            continue
        vals = pd.to_numeric(non_missing(df[col]), errors="coerce").dropna()
        if vals.empty:
            continue
        zero   = (vals == 0).sum()
        lt1    = (vals < 1).sum()
        gt10   = (vals > 10).sum()
        print(f"  {label:<16} {zero:>9d} {lt1:>9d} {gt10:>10d}  "
              f"median={vals.median():.3f}  max={vals.max():.3f}")


def print_prs_flags(df, total):
    flags = {
        "PRS minimum columns present":     "feature2_prs_minimum_columns_present",
        "Close to GWAS-SSF schema":        "feature2_close_to_gwas_ssf",
        "Requires column renaming":        "feature2_requires_column_renaming",
        "Requires decompression":          "feature2_requires_decompression_or_extraction",
        "Requires effect conversion":      "feature2_requires_effect_conversion_or_to_log_or",
        "Cannot normalise safely":         "feature2_cannot_normalise_safely",
    }
    for label, col in flags.items():
        if col in df.columns:
            print_boolean(df[col], label, total)


# ---------------------------------------------------------------------------
# Main synthesis report
# ---------------------------------------------------------------------------

def run_synthesis(df):
    total = len(df)

    print("=" * 90)
    print("GWASRANKER FEATURE1 + FEATURE2  —  FILE AND SCHEMA VARIATION SYNTHESIS")
    print(f"Total candidate GWAS files: {total}")
    print("=" * 90)

    # ── FEATURE 1 ──────────────────────────────────────────────────────────
    sep("FEATURE 1  ·  File-Level Structure")

    sep("1.1  File Extensions", char="·")
    if "feature1_file_extension" in df.columns:
        print_categorical(df["feature1_file_extension"], "File extension", total)

    sep("1.2  Compression", char="·")
    if "feature1_compression_type" in df.columns:
        print_categorical(df["feature1_compression_type"], "Compression type", total)
    if "feature1_compression_detection_source" in df.columns:
        print_categorical(df["feature1_compression_detection_source"],
                          "Detection source", total)

    sep("1.3  Delimiter", char="·")
    if "feature1_delimiter" in df.columns:
        print_categorical(df["feature1_delimiter"], "Delimiter", total)

    sep("1.4  File Size Distribution", char="·")
    print_file_sizes(df, total)

    sep("1.5  Row Count Distribution", char="·")
    print_row_counts(df, total)

    sep("1.6  Number of Columns in Raw File", char="·")
    print_column_counts(df, total)

    sep("1.7  Header Detection", char="·")
    if "feature1_header_detected" in df.columns:
        print_boolean(df["feature1_header_detected"], "Header detected", total)

    sep("1.8  Read Status", char="·")
    if "feature1_read_status" in df.columns:
        print_categorical(df["feature1_read_status"], "Read status", total)

    # ── FEATURE 2 ──────────────────────────────────────────────────────────
    sep("FEATURE 2  ·  Schema Mapping and Column Compatibility")

    sep("2.1  Raw Header Diversity  (schema signatures)", char="·")
    for col, label in [
        ("feature2_raw_header_signature",            "Unique raw header signatures"),
        ("feature2_normalised_header_signature",     "Unique normalised header signatures"),
        ("feature2_mapped_standard_header_signature","Unique mapped standard header signatures"),
    ]:
        if col in df.columns:
            n_unique = non_missing(df[col]).nunique()
            n_present = non_missing(df[col]).count()
            print(f"  {label}: {n_unique} distinct patterns across {n_present} files")

    sep("2.2  Mapped Standard Column Prevalence", char="·")
    print(f"  How many of {total} files successfully mapped each standard GWAS column:")
    print_mapped_column_prevalence(df, total)

    sep("2.3  Number of Mapped Standard Columns per File", char="·")
    for col, label in [
        ("feature2_n_unique_mapped_standard_columns_in_file", "Unique mapped standard columns"),
        ("feature2_n_unmapped_columns",                       "Unmapped / non-standard columns"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    sep("2.4  Effect Column Type", char="·")
    if "feature2_effect_column_type" in df.columns:
        print_categorical(df["feature2_effect_column_type"], "Effect column type", total)

    sep("2.5  P-value Format", char="·")
    if "feature2_pvalue_format" in df.columns:
        print_categorical(df["feature2_pvalue_format"], "P-value format", total)

    sep("2.6  Chromosome / Position / SNP Formats", char="·")
    for col, label in [
        ("feature2_chromosome_format", "Chromosome format"),
        ("feature2_bp_format",         "Base-pair position format"),
        ("feature2_snp_format",        "SNP identifier format"),
    ]:
        if col in df.columns:
            print_categorical(df[col], label, total)

    sep("2.7  Allele Formats", char="·")
    for col, label in [
        ("feature2_a1_allele_format", "Effect allele (A1) format"),
        ("feature2_a2_allele_format", "Other allele (A2) format"),
    ]:
        if col in df.columns:
            print_categorical(df[col], label, total)

    sep("2.8  Per-File Missingness and Duplication Distributions", char="·")
    print_missingness_distributions(df, total)

    sep("2.9  PRS Compatibility and Preprocessing Requirements", char="·")
    print_prs_flags(df, total)

    sep("2.10  Unmapped Column Names  (most common non-standard columns)", char="·")
    if "feature2_unmapped_columns_joined" in df.columns:
        # Explode pipe-separated values
        all_unmapped = []
        for val in non_missing(df["feature2_unmapped_columns_joined"]):
            all_unmapped.extend([v.strip() for v in val.split("|") if v.strip()])
        if all_unmapped:
            counts = pd.Series(all_unmapped).value_counts().head(20)
            print(f"  Top non-standard column names found across files:")
            for val, cnt in counts.items():
                print(f"    {val:<40s}  {cnt:>4d} files  ({pct(cnt, total):.1f}%)")

    sep()
    print("END OF FEATURE1 / FEATURE2 SYNTHESIS REPORT")
    sep()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Print Feature1 and Feature2 synthesis report. Saves nothing."
    )
    parser.add_argument(
        "--input-file",
        default=DEFAULT_INPUT_FILE,
        help=f"Input merged CSV. Default: {DEFAULT_INPUT_FILE}",
    )
    args = parser.parse_args()

    path = Path(args.input_file)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    print(f"Loading {path} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(df)} rows × {len(df.columns)} columns.\n")

    run_synthesis(df)


if __name__ == "__main__":
    main()