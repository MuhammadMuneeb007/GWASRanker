#!/usr/bin/env python3
"""
Synthesis analysis for Feature6 (reference-panel + heritability subset) and Feature7.
Prints a structured synthesis report to stdout. Saves nothing.
"""

import argparse
from pathlib import Path
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


def to_numeric(series):
    return pd.to_numeric(non_missing(series), errors="coerce").dropna()


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
# Generic printers
# ---------------------------------------------------------------------------

def print_numeric_dist(series, label, total):
    vals = to_numeric(series)
    if vals.empty:
        print(f"  {label}: no numeric values")
        return
    q = vals.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    print(f"  {label}  (n = {len(vals)} / {total})")
    print(f"    min={vals.min():.4g}  p5={q[0.05]:.4g}  median={q[0.50]:.4g}"
          f"  mean={vals.mean():.4g}  p95={q[0.95]:.4g}  max={vals.max():.4g}"
          f"  sd={vals.std():.4g}")


def print_boolean(series, label, total):
    vals = non_missing(series)
    if vals.empty:
        print(f"  {label}: all missing")
        return
    lower = vals.str.lower()
    t = lower.isin({"true", "1", "yes"}).sum()
    f = lower.isin({"false", "0", "no"}).sum()
    other = len(vals) - t - f
    print(f"  {label}")
    print(f"    True:  {t:>5d} / {total}  ({pct(t, total):>5.1f}%)")
    print(f"    False: {f:>5d} / {total}  ({pct(f, total):>5.1f}%)")
    if other:
        print(f"    Other/missing: {other}")


def print_categorical(series, label, total, max_show=10):
    vals = non_missing(series)
    if vals.empty:
        print(f"  {label}: all missing")
        return
    counts = vals.value_counts()
    print(f"  {label}  (n non-missing = {len(vals)} / {total})")
    for val, cnt in counts.head(max_show).items():
        bar = "█" * int(20 * cnt / counts.iloc[0])
        print(f"    {val:<55s}  {cnt:>5d}  ({pct(cnt, total):>5.1f}%)  {bar}")
    if len(counts) > max_show:
        print(f"    ... and {len(counts) - max_show} more unique values")


def print_numeric_zero_nonzero(series, label, total):
    vals = to_numeric(series)
    if vals.empty:
        print(f"  {label}: no numeric values")
        return
    zero = (vals == 0).sum()
    nonzero = (vals > 0).sum()
    print(f"  {label}  (n = {len(vals)} / {total})")
    print(f"    = 0:   {zero:>5d} files  ({pct(zero, total):.1f}%)")
    print(f"    > 0:   {nonzero:>5d} files  ({pct(nonzero, total):.1f}%)")
    if nonzero > 0:
        nz = vals[vals > 0]
        print(f"    among non-zero:  median={nz.median():.4g}  "
              f"p95={nz.quantile(0.95):.4g}  max={nz.max():.4g}")


# ---------------------------------------------------------------------------
# Feature6 — Reference AF sections
# ---------------------------------------------------------------------------

def section_reference_af(df, total):
    sep("6.R1  Reference Allele-Frequency Overlap and Correlation (Feature6)", char="·")

    for col, label in [
        ("feature6_reference_overlap",              "Reference overlap count"),
        ("feature6_reference_af_correlation",       "Reference AF correlation"),
        ("feature6_reference_mean_abs_af_difference","Mean absolute AF difference"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    # AF correlation quality buckets
    col = "feature6_reference_af_correlation"
    if col in df.columns:
        vals = to_numeric(df[col])
        if not vals.empty:
            buckets = [
                ("< 0.80  (poor)",        float("-inf"), 0.80),
                ("0.80 – 0.95  (moderate)", 0.80,        0.95),
                ("0.95 – 0.99  (good)",    0.95,         0.99),
                ("≥ 0.99  (excellent)",    0.99,         float("inf")),
            ]
            print(f"  Reference AF correlation buckets  (n = {len(vals)} / {total})")
            for label, lo, hi in buckets:
                cnt = ((vals >= lo) & (vals < hi)).sum()
                if lo == float("-inf"):
                    cnt = (vals < 0.80).sum()
                print(f"    {label:<35s}  {cnt:>4d} files  ({pct(cnt, total):.1f}%)")

    for col, label in [
        ("feature6_reference_af_file",                "Reference AF file"),
        ("feature6_reference_af_id_column",           "Reference AF ID column"),
        ("feature6_reference_af_column",              "Reference AF column"),
        ("feature6_reference_af_resolved_path",       "Reference AF resolved path"),
    ]:
        if col in df.columns:
            print_categorical(df[col], label, total)


def section_hapmap3(df, total):
    sep("6.R2  HapMap3 Reference Overlap (Feature6)", char="·")

    for col, label in [
        ("feature6_hapmap3_overlap",          "HapMap3 overlap count"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    for col, label in [
        ("feature6_hapmap3_rsid_available",   "HapMap3 rsID available"),
        ("feature6_hapmap3_chrpos_fallback",  "HapMap3 chr:pos fallback used"),
    ]:
        if col in df.columns:
            print_boolean(df[col], label, total)

    for col, label in [
        ("feature6_hapmap3_match_method",     "HapMap3 match method"),
        ("feature6_hapmap3_reference",        "HapMap3 reference file"),
    ]:
        if col in df.columns:
            print_categorical(df[col], label, total)

    # HapMap3 overlap buckets
    col = "feature6_hapmap3_overlap"
    if col in df.columns:
        vals = to_numeric(df[col])
        if not vals.empty:
            buckets = [
                ("< 500K",         0,         500_000),
                ("500K – 1M",      500_000,   1_000_000),
                ("1M – 1.2M",    1_000_000,   1_200_000),
                ("> 1.2M",       1_200_000,   float("inf")),
            ]
            print(f"  HapMap3 overlap count buckets  (n = {len(vals)} / {total})")
            for label, lo, hi in buckets:
                cnt = ((vals >= lo) & (vals < hi)).sum()
                print(f"    {label:<25s}  {cnt:>4d} files  ({pct(cnt, total):.1f}%)")


# ---------------------------------------------------------------------------
# Feature7 sections
# ---------------------------------------------------------------------------

def section_feature7_hapmap3(df, total):
    sep("7.1  HapMap3 Reference Overlap (Feature7)", char="·")

    for col in df.columns:
        if not col.lower().startswith("feature7_"):
            continue
        if "hapmap3" not in col.lower():
            continue
        vals = non_missing(df[col])
        if vals.empty:
            continue
        lower = vals.str.lower().unique()
        if set(lower).issubset({"true", "false", "0", "1"}):
            print_boolean(df[col], col, total)
        else:
            try:
                pd.to_numeric(vals, errors="raise")
                print_numeric_dist(df[col], col, total)
            except Exception:
                print_categorical(df[col], col, total)


def section_feature7_dbsnp(df, total):
    sep("7.2  dbSNP151 Reference Overlap (Feature7)", char="·")

    for col in df.columns:
        if not col.lower().startswith("feature7_"):
            continue
        if "dbsnp" not in col.lower():
            continue
        vals = non_missing(df[col])
        if vals.empty:
            continue
        lower = vals.str.lower().unique()
        if set(lower).issubset({"true", "false", "0", "1"}):
            print_boolean(df[col], col, total)
        else:
            try:
                pd.to_numeric(vals, errors="raise")
                print_numeric_dist(df[col], col, total)
            except Exception:
                print_categorical(df[col], col, total)


def section_feature7_1kg(df, total):
    sep("7.3  1000 Genomes EUR Reference Overlap (Feature7)", char="·")

    for col in df.columns:
        if not col.lower().startswith("feature7_"):
            continue
        col_lower = col.lower()
        if not any(t in col_lower for t in ["1000g", "1kg", "eur", "thousand_genomes"]):
            continue
        vals = non_missing(df[col])
        if vals.empty:
            continue
        lower = vals.str.lower().unique()
        if set(lower).issubset({"true", "false", "0", "1"}):
            print_boolean(df[col], col, total)
        else:
            try:
                pd.to_numeric(vals, errors="raise")
                print_numeric_dist(df[col], col, total)
            except Exception:
                print_categorical(df[col], col, total)


def section_feature7_plink_bim(df, total):
    sep("7.4  PLINK BIM Reference Overlap (Feature7)", char="·")

    for col in df.columns:
        if not col.lower().startswith("feature7_"):
            continue
        if not any(t in col.lower() for t in ["plink", "bim"]):
            continue
        vals = non_missing(df[col])
        if vals.empty:
            continue
        lower = vals.str.lower().unique()
        if set(lower).issubset({"true", "false", "0", "1"}):
            print_boolean(df[col], col, total)
        else:
            try:
                pd.to_numeric(vals, errors="raise")
                print_numeric_dist(df[col], col, total)
            except Exception:
                print_categorical(df[col], col, total)


def section_feature7_other(df, total):
    sep("7.5  Other Feature7 Reference and Clumping Features", char="·")

    skip_terms = ["hapmap3", "dbsnp", "1000g", "1kg", "eur",
                  "thousand_genomes", "plink", "bim",
                  "file_path", "path", "accessionid", "phenotype",
                  "status", "error", "feature7_file"]

    for col in df.columns:
        if not col.lower().startswith("feature7_"):
            continue
        col_lower = col.lower()
        if any(t in col_lower for t in skip_terms):
            continue
        vals = non_missing(df[col])
        if vals.empty:
            continue
        lower = vals.str.lower().unique()
        if set(lower).issubset({"true", "false", "0", "1"}):
            print_boolean(df[col], col, total)
        else:
            try:
                pd.to_numeric(vals, errors="raise")
                print_numeric_dist(df[col], col, total)
            except Exception:
                print_categorical(df[col], col, total)


# ---------------------------------------------------------------------------
# Main synthesis
# ---------------------------------------------------------------------------

def run_synthesis(df):
    total = len(df)

    print("=" * 90)
    print("GWASRANKER FEATURE6 (REFERENCE + HERITABILITY SUBSET) + FEATURE7")
    print("REFERENCE-PANEL AND HERITABILITY SYNTHESIS")
    print(f"Total candidate GWAS files: {total}")
    print("=" * 90)

    sep("FEATURE 6  ·  Reference Allele-Frequency and HapMap3 Overlap")
    section_reference_af(df, total)
    section_hapmap3(df, total)

    sep("FEATURE 7  ·  External Reference-Panel Overlaps")
    section_feature7_hapmap3(df, total)
    section_feature7_dbsnp(df, total)
    section_feature7_1kg(df, total)
    section_feature7_plink_bim(df, total)
    section_feature7_other(df, total)

    sep()
    print("END OF REFERENCE-PANEL AND HERITABILITY SYNTHESIS REPORT")
    sep()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Print reference-panel and heritability synthesis (Feature6 ref subset + Feature7). Saves nothing."
    )
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
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