#!/usr/bin/env python3
"""
Synthesis analysis for Feature8 (target-genotype compatibility) and
selected Feature9 (PRS-ready target-overlapping variant counts).
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


def sep(title="", width=90, char="-"):
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
        bar = "#" * int(20 * cnt / counts.iloc[0])
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
# Feature8 sections
# ---------------------------------------------------------------------------

def section_f8_variant_counts(df, total):
    sep("8.1  GWAS and Target BIM Variant Counts (Feature8)", char=".")

    for col, label in [
        ("feature8_final_gwas_variant_count", "GWAS variant count"),
        ("feature8_target_bim_variant_count", "Target BIM variant count"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)


def section_f8_overlap(df, total):
    sep("8.2  GWAS-Target Variant Overlap (Feature8)", char=".")

    for col, label in [
        ("feature8_rsid_overlap",             "rsID overlap count"),
        ("feature8_chrpos_overlap",           "Chr:pos overlap count"),
        ("feature8_chrposallele_overlap",     "Chr:pos:allele overlap count"),
        ("feature8_rsid_overlap_pct_of_gwas", "rsID overlap % of GWAS"),
        ("feature8_chrpos_overlap_pct_of_gwas","Chr:pos overlap % of GWAS"),
        ("feature8_chrposallele_overlap_pct_of_gwas","Chr:pos:allele overlap % of GWAS"),
        ("feature8_rsid_overlap_pct_of_target","rsID overlap % of target"),
        ("feature8_chrpos_overlap_pct_of_target","Chr:pos overlap % of target"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    # Overlap percentage buckets for primary overlap metric
    for col, label in [
        ("feature8_rsid_overlap_pct_of_gwas",  "rsID overlap % of GWAS"),
        ("feature8_chrpos_overlap_pct_of_gwas", "Chr:pos overlap % of GWAS"),
    ]:
        if col not in df.columns:
            continue
        vals = to_numeric(df[col])
        if vals.empty:
            continue
        buckets = [
            ("= 0%          (no overlap)",   0,    0.001),
            ("0 – 25%       (low)",          0.001, 25),
            ("25 – 50%      (moderate)",     25,    50),
            ("50 – 80%      (good)",         50,    80),
            ("> 80%         (high)",         80,    float("inf")),
        ]
        print(f"  {label} buckets  (n = {len(vals)} / {total})")
        for blabel, lo, hi in buckets:
            cnt = ((vals >= lo) & (vals < hi)).sum()
            print(f"    {blabel:<35s}  {cnt:>4d} files  ({pct(cnt, total):.1f}%)")


def section_f8_allele_matching(df, total):
    sep("8.3  Allele Matching Categories (Feature8)", char=".")

    for col, label in [
        ("feature8_direct_allele_match_count",    "Direct allele match count"),
        ("feature8_flipped_allele_match_count",   "Flipped allele match count"),
        ("feature8_strand_flip_match_count",      "Strand-flip match count"),
        ("feature8_allele_mismatch_count",        "Allele mismatch count"),
        ("feature8_palindromic_overlap_count",    "Palindromic overlap count"),
        ("feature8_missing_allele_count",         "Missing allele count"),
        ("feature8_direct_allele_match_pct",      "Direct allele match %"),
        ("feature8_flipped_allele_match_pct",     "Flipped allele match %"),
        ("feature8_strand_flip_match_pct",        "Strand-flip match %"),
        ("feature8_allele_mismatch_pct",          "Allele mismatch %"),
        ("feature8_palindromic_overlap_pct",      "Palindromic overlap %"),
        ("feature8_missing_allele_pct",           "Missing allele %"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    # Zero/nonzero for mismatch
    for col, label in [
        ("feature8_allele_mismatch_count",     "Allele mismatch count"),
        ("feature8_palindromic_overlap_count", "Palindromic overlap count"),
    ]:
        if col in df.columns:
            print_numeric_zero_nonzero(df[col], label, total)


def section_f8_maf(df, total):
    sep("8.4  GWAS-Target MAF Concordance (Feature8)", char=".")

    for col, label in [
        ("feature8_maf_correlation",          "GWAS-target MAF correlation"),
        ("feature8_maf_mean_abs_difference",  "Mean absolute MAF difference"),
        ("feature8_maf_available",            "MAF available"),
    ]:
        if col not in df.columns:
            continue
        vals = non_missing(df[col])
        lower = vals.str.lower().unique()
        if set(lower).issubset({"true", "false", "0", "1"}):
            print_boolean(df[col], label, total)
        else:
            print_numeric_dist(df[col], label, total)

    # MAF correlation buckets
    col = "feature8_maf_correlation"
    if col in df.columns:
        vals = to_numeric(df[col])
        if not vals.empty:
            buckets = [
                ("< 0.80  (poor)",          float("-inf"), 0.80),
                ("0.80 – 0.95  (moderate)", 0.80,          0.95),
                ("0.95 – 0.99  (good)",     0.95,          0.99),
                (">= 0.99  (excellent)",    0.99,          float("inf")),
            ]
            print(f"  MAF correlation buckets  (n = {len(vals)} / {total})")
            for label, lo, hi in buckets:
                cnt = ((vals >= lo) & (vals < hi)).sum()
                if lo == float("-inf"):
                    cnt = (vals < 0.80).sum()
                print(f"    {label:<35s}  {cnt:>4d} files  ({pct(cnt, total):.1f}%)")


def section_f8_usable_variants(df, total):
    sep("8.5  Final Usable Target-Overlapping Variants (Feature8)", char=".")

    for col, label in [
        ("feature8_final_usable_variant_count",   "Final usable variant count"),
        ("feature8_final_usable_variant_pct_of_gwas","Final usable % of GWAS"),
        ("feature8_final_usable_variant_pct_of_target","Final usable % of target"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    print_numeric_zero_nonzero(
        df.get("feature8_final_usable_variant_count", pd.Series(dtype=float)),
        "Final usable variant count (zero vs non-zero)", total
    )

    # Usable variant count buckets
    col = "feature8_final_usable_variant_count"
    if col in df.columns:
        vals = to_numeric(df[col])
        if not vals.empty:
            buckets = [
                ("0          (no usable variants)", 0,         1),
                ("1 – 50K",                         1,         50_000),
                ("50K – 500K",                      50_000,    500_000),
                ("500K – 2M",                       500_000,   2_000_000),
                ("> 2M",                            2_000_000, float("inf")),
            ]
            print(f"  Final usable variant count buckets  (n = {len(vals)} / {total})")
            for label, lo, hi in buckets:
                cnt = ((vals >= lo) & (vals < hi)).sum()
                print(f"    {label:<38s}  {cnt:>4d} files  ({pct(cnt, total):.1f}%)")


def section_f8_target_sample(df, total):
    sep("8.6  Target Sample and Phenotype Metadata (Feature8)", char=".")

    for col, label in [
        ("feature8_target_sample_size",   "Target sample size"),
        ("feature8_target_case_count",    "Target case count"),
        ("feature8_target_control_count", "Target control count"),
        ("feature8_case_fraction",        "Target case fraction"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    for col, label in [
        ("feature8_phenotype_type",       "Target phenotype type"),
        ("feature8_target_build",         "Target genome build"),
    ]:
        if col in df.columns:
            print_categorical(df[col], label, total)


def section_f8_processing(df, total):
    sep("8.7  Feature8 Processing Flags", char=".")

    # Auto-detect remaining boolean and categorical feature8 columns
    # not already covered in the above sections
    covered = {
        "feature8_final_gwas_variant_count",
        "feature8_target_bim_variant_count",
        "feature8_rsid_overlap",
        "feature8_chrpos_overlap",
        "feature8_chrposallele_overlap",
        "feature8_rsid_overlap_pct_of_gwas",
        "feature8_chrpos_overlap_pct_of_gwas",
        "feature8_chrposallele_overlap_pct_of_gwas",
        "feature8_rsid_overlap_pct_of_target",
        "feature8_chrpos_overlap_pct_of_target",
        "feature8_direct_allele_match_count",
        "feature8_flipped_allele_match_count",
        "feature8_strand_flip_match_count",
        "feature8_allele_mismatch_count",
        "feature8_palindromic_overlap_count",
        "feature8_missing_allele_count",
        "feature8_direct_allele_match_pct",
        "feature8_flipped_allele_match_pct",
        "feature8_strand_flip_match_pct",
        "feature8_allele_mismatch_pct",
        "feature8_palindromic_overlap_pct",
        "feature8_missing_allele_pct",
        "feature8_maf_correlation",
        "feature8_maf_mean_abs_difference",
        "feature8_maf_available",
        "feature8_final_usable_variant_count",
        "feature8_final_usable_variant_pct_of_gwas",
        "feature8_final_usable_variant_pct_of_target",
        "feature8_target_sample_size",
        "feature8_target_case_count",
        "feature8_target_control_count",
        "feature8_case_fraction",
        "feature8_phenotype_type",
        "feature8_target_build",
    }

    skip_terms = ["file", "path", "accessionid", "phenotype_name",
                  "status", "error", "log", "combined_file"]

    for col in df.columns:
        if not col.lower().startswith("feature8_"):
            continue
        if col in covered:
            continue
        if any(t in col.lower() for t in skip_terms):
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
# Feature9 selected sections
# ---------------------------------------------------------------------------

def section_f9_prs_ready(df, total):
    sep("9.1  PRS-Ready Target-Overlapping Variant Counts (Feature9 selected)", char=".")

    for col, label in [
        ("feature9_prs_full_ready_and_in_target", "PRS full-ready and in-target count"),
        ("feature9_prs_ready_and_in_target",      "PRS ready and in-target count"),
        ("feature9_target_bim_variant_count",     "Target BIM variant count (Feature9)"),
        ("feature9_final_gwas_variant_count",     "Final GWAS variant count (Feature9)"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)
            print_numeric_zero_nonzero(df[col], f"{label} (zero vs non-zero)", total)

    # Auto-detect remaining in_target / overlap feature9 columns
    skip_terms = ["file", "path", "accessionid", "phenotype",
                  "status", "error", "log", "combined_file",
                  "fisher", "clump", "prune", "plink",
                  "bed", "bim_file", "fam", "bfile"]

    for col in df.columns:
        col_lower = col.lower()
        if not col_lower.startswith("feature9_"):
            continue
        if any(t in col_lower for t in skip_terms):
            continue
        if not any(t in col_lower for t in [
            "in_target", "target_overlap", "overlap_with_target",
            "target_compatible", "target_allele", "target_maf",
            "maf_correlation", "maf_abs",
        ]):
            continue
        # skip already covered
        if col in {
            "feature9_prs_full_ready_and_in_target",
            "feature9_prs_ready_and_in_target",
            "feature9_target_bim_variant_count",
            "feature9_final_gwas_variant_count",
        }:
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
    print("GWASRANKER FEATURE8 + FEATURE9 (SELECTED)")
    print("TARGET-GENOTYPE COMPATIBILITY SYNTHESIS")
    print(f"Total candidate GWAS files: {total}")
    print("=" * 90)

    sep("FEATURE 8  .  Target-Genotype Compatibility")
    section_f8_variant_counts(df, total)
    section_f8_overlap(df, total)
    section_f8_allele_matching(df, total)
    section_f8_maf(df, total)
    section_f8_usable_variants(df, total)
    section_f8_target_sample(df, total)
    section_f8_processing(df, total)

    sep("FEATURE 9 (SELECTED)  .  PRS-Ready Target-Overlapping Variants")
    section_f9_prs_ready(df, total)

    sep()
    print("END OF TARGET-GENOTYPE COMPATIBILITY SYNTHESIS REPORT")
    sep()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Print target-genotype compatibility synthesis (Feature8 + selected Feature9). Saves nothing."
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