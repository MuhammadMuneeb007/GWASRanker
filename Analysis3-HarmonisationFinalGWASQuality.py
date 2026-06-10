#!/usr/bin/env python3
"""
Synthesis analysis for Feature3 and Feature5 columns.
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


def print_numeric_dist(series, label, total):
    vals = non_missing(series)
    numeric = pd.to_numeric(vals, errors="coerce").dropna()
    if numeric.empty:
        print(f"  {label}: no numeric values")
        return
    q = numeric.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    print(f"  {label}  (n = {len(numeric)} / {total})")
    print(f"    min={numeric.min():.4g}  p5={q[0.05]:.4g}  median={q[0.50]:.4g}"
          f"  mean={numeric.mean():.4g}  p95={q[0.95]:.4g}  max={numeric.max():.4g}"
          f"  sd={numeric.std():.4g}")


def print_boolean(series, label, total):
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
        print(f"    Other/missing: {other_count:>5d}")


# ---------------------------------------------------------------------------
# Section-specific helpers
# ---------------------------------------------------------------------------

def print_genome_build(df, total):
    """Build detection outcomes."""
    for col, label in [
        ("feature3_genome_build",         "Inferred genome build (Feature3)"),
        ("feature3_genome_build_is_known","Build successfully inferred"),
        ("feature3_genome_build_is_grch37","Build is GRCh37/hg19"),
        ("feature3_genome_build_is_grch38","Build is GRCh38/hg38"),
        ("feature5_genome_build",          "Genome build used in FinalGWAS"),
    ]:
        if col not in df.columns:
            continue
        s = df[col]
        dtype = non_missing(s).str.lower().unique()
        if set(dtype).issubset({"true","false","0","1"}):
            print_boolean(s, label, total)
        else:
            print_categorical(s, label, total)


def print_gwaslab_loading(df, total):
    """GWASLab load success, variant counts, QC removal."""

    # Load success
    for col, label in [
        ("feature3_gwaslab_load_success",  "GWASLab load success"),
        ("feature3_gwaslab_chr_numeric_success", "CHR parsed as numeric"),
        ("feature3_gwaslab_pos_numeric_success", "POS parsed as numeric"),
        ("feature3_gwaslab_p_numeric_success",   "P parsed as numeric"),
        ("feature3_gwaslab_beta_numeric_success","BETA parsed as numeric"),
        ("feature3_gwaslab_se_numeric_success",  "SE parsed as numeric"),
    ]:
        if col in df.columns:
            print_boolean(df[col], label, total)

    print()

    # Variant counts
    for col, label in [
        ("feature3_gwaslab_loaded_variant_count",    "GWASLab loaded variant count"),
        ("feature3_gwaslab_qc_pass_variant_count",   "GWASLab QC-pass variant count"),
        ("feature3_gwaslab_qc_removed_variant_count","GWASLab QC-removed variant count"),
        ("feature3_gwaslab_qc_removed_variant_percentage","GWASLab QC-removed (%)"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    # QC removal bucket breakdown
    col = "feature3_gwaslab_qc_removed_variant_percentage"
    if col in df.columns:
        vals = pd.to_numeric(non_missing(df[col]), errors="coerce").dropna()
        buckets = [
            ("0% removed",     0,   0.001),
            ("<1% removed",    0.001, 1),
            ("1–5% removed",   1,   5),
            ("5–20% removed",  5,  20),
            (">20% removed",  20, float("inf")),
        ]
        print(f"  QC-removal rate breakdown  (n = {len(vals)} / {total})")
        for label, lo, hi in buckets:
            cnt = ((vals >= lo) & (vals < hi)).sum()
            print(f"    {label:<22s}  {cnt:>4d} files  ({pct(cnt, total):.1f}%)")


def print_gwaslab_inferred_format(df, total):
    """Top inferred GWASLab formats (rank 1 only)."""
    col = "feature3_gwaslab_inferred_format_1"
    score_col = "feature3_gwaslab_inferred_format_1_score"
    if col not in df.columns:
        return
    print_categorical(df[col], "GWASLab top inferred format", total)
    if score_col in df.columns:
        print_numeric_dist(df[score_col], "Top inferred format score", total)


def print_build_match_ratio(df, total):
    """hg19 vs hg38 match counts and ratio."""
    for col, label in [
        ("feature3_gwaslab_build_hg19_match_count", "hg19/GRCh37 build-match count"),
        ("feature3_gwaslab_build_hg38_match_count", "hg38/GRCh38 build-match count"),
        ("feature3_gwaslab_build_match_ratio",       "Build-match ratio (winning build)"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)


def print_duplicate_variants(df, total):
    """Duplicate SNP IDs, chr-pos, and chr-pos-allele duplicates."""
    for col, label in [
        ("feature3_gwaslab_duplicated_snpid_count",        "Dup SNP ID count (pre-QC, GWASLab)"),
        ("feature3_gwaslab_duplicated_chr_pos_count",      "Dup chr:pos count (pre-QC, GWASLab)"),
        ("feature3_gwaslab_duplicated_chr_pos_ea_nea_count","Dup chr:pos:EA:NEA count (pre-QC)"),
        ("feature5_duplicated_snpid_count",                "Dup SNP ID count (FinalGWAS)"),
        ("feature5_duplicated_snpid_percentage",           "Dup SNP ID % (FinalGWAS)"),
        ("feature5_duplicated_chr_pos_count",              "Dup chr:pos count (FinalGWAS)"),
        ("feature5_duplicated_chr_pos_percentage",         "Dup chr:pos % (FinalGWAS)"),
        ("feature5_duplicated_chr_pos_ea_nea_count",       "Dup chr:pos:EA:NEA count (FinalGWAS)"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)


def print_finalgwas_size(df, total):
    for col, label in [
        ("feature5_final_gwas_size_mb",              "FinalGWAS file size (MB)"),
        ("feature5_final_gwas_loaded_variant_count", "FinalGWAS loaded variant count"),
        ("feature5_autosomal_variant_count",         "Autosomal variant count"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    # Scale buckets for variant count
    col = "feature5_final_gwas_loaded_variant_count"
    if col in df.columns:
        vals = pd.to_numeric(non_missing(df[col]), errors="coerce").dropna()
        buckets = [
            ("<500K",      0,         500_000),
            ("500K–2M",    500_000,  2_000_000),
            ("2M–10M",   2_000_000, 10_000_000),
            (">10M",    10_000_000, float("inf")),
        ]
        print(f"  FinalGWAS variant scale breakdown  (n = {len(vals)} / {total})")
        for label, lo, hi in buckets:
            cnt = ((vals >= lo) & (vals < hi)).sum()
            print(f"    {label:<12s}  {cnt:>4d} files  ({pct(cnt, total):.1f}%)")


def print_chromosome_representation(df, total):
    """Autosome presence and non-autosomal contamination."""
    for col, label in [
        ("feature6_all_22_autosomes_present", "All 22 autosomes present"),
        ("feature5_non_autosomal_percentage", "Non-autosomal variant %"),
        ("feature5_invalid_chromosome_percentage", "Invalid chromosome %"),
        ("feature5_missing_chr_percentage",   "Missing CHR %"),
        ("feature5_missing_pos_percentage",   "Missing POS %"),
        ("feature5_missing_snpid_percentage", "Missing SNPID %"),
    ]:
        if col not in df.columns:
            continue
        vals = non_missing(df[col])
        if set(vals.str.lower().unique()).issubset({"true","false","0","1"}):
            print_boolean(df[col], label, total)
        else:
            print_numeric_dist(df[col], label, total)


def print_allele_composition(df, total):
    """SNP vs indel vs palindromic vs multi-character breakdown."""
    for col, label in [
        ("feature5_single_base_snp_percentage",      "Single-base SNP %"),
        ("feature5_indel_percentage",                "Indel %"),
        ("feature5_multi_character_allele_percentage","Multi-character allele %"),
        ("feature5_palindromic_snp_percentage",      "Palindromic (A/T, C/G) SNP %"),
        ("feature5_non_acgt_allele_percentage",      "Non-ACGT allele %"),
        ("feature5_symbolic_allele_percentage",      "Symbolic allele %"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)


def print_pvalue_summary(df, total):
    for col, label in [
        ("feature5_min_p",                          "Minimum P-value"),
        ("feature5_median_p",                       "Median P-value"),
        ("feature5_genome_wide_significant_count",  "Genome-wide significant variants (P≤5e-8)"),
        ("feature5_genome_wide_significant_percentage","Genome-wide significant %"),
        ("feature5_suggestive_significant_count",   "Suggestive significant variants (P≤1e-5)"),
        ("feature5_suggestive_significant_percentage","Suggestive significant %"),
        ("feature5_p_equal_one_count",              "Variants with P=1"),
        ("feature5_p_z_concordance_percentage",     "P–Z concordance %"),
        ("feature5_p_z_correlation",                "P–Z correlation"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    # Signal presence breakdown
    col = "feature5_genome_wide_significant_count"
    if col in df.columns:
        vals = pd.to_numeric(non_missing(df[col]), errors="coerce").dropna()
        zero = (vals == 0).sum()
        lt10 = ((vals > 0) & (vals < 10)).sum()
        lt100 = ((vals >= 10) & (vals < 100)).sum()
        ge100 = (vals >= 100).sum()
        print(f"  Genome-wide significant locus count breakdown  (n = {len(vals)} / {total})")
        print(f"    0 hits:      {zero:>4d} files  ({pct(zero, total):.1f}%)")
        print(f"    1–9 hits:    {lt10:>4d} files  ({pct(lt10, total):.1f}%)")
        print(f"    10–99 hits:  {lt100:>4d} files  ({pct(lt100, total):.1f}%)")
        print(f"    ≥100 hits:   {ge100:>4d} files  ({pct(ge100, total):.1f}%)")


def print_frequency_spectrum(df, total):
    for col, label in [
        ("feature5_rare_variant_percentage",         "Rare variant % (MAF<1%)"),
        ("feature5_low_frequency_variant_percentage","Low-frequency variant % (1–5%)"),
        ("feature5_common_variant_percentage",       "Common variant % (MAF>5%)"),
        ("feature5_median_eaf",                      "Median effect-allele frequency"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)


def print_sample_size(df, total):
    for col, label in [
        ("feature5_has_total_N",    "Has total N column"),
        ("feature5_has_N_cases",    "Has N_cases column"),
        ("feature5_has_N_controls", "Has N_controls column"),
        ("feature5_N_constant_or_variable", "N reporting style (constant vs variable)"),
    ]:
        if col not in df.columns:
            continue
        vals = non_missing(df[col])
        if set(vals.str.lower().unique()).issubset({"true","false","0","1"}):
            print_boolean(df[col], label, total)
        else:
            print_categorical(df[col], label, total)

    for col, label in [
        ("feature5_median_N",          "Median total N"),
        ("feature5_min_N",             "Min total N"),
        ("feature5_max_N",             "Max total N"),
        ("feature5_median_N_cases",    "Median N cases"),
        ("feature5_median_N_controls", "Median N controls"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)


def print_effect_column(df, total):
    for col, label in [
        ("feature5_effect_column_type", "Effect column type in FinalGWAS"),
    ]:
        if col in df.columns:
            print_categorical(df[col], label, total)

    for col, label in [
        ("feature5_or_extreme_percentage",          "Extreme OR % (|OR|>100 or <0.01)"),
        ("feature5_beta_extreme_abs_gt_10_percentage","Extreme |BETA|>10 %"),
        ("feature5_beta_se_z_concordance_percentage","BETA/SE/Z concordance %"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)


def print_tool_compatibility(df, total):
    flags = {
        "PRS minimum columns present":   "feature5_prs_minimum_columns_present",
        "LDSC minimum columns present":  "feature5_ldsc_minimum_columns_present",
        "PLINK minimum columns present": "feature5_plink_minimum_columns_present",
        "GWAS-SSF mandatory columns":    "feature5_gwas_ssf_mandatory_columns_present",
    }
    for label, col in flags.items():
        if col in df.columns:
            print_boolean(df[col], label, total)


def print_standard_columns_finalgwas(df, total):
    """Which standard columns are present in FinalGWAS."""
    cols_of_interest = {
        "SNPID": "feature5_has_SNPID",
        "CHR":   "feature5_has_CHR",
        "POS":   "feature5_has_POS",
        "EA":    "feature5_has_EA",
        "NEA":   "feature5_has_NEA",
        "BETA":  "feature5_has_BETA",
        "OR":    "feature5_has_OR",
        "Z":     "feature5_has_Z",
        "SE":    "feature5_has_SE",
        "P":     "feature5_has_P",
        "N":     "feature5_has_N",
        "N_CASES":    "feature5_has_N_CASES",
        "N_CONTROLS": "feature5_has_N_CONTROLS",
        "EAF":   "feature5_has_EAF",
        "MAF":   "feature5_has_MAF",
        "INFO":  "feature5_has_INFO",
    }
    print(f"\n  {'Column':<14} {'Present N':>10} {'Present %':>10}")
    print(f"  {'─'*14} {'─'*10} {'─'*10}")
    for col_name, col in cols_of_interest.items():
        if col not in df.columns:
            continue
        vals = non_missing(df[col])
        count = (vals.str.lower() == "true").sum()
        print(f"  {col_name:<14} {count:>10d} {pct(count, total):>9.1f}%")


# ---------------------------------------------------------------------------
# Main synthesis report
# ---------------------------------------------------------------------------

def run_synthesis(df):
    total = len(df)

    print("=" * 90)
    print("GWASRANKER FEATURE3 + FEATURE5  —  HARMONISATION AND FINALGWAS QUALITY SYNTHESIS")
    print(f"Total candidate GWAS files: {total}")
    print("=" * 90)

    # ── FEATURE 3 ──────────────────────────────────────────────────────────
    sep("FEATURE 3  ·  Harmonisation and GWASLab Processing")

    sep("3.1  Genome-Build Detection", char="·")
    print_genome_build(df, total)

    sep("3.2  GWASLab Loading and Column Parsing", char="·")
    print_gwaslab_loading(df, total)

    sep("3.3  GWASLab Inferred File Format", char="·")
    print_gwaslab_inferred_format(df, total)

    sep("3.4  Build-Match Evidence (hg19 vs hg38)", char="·")
    print_build_match_ratio(df, total)

    sep("3.5  Duplicate Variants (Pre-QC, GWASLab)", char="·")
    print_duplicate_variants(df, total)

    # ── FEATURE 5 ──────────────────────────────────────────────────────────
    sep("FEATURE 5  ·  FinalGWAS File Quality")

    sep("5.1  FinalGWAS Size and Variant Scale", char="·")
    print_finalgwas_size(df, total)

    sep("5.2  Standard Columns Present in FinalGWAS", char="·")
    print_standard_columns_finalgwas(df, total)

    sep("5.3  Chromosome and Coordinate Validity", char="·")
    print_chromosome_representation(df, total)

    sep("5.4  Allele Composition and Variant Types", char="·")
    print_allele_composition(df, total)

    sep("5.5  Duplicate Variants (FinalGWAS)", char="·")
    # already called above for FinalGWAS columns — re-call filtered to feature5
    for col, label in [
        ("feature5_duplicated_snpid_count",        "Dup SNP ID count"),
        ("feature5_duplicated_snpid_percentage",   "Dup SNP ID %"),
        ("feature5_duplicated_chr_pos_count",      "Dup chr:pos count"),
        ("feature5_duplicated_chr_pos_percentage", "Dup chr:pos %"),
        ("feature5_duplicated_chr_pos_ea_nea_count","Dup chr:pos:EA:NEA count"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    sep("5.6  P-value Distribution and Association Signal", char="·")
    print_pvalue_summary(df, total)

    sep("5.7  Variant Frequency Spectrum", char="·")
    print_frequency_spectrum(df, total)

    sep("5.8  Sample-Size Reporting", char="·")
    print_sample_size(df, total)

    sep("5.9  Effect-Size Column and Validity", char="·")
    print_effect_column(df, total)

    sep("5.10  Downstream Tool Compatibility Flags", char="·")
    print_tool_compatibility(df, total)

    sep()
    print("END OF FEATURE3 / FEATURE5 SYNTHESIS REPORT")
    sep()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Print Feature3 and Feature5 synthesis report. Saves nothing."
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