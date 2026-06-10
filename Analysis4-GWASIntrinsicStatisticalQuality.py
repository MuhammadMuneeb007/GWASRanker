#!/usr/bin/env python3
"""
Synthesis analysis for GWAS-intrinsic statistical quality:
Feature4 and Feature6 only (excluding reference-panel overlap columns).
Prints a structured synthesis report to stdout. Saves nothing.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

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
# Column selection — Feature4 and Feature6 only
# ---------------------------------------------------------------------------

def is_intrinsic_column(column):
    col = column.lower()

    if col.startswith("feature4_"):
        return True

    if col.startswith("feature6_"):
        exclude = [
            "reference_af_file", "reference_af_resolved_path",
            "reference_af_id_column", "reference_af_column",
            "reference_overlap", "reference_af_correlation",
            "reference_mean_abs_af_difference",
            "hapmap3_reference", "hapmap3_reference_resolved_path",
            "hapmap3_reference_id_column", "hapmap3_overlap",
            "hapmap3_match_method", "hapmap3_rsid_available",
            "hapmap3_chrpos_fallback",
            "ldsc_ref_ld_chr", "ldsc_w_ld_chr",
            "feature6_file", "gwas_standard_path", "combined_file",
        ]
        return not any(term in col for term in exclude)

    return False


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
    vals = pd.to_numeric(non_missing(series), errors="coerce").dropna()
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


def print_numeric_zero_nonzero(series, label, total):
    vals = pd.to_numeric(non_missing(series), errors="coerce").dropna()
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
# Heritability table printer
# ---------------------------------------------------------------------------

def print_heritability_table(df, total):
    """
    Print a detailed heritability summary table covering LDSC h2,
    intercept, ratio, N x h2, and removal rates — with clear
    indication of how much these vary across GWAS files.
    """
    sep("HERITABILITY VARIATION SUMMARY TABLE", char="=")
    print(f"  This table shows how much heritability-related features vary")
    print(f"  across the {total} candidate GWAS files.")
    print()

    metrics = [
        ("feature6_ldsc_h2",           "LDSC h2 (heritability estimate)"),
        ("feature6_ldsc_h2_se",        "LDSC h2 SE"),
        ("feature6_ldsc_intercept",    "LDSC intercept"),
        ("feature6_ldsc_intercept_se", "LDSC intercept SE"),
        ("feature6_ldsc_ratio",        "LDSC ratio (confounding / signal)"),
        ("feature6_ldsc_ratio_se",     "LDSC ratio SE"),
        ("feature6_ldsc_mean_chi2",    "LDSC mean chi-square"),
        ("feature6_ldsc_lambda_gc",    "LDSC lambda GC"),
        ("feature6_n_times_h2",        "N × h2 (effective signal)"),
        ("feature6_ldsc_clean_removed_variant_percentage",
                                       "LDSC variant removal %"),
    ]

    header = (f"  {'Metric':<42s} {'n':>5s} {'Missing%':>9s} "
              f"{'Min':>10s} {'p5':>10s} {'Median':>10s} "
              f"{'Mean':>10s} {'p95':>10s} {'Max':>10s} {'SD':>10s}")
    print(header)
    print("  " + "─" * (len(header) - 2))

    for col, label in metrics:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(non_missing(df[col]), errors="coerce").dropna()
        miss_count = total - len(vals)
        miss_pct = pct(miss_count, total)
        if vals.empty:
            print(f"  {label:<42s} {'0':>5s} {'100.0':>9s}  {'—':>10s}" * 6)
            continue
        q = vals.quantile([0.05, 0.95])
        print(f"  {label:<42s} {len(vals):>5d} {miss_pct:>8.1f}%"
              f" {vals.min():>10.4g} {q[0.05]:>10.4g} {vals.median():>10.4g}"
              f" {vals.mean():>10.4g} {q[0.95]:>10.4g} {vals.max():>10.4g}"
              f" {vals.std():>10.4g}")

    print()

    # h2 range buckets — key synthesis point
    col = "feature6_ldsc_h2"
    if col in df.columns:
        vals = pd.to_numeric(non_missing(df[col]), errors="coerce").dropna()
        print(f"  LDSC h2 distribution buckets  (n = {len(vals)} / {total})")
        buckets = [
            ("Negative h2  (estimation artefact)", float("-inf"), 0),
            ("h2 = 0 – 0.02  (near-zero)",          0,    0.02),
            ("h2 = 0.02 – 0.10  (low)",              0.02, 0.10),
            ("h2 = 0.10 – 0.30  (moderate)",         0.10, 0.30),
            ("h2 > 0.30  (high)",                    0.30, float("inf")),
        ]
        for label, lo, hi in buckets:
            cnt = ((vals >= lo) & (vals < hi)).sum()
            if lo == float("-inf"):
                cnt = (vals < 0).sum()
            print(f"    {label:<45s}  {cnt:>4d} files  ({pct(cnt, total):.1f}%)")

    print()

    # N x h2 variation — critical for PRS signal
    col = "feature6_n_times_h2"
    if col in df.columns:
        vals = pd.to_numeric(non_missing(df[col]), errors="coerce").dropna()
        print(f"  N × h2 distribution buckets  (n = {len(vals)} / {total})")
        buckets = [
            ("Negative N×h2", float("-inf"), 0),
            ("0 – 1,000",      0,     1_000),
            ("1,000 – 10,000", 1_000, 10_000),
            ("10,000 – 50,000",10_000,50_000),
            ("> 50,000",       50_000, float("inf")),
        ]
        for label, lo, hi in buckets:
            cnt = ((vals >= lo) & (vals < hi)).sum()
            if lo == float("-inf"):
                cnt = (vals < 0).sum()
            print(f"    {label:<45s}  {cnt:>4d} files  ({pct(cnt, total):.1f}%)")

    print()

    # LDSC intercept interpretation
    col = "feature6_ldsc_intercept"
    if col in df.columns:
        vals = pd.to_numeric(non_missing(df[col]), errors="coerce").dropna()
        below1 = (vals < 1.0).sum()
        near1  = ((vals >= 1.0) & (vals < 1.05)).sum()
        mod    = ((vals >= 1.05) & (vals < 1.10)).sum()
        high   = (vals >= 1.10).sum()
        print(f"  LDSC intercept interpretation  (n = {len(vals)} / {total})")
        print(f"    < 1.00  (deflation / negative bias):  {below1:>4d} files  ({pct(below1, total):.1f}%)")
        print(f"    1.00 – 1.05  (near-nominal):          {near1:>4d} files  ({pct(near1, total):.1f}%)")
        print(f"    1.05 – 1.10  (moderate inflation):    {mod:>4d} files  ({pct(mod, total):.1f}%)")
        print(f"    ≥ 1.10  (notable inflation):          {high:>4d} files  ({pct(high, total):.1f}%)")


# ---------------------------------------------------------------------------
# Section printers
# ---------------------------------------------------------------------------

def section_sample_size(df, total):
    sep("4.1  Sample-Size Reporting (Feature4)", char="·")
    for col, label in [
        ("feature4_has_total_N",         "Has total N column"),
        ("feature4_has_variant_level_N", "N is variant-level (not constant)"),
        ("feature4_has_N_cases",         "Has N_cases column"),
        ("feature4_has_N_controls",      "Has N_controls column"),
    ]:
        if col in df.columns:
            print_boolean(df[col], label, total)

    if "feature4_N_constant_or_variable" in df.columns:
        print_categorical(df["feature4_N_constant_or_variable"],
                          "N reporting style", total)

    for col, label in [
        ("feature4_median_N",            "Median total N per file"),
        ("feature4_min_N",               "Min total N per file"),
        ("feature4_max_N",               "Max total N per file"),
        ("feature4_N_variability_ratio", "N variability ratio (max/min)"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)


def section_variant_quality(df, total):
    sep("4.2  Variant Identifier and Allele Quality (Feature4)", char="·")
    for col, label in [
        ("feature4_palindromic_snp_percentage",        "Palindromic SNP %"),
        ("feature4_non_acgt_allele_percentage",        "Non-ACGT allele %"),
        ("feature4_indel_percentage",                  "Indel %"),
        ("feature4_multi_character_allele_percentage", "Multi-character allele %"),
        ("feature4_missing_variant_id_percentage",     "Missing variant ID %"),
        ("feature4_duplicated_variant_id_percentage",  "Duplicated variant ID %"),
        ("feature4_unique_variant_count",              "Unique variant count"),
        ("feature4_rsid_percentage",                   "rsID available %"),
        ("feature4_chrpos_percentage",                 "Chr:pos available %"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    for col, label in [
        ("feature4_has_rsid_column",    "Has rsID column"),
        ("feature4_has_chr_bp_columns", "Has CHR+BP columns"),
        ("feature4_has_chr_bp_alleles", "Has CHR+BP+alleles"),
        ("feature4_allele_case_mixed",  "Allele case mixed"),
        ("feature4_a1_has_lowercase",   "A1 has lowercase"),
        ("feature4_a2_has_lowercase",   "A2 has lowercase"),
    ]:
        if col in df.columns:
            print_boolean(df[col], label, total)

    if "feature4_snp_id_format" in df.columns:
        print_categorical(df["feature4_snp_id_format"], "SNP ID format", total)


def section_pvalue_feature4(df, total):
    sep("4.3  P-value Completeness and Association Signal (Feature4)", char="·")
    if "feature4_has_p_column" in df.columns:
        print_boolean(df["feature4_has_p_column"], "Has P-value column", total)

    for col, label in [
        ("feature4_min_p",    "Minimum P-value per file"),
        ("feature4_median_p", "Median P-value per file"),
        ("feature4_p_equal_one_count",               "Variants with P=1 (count)"),
        ("feature4_genome_wide_significant_count",   "Genome-wide significant count"),
        ("feature4_genome_wide_significant_percentage","Genome-wide significant %"),
        ("feature4_suggestive_significant_count",    "Suggestive significant count"),
        ("feature4_suggestive_significant_percentage","Suggestive significant %"),
        ("feature4_p_scientific_notation_percentage","P in scientific notation %"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    # Signal presence breakdown
    col = "feature4_genome_wide_significant_count"
    if col in df.columns:
        vals = pd.to_numeric(non_missing(df[col]), errors="coerce").dropna()
        zero   = (vals == 0).sum()
        lt10   = ((vals > 0) & (vals < 10)).sum()
        lt100  = ((vals >= 10) & (vals < 100)).sum()
        ge100  = (vals >= 100).sum()
        print(f"  GW-significant hit count breakdown  (n = {len(vals)} / {total})")
        print(f"    0 hits:      {zero:>4d} files  ({pct(zero, total):.1f}%)")
        print(f"    1–9 hits:    {lt10:>4d} files  ({pct(lt10, total):.1f}%)")
        print(f"    10–99 hits:  {lt100:>4d} files  ({pct(lt100, total):.1f}%)")
        print(f"    ≥100 hits:   {ge100:>4d} files  ({pct(ge100, total):.1f}%)")


def section_info_eaf(df, total):
    sep("4.4  INFO Score and Allele-Frequency Availability (Feature4)", char="·")
    for col, label in [
        ("feature4_has_info_column", "Has INFO column"),
        ("feature4_has_eaf_column",  "Has EAF column"),
        ("feature4_has_maf_column",  "Has MAF column"),
    ]:
        if col in df.columns:
            print_boolean(df[col], label, total)

    for col, label in [
        ("feature4_median_info",                 "Median INFO score"),
        ("feature4_info_missing_percentage",     "INFO missing %"),
        ("feature4_info_below_0_3_percentage",   "INFO < 0.3 %"),
        ("feature4_info_below_0_8_percentage",   "INFO < 0.8 %"),
        ("feature4_median_eaf",                  "Median EAF"),
        ("feature4_rare_variant_percentage",     "Rare variant % (MAF<1%)"),
        ("feature4_low_freq_variant_percentage", "Low-frequency % (1–5%)"),
        ("feature4_common_variant_percentage",   "Common variant % (>5%)"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)


def section_inflation(df, total):
    sep("6.1  Statistical Inflation and Signal Enrichment (Feature6)", char="·")
    for col, label in [
        ("feature6_lambda_gc",               "Genomic-control lambda (GC)"),
        ("feature6_mean_chi2",               "Mean chi-square statistic"),
        ("feature6_qq_lambda_median",        "QQ lambda (median)"),
        ("feature6_qq_lambda_top5pct",       "QQ lambda (top 5%)"),
        ("feature6_qq_lambda_top1pct",       "QQ lambda (top 1%)"),
        ("feature6_lambda_gc_top_1pct",      "GC lambda (top 1%)"),
        ("feature6_lambda_top1_to_median_ratio","Lambda top-1%/median ratio"),
        ("feature6_qq_top1_to_median_ratio", "QQ top-1%/median ratio"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    if "feature6_qq_inflation_pattern" in df.columns:
        print_categorical(df["feature6_qq_inflation_pattern"],
                          "QQ inflation pattern", total)

    if "feature6_z_consistency_fail_gt_10pct" in df.columns:
        print_boolean(df["feature6_z_consistency_fail_gt_10pct"],
                      "Z-score consistency failure >10%", total)

    if "feature6_inconsistent_z_pct" in df.columns:
        print_numeric_dist(df["feature6_inconsistent_z_pct"],
                           "Inconsistent Z-score %", total)


def section_beta_se(df, total):
    sep("6.2  Beta and SE Distributions (Feature6)", char="·")
    for col, label in [
        ("feature6_beta_mean",                    "Beta mean"),
        ("feature6_beta_std",                     "Beta SD"),
        ("feature6_beta_skewness",                "Beta skewness"),
        ("feature6_beta_kurtosis",                "Beta kurtosis"),
        ("feature6_beta_p95_abs",                 "Beta |p95|"),
        ("feature6_beta_p99_abs",                 "Beta |p99|"),
        ("feature6_beta_outlier_pct_abs_z_gt_5",  "Beta outlier % (|z|>5)"),
        ("feature6_beta_outlier_pct_abs_z_gt_10", "Beta outlier % (|z|>10)"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    if "feature6_beta_distribution_flag" in df.columns:
        print_categorical(df["feature6_beta_distribution_flag"],
                          "Beta distribution flag", total)

    print()
    for col, label in [
        ("feature6_se_mean", "SE mean"),
        ("feature6_se_std",  "SE SD"),
        ("feature6_se_cv",   "SE coefficient of variation"),
        ("feature6_se_min",  "SE minimum"),
        ("feature6_se_max",  "SE maximum"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    for col, label in [
        ("feature6_se_bimodal_flag",     "SE bimodal flag"),
        ("feature6_se_distribution_flag","SE distribution flag"),
    ]:
        if col not in df.columns:
            continue
        s = non_missing(df[col])
        if set(s.str.lower().unique()).issubset({"true", "false", "0", "1"}):
            print_boolean(df[col], label, total)
        else:
            print_categorical(df[col], label, total)


def section_heritability(df, total):
    sep("6.3  Heritability and LDSC Metrics (Feature6)", char="·")

    print()
    print("  NOTE: Heritability estimates varied substantially across candidate")
    print("  GWAS files, spanning negative estimates (LDSC artefacts), near-zero,")
    print("  and high estimates, with similarly large variation in LDSC intercept,")
    print("  ratio, and N×h2. See the heritability variation table below.")
    print()

    # Availability flags first
    for col, label in [
        ("feature6_heritability_attempted", "Heritability attempted"),
        ("feature6_heritability_success",   "Heritability succeeded"),
        ("feature6_ldsc_h2_attempted",      "LDSC h2 attempted"),
        ("feature6_ldsc_h2_success",        "LDSC h2 succeeded"),
        ("feature6_n_times_h2_available",   "N × h2 available"),
    ]:
        if col in df.columns:
            print_boolean(df[col], label, total)

    print()

    # LDSC source flags
    for col, label in [
        ("feature6_ldsc_beta_source", "LDSC beta source"),
        ("feature6_ldsc_z_source",    "LDSC Z source"),
        ("feature6_ldsc_se_source",   "LDSC SE source"),
    ]:
        if col in df.columns:
            print_categorical(df[col], label, total)

    print()

    # LDSC cleaning variant counts
    for col, label in [
        ("feature6_ldsc_clean_pre_variant_count",        "LDSC pre-cleaning variant count"),
        ("feature6_ldsc_clean_post_variant_count",       "LDSC post-cleaning variant count"),
        ("feature6_ldsc_clean_removed_variant_count",    "LDSC removed variant count"),
        ("feature6_ldsc_clean_removed_variant_percentage","LDSC removed variant %"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    # LDSC removal rate breakdown
    col = "feature6_ldsc_clean_removed_variant_percentage"
    if col in df.columns:
        vals = pd.to_numeric(non_missing(df[col]), errors="coerce").dropna()
        buckets = [
            ("<5% removed",    0,   5),
            ("5–20% removed",  5,  20),
            ("20–50% removed", 20, 50),
            (">50% removed",   50, float("inf")),
        ]
        print(f"  LDSC removal rate breakdown  (n = {len(vals)} / {total})")
        for label, lo, hi in buckets:
            cnt = ((vals >= lo) & (vals < hi)).sum()
            print(f"    {label:<22s}  {cnt:>4d} files  ({pct(cnt, total):.1f}%)")

    print()

    # Now print the full heritability variation table
    print_heritability_table(df, total)


def section_chromosomal_coverage(df, total):
    sep("6.4  Chromosomal Coverage and Variant Density (Feature6)", char="·")
    if "feature6_all_22_autosomes_present" in df.columns:
        print_boolean(df["feature6_all_22_autosomes_present"],
                      "All 22 autosomes present", total)

    for col, label in [
        ("feature6_chr22_variant_count",           "Chr22 variant count"),
        ("feature6_variant_density_median_per_mb", "Median variant density (per Mb)"),
        ("feature6_variant_density_min_chr_per_mb","Min chr variant density (per Mb)"),
        ("feature6_zero_coverage_1mb_bin_percentage","Zero-coverage 1Mb bin %"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)


def section_effect_frequency(df, total):
    sep("6.5  Effect–Frequency Relationship and Independent Loci (Feature6)", char="·")
    if "feature6_abs_beta_log_maf_spearman" in df.columns:
        print_numeric_dist(df["feature6_abs_beta_log_maf_spearman"],
                           "|Beta|–log(MAF) Spearman correlation", total)
    if "feature6_effect_frequency_relationship_note" in df.columns:
        print_categorical(df["feature6_effect_frequency_relationship_note"],
                          "Effect–frequency relationship note", total)

    if "feature6_n_genome_wide_significant_window_loci" in df.columns:
        print_numeric_zero_nonzero(
            df["feature6_n_genome_wide_significant_window_loci"],
            "Independent GW-significant loci (window-based)", total)
        print_numeric_dist(
            df["feature6_n_genome_wide_significant_window_loci"],
            "Independent locus count distribution", total)

        # Locus count breakdown
        vals = pd.to_numeric(
            non_missing(df["feature6_n_genome_wide_significant_window_loci"]),
            errors="coerce").dropna()
        buckets = [
            ("0 loci (no GW signal)",   0,   1),
            ("1–10 loci",               1,  11),
            ("11–50 loci",             11,  51),
            ("51–200 loci",            51, 201),
            (">200 loci",             201, float("inf")),
        ]
        print(f"  Independent locus count breakdown  (n = {len(vals)} / {total})")
        for label, lo, hi in buckets:
            cnt = ((vals >= lo) & (vals < hi)).sum()
            print(f"    {label:<28s}  {cnt:>4d} files  ({pct(cnt, total):.1f}%)")


# ---------------------------------------------------------------------------
# Main synthesis
# ---------------------------------------------------------------------------

def run_synthesis(df):
    total = len(df)

    print("=" * 90)
    print("GWASRANKER FEATURE4 + FEATURE6")
    print("GWAS-INTRINSIC STATISTICAL QUALITY SYNTHESIS")
    print(f"Total candidate GWAS files: {total}")
    print("=" * 90)

    sep("FEATURE 4  ·  Sample Size, Variant Quality, P-value, INFO, EAF")
    section_sample_size(df, total)
    section_variant_quality(df, total)
    section_pvalue_feature4(df, total)
    section_info_eaf(df, total)

    sep("FEATURE 6  ·  Inflation, Effect Distributions, Heritability, Coverage")
    section_inflation(df, total)
    section_beta_se(df, total)
    section_heritability(df, total)
    section_chromosomal_coverage(df, total)
    section_effect_frequency(df, total)

    sep()
    print("END OF GWAS-INTRINSIC STATISTICAL QUALITY SYNTHESIS REPORT")
    sep()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Print GWAS-intrinsic statistical quality synthesis (Feature4+6 only). Saves nothing."
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