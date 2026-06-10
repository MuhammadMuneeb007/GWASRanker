#!/usr/bin/env python3
"""
Synthesis analysis for Feature9: PRS readiness, clumping, pruning,
effect-size distributions, and per-phenotype clumping tables.
Prints a structured synthesis report to stdout. Saves nothing.
"""

import argparse
from pathlib import Path
import pandas as pd

DEFAULT_INPUT_FILE = "AllFeatures_PRS_Merged.csv"

MISSING_TOKENS = {"", "nan", "na", "n/a", "none", "null", ".", "<na>"}

CLUMP_THRESHOLDS = [
    ("p5e8",  "P <= 5e-8  (GW significant)"),
    ("p1e6",  "P <= 1e-6"),
    ("p1e5",  "P <= 1e-5"),
    ("p1e4",  "P <= 1e-4"),
    ("p1e3",  "P <= 1e-3"),
    ("p0_05", "P <= 0.05"),
    ("p1",    "P <= 1.0  (all variants)"),
]


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
        print("─" * width)


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
# Helper: get phenotype column
# ---------------------------------------------------------------------------

def get_pheno_col(df):
    for c in ["feature9_phenotype", "phenotype", "feature1_phenotype"]:
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------------------------------
# Section 9.1 — PRS readiness flags
# ---------------------------------------------------------------------------

def section_prs_readiness(df, total):
    sep("9.1  PRS Readiness Flags (Feature9)", char=".")

    for col, label in [
        ("feature9_prs_has_snpid_count",    "Has SNPID count"),
        ("feature9_prs_has_valid_p_count",  "Has valid P count"),
        ("feature9_prs_has_effect_count",   "Has effect (BETA/OR) count"),
        ("feature9_prs_has_alleles_count",  "Has alleles count"),
        ("feature9_prs_clump_ready_count",  "Clump-ready count"),
        ("feature9_prs_score_ready_count",  "Score-ready count"),
        ("feature9_prs_full_ready_count",   "Fully PRS-ready count"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    for col, label in [
        ("feature9_prs_has_snpid_pct",    "Has SNPID %"),
        ("feature9_prs_has_valid_p_pct",  "Has valid P %"),
        ("feature9_prs_has_effect_pct",   "Has effect %"),
        ("feature9_prs_has_alleles_pct",  "Has alleles %"),
        ("feature9_prs_clump_ready_pct",  "Clump-ready %"),
        ("feature9_prs_score_ready_pct",  "Score-ready %"),
        ("feature9_prs_full_ready_pct",   "Fully PRS-ready %"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    # Full-ready bucket breakdown
    col = "feature9_prs_full_ready_count"
    if col in df.columns:
        vals = to_numeric(df[col])
        if not vals.empty:
            buckets = [
                ("= 0           (no ready variants)", 0,       1),
                ("1 – 100K",                          1,       100_000),
                ("100K – 500K",                       100_000, 500_000),
                ("500K – 2M",                         500_000, 2_000_000),
                ("> 2M",                              2_000_000, float("inf")),
            ]
            print(f"  Full PRS-ready count buckets  (n = {len(vals)} / {total})")
            for label, lo, hi in buckets:
                cnt = ((vals >= lo) & (vals < hi)).sum()
                print(f"    {label:<38s}  {cnt:>4d} files  ({pct(cnt, total):.1f}%)")

    print()
    for col, label in [
        ("feature9_prs_full_ready_and_in_target_count",          "PRS-ready and in-target count"),
        ("feature9_prs_full_ready_and_in_target_pct_of_gwas",    "PRS-ready in-target % of GWAS"),
        ("feature9_prs_full_ready_and_in_target_pct_of_ready",   "PRS-ready in-target % of ready"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    for col, label in [
        ("feature9_prs_duplicate_snpid_count",               "Duplicate SNPID count"),
        ("feature9_prs_duplicate_snpid_pct_of_snpid_available","Duplicate SNPID % of SNPID-available"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)


# ---------------------------------------------------------------------------
# Section 9.2 — Effect size distributions
# ---------------------------------------------------------------------------

def section_effect_distributions(df, total):
    sep("9.2  Effect Size Distributions (Feature9)", char=".")

    print(f"\n  -- Beta --")
    for col, label in [
        ("feature9_effect_beta_available_count", "Beta available count"),
        ("feature9_effect_beta_available_pct",   "Beta available %"),
        ("feature9_effect_beta_from_or_count",   "Beta derived from OR count"),
        ("feature9_beta_mean",                   "Beta mean"),
        ("feature9_beta_sd",                     "Beta SD"),
        ("feature9_beta_min",                    "Beta min"),
        ("feature9_beta_max",                    "Beta max"),
        ("feature9_beta_median",                 "Beta median"),
        ("feature9_beta_median_abs",             "Beta median |abs|"),
        ("feature9_beta_abs_q90",                "Beta |abs| q90"),
        ("feature9_beta_abs_q95",                "Beta |abs| q95"),
        ("feature9_beta_abs_q99",                "Beta |abs| q99"),
        ("feature9_beta_skewness",               "Beta skewness"),
        ("feature9_beta_kurtosis",               "Beta kurtosis"),
        ("feature9_extreme_beta_abs_gt_1_count", "Extreme |beta| > 1 count"),
        ("feature9_extreme_beta_abs_gt_1_pct",   "Extreme |beta| > 1 %"),
        ("feature9_extreme_beta_abs_gt_2_count", "Extreme |beta| > 2 count"),
        ("feature9_extreme_beta_abs_gt_5_count", "Extreme |beta| > 5 count"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    print(f"\n  -- OR --")
    for col, label in [
        ("feature9_or_available_count",      "OR available count"),
        ("feature9_or_available_pct",        "OR available %"),
        ("feature9_or_median",               "OR median"),
        ("feature9_or_q95",                  "OR q95"),
        ("feature9_or_q99",                  "OR q99"),
        ("feature9_extreme_or_gt_10_count",  "Extreme OR > 10 count"),
        ("feature9_extreme_or_gt_10_pct",    "Extreme OR > 10 %"),
        ("feature9_extreme_or_lt_0_1_count", "Extreme OR < 0.1 count"),
        ("feature9_extreme_or_lt_0_1_pct",   "Extreme OR < 0.1 %"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    print(f"\n  -- SE --")
    for col, label in [
        ("feature9_se_available_count",          "SE available count"),
        ("feature9_se_available_pct",            "SE available %"),
        ("feature9_se_median",                   "SE median"),
        ("feature9_se_q95",                      "SE q95"),
        ("feature9_se_q99",                      "SE q99"),
        ("feature9_se_zero_or_negative_count",   "SE zero or negative count"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    print(f"\n  -- Z --")
    for col, label in [
        ("feature9_z_available_count",   "Z available count"),
        ("feature9_z_available_pct",     "Z available %"),
        ("feature9_z_abs_median",        "Z |abs| median"),
        ("feature9_z_abs_q95",           "Z |abs| q95"),
        ("feature9_z_abs_q99",           "Z |abs| q99"),
        ("feature9_z_abs_max",           "Z |abs| max"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    print(f"\n  -- P --")
    for col, label in [
        ("feature9_p_available_count",  "P available count"),
        ("feature9_p_available_pct",    "P available %"),
        ("feature9_p_valid_count",      "P valid count"),
        ("feature9_p_valid_pct",        "P valid %"),
        ("feature9_p_missing_count",    "P missing count"),
        ("feature9_p_invalid_count",    "P invalid count"),
        ("feature9_p_zero_count",       "P = 0 count"),
        ("feature9_p_min",              "P minimum"),
        ("feature9_p_median",           "P median"),
        ("feature9_neglog10p_max",      "-log10(P) max"),
        ("feature9_neglog10p_median",   "-log10(P) median"),
        ("feature9_neglog10p_q95",      "-log10(P) q95"),
        ("feature9_neglog10p_q99",      "-log10(P) q99"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    # P threshold bin counts
    print(f"\n  -- P threshold hit counts --")
    for col, label in [
        ("feature9_p_le_5e8_count",   "P <= 5e-8 count"),
        ("feature9_p_le_5e8_pct",     "P <= 5e-8 %"),
        ("feature9_p_le_1e6_count",   "P <= 1e-6 count"),
        ("feature9_p_le_1e5_count",   "P <= 1e-5 count"),
        ("feature9_p_le_1e4_count",   "P <= 1e-4 count"),
        ("feature9_p_le_1e3_count",   "P <= 1e-3 count"),
        ("feature9_p_le_1e2_count",   "P <= 0.01 count"),
        ("feature9_p_le_0_05_count",  "P <= 0.05 count"),
        ("feature9_p_le_0_1_count",   "P <= 0.10 count"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)


# ---------------------------------------------------------------------------
# Section 9.3 — Clumping results
# ---------------------------------------------------------------------------

def section_clumping(df, total):
    sep("9.3  LD Clumping Results by P-value Threshold (Feature9)", char=".")

    print(f"\n  Clumping parameters:")
    for col, label in [
        ("feature9_clump_r2",      "Clump r2 threshold"),
        ("feature9_clump_kb",      "Clump window (kb)"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    for tag, desc in CLUMP_THRESHOLDS:
        count_col = f"feature9_clump_{tag}_count"
        status_col = f"feature9_clump_{tag}_status"
        chr_col = f"feature9_clump_{tag}_chromosome_count"
        minp_col = f"feature9_clump_{tag}_min_p"
        medp_col = f"feature9_clump_{tag}_median_p"
        sp2_mean_col = f"feature9_clump_{tag}_mean_sp2_count"
        sp2_max_col = f"feature9_clump_{tag}_max_sp2_count"

        print(f"\n  ── {desc} ──")

        if status_col in df.columns:
            print_categorical(df[status_col], f"Clump status ({tag})", total)

        if count_col in df.columns:
            print_numeric_zero_nonzero(df[count_col],
                                       f"Clump count ({tag}) zero vs non-zero", total)
            print_numeric_dist(df[count_col], f"Clump count ({tag})", total)

            # Bucket breakdown
            vals = to_numeric(df[count_col])
            if not vals.empty:
                buckets = [
                    ("0 clumps",      0,    1),
                    ("1 – 10",        1,   11),
                    ("11 – 50",      11,   51),
                    ("51 – 200",     51,  201),
                    ("201 – 1000", 201, 1001),
                    ("> 1000",     1001, float("inf")),
                ]
                print(f"    Clump count buckets  (n = {len(vals)} / {total})")
                for label, lo, hi in buckets:
                    cnt = ((vals >= lo) & (vals < hi)).sum()
                    print(f"      {label:<18s}  {cnt:>4d} files  ({pct(cnt, total):.1f}%)")

        for col, label in [
            (chr_col,      f"Chromosomes with clumps ({tag})"),
            (minp_col,     f"Min P in clumps ({tag})"),
            (medp_col,     f"Median P in clumps ({tag})"),
            (sp2_mean_col, f"Mean SP2 count ({tag})"),
            (sp2_max_col,  f"Max SP2 count ({tag})"),
        ]:
            if col in df.columns:
                print_numeric_dist(df[col], label, total)


# ---------------------------------------------------------------------------
# Section 9.4 — Pruning results
# ---------------------------------------------------------------------------

def section_pruning(df, total):
    sep("9.4  LD Pruning Results (Feature9)", char=".")

    for col, label in [
        ("feature9_prune_window_kb",    "Prune window (kb)"),
        ("feature9_prune_step_size",    "Prune step size"),
        ("feature9_prune_r2",           "Prune r2 threshold"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    if "feature9_prune_status" in df.columns:
        print_categorical(df["feature9_prune_status"], "Prune status", total)

    for col, label in [
        ("feature9_prune_in_count",          "Prune-in variant count"),
        ("feature9_prune_out_count",         "Prune-out variant count"),
        ("feature9_prune_total_input_count", "Prune total input count"),
        ("feature9_prune_in_pct_of_input",   "Prune-in % of input"),
        ("feature9_prune_out_pct_of_input",  "Prune-out % of input"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    # Prune-in bucket breakdown
    col = "feature9_prune_in_count"
    if col in df.columns:
        vals = to_numeric(df[col])
        if not vals.empty:
            buckets = [
                ("= 0",           0,       1),
                ("1 – 100K",      1,       100_000),
                ("100K – 300K",   100_000, 300_000),
                ("300K – 500K",   300_000, 500_000),
                ("> 500K",        500_000, float("inf")),
            ]
            print(f"  Prune-in count buckets  (n = {len(vals)} / {total})")
            for label, lo, hi in buckets:
                cnt = ((vals >= lo) & (vals < hi)).sum()
                print(f"    {label:<18s}  {cnt:>4d} files  ({pct(cnt, total):.1f}%)")


# ---------------------------------------------------------------------------
# Section 9.5 — Target Fisher concordance
# ---------------------------------------------------------------------------

def section_fisher_concordance(df, total):
    sep("9.5  Target Fisher GWAS Concordance (Feature9)", char=".")

    if "feature9_target_fisher_available" in df.columns:
        print_boolean(df["feature9_target_fisher_available"],
                      "Fisher association available", total)

    if "feature9_target_assoc_method" in df.columns:
        print_categorical(df["feature9_target_assoc_method"],
                          "Association method", total)

    for col, label in [
        ("feature9_target_fisher_input_snp_count",          "Fisher input SNP count"),
        ("feature9_target_fisher_assoc_variant_count",      "Fisher assoc variant count"),
        ("feature9_target_fisher_common_snp_count",         "Fisher common SNP count"),
        ("feature9_target_fisher_allele_compatible_count",  "Fisher allele-compatible count"),
        ("feature9_target_fisher_allele_compatible_pct",    "Fisher allele-compatible %"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    print(f"\n  -- Allele matching in Fisher --")
    for col, label in [
        ("feature9_target_fisher_allele_direct_match_count",       "Direct match count"),
        ("feature9_target_fisher_allele_flip_match_count",         "Flip match count"),
        ("feature9_target_fisher_allele_strand_match_count",       "Strand match count"),
        ("feature9_target_fisher_allele_strand_flip_match_count",  "Strand-flip match count"),
        ("feature9_target_fisher_palindromic_ambiguous_count",     "Palindromic ambiguous count"),
        ("feature9_target_fisher_allele_mismatch_count",           "Allele mismatch count"),
        ("feature9_target_fisher_allele_missing_count",            "Allele missing count"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    print(f"\n  -- Beta and P concordance --")
    for col, label in [
        ("feature9_target_fisher_beta_corr_raw_all_common",       "Beta correlation (raw, all common)"),
        ("feature9_target_fisher_beta_corr_signed_compatible",    "Beta correlation (signed, compatible)"),
        ("feature9_target_fisher_abs_beta_corr_compatible",       "Abs beta correlation (compatible)"),
        ("feature9_target_fisher_beta_abs_diff_mean_compatible",  "Beta abs diff mean (compatible)"),
        ("feature9_target_fisher_beta_abs_diff_median_compatible","Beta abs diff median (compatible)"),
        ("feature9_target_fisher_pvalue_corr_all_common",         "P-value correlation (all common)"),
        ("feature9_target_fisher_neglog10p_corr_all_common",      "-log10P correlation (all common)"),
        ("feature9_target_fisher_pvalue_corr_compatible",         "P-value correlation (compatible)"),
        ("feature9_target_fisher_neglog10p_corr_compatible",      "-log10P correlation (compatible)"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)


# ---------------------------------------------------------------------------
# Section 9.6 — Per-phenotype clumping summary table
# ---------------------------------------------------------------------------

def section_per_phenotype_clumping(df, total):
    sep("9.6  Per-Phenotype Clumping Summary Table (Feature9)", char=".")

    pheno_col = get_pheno_col(df)
    if pheno_col is None:
        print("  No phenotype column found — skipping per-phenotype table.")
        return

    phenotypes = sorted(non_missing(df[pheno_col]).unique())

    # ── Summary table: one row per phenotype, one column per threshold ──
    print()
    hdr_fixed = f"  {'Phenotype':<38s} {'Files':>5}"
    hdr_thresholds = "".join(
        f"  {tag:>8s}" for tag, _ in CLUMP_THRESHOLDS
    )
    print(hdr_fixed + hdr_thresholds + "  {'prune_in':>9}")
    divider = "  " + "-" * (38 + 5 + 10 * len(CLUMP_THRESHOLDS) + 11)
    print(divider)
    print(f"  {'':38s} {'':5}  " +
          "  ".join(f"{'median':>8}" for _ in CLUMP_THRESHOLDS) +
          f"  {'median':>9}")
    print(divider)

    for pheno in phenotypes:
        mask = non_missing(df[pheno_col]) == pheno
        grp = df[mask]
        n = len(grp)

        clump_medians = []
        for tag, _ in CLUMP_THRESHOLDS:
            col = f"feature9_clump_{tag}_count"
            vals = to_numeric(grp[col]) if col in grp.columns else pd.Series(dtype=float)
            clump_medians.append(
                f"{vals.median():>8.0f}" if not vals.empty else f"{'—':>8}"
            )

        prune_col = "feature9_prune_in_count"
        prune_vals = to_numeric(grp[prune_col]) if prune_col in grp.columns else pd.Series(dtype=float)
        prune_med = f"{prune_vals.median():>9.0f}" if not prune_vals.empty else f"{'—':>9}"

        print(f"  {pheno:<38s} {n:>5d}  " +
              "  ".join(clump_medians) +
              f"  {prune_med}")

    print()

    # ── Detailed per-phenotype blocks ──
    sep("9.6b  Detailed Per-Phenotype Clumping Blocks", char=".")

    for pheno in phenotypes:
        mask = non_missing(df[pheno_col]) == pheno
        grp = df[mask]
        n = len(grp)

        print(f"\n  Phenotype : {pheno}  ({n} files)")
        print(f"  {'Threshold':<30s} {'n_success':>9} {'n_zero':>7} "
              f"{'median':>8} {'p95':>8} {'max':>8} {'med_chr':>8} {'med_sp2':>8}")
        print(f"  {'─'*30} {'─'*9} {'─'*7} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

        for tag, desc in CLUMP_THRESHOLDS:
            count_col  = f"feature9_clump_{tag}_count"
            status_col = f"feature9_clump_{tag}_status"
            chr_col    = f"feature9_clump_{tag}_chromosome_count"
            sp2_col    = f"feature9_clump_{tag}_mean_sp2_count"

            counts = to_numeric(grp[count_col]) if count_col in grp.columns \
                else pd.Series(dtype=float)
            chrs   = to_numeric(grp[chr_col])   if chr_col   in grp.columns \
                else pd.Series(dtype=float)
            sp2s   = to_numeric(grp[sp2_col])   if sp2_col   in grp.columns \
                else pd.Series(dtype=float)

            if status_col in grp.columns:
                n_success = (non_missing(grp[status_col]).str.lower() == "success").sum()
            else:
                n_success = 0

            n_zero   = int((counts == 0).sum()) if not counts.empty else 0
            med      = f"{counts.median():>8.1f}" if not counts.empty else f"{'—':>8}"
            p95      = f"{counts.quantile(0.95):>8.1f}" if not counts.empty else f"{'—':>8}"
            mx       = f"{counts.max():>8.1f}" if not counts.empty else f"{'—':>8}"
            med_chr  = f"{chrs.median():>8.1f}" if not chrs.empty else f"{'—':>8}"
            med_sp2  = f"{sp2s.median():>8.2f}" if not sp2s.empty else f"{'—':>8}"

            print(f"  {desc:<30s} {n_success:>9d} {n_zero:>7d} "
                  f"{med} {p95} {mx} {med_chr} {med_sp2}")

        # Pruning for this phenotype
        prune_col = "feature9_prune_in_count"
        prune_pct_col = "feature9_prune_in_pct_of_input"
        pvals = to_numeric(grp[prune_col]) if prune_col in grp.columns \
            else pd.Series(dtype=float)
        ppct  = to_numeric(grp[prune_pct_pct_col := prune_pct_col]) \
            if prune_pct_col in grp.columns else pd.Series(dtype=float)

        if not pvals.empty:
            print(f"\n  Pruning: median in-set = {pvals.median():.0f}  "
                  f"(range {pvals.min():.0f}–{pvals.max():.0f})  "
                  f"median % of input = "
                  f"{ppct.median():.1f}%" if not ppct.empty else "")


# ---------------------------------------------------------------------------
# Section 9.7 — PLINK clump input diagnostics
# ---------------------------------------------------------------------------

def section_clump_input(df, total):
    sep("9.7  PLINK Clump Input Diagnostics (Feature9)", char=".")

    for col, label in [
        ("feature9_plink_clump_input_variant_count",      "Clump input variant count"),
        ("feature9_plink_clump_input_pct_of_final_gwas",  "Clump input % of final GWAS"),
        ("feature9_plink_extract_variant_count",          "Extract variant count"),
        ("feature9_plink_extract_pct_of_final_gwas",      "Extract % of final GWAS"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)
            print_numeric_zero_nonzero(df[col], f"{label} (zero vs non-zero)", total)


# ---------------------------------------------------------------------------
# Main synthesis
# ---------------------------------------------------------------------------

def run_synthesis(df):
    total = len(df)

    print("=" * 90)
    print("GWASRANKER FEATURE9")
    print("PRS READINESS, CLUMPING, PRUNING AND TARGET CONCORDANCE SYNTHESIS")
    print(f"Total candidate GWAS files: {total}")
    print("=" * 90)

    sep("FEATURE 9  .  PRS Readiness and Clumping")
    section_prs_readiness(df, total)
    section_effect_distributions(df, total)
    section_clumping(df, total)
    section_pruning(df, total)
    section_fisher_concordance(df, total)
    section_clump_input(df, total)
    section_per_phenotype_clumping(df, total)

    sep()
    print("END OF PRS READINESS, CLUMPING AND PRUNING SYNTHESIS REPORT")
    sep()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Print PRS readiness, clumping and pruning synthesis (Feature9). Saves nothing."
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