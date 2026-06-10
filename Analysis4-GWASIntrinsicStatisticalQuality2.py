#!/usr/bin/env python3
"""
Per-phenotype heritability and GWAS signal variation analysis.
Prints a concise summary to stdout. Saves nothing.
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


def sep(title="", width=110, char="─"):
    if title:
        side = (width - len(title) - 2) // 2
        print(f"\n{'─' * side} {title} {'─' * (width - side - len(title) - 2)}")
    else:
        print(char * width)


# ---------------------------------------------------------------------------
# Per-phenotype summary
# ---------------------------------------------------------------------------

def summarise_phenotype(pheno_name, grp, total_files):
    """
    For a single phenotype group, print a concise one-block summary
    covering: number of GWAS files, heritability, inflation,
    signal strength, and sample size.
    """
    n = len(grp)

    # ── Heritability ──
    h2       = to_numeric(grp.get("feature6_ldsc_h2",        pd.Series(dtype=float)))
    n_times  = to_numeric(grp.get("feature6_n_times_h2",     pd.Series(dtype=float)))
    intercept= to_numeric(grp.get("feature6_ldsc_intercept", pd.Series(dtype=float)))
    ratio    = to_numeric(grp.get("feature6_ldsc_ratio",     pd.Series(dtype=float)))

    # ── Inflation ──
    lambda_gc= to_numeric(grp.get("feature6_lambda_gc",      pd.Series(dtype=float)))
    mean_chi2= to_numeric(grp.get("feature6_mean_chi2",      pd.Series(dtype=float)))

    # ── Signal ──
    gw_count = to_numeric(grp.get("feature4_genome_wide_significant_count", pd.Series(dtype=float)))
    loci     = to_numeric(grp.get("feature6_n_genome_wide_significant_window_loci", pd.Series(dtype=float)))
    min_p    = to_numeric(grp.get("feature4_min_p",          pd.Series(dtype=float)))

    # ── Sample size ──
    median_n = to_numeric(grp.get("feature4_median_N",       pd.Series(dtype=float)))

    # ── Variant quality ──
    dup_id   = to_numeric(grp.get("feature4_duplicated_variant_id_percentage", pd.Series(dtype=float)))
    missing_id = to_numeric(grp.get("feature4_missing_variant_id_percentage",  pd.Series(dtype=float)))

    def fmt(series, fmt_str=".3g"):
        if series.empty:
            return "—"
        if len(series) == 1:
            return f"{series.iloc[0]:{fmt_str}}"
        return f"{series.min():{fmt_str}} – {series.max():{fmt_str}}  (median {series.median():{fmt_str}})"

    def fmt1(val, fmt_str=".3g"):
        if pd.isna(val):
            return "—"
        return f"{val:{fmt_str}}"

    print(f"\n  Phenotype : {pheno_name}")
    print(f"  GWAS files: {n}  |  "
          f"h2 succeeded: {len(h2)}/{n}  |  "
          f"Median N: {fmt1(median_n.median() if not median_n.empty else float('nan'), '.5g')}")

    # Heritability block
    print(f"  ── Heritability ──────────────────────────────────────────────────────")
    print(f"    h2 range        : {fmt(h2)}")
    print(f"    N×h2 range      : {fmt(n_times, '.5g')}")
    print(f"    Intercept range : {fmt(intercept)}")
    print(f"    Ratio range     : {fmt(ratio)}")

    # h2 bucket counts
    if not h2.empty:
        neg    = (h2 < 0).sum()
        low    = ((h2 >= 0) & (h2 < 0.02)).sum()
        mid    = ((h2 >= 0.02) & (h2 < 0.10)).sum()
        mod    = ((h2 >= 0.10) & (h2 < 0.30)).sum()
        high   = (h2 >= 0.30).sum()
        parts  = []
        if neg:  parts.append(f"negative={neg}")
        if low:  parts.append(f"near-zero={low}")
        if mid:  parts.append(f"low={mid}")
        if mod:  parts.append(f"moderate={mod}")
        if high: parts.append(f"high={high}")
        print(f"    h2 buckets      : {', '.join(parts) if parts else '—'}")

    # Inflation block
    print(f"  ── Inflation ─────────────────────────────────────────────────────────")
    print(f"    λ_GC range      : {fmt(lambda_gc)}")
    print(f"    Mean χ² range   : {fmt(mean_chi2)}")

    # Signal block
    print(f"  ── Association Signal ────────────────────────────────────────────────")
    print(f"    GW hits range   : {fmt(gw_count, '.4g')}")
    print(f"    Indep loci range: {fmt(loci, '.4g')}")
    if not min_p.empty:
        best_p = min_p.min()
        print(f"    Best P across files: {best_p:.3g}")

    # Variant quality
    print(f"  ── Variant Quality ───────────────────────────────────────────────────")
    print(f"    Dup ID %        : {fmt(dup_id)}")
    print(f"    Missing ID %    : {fmt(missing_id)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis(df):
    total = len(df)

    # Identify phenotype column
    pheno_col = None
    for candidate in ["phenotype", "feature1_phenotype", "feature4_phenotype"]:
        if candidate in df.columns:
            pheno_col = candidate
            break

    if pheno_col is None:
        print("ERROR: No phenotype column found.")
        return

    phenotypes = non_missing(df[pheno_col]).unique()
    phenotypes = sorted(phenotypes)

    print("=" * 110)
    print("GWASRANKER  —  PER-PHENOTYPE HERITABILITY AND GWAS VARIATION SUMMARY")
    print(f"Total files: {total}  |  Unique phenotypes: {len(phenotypes)}")
    print("=" * 110)

    # ── Overall heritability variation reminder ──
    sep("OVERALL HERITABILITY VARIATION ACROSS ALL FILES")
    h2_all = to_numeric(df.get("feature6_ldsc_h2", pd.Series(dtype=float)))
    nx_all = to_numeric(df.get("feature6_n_times_h2", pd.Series(dtype=float)))
    ic_all = to_numeric(df.get("feature6_ldsc_intercept", pd.Series(dtype=float)))

    if not h2_all.empty:
        print(f"  LDSC h2      : min={h2_all.min():.3g}  median={h2_all.median():.3g}"
              f"  mean={h2_all.mean():.3g}  max={h2_all.max():.3g}  "
              f"(n={len(h2_all)}/{total})")
    if not nx_all.empty:
        print(f"  N × h2       : min={nx_all.min():.3g}  median={nx_all.median():.3g}"
              f"  mean={nx_all.mean():.3g}  max={nx_all.max():.3g}  "
              f"(n={len(nx_all)}/{total})")
    if not ic_all.empty:
        print(f"  Intercept    : min={ic_all.min():.3g}  median={ic_all.median():.3g}"
              f"  mean={ic_all.mean():.3g}  max={ic_all.max():.3g}  "
              f"(n={len(ic_all)}/{total})")

    # ── Per-phenotype blocks ──
    sep("PER-PHENOTYPE BREAKDOWN")

    # Summary table header
    print()
    hdr = (f"  {'Phenotype':<30s} {'Files':>5} {'h2 median':>10} "
           f"{'h2 range':>22} {'N×h2 median':>13} "
           f"{'λ_GC median':>12} {'GW loci median':>15} {'Best P':>12}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for pheno in phenotypes:
        mask = non_missing(df[pheno_col]) == pheno
        grp  = df[mask]
        n    = len(grp)

        h2   = to_numeric(grp.get("feature6_ldsc_h2",        pd.Series(dtype=float)))
        nx   = to_numeric(grp.get("feature6_n_times_h2",     pd.Series(dtype=float)))
        lgc  = to_numeric(grp.get("feature6_lambda_gc",      pd.Series(dtype=float)))
        loci = to_numeric(grp.get("feature6_n_genome_wide_significant_window_loci",
                                   pd.Series(dtype=float)))
        mp   = to_numeric(grp.get("feature4_min_p",          pd.Series(dtype=float)))

        def v(s, fmt=".3g"):
            return f"{s.median():{fmt}}" if not s.empty else "—"

        def rng(s, fmt=".3g"):
            if s.empty:
                return "—"
            if s.min() == s.max():
                return f"{s.min():{fmt}}"
            return f"{s.min():{fmt}}–{s.max():{fmt}}"

        best_p = f"{mp.min():.2g}" if not mp.empty else "—"

        print(f"  {pheno:<30s} {n:>5d} {v(h2):>10} "
              f"{rng(h2):>22} {v(nx, '.4g'):>13} "
              f"{v(lgc):>12} {v(loci, '.4g'):>15} {best_p:>12}")

    # ── Detailed per-phenotype blocks ──
    sep("DETAILED PER-PHENOTYPE BLOCKS")
    for pheno in phenotypes:
        mask = non_missing(df[pheno_col]) == pheno
        grp  = df[mask]
        sep(pheno, char="·")
        summarise_phenotype(pheno, grp, total)

    sep()
    print("END OF PER-PHENOTYPE HERITABILITY VARIATION REPORT")
    sep()


def main():
    parser = argparse.ArgumentParser(
        description="Per-phenotype heritability and GWAS variation. Prints to screen. Saves nothing."
    )
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
    args = parser.parse_args()

    path = Path(args.input_file)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    print(f"Loading {path} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(df)} rows × {len(df.columns)} columns.\n")

    run_analysis(df)


if __name__ == "__main__":
    main()