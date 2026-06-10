#!/usr/bin/env python3
"""
Synthesis analysis for PRS performance outcomes (PLINK C+T and PRSice-2).
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


def sep(title="", width=110, char="-"):
    if title:
        side = (width - len(title) - 2) // 2
        print(f"\n{'─' * side} {title} {'─' * (width - side - len(title) - 2)}")
    else:
        print("─" * width)


def get_pheno_col(df):
    for c in ["phenotype", "feature1_phenotype", "feature9_phenotype"]:
        if c in df.columns:
            return c
    return None


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


# ---------------------------------------------------------------------------
# Helper: summarise one performance series with buckets
# ---------------------------------------------------------------------------

def summarise_performance(series, label, total, metric="AUC"):
    vals = to_numeric(series)
    if vals.empty:
        print(f"  {label}: no numeric values")
        return
    q = vals.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    print(f"  {label}  (n = {len(vals)} / {total})")
    print(f"    min={vals.min():.4f}  p5={q[0.05]:.4f}  p25={q[0.25]:.4f}"
          f"  median={q[0.50]:.4f}  p75={q[0.75]:.4f}  p95={q[0.95]:.4f}"
          f"  max={vals.max():.4f}  mean={vals.mean():.4f}  sd={vals.std():.4f}")

    if metric == "AUC":
        buckets = [
            ("< 0.50  (worse than chance)",   float("-inf"), 0.50),
            ("0.50 – 0.55  (near-null)",       0.50,          0.55),
            ("0.55 – 0.60  (low)",             0.55,          0.60),
            ("0.60 – 0.70  (moderate)",        0.60,          0.70),
            ("0.70 – 0.80  (good)",            0.70,          0.80),
            (">= 0.80      (excellent)",        0.80,          float("inf")),
        ]
    else:
        buckets = [
            ("< 0.00  (negative R2)",          float("-inf"), 0.00),
            ("0.00 – 0.01  (near-null)",        0.00,          0.01),
            ("0.01 – 0.05  (low)",              0.01,          0.05),
            ("0.05 – 0.10  (moderate)",         0.05,          0.10),
            (">= 0.10      (good)",             0.10,          float("inf")),
        ]

    for blabel, lo, hi in buckets:
        if lo == float("-inf"):
            cnt = (vals < hi).sum()
        elif hi == float("inf"):
            cnt = (vals >= lo).sum()
        else:
            cnt = ((vals >= lo) & (vals < hi)).sum()
        if cnt > 0:
            print(f"    {blabel:<40s}  {cnt:>4d} files  ({pct(cnt, total):.1f}%)")


# ---------------------------------------------------------------------------
# Section: overall run status
# ---------------------------------------------------------------------------

def section_run_status(df, total):
    sep("P.1  PRS Run Status Overview", char=".")

    for col, label in [
        ("plink_best_result_status",   "PLINK best result status"),
        ("prsice2_best_result_status", "PRSice-2 best result status"),
        ("plink_run_summary_status",   "PLINK run summary status"),
        ("prsice2_run_summary_status", "PRSice-2 run summary status"),
        ("prs_performance_status",     "Overall PRS performance status"),
    ]:
        if col in df.columns:
            print_categorical(df[col], label, total)

    for col, label in [
        ("plink_n_folds",   "PLINK number of CV folds"),
        ("prsice2_n_folds", "PRSice-2 number of CV folds"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    for col, label in [
        ("plink_performance_metric",   "PLINK performance metric"),
        ("prsice2_performance_metric", "PRSice-2 performance metric"),
        ("prsice2_prsice_model",       "PRSice-2 model type"),
        ("plink_prsice_model",         "PLINK model type"),
    ]:
        if col in df.columns:
            print_categorical(df[col], label, total)


# ---------------------------------------------------------------------------
# Section: PLINK best P-value threshold
# ---------------------------------------------------------------------------

def section_plink_threshold(df, total):
    sep("P.2  PLINK Best P-value Threshold", char=".")

    if "plink_pvalue_label" in df.columns:
        print_categorical(df["plink_pvalue_label"], "PLINK best P-value threshold label", total)

    if "plink_pvalue" in df.columns:
        print_numeric_dist(df["plink_pvalue"], "PLINK best P-value threshold (numeric)", total)

    if "prsice2_pvalue_label" in df.columns:
        print_categorical(df["prsice2_pvalue_label"], "PRSice-2 best P-value threshold label", total)

    if "prsice2_pvalue" in df.columns:
        print_numeric_dist(df["prsice2_pvalue"], "PRSice-2 best P-value threshold (numeric)", total)


# ---------------------------------------------------------------------------
# Section: PLINK performance
# ---------------------------------------------------------------------------

def section_plink_performance(df, total):
    sep("P.3  PLINK C+T PRS Performance", char=".")

    # Detect metric
    metric = "AUC"
    if "plink_performance_metric" in df.columns:
        vals = non_missing(df["plink_performance_metric"])
        if not vals.empty and "r2" in vals.iloc[0].lower():
            metric = "R2"

    print(f"\n  -- Pure PRS model (PRS score alone) --")
    for col, label in [
        ("plink_Train_pure_prs_mean", "Train pure PRS"),
        ("plink_Test_pure_prs_mean",  "Test pure PRS"),
        ("plink_pure_train_test_sum", "Pure PRS train+test sum"),
    ]:
        if col in df.columns:
            summarise_performance(df[col], label, total, metric)

    print(f"\n  -- Null model (covariates only, no PRS) --")
    for col, label in [
        ("plink_Train_null_model_mean", "Train null model"),
        ("plink_Test_null_model_mean",  "Test null model"),
        ("plink_null_train_test_sum",   "Null model train+test sum"),
    ]:
        if col in df.columns:
            summarise_performance(df[col], label, total, metric)

    print(f"\n  -- Best model (PRS + covariates) --")
    for col, label in [
        ("plink_Train_best_model_mean", "Train best model"),
        ("plink_Test_best_model_mean",  "Test best model"),
        ("plink_best_train_test_sum",   "Best model train+test sum"),
    ]:
        if col in df.columns:
            summarise_performance(df[col], label, total, metric)

    print(f"\n  -- Generalisation and incremental gain --")
    for col, label in [
        ("plink_generalisation_gap",                    "Generalisation gap (pure PRS train-test)"),
        ("plink_best_model_generalisation_gap",         "Best model generalisation gap"),
        ("plink_test_incremental_auc_best_minus_null",  "Test incremental AUC: best minus null"),
        ("plink_test_incremental_auc_best_minus_pure_prs","Test incremental AUC: best minus pure PRS"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    # Incremental gain buckets
    col = "plink_test_incremental_auc_best_minus_null"
    if col in df.columns:
        vals = to_numeric(df[col])
        if not vals.empty:
            buckets = [
                ("<= 0.00  (no gain)",          float("-inf"), 0.001),
                ("0.001 – 0.01  (marginal)",    0.001,         0.01),
                ("0.01  – 0.05  (modest)",      0.01,          0.05),
                ("0.05  – 0.10  (moderate)",    0.05,          0.10),
                (">= 0.10       (substantial)", 0.10,          float("inf")),
            ]
            print(f"  Incremental AUC (best minus null) buckets  (n = {len(vals)} / {total})")
            for blabel, lo, hi in buckets:
                if lo == float("-inf"):
                    cnt = (vals < hi).sum()
                elif hi == float("inf"):
                    cnt = (vals >= lo).sum()
                else:
                    cnt = ((vals >= lo) & (vals < hi)).sum()
                print(f"    {blabel:<40s}  {cnt:>4d} files  ({pct(cnt, total):.1f}%)")


# ---------------------------------------------------------------------------
# Section: PRSice-2 performance
# ---------------------------------------------------------------------------

def section_prsice2_performance(df, total):
    sep("P.4  PRSice-2 PRS Performance", char=".")

    metric = "AUC"
    if "prsice2_performance_metric" in df.columns:
        vals = non_missing(df["prsice2_performance_metric"])
        if not vals.empty and "r2" in vals.iloc[0].lower():
            metric = "R2"

    print(f"\n  -- Pure PRS model --")
    for col, label in [
        ("prsice2_Train_pure_prs_mean", "Train pure PRS"),
        ("prsice2_Test_pure_prs_mean",  "Test pure PRS"),
        ("prsice2_pure_train_test_sum", "Pure PRS train+test sum"),
    ]:
        if col in df.columns:
            summarise_performance(df[col], label, total, metric)

    print(f"\n  -- Null model --")
    for col, label in [
        ("prsice2_Train_null_model_mean", "Train null model"),
        ("prsice2_Test_null_model_mean",  "Test null model"),
        ("prsice2_null_train_test_sum",   "Null model train+test sum"),
    ]:
        if col in df.columns:
            summarise_performance(df[col], label, total, metric)

    print(f"\n  -- Best model --")
    for col, label in [
        ("prsice2_Train_best_model_mean", "Train best model"),
        ("prsice2_Test_best_model_mean",  "Test best model"),
        ("prsice2_best_train_test_sum",   "Best model train+test sum"),
    ]:
        if col in df.columns:
            summarise_performance(df[col], label, total, metric)

    print(f"\n  -- Generalisation and incremental gain --")
    for col, label in [
        ("prsice2_generalisation_gap",                     "Generalisation gap (pure PRS train-test)"),
        ("prsice2_best_model_generalisation_gap",          "Best model generalisation gap"),
        ("prsice2_test_incremental_auc_best_minus_null",   "Test incremental AUC: best minus null"),
        ("prsice2_test_incremental_auc_best_minus_pure_prs","Test incremental AUC: best minus pure PRS"),
    ]:
        if col in df.columns:
            print_numeric_dist(df[col], label, total)

    col = "prsice2_test_incremental_auc_best_minus_null"
    if col in df.columns:
        vals = to_numeric(df[col])
        if not vals.empty:
            buckets = [
                ("<= 0.00  (no gain)",          float("-inf"), 0.001),
                ("0.001 – 0.01  (marginal)",    0.001,         0.01),
                ("0.01  – 0.05  (modest)",      0.01,          0.05),
                ("0.05  – 0.10  (moderate)",    0.05,          0.10),
                (">= 0.10       (substantial)", 0.10,          float("inf")),
            ]
            print(f"  Incremental AUC (best minus null) buckets  (n = {len(vals)} / {total})")
            for blabel, lo, hi in buckets:
                if lo == float("-inf"):
                    cnt = (vals < hi).sum()
                elif hi == float("inf"):
                    cnt = (vals >= lo).sum()
                else:
                    cnt = ((vals >= lo) & (vals < hi)).sum()
                print(f"    {blabel:<40s}  {cnt:>4d} files  ({pct(cnt, total):.1f}%)")


# ---------------------------------------------------------------------------
# Section: PLINK vs PRSice-2 comparison
# ---------------------------------------------------------------------------

def section_tool_comparison(df, total):
    sep("P.5  PLINK vs PRSice-2 Test Performance Comparison", char=".")

    p_test = to_numeric(df["plink_Test_best_model_mean"]) \
        if "plink_Test_best_model_mean" in df.columns else pd.Series(dtype=float)
    r_test = to_numeric(df["prsice2_Test_best_model_mean"]) \
        if "prsice2_Test_best_model_mean" in df.columns else pd.Series(dtype=float)

    common_idx = p_test.index.intersection(r_test.index)
    if len(common_idx) > 0:
        p_vals = p_test.loc[common_idx]
        r_vals = r_test.loc[common_idx]
        diff = p_vals - r_vals
        print(f"  PLINK vs PRSice-2 test best model (n paired = {len(common_idx)})")
        print(f"    PLINK  median={p_vals.median():.4f}  mean={p_vals.mean():.4f}"
              f"  min={p_vals.min():.4f}  max={p_vals.max():.4f}")
        print(f"    PRSice median={r_vals.median():.4f}  mean={r_vals.mean():.4f}"
              f"  min={r_vals.min():.4f}  max={r_vals.max():.4f}")
        print(f"    Diff (PLINK-PRSice)  median={diff.median():.4f}"
              f"  mean={diff.mean():.4f}  min={diff.min():.4f}  max={diff.max():.4f}")
        plink_better = (diff > 0).sum()
        prsice_better = (diff < 0).sum()
        equal = (diff == 0).sum()
        print(f"    PLINK better: {plink_better}  PRSice better: {prsice_better}"
              f"  Equal: {equal}")


# ---------------------------------------------------------------------------
# Section: per-phenotype performance table
# ---------------------------------------------------------------------------

def section_per_phenotype_performance(df, total):
    sep("P.6  Per-Phenotype PRS Performance Summary", char=".")

    pheno_col = get_pheno_col(df)
    if pheno_col is None:
        print("  No phenotype column found.")
        return

    phenotypes = sorted(non_missing(df[pheno_col]).unique())

    # ── Determine metric label ──
    metric_label = "AUC"
    if "plink_performance_metric" in df.columns:
        vals = non_missing(df["plink_performance_metric"])
        if not vals.empty and "r2" in vals.iloc[0].lower():
            metric_label = "R2"

    # ── Summary table ──
    print()
    hdr = (f"  {'Phenotype':<38s} {'n':>4}  "
           f"{'PLINK test best':^20s}  "
           f"{'PLINK test pure':^20s}  "
           f"{'PLINK test null':^20s}  "
           f"{'PRSice test best':^20s}  "
           f"{'PLINK incr':^12s}  "
           f"{'PRSice incr':^12s}")
    print(hdr)
    print(f"  {'':38s} {'':4}  "
          f"{'min / med / max':^20s}  "
          f"{'min / med / max':^20s}  "
          f"{'min / med / max':^20s}  "
          f"{'min / med / max':^20s}  "
          f"{'med':^12s}  "
          f"{'med':^12s}")
    print("  " + "-" * (len(hdr) - 2))

    for pheno in phenotypes:
        mask = non_missing(df[pheno_col]) == pheno
        grp = df[mask]
        n = len(grp)

        def mmm(col):
            vals = to_numeric(grp[col]) if col in grp.columns else pd.Series(dtype=float)
            if vals.empty:
                return f"{'—':^20s}"
            return f"{vals.min():.3f}/{vals.median():.3f}/{vals.max():.3f}".center(20)

        def med(col):
            vals = to_numeric(grp[col]) if col in grp.columns else pd.Series(dtype=float)
            if vals.empty:
                return f"{'—':^12s}"
            return f"{vals.median():.4f}".center(12)

        print(f"  {pheno:<38s} {n:>4d}  "
              f"{mmm('plink_Test_best_model_mean')}  "
              f"{mmm('plink_Test_pure_prs_mean')}  "
              f"{mmm('plink_Test_null_model_mean')}  "
              f"{mmm('prsice2_Test_best_model_mean')}  "
              f"{med('plink_test_incremental_auc_best_minus_null')}  "
              f"{med('prsice2_test_incremental_auc_best_minus_null')}")

    # ── Detailed per-phenotype blocks ──
    sep("P.6b  Detailed Per-Phenotype Performance Blocks", char=".")

    for pheno in phenotypes:
        mask = non_missing(df[pheno_col]) == pheno
        grp = df[mask]
        n = len(grp)

        print(f"\n  Phenotype : {pheno}  ({n} files)")

        for tool, prefix in [("PLINK C+T", "plink"), ("PRSice-2", "prsice2")]:
            print(f"\n    [{tool}]")
            print(f"    {'Metric':<45s} {'n':>4} {'min':>8} {'p25':>8} "
                  f"{'median':>8} {'p75':>8} {'max':>8} {'mean':>8}")
            print(f"    {'─'*45} {'─'*4} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

            metrics = [
                (f"{prefix}_Test_pure_prs_mean",                   "Test pure PRS"),
                (f"{prefix}_Train_pure_prs_mean",                  "Train pure PRS"),
                (f"{prefix}_Test_null_model_mean",                  "Test null model"),
                (f"{prefix}_Train_null_model_mean",                 "Train null model"),
                (f"{prefix}_Test_best_model_mean",                  "Test best model"),
                (f"{prefix}_Train_best_model_mean",                 "Train best model"),
                (f"{prefix}_generalisation_gap",                    "Generalisation gap (pure)"),
                (f"{prefix}_best_model_generalisation_gap",         "Generalisation gap (best)"),
                (f"{prefix}_test_incremental_auc_best_minus_null",  "Incremental: best minus null"),
                (f"{prefix}_test_incremental_auc_best_minus_pure_prs","Incremental: best minus pure"),
            ]

            for col, label in metrics:
                vals = to_numeric(grp[col]) if col in grp.columns \
                    else pd.Series(dtype=float)
                if vals.empty:
                    print(f"    {label:<45s} {'—':>4} {'—':>8} {'—':>8} "
                          f"{'—':>8} {'—':>8} {'—':>8} {'—':>8}")
                    continue
                q = vals.quantile([0.25, 0.75])
                print(f"    {label:<45s} {len(vals):>4d} "
                      f"{vals.min():>8.4f} {q[0.25]:>8.4f} "
                      f"{vals.median():>8.4f} {q[0.75]:>8.4f} "
                      f"{vals.max():>8.4f} {vals.mean():>8.4f}")

            # Best P-value threshold breakdown
            pval_col = f"{prefix}_pvalue_label"
            if pval_col in grp.columns:
                pval_vals = non_missing(grp[pval_col])
                if not pval_vals.empty:
                    counts = pval_vals.value_counts()
                    print(f"\n    Best P-value threshold distribution ({tool}):")
                    for val, cnt in counts.items():
                        print(f"      {val:<30s}  {cnt:>4d} files  ({pct(cnt, n):.1f}%)")


# ---------------------------------------------------------------------------
# Section: generalisation gap analysis
# ---------------------------------------------------------------------------

def section_generalisation(df, total):
    sep("P.7  Generalisation Gap and Overfitting Analysis", char=".")

    print(f"\n  A positive generalisation gap (train > test) indicates overfitting.")
    print(f"  A negative gap indicates the test set outperformed training (unusual).")
    print()

    for tool, prefix in [("PLINK C+T", "plink"), ("PRSice-2", "prsice2")]:
        print(f"  [{tool}]")
        for col, label in [
            (f"{prefix}_generalisation_gap",             "Pure PRS generalisation gap (train-test)"),
            (f"{prefix}_best_model_generalisation_gap",  "Best model generalisation gap (train-test)"),
        ]:
            if col not in df.columns:
                continue
            vals = to_numeric(df[col])
            if vals.empty:
                print(f"  {label}: no numeric values")
                continue
            overfit = (vals > 0.05).sum()
            ok      = ((vals >= -0.05) & (vals <= 0.05)).sum()
            underfit = (vals < -0.05).sum()
            print(f"  {label}  (n = {len(vals)} / {total})")
            print(f"    min={vals.min():.4f}  median={vals.median():.4f}"
                  f"  mean={vals.mean():.4f}  max={vals.max():.4f}"
                  f"  sd={vals.std():.4f}")
            print(f"    Gap > 0.05 (overfit):     {overfit:>4d} files  ({pct(overfit, total):.1f}%)")
            print(f"    Gap -0.05 to 0.05 (OK):   {ok:>4d} files  ({pct(ok, total):.1f}%)")
            print(f"    Gap < -0.05 (unusual):    {underfit:>4d} files  ({pct(underfit, total):.1f}%)")
        print()


# ---------------------------------------------------------------------------
# Main synthesis
# ---------------------------------------------------------------------------

def run_synthesis(df):
    total = len(df)

    print("=" * 110)
    print("GWASRANKER  —  PRS PERFORMANCE OUTCOMES SYNTHESIS")
    print("PLINK C+T AND PRSice-2 TRAIN / TEST / PURE / NULL / BEST MODEL")
    print(f"Total candidate GWAS files: {total}")
    print("=" * 110)

    sep("PRS PERFORMANCE  .  PLINK C+T and PRSice-2")
    section_run_status(df, total)
    section_plink_threshold(df, total)
    section_plink_performance(df, total)
    section_prsice2_performance(df, total)
    section_tool_comparison(df, total)
    section_generalisation(df, total)
    section_per_phenotype_performance(df, total)

    sep()
    print("END OF PRS PERFORMANCE OUTCOMES SYNTHESIS REPORT")
    sep()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Print PRS performance synthesis (PLINK + PRSice-2). Saves nothing."
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