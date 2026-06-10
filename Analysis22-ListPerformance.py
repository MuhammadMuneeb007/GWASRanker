#!/usr/bin/env python3
"""
Analysis22-ListPerformance.py

Print a comprehensive performance summary for a trained model.

Usage:
    python Analysis22-ListPerformance.py 1    # Dataset 1 (Feature1-6)
    python Analysis22-ListPerformance.py 2    # Dataset 2 (Feature1-8)
    python Analysis22-ListPerformance.py 3    # Dataset 3 (Feature1-9)

Prints to screen only. Saves nothing.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FEATURE_END_MAP = {1: 6, 2: 8, 3: 9}

TARGET_SHORT = {
    "target_plink_train_pure_prs":       "Train-PRS",
    "target_plink_test_pure_prs":        "Test-PRS",
    "target_plink_pure_prs_gap_quality": "Gap-Quality",
}

MODEL_ORDER = [
    "RidgeCV",
    "LassoCV",
    "ElasticNetCV",
    "RandomForest",
    "ExtraTrees",
    "GradientBoosting",
    "HistGradientBoosting",
]

WIDTH = 130


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def sep(title="", char="─"):
    if title:
        side = (WIDTH - len(title) - 2) // 2
        print(f"\n{'─' * side} {title} {'─' * (WIDTH - side - len(title) - 2)}")
    else:
        print("─" * WIDTH)


def header(title):
    print()
    print("=" * WIDTH)
    print(f"  {title}")
    print("=" * WIDTH)


def load(path, label):
    if not path.exists():
        print(f"  [MISSING] {label}: {path}")
        return None
    df = pd.read_csv(path, low_memory=False)
    print(f"  [OK]      {label}: {path}  ({len(df)} rows)")
    return df


def fmt(val, decimals=4):
    if pd.isna(val):
        return "      —"
    return f"{val:>{decimals + 6}.{decimals}f}"


def fmt2(val, decimals=3):
    if pd.isna(val):
        return "    —"
    return f"{val:>{decimals + 4}.{decimals}f}"


def pct(val):
    if pd.isna(val):
        return "    —"
    return f"{val * 100:>6.1f}%"


def sig(p):
    if pd.isna(p):
        return "  NA"
    if p < 0.001:
        return " ***"
    if p < 0.01:
        return "  **"
    if p < 0.05:
        return "   *"
    return "  ns"


# ---------------------------------------------------------------------------
# Section 1: Dataset overview
# ---------------------------------------------------------------------------

def print_dataset_overview(model_num, feature_end, feature_file, target_file):
    sep("1. DATASET OVERVIEW")

    features = pd.read_csv(feature_file, low_memory=False) if feature_file.exists() else None
    targets  = pd.read_csv(target_file,  low_memory=False) if target_file.exists()  else None

    if features is None or targets is None:
        print("  [ERROR] Feature or target file missing.")
        return

    id_cols = ["phenotype", "accessionId"]
    feat_cols = [c for c in features.columns if c not in id_cols]

    print(f"  Model number          : {model_num}")
    print(f"  Feature range         : Feature1 – Feature{feature_end}")
    print(f"  Total feature columns : {len(feat_cols)}")
    print(f"  Total rows            : {len(features)}")
    print(f"  Phenotypes            : {features['phenotype'].nunique()}")
    print(f"  GWAS files            : {features['accessionId'].nunique()}")

    print()
    print(f"  {'Phenotype':<45s}  {'N GWAS':>7}")
    print(f"  {'─' * 45}  {'─' * 7}")
    counts = features.groupby("phenotype").size().sort_values(ascending=False)
    for pheno, n in counts.items():
        print(f"  {pheno:<45s}  {n:>7d}")

    print()
    target_cols = [c for c in targets.columns if c.startswith("target_")]
    print(f"  {'Target':<42s}  {'Non-missing':>12}  {'Mean':>10}  {'SD':>10}  {'Min':>10}  {'Max':>10}")
    print(f"  {'─' * 42}  {'─' * 12}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 10}")
    for tc in target_cols:
        col = pd.to_numeric(targets[tc], errors="coerce")
        print(f"  {tc:<42s}  {col.notna().sum():>12d}  "
              f"{col.mean():>10.4f}  {col.std():>10.4f}  "
              f"{col.min():>10.4f}  {col.max():>10.4f}")


# ---------------------------------------------------------------------------
# Section 2: All-model LOPO comparison
# ---------------------------------------------------------------------------

def print_all_model_comparison(model_comparison):
    sep("2. ALL-MODEL LOPO PERFORMANCE COMPARISON")

    if model_comparison is None:
        print("  [MISSING] model_comparison_summary.csv")
        return

    targets = model_comparison["target"].unique() if "target" in model_comparison.columns else []

    for target in sorted(targets):
        tshort = TARGET_SHORT.get(target, target)
        print(f"\n  Target: {tshort}  ({target})")

        sub = model_comparison[model_comparison["target"] == target].copy()

        # Sort by mean spearman descending
        if "spearman_rank_r_mean" in sub.columns:
            sub = sub.sort_values("spearman_rank_r_mean", ascending=False)

        cols = [
            ("model",                              "Model",              "<20"),
            ("n_heldout_phenotypes",               "Folds",              ">6"),
            ("total_test_gwas",                    "N-test",             ">7"),
            ("spearman_rank_r_mean",               "Spearman",           ">9"),
            ("spearman_rank_r_sd",                 "Sp-SD",              ">7"),
            ("spearman_rank_r_ci95_low",           "CI-lo",              ">7"),
            ("spearman_rank_r_ci95_high",          "CI-hi",              ">7"),
            ("kendall_tau_mean",                   "Kendall",            ">8"),
            ("top1_correct_mean",                  "Top1",               ">6"),
            ("top3_overlap_fraction_mean",         "Top3",               ">6"),
            ("ndcg_at_3_mean",                     "NDCG@3",             ">7"),
            ("ndcg_at_5_mean",                     "NDCG@5",             ">7"),
            ("mean_absolute_rank_error_mean",      "MARE",               ">7"),
            ("ranking_statistically_significant",  "Sig",                ">5"),
            ("combined_permutation_p_spearman",    "Perm-P",             ">8"),
        ]

        # Header
        hdr = "  "
        for col, label, fmt_str in cols:
            w = int(fmt_str.replace("<", "").replace(">", ""))
            align = "<" if "<" in fmt_str else ">"
            hdr += f"{label:{align}{w}}  "
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))

        for _, row in sub.iterrows():
            line = "  "
            for col, label, fmt_str in cols:
                w = int(fmt_str.replace("<", "").replace(">", ""))
                align = "<" if "<" in fmt_str else ">"
                val = row.get(col, np.nan)
                if col == "model":
                    line += f"{str(val):{align}{w}}  "
                elif col in ("n_heldout_phenotypes", "total_test_gwas"):
                    v = int(val) if not pd.isna(val) else 0
                    line += f"{v:{align}{w}}  "
                elif col == "ranking_statistically_significant":
                    v = "Yes" if str(val).lower() in ("true", "1", "yes") else "No"
                    line += f"{v:{align}{w}}  "
                elif col == "combined_permutation_p_spearman":
                    v = f"{val:.3g}" if not pd.isna(val) else "—"
                    line += f"{v:{align}{w}}  "
                else:
                    v = f"{val:.4f}" if not pd.isna(val) else "—"
                    line += f"{v:{align}{w}}  "
            print(line)


# ---------------------------------------------------------------------------
# Section 3: Best model per target summary
# ---------------------------------------------------------------------------

def print_best_model_summary(best_models, model_comparison):
    sep("3. BEST MODEL PER TARGET")

    if best_models is None:
        print("  [MISSING] best_model_per_target.csv")
        return

    for _, row in best_models.iterrows():
        target = row.get("target", "?")
        model  = row.get("model",  "?")
        tshort = TARGET_SHORT.get(target, target)

        print(f"\n  Target : {tshort}")
        print(f"  Model  : {model}")

        if model_comparison is not None and "target" in model_comparison.columns:
            mc = model_comparison[
                (model_comparison["target"] == target) &
                (model_comparison["model"]  == model)
            ]
            if len(mc) > 0:
                r = mc.iloc[0]
                print(f"  Folds  : {int(r.get('n_heldout_phenotypes', 0))}")
                print(f"  N-test : {int(r.get('total_test_gwas', 0))}")

                for metric, label in [
                    ("spearman_rank_r_mean",          "Spearman rho (mean)"),
                    ("spearman_rank_r_sd",            "Spearman rho (SD)"),
                    ("spearman_rank_r_ci95_low",      "Spearman 95% CI low"),
                    ("spearman_rank_r_ci95_high",     "Spearman 95% CI high"),
                    ("kendall_tau_mean",               "Kendall tau (mean)"),
                    ("top1_correct_mean",              "Top-1 correct (mean)"),
                    ("top3_overlap_fraction_mean",     "Top-3 overlap (mean)"),
                    ("top5_overlap_fraction_mean",     "Top-5 overlap (mean)"),
                    ("ndcg_at_3_mean",                 "NDCG@3 (mean)"),
                    ("ndcg_at_5_mean",                 "NDCG@5 (mean)"),
                    ("mean_absolute_rank_error_mean",  "MARE (mean)"),
                    ("value_r2_mean",                  "Value R2 (mean)"),
                    ("combined_permutation_p_spearman","Combined perm p (Spearman)"),
                    ("ranking_statistically_significant","Statistically significant"),
                    ("ranking_significance_label",     "Significance label"),
                ]:
                    val = r.get(metric, np.nan)
                    if isinstance(val, float) and not pd.isna(val):
                        print(f"    {label:<40s}: {val:.4f}")
                    elif not pd.isna(val) if not isinstance(val, float) else False:
                        print(f"    {label:<40s}: {val}")


# ---------------------------------------------------------------------------
# Section 4: Per-fold LOPO results
# ---------------------------------------------------------------------------

def print_per_fold_results(fold_metrics, best_models):
    sep("4. PER-FOLD LOPO RESULTS (ALL TARGETS, BEST MODEL)")

    if fold_metrics is None:
        print("  [MISSING] lopo_fold_metrics.csv")
        return

    if best_models is None:
        print("  [MISSING] best_model_per_target.csv")
        return

    for _, brow in best_models.iterrows():
        target     = brow.get("target", "?")
        model_name = brow.get("model",  "?")
        tshort     = TARGET_SHORT.get(target, target)

        sub = fold_metrics[
            (fold_metrics["target"] == target) &
            (fold_metrics["model"]  == model_name)
        ].copy()

        if len(sub) == 0:
            print(f"\n  Target: {tshort} | Model: {model_name}  — no fold data found")
            continue

        sub = sub.sort_values("heldout_phenotype") if "heldout_phenotype" in sub.columns else sub

        print(f"\n  Target: {tshort}  |  Model: {model_name}  |  Folds: {len(sub)}")
        print()

        cols = [
            ("heldout_phenotype",             "Held-out phenotype",   "<45"),
            ("n_test_gwas",                   "N",                    ">4"),
            ("spearman_rank_r",               "Spearman",             ">9"),
            ("kendall_tau",                   "Kendall",              ">8"),
            ("top1_correct",                  "Top1",                 ">5"),
            ("top3_overlap_fraction",         "Top3",                 ">6"),
            ("top5_overlap_fraction",         "Top5",                 ">6"),
            ("ndcg_at_3",                     "NDCG@3",               ">7"),
            ("ndcg_at_5",                     "NDCG@5",               ">7"),
            ("mean_absolute_rank_error",      "MARE",                 ">7"),
            ("value_r2",                      "Val-R2",               ">7"),
            ("random_spearman_empirical_p",   "Perm-P",               ">8"),
        ]

        hdr = "  "
        for col, label, fmt_str in cols:
            w = int(fmt_str.replace("<", "").replace(">", ""))
            align = "<" if "<" in fmt_str else ">"
            hdr += f"{label:{align}{w}}  "
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))

        for _, row in sub.iterrows():
            line = "  "
            for col, label, fmt_str in cols:
                w = int(fmt_str.replace("<", "").replace(">", ""))
                align = "<" if "<" in fmt_str else ">"
                val = row.get(col, np.nan)
                if col == "heldout_phenotype":
                    line += f"{str(val):{align}{w}}  "
                elif col in ("n_test_gwas",):
                    v = int(val) if not pd.isna(val) else 0
                    line += f"{v:{align}{w}}  "
                elif col in ("top1_correct",):
                    v = int(val) if not pd.isna(val) else "—"
                    line += f"{str(v):{align}{w}}  "
                elif col == "random_spearman_empirical_p":
                    v = f"{val:.3g}" if not pd.isna(val) else "—"
                    line += f"{v:{align}{w}}  "
                else:
                    v = f"{val:.4f}" if not pd.isna(val) else "—"
                    line += f"{v:{align}{w}}  "
            print(line)

        # Summary row
        print("  " + "─" * (len(hdr) - 2))
        line = "  "
        for col, label, fmt_str in cols:
            w = int(fmt_str.replace("<", "").replace(">", ""))
            align = "<" if "<" in fmt_str else ">"
            if col == "heldout_phenotype":
                line += f"{'MEAN':{align}{w}}  "
            elif col == "n_test_gwas":
                v = int(sub[col].sum()) if col in sub.columns else 0
                line += f"{v:{align}{w}}  "
            elif col == "top1_correct":
                v = f"{sub[col].mean():.2f}" if col in sub.columns else "—"
                line += f"{v:{align}{w}}  "
            else:
                v = f"{sub[col].mean():.4f}" if col in sub.columns and sub[col].notna().any() else "—"
                line += f"{v:{align}{w}}  "
        print(line)


# ---------------------------------------------------------------------------
# Section 5: Per-fold per-GWAS predictions
# ---------------------------------------------------------------------------

def print_per_fold_predictions(lopo_predictions, best_models):
    sep("5. PER-FOLD GWAS-LEVEL PREDICTIONS (BEST MODEL, ALL TARGETS)")

    if lopo_predictions is None:
        print("  [MISSING] lopo_all_predictions.csv")
        return

    if best_models is None:
        print("  [MISSING] best_model_per_target.csv")
        return

    for _, brow in best_models.iterrows():
        target     = brow.get("target", "?")
        model_name = brow.get("model",  "?")
        tshort     = TARGET_SHORT.get(target, target)

        sub = lopo_predictions[
            (lopo_predictions["target"] == target) &
            (lopo_predictions["model"]  == model_name)
        ].copy()

        if len(sub) == 0:
            print(f"\n  Target: {tshort} | Model: {model_name}  — no prediction data found")
            continue

        print(f"\n  Target: {tshort}  |  Model: {model_name}  |  Rows: {len(sub)}")
        print()

        cols = [
            ("heldout_phenotype",  "Held-out phenotype",  "<45"),
            ("accessionId",        "AccessionId",         "<20"),
            ("actual_value",       "Actual",              ">9"),
            ("predicted_value",    "Predicted",           ">10"),
            ("actual_rank",        "Act-Rank",            ">9"),
            ("predicted_rank",     "Pred-Rank",           ">10"),
            ("absolute_rank_error","Rank-Err",            ">9"),
        ]

        hdr = "  "
        for col, label, fmt_str in cols:
            w = int(fmt_str.replace("<", "").replace(">", ""))
            align = "<" if "<" in fmt_str else ">"
            hdr += f"{label:{align}{w}}  "
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))

        for pheno, grp in sub.groupby("heldout_phenotype"):
            grp = grp.sort_values("actual_rank") if "actual_rank" in grp.columns else grp
            for _, row in grp.iterrows():
                line = "  "
                for col, label, fmt_str in cols:
                    w = int(fmt_str.replace("<", "").replace(">", ""))
                    align = "<" if "<" in fmt_str else ">"
                    val = row.get(col, np.nan)
                    if col in ("heldout_phenotype", "accessionId"):
                        line += f"{str(val):{align}{w}}  "
                    elif col in ("actual_rank", "predicted_rank", "absolute_rank_error"):
                        v = int(val) if not pd.isna(val) else "—"
                        line += f"{str(v):{align}{w}}  "
                    else:
                        v = f"{val:.4f}" if not pd.isna(val) else "—"
                        line += f"{v:{align}{w}}  "
                print(line)
            print()


# ---------------------------------------------------------------------------
# Section 6: Feature importance summary
# ---------------------------------------------------------------------------

def print_feature_importance(fold_importance_summary, best_models, top_n=30):
    sep(f"6. TOP {top_n} FEATURES PER TARGET (BEST MODEL, LOPO FOLD SUMMARY)")

    if fold_importance_summary is None:
        print("  [MISSING] feature_importance_summary.csv")
        return

    if best_models is None:
        print("  [MISSING] best_model_per_target.csv")
        return

    for _, brow in best_models.iterrows():
        target     = brow.get("target", "?")
        model_name = brow.get("model",  "?")
        tshort     = TARGET_SHORT.get(target, target)

        sub = fold_importance_summary[
            (fold_importance_summary["target"] == target) &
            (fold_importance_summary["model"]  == model_name)
        ].copy()

        if len(sub) == 0:
            print(f"\n  Target: {tshort} | Model: {model_name}  — no importance data found")
            continue

        if "mean_abs_importance_value" in sub.columns:
            sub = sub.sort_values("mean_abs_importance_value", ascending=False)

        print(f"\n  Target: {tshort}  |  Model: {model_name}  |  Total features: {len(sub)}")
        print()

        cols = [
            ("feature_rank_within_target_model", "Rank",   ">5"),
            ("feature",                           "Feature","<55"),
            ("importance_type",                   "Type",   "<25"),
            ("mean_importance_value",             "Mean",   ">9"),
            ("sd_importance_value",               "SD",     ">7"),
            ("importance_value_ci95_low",         "CI-lo",  ">8"),
            ("importance_value_ci95_high",        "CI-hi",  ">8"),
            ("mean_abs_importance_value",         "|Mean|", ">8"),
            ("n_folds_with_importance",           "Folds",  ">6"),
        ]

        hdr = "  "
        for col, label, fmt_str in cols:
            w = int(fmt_str.replace("<", "").replace(">", ""))
            align = "<" if "<" in fmt_str else ">"
            hdr += f"{label:{align}{w}}  "
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))

        for _, row in sub.head(top_n).iterrows():
            line = "  "
            for col, label, fmt_str in cols:
                w = int(fmt_str.replace("<", "").replace(">", ""))
                align = "<" if "<" in fmt_str else ">"
                val = row.get(col, np.nan)
                if col in ("feature", "importance_type"):
                    line += f"{str(val):{align}{w}}  "
                elif col in ("feature_rank_within_target_model", "n_folds_with_importance"):
                    v = int(val) if not pd.isna(val) else "—"
                    line += f"{str(v):{align}{w}}  "
                else:
                    v = f"{val:.5g}" if not pd.isna(val) else "—"
                    line += f"{v:{align}{w}}  "
            print(line)


# ---------------------------------------------------------------------------
# Section 7: Feature stability (Final analysis)
# ---------------------------------------------------------------------------

def print_feature_stability(feature_stability, best_models, top_n=30):
    sep(f"7. TOP {top_n} STABLE FEATURES PER TARGET (FINAL ANALYSIS OUTPUT)")

    if feature_stability is None:
        print("  [MISSING] top_feature_stability.csv")
        return

    if best_models is None:
        print("  [MISSING] best_model_per_target.csv")
        return

    for _, brow in best_models.iterrows():
        target     = brow.get("target", "?")
        model_name = brow.get("model",  "?")
        tshort     = TARGET_SHORT.get(target, target)

        sub = feature_stability[
            (feature_stability["target"] == target) &
            (feature_stability["model"]  == model_name)
        ].copy()

        if len(sub) == 0:
            print(f"\n  Target: {tshort} | Model: {model_name}  — no stability data found")
            continue

        if "mean_abs_importance_value" in sub.columns:
            sub = sub.sort_values("mean_abs_importance_value", ascending=False)

        print(f"\n  Target: {tshort}  |  Model: {model_name}  |  Features: {len(sub)}")
        print()

        cols = [
            ("stability_rank_within_target", "Rank",    ">5"),
            ("feature",                       "Feature", "<50"),
            ("mean_importance_value",         "Mean",    ">9"),
            ("importance_ci95_low",           "CI-lo",   ">8"),
            ("importance_ci95_high",          "CI-hi",   ">8"),
            ("mean_abs_importance_value",     "|Mean|",  ">8"),
            ("t_p_vs_zero",                   "T-p",     ">8"),
            ("wilcoxon_p_vs_zero",            "W-p",     ">8"),
            ("selection_frequency",           "Sel-Freq",">9"),
            ("dominant_sign",                 "Sign",    "<10"),
            ("sign_consistency",              "SignCons",">9"),
            ("stable_feature",                "Stable",  ">7"),
        ]

        hdr = "  "
        for col, label, fmt_str in cols:
            w = int(fmt_str.replace("<", "").replace(">", ""))
            align = "<" if "<" in fmt_str else ">"
            hdr += f"{label:{align}{w}}  "
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))

        for _, row in sub.head(top_n).iterrows():
            line = "  "
            for col, label, fmt_str in cols:
                w = int(fmt_str.replace("<", "").replace(">", ""))
                align = "<" if "<" in fmt_str else ">"
                val = row.get(col, np.nan)
                if col == "feature":
                    line += f"{str(val):{align}{w}}  "
                elif col == "dominant_sign":
                    line += f"{str(val):{align}{w}}  "
                elif col == "stable_feature":
                    v = "Yes" if str(val).lower() in ("true", "1") else "No"
                    line += f"{v:{align}{w}}  "
                elif col == "stability_rank_within_target":
                    v = int(val) if not pd.isna(val) else "—"
                    line += f"{str(v):{align}{w}}  "
                elif col in ("t_p_vs_zero", "wilcoxon_p_vs_zero"):
                    v = f"{val:.3g}" if not pd.isna(val) else "—"
                    line += f"{v:{align}{w}}  "
                else:
                    v = f"{val:.4f}" if not pd.isna(val) else "—"
                    line += f"{v:{align}{w}}  "
            print(line)


# ---------------------------------------------------------------------------
# Section 8: Final all-data rankings per phenotype
# ---------------------------------------------------------------------------

def print_final_rankings(final_rankings, best_models):
    sep("8. FINAL ALL-DATA RANKINGS PER PHENOTYPE (IN-SAMPLE)")

    if final_rankings is None:
        print("  [MISSING] final_all_data_rankings_by_phenotype.csv")
        print("  Note: these are in-sample rankings, not validation performance.")
        return

    if best_models is None:
        print("  [MISSING] best_model_per_target.csv")
        return

    for _, brow in best_models.iterrows():
        target     = brow.get("target", "?")
        model_name = brow.get("model",  "?")
        tshort     = TARGET_SHORT.get(target, target)

        sub = final_rankings[
            (final_rankings["target"] == target) &
            (final_rankings["model"]  == model_name)
        ].copy() if "target" in final_rankings.columns else final_rankings.copy()

        if len(sub) == 0:
            print(f"\n  Target: {tshort} | Model: {model_name}  — no ranking data found")
            continue

        print(f"\n  Target: {tshort}  |  Model: {model_name}  |  Total GWAS: {len(sub)}")
        print(f"  NOTE: Rankings are in-sample (final model trained on all data).")
        print()

        for pheno, grp in sub.groupby("phenotype"):
            grp = grp.sort_values("final_model_rank") if "final_model_rank" in grp.columns \
                else grp.sort_values("actual_rank") if "actual_rank" in grp.columns else grp

            print(f"  Phenotype: {pheno}  ({len(grp)} GWAS files)")

            cols = [
                ("final_model_rank", "Pred-Rank", ">9"),
                ("actual_rank",      "Act-Rank",  ">9"),
                ("accessionId",      "AccessionId","<20"),
                ("actual_value",     "Actual",    ">9"),
                ("final_model_score","Score",      ">9"),
                ("absolute_rank_error","RankErr",  ">8"),
            ]

            hdr = "    "
            for col, label, fmt_str in cols:
                w = int(fmt_str.replace("<", "").replace(">", ""))
                align = "<" if "<" in fmt_str else ">"
                hdr += f"{label:{align}{w}}  "
            print(hdr)
            print("    " + "─" * (len(hdr) - 4))

            for _, row in grp.iterrows():
                line = "    "
                for col, label, fmt_str in cols:
                    w = int(fmt_str.replace("<", "").replace(">", ""))
                    align = "<" if "<" in fmt_str else ">"
                    val = row.get(col, np.nan)
                    if col == "accessionId":
                        line += f"{str(val):{align}{w}}  "
                    elif col in ("final_model_rank", "actual_rank", "absolute_rank_error"):
                        v = int(val) if not pd.isna(val) else "—"
                        line += f"{str(v):{align}{w}}  "
                    else:
                        v = f"{val:.4f}" if not pd.isna(val) else "—"
                        line += f"{v:{align}{w}}  "
                print(line)
            print()


# ---------------------------------------------------------------------------
# Section 9: Final ranking metrics summary
# ---------------------------------------------------------------------------

def print_final_metric_summary(final_metric_summary, lopo_validation_summary, best_models):
    sep("9. FINAL METRICS SUMMARY: LOPO VALIDATION vs FINAL IN-SAMPLE")

    if best_models is None:
        print("  [MISSING] best_model_per_target.csv")
        return

    for _, brow in best_models.iterrows():
        target     = brow.get("target", "?")
        model_name = brow.get("model",  "?")
        tshort     = TARGET_SHORT.get(target, target)

        print(f"\n  Target: {tshort}  |  Model: {model_name}")
        print(f"  {'Metric':<45s}  {'LOPO (honest)':>15}  {'Final in-sample':>16}")
        print(f"  {'─' * 45}  {'─' * 15}  {'─' * 16}")

        lopo_row = None
        if lopo_validation_summary is not None and "target" in lopo_validation_summary.columns:
            lr = lopo_validation_summary[
                (lopo_validation_summary["target"] == target) &
                (lopo_validation_summary["model"]  == model_name)
            ]
            if len(lr) > 0:
                lopo_row = lr.iloc[0]

        final_row = None
        if final_metric_summary is not None and "target" in final_metric_summary.columns:
            fr = final_metric_summary[
                (final_metric_summary["target"] == target) &
                (final_metric_summary["model"]  == model_name)
            ]
            if len(fr) > 0:
                final_row = fr.iloc[0]

        metrics = [
            ("spearman_rank_r_mean",          "Spearman rho (mean)"),
            ("spearman_rank_r_sd",            "Spearman rho (SD)"),
            ("spearman_rank_r_ci95_low",      "Spearman 95% CI low"),
            ("spearman_rank_r_ci95_high",     "Spearman 95% CI high"),
            ("kendall_tau_mean",               "Kendall tau (mean)"),
            ("top1_correct_mean",              "Top-1 correct (mean)"),
            ("top3_overlap_fraction_mean",     "Top-3 overlap (mean)"),
            ("top5_overlap_fraction_mean",     "Top-5 overlap (mean)"),
            ("ndcg_at_3_mean",                 "NDCG@3 (mean)"),
            ("ndcg_at_5_mean",                 "NDCG@5 (mean)"),
            ("mean_absolute_rank_error_mean",  "MARE (mean)"),
            ("value_r2_mean",                  "Value R2 (mean)"),
        ]

        for col, label in metrics:
            lv = lopo_row.get(col, np.nan)  if lopo_row  is not None else np.nan
            fv = final_row.get(col, np.nan) if final_row is not None else np.nan
            ls = f"{lv:.4f}" if not pd.isna(lv) else "—"
            fs = f"{fv:.4f}" if not pd.isna(fv) else "—"
            print(f"  {label:<45s}  {ls:>15}  {fs:>16}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python Analysis22-ListPerformance.py <model_number>")
        print("  model_number: 1, 2, or 3")
        sys.exit(1)

    model_num = int(sys.argv[1])

    if model_num not in FEATURE_END_MAP:
        print(f"[ERROR] model_number must be 1, 2, or 3. Got: {model_num}")
        sys.exit(1)

    feature_end = FEATURE_END_MAP[model_num]
    model_label = f"Model{model_num}"
    model_tag   = f"model{model_num}"

    model_dir    = Path(model_label)
    training_dir = model_dir / "Training"
    final_dir    = model_dir / "Final"

    feature_file = model_dir / f"{model_tag}_features.csv"
    target_file  = model_dir / f"{model_tag}_target.csv"

    header(f"MODEL {model_num}  |  Feature1-Feature{feature_end}  |  GWAS RANKING PERFORMANCE REPORT")

    print(f"\n  Loading files from {model_dir}/")
    print()

    model_comparison       = load(training_dir / "05_model_comparison_summary.csv",      "Model comparison")
    best_models            = load(training_dir / "06_best_model_per_target.csv",          "Best models")
    fold_metrics           = load(training_dir / "04_lopo_fold_metrics.csv",              "Fold metrics")
    lopo_predictions       = load(training_dir / "03_lopo_all_predictions.csv",           "LOPO predictions")
    fold_importance_sum    = load(training_dir / "08_feature_importance_summary.csv",     "Feature importance summary")
    feature_stability      = load(final_dir    / "05_top_feature_stability.csv",          "Feature stability")
    final_rankings         = load(final_dir    / "09_final_all_data_rankings_by_phenotype.csv", "Final rankings")
    final_metric_summary   = load(final_dir    / "11_final_ranking_metrics_summary.csv",  "Final metric summary")
    lopo_validation_sum    = load(final_dir    / "02_lopo_validation_summary_selected_models.csv", "LOPO validation summary")

    print_dataset_overview(model_num, feature_end, feature_file, target_file)
    print_all_model_comparison(model_comparison)
    print_best_model_summary(best_models, model_comparison)
    print_per_fold_results(fold_metrics, best_models)
    print_per_fold_predictions(lopo_predictions, best_models)
    print_feature_importance(fold_importance_sum, best_models)
    print_feature_stability(feature_stability, best_models)
    print_final_rankings(final_rankings, best_models)
    print_final_metric_summary(final_metric_summary, lopo_validation_sum, best_models)

    sep()
    print(f"  END OF MODEL {model_num} PERFORMANCE REPORT")
    sep()


if __name__ == "__main__":
    main()