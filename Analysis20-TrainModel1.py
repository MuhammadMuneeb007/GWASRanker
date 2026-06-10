#!/usr/bin/env python3

"""
Analysis20-TrainModel1.py

Train leakage-safe GWAS ranking models for a configured model dataset.

Input files
-----------
Model-specific files such as:
Model1/model1_features.csv
Model1/model1_target.csv

These files should be generated first using:

    Analysis19-GenerateData1.py

Main outputs
------------
ModelX/Training/
    00_training_metadata.json
    01_feature_columns_used.csv
    02_target_columns_used.csv
    03_lopo_all_predictions.csv
    04_lopo_fold_metrics.csv
    05_model_comparison_summary.csv
    06_best_model_per_target.csv
    07_feature_importance_all_folds.csv
    08_feature_importance_summary.csv
    09_top_features_best_models.csv
    10_final_model_manifest.csv
    11_target_correlation_matrix.csv
    12_publication_summary.md
    13_publication_summary.tex
    14_leakage_audit_training.csv
    models/

Validation design
-----------------
Leave-one-phenotype-out cross-validation.

For each phenotype:
    train on all other phenotypes
    test on the held-out phenotype

This evaluates whether the model can rank GWAS files for a new phenotype.

Targets
-------
Expected target columns:

    target_plink_train_pure_prs
    target_plink_test_pure_prs
    target_plink_pure_prs_gap_quality

Higher is better for all targets.

Important
---------
Imputation and scaling are performed inside the sklearn Pipeline.
This avoids preprocessing leakage from held-out phenotypes.

Author
------
Generated for GWAS ranking publication analysis.
"""

from pathlib import Path
from datetime import datetime
import json
import platform
import warnings

import numpy as np
import pandas as pd

from scipy.stats import (
    spearmanr,
    kendalltau,
    ttest_1samp,
    wilcoxon,
    combine_pvalues,
)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

try:
    import joblib
except ImportError:
    joblib = None


warnings.filterwarnings("ignore")


# =============================================================================
# User settings
# =============================================================================

MODEL_NUMBER = 1
FEATURE_START = 1
FEATURE_END = 6

MODEL_LABEL = f"Model{MODEL_NUMBER}"
MODEL_TAG = f"model{MODEL_NUMBER}"
FEATURE_RANGE_TEXT = f"Feature{FEATURE_START}-Feature{FEATURE_END}"
FEATURE_FILE = Path(MODEL_LABEL) / f"{MODEL_TAG}_features.csv"
TARGET_FILE = Path(MODEL_LABEL) / f"{MODEL_TAG}_target.csv"
OUTPUT_DIR = Path(MODEL_LABEL) / "Training"

RANDOM_SEED = 42

ID_COLUMNS = ["phenotype", "accessionId"]

TARGET_COLUMNS = [
    "target_plink_train_pure_prs",
    "target_plink_test_pure_prs",
    "target_plink_pure_prs_gap_quality",
]

MIN_ROWS_FOR_TARGET = 20
MIN_PHENOTYPES_FOR_TARGET = 2
MIN_GWAS_IN_HELDOUT_PHENOTYPE = 2
MIN_TRAIN_ROWS = 10

N_BOOTSTRAP = 2000
N_PERMUTATIONS = 1000

SAVE_FINAL_MODELS = True
SAVE_LOPO_MODELS = False

TOP_N_FEATURES_PER_MODEL = 100
TOP_N_FEATURES_PRINT = 30

PRINT_FOLD_RESULTS = True


# =============================================================================
# Helpers
# =============================================================================

def print_section(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def print_subsection(title: str) -> None:
    print()
    print("-" * 100)
    print(title)
    print("-" * 100)


def configure_model(model_number: int, feature_end: int, output_dir: str | Path | None = None) -> None:
    global MODEL_NUMBER, FEATURE_END
    global MODEL_LABEL, MODEL_TAG, FEATURE_RANGE_TEXT
    global FEATURE_FILE, TARGET_FILE, OUTPUT_DIR

    MODEL_NUMBER = int(model_number)
    FEATURE_END = int(feature_end)
    MODEL_LABEL = f"Model{MODEL_NUMBER}"
    MODEL_TAG = f"model{MODEL_NUMBER}"
    FEATURE_RANGE_TEXT = f"Feature{FEATURE_START}-Feature{FEATURE_END}"
    FEATURE_FILE = Path(MODEL_LABEL) / f"{MODEL_TAG}_features.csv"
    TARGET_FILE = Path(MODEL_LABEL) / f"{MODEL_TAG}_target.csv"
    OUTPUT_DIR = Path(output_dir) if output_dir is not None else Path(MODEL_LABEL) / "Training"


def safe_name(x) -> str:
    x = str(x)

    for bad in [" ", "/", "\\", ":", ";", ",", "(", ")", "[", "]", "{", "}", "|", "@", "#"]:
        x = x.replace(bad, "_")

    while "__" in x:
        x = x.replace("__", "_")

    return x.strip("_")


def clean_numeric(series: pd.Series) -> pd.Series:
    """
    Safely convert a column to numeric.
    """

    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)

    if pd.api.types.is_numeric_dtype(series):
        out = pd.to_numeric(series, errors="coerce")
        out = out.replace([np.inf, -np.inf], np.nan)
        return out

    x = series.astype(str).str.strip()

    replacements = {
        "": np.nan,
        "nan": np.nan,
        "NaN": np.nan,
        "NAN": np.nan,
        "None": np.nan,
        "none": np.nan,
        "NONE": np.nan,
        "NA": np.nan,
        "N/A": np.nan,
        "na": np.nan,
        "n/a": np.nan,
        ".": np.nan,
        "null": np.nan,
        "NULL": np.nan,
        "inf": np.nan,
        "-inf": np.nan,
        "Inf": np.nan,
        "-Inf": np.nan,
        "Infinity": np.nan,
        "-Infinity": np.nan,
        "True": "1",
        "False": "0",
        "TRUE": "1",
        "FALSE": "0",
        "true": "1",
        "false": "0",
        "Yes": "1",
        "No": "0",
        "YES": "1",
        "NO": "0",
        "yes": "1",
        "no": "0",
    }

    x = x.replace(replacements)
    x = x.astype(str).str.strip()
    x = x.str.replace(",", "", regex=False)
    x = x.str.replace("%", "", regex=False)

    out = pd.to_numeric(x, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)

    return out


def rank_values(values, higher_is_better=True):
    values = pd.Series(values)

    ranks = values.rank(
        method="min",
        ascending=not higher_is_better,
    )

    return ranks.to_numpy()


def dcg_at_k(relevance, k):
    relevance = np.asarray(relevance, dtype=float)[:k]

    if len(relevance) == 0:
        return np.nan

    discounts = np.log2(np.arange(2, len(relevance) + 2))

    return float(np.sum(relevance / discounts))


def ndcg_at_k(actual_value, predicted_value, k=3):
    actual_value = np.asarray(actual_value, dtype=float)
    predicted_value = np.asarray(predicted_value, dtype=float)

    if len(actual_value) < 2:
        return np.nan

    min_val = np.nanmin(actual_value)
    relevance = actual_value - min_val

    if np.nanmax(relevance) == 0:
        return np.nan

    predicted_order = np.argsort(-predicted_value)
    ideal_order = np.argsort(-actual_value)

    dcg = dcg_at_k(relevance[predicted_order], k)
    idcg = dcg_at_k(relevance[ideal_order], k)

    if idcg == 0 or pd.isna(idcg):
        return np.nan

    return float(dcg / idcg)


def topk_overlap_fraction(actual_value, predicted_value, k=3):
    actual_value = np.asarray(actual_value, dtype=float)
    predicted_value = np.asarray(predicted_value, dtype=float)

    n = len(actual_value)

    if n == 0:
        return np.nan, 0

    k_eff = min(k, n)

    true_top = set(np.argsort(-actual_value)[:k_eff])
    pred_top = set(np.argsort(-predicted_value)[:k_eff])

    overlap = len(true_top.intersection(pred_top))

    return float(overlap / max(1, k_eff)), int(overlap)


def top1_correct(actual_value, predicted_value):
    actual_value = np.asarray(actual_value, dtype=float)
    predicted_value = np.asarray(predicted_value, dtype=float)

    if len(actual_value) == 0:
        return np.nan

    true_best = set(np.where(actual_value == np.nanmax(actual_value))[0])
    pred_best = set(np.where(predicted_value == np.nanmax(predicted_value))[0])

    return int(len(true_best.intersection(pred_best)) > 0)


# =============================================================================
# Data loading and leakage checks
# =============================================================================

def load_model_data():
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"Missing feature file: {FEATURE_FILE}")

    if not TARGET_FILE.exists():
        raise FileNotFoundError(f"Missing target file: {TARGET_FILE}")

    print_section(f"READING {MODEL_LABEL.upper()} FEATURES AND TARGETS")

    features = pd.read_csv(FEATURE_FILE, low_memory=False)
    targets = pd.read_csv(TARGET_FILE, low_memory=False)

    required = set(ID_COLUMNS)

    if not required.issubset(features.columns):
        raise ValueError(f"Feature file must contain columns: {ID_COLUMNS}")

    if not required.issubset(targets.columns):
        raise ValueError(f"Target file must contain columns: {ID_COLUMNS}")

    available_targets = [c for c in TARGET_COLUMNS if c in targets.columns]

    if len(available_targets) == 0:
        raise ValueError(
            f"No expected target columns found. Expected one or more of: {TARGET_COLUMNS}"
        )

    print(f"[FEATURE ROWS]       {len(features)}")
    print(f"[FEATURE COLUMNS]    {len(features.columns)}")
    print(f"[TARGET ROWS]        {len(targets)}")
    print(f"[TARGET COLUMNS]     {len(targets.columns)}")
    print(f"[TARGETS AVAILABLE]  {available_targets}")

    merged = features.merge(
        targets[ID_COLUMNS + available_targets],
        on=ID_COLUMNS,
        how="inner",
        validate="one_to_one",
    )

    print(f"[MERGED ROWS]        {len(merged)}")
    print(f"[PHENOTYPES]         {merged['phenotype'].nunique()}")
    print(f"[GWAS FILES]         {merged['accessionId'].nunique()}")

    return merged, available_targets


def get_feature_columns(df, target_cols):
    """
    Use all non-ID and non-target columns from the configured feature file.

    This script still performs a leakage audit in case the feature file was
    manually edited.
    """

    feature_cols = [
        c for c in df.columns
        if c not in ID_COLUMNS
        and c not in target_cols
    ]

    audit_rows = []

    for col in feature_cols:
        lower = col.lower()

        allowed_prefix = any(lower.startswith(f"feature{i}_") for i in range(FEATURE_START, FEATURE_END + 1))

        later_feature_prefixes = tuple([f"feature{i}_" for i in range(FEATURE_END + 1, 13)])
        forbidden_prefix = lower.startswith(
            (
                "target_",
                "plink_",
                "prsice2_",
            ) + later_feature_prefixes
        )

        forbidden_terms = [
            "pure_prs",
            "best_model",
            "train_pure_prs",
            "test_pure_prs",
            "performance",
            "gap",
            "auc",
            "accuracy",
            "r2",
        ]

        contains_forbidden_term = any(term in lower for term in forbidden_terms)

        possible_leakage = (
            not allowed_prefix
            or forbidden_prefix
            or contains_forbidden_term
        )

        audit_rows.append(
            {
                "feature": col,
                f"allowed_prefix_feature{FEATURE_START}_to_feature{FEATURE_END}": allowed_prefix,
                "forbidden_prefix": forbidden_prefix,
                "contains_forbidden_metric_term": contains_forbidden_term,
                "possible_leakage": possible_leakage,
            }
        )

    audit = pd.DataFrame(audit_rows)

    leakage_count = int(audit["possible_leakage"].sum())

    print_section("TRAINING LEAKAGE AUDIT")

    print(f"[FEATURE COLUMNS]              {len(feature_cols)}")
    print(f"[POSSIBLE LEAKAGE FEATURES]    {leakage_count}")

    if leakage_count > 0:
        print()
        print("[FAILED] Suspicious features found:")
        print(audit[audit["possible_leakage"]].to_string(index=False))
        raise RuntimeError(f"Leakage audit failed. Fix {MODEL_TAG}_features.csv first.")

    print(f"[PASSED] Feature matrix uses only {FEATURE_RANGE_TEXT} predictors.")

    return feature_cols, audit


def force_numeric_features(df, feature_cols, target_cols):
    df = df.copy()

    for col in feature_cols:
        df[col] = clean_numeric(df[col])

    for col in target_cols:
        df[col] = clean_numeric(df[col])

    return df


# =============================================================================
# Model definitions
# =============================================================================

def build_models(n_train_rows):
    """
    Build multiple models.

    Notes:
    - Imputation is inside the pipeline.
    - Scaling is inside the pipeline for linear models.
    - VarianceThreshold removes constant columns inside each fold.
    """

    cv_folds = min(5, max(2, n_train_rows // 20))

    models = {
        "RidgeCV": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("variance", VarianceThreshold(threshold=0.0)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    RidgeCV(
                        alphas=np.logspace(-4, 4, 100),
                        cv=cv_folds,
                    ),
                ),
            ]
        ),

        "LassoCV": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("variance", VarianceThreshold(threshold=0.0)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LassoCV(
                        alphas=np.logspace(-4, 1, 80),
                        cv=cv_folds,
                        max_iter=30000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),

        "ElasticNetCV": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("variance", VarianceThreshold(threshold=0.0)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNetCV(
                        alphas=np.logspace(-4, 1, 80),
                        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                        cv=cv_folds,
                        max_iter=30000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),

        "RandomForest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("variance", VarianceThreshold(threshold=0.0)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        max_depth=None,
                        min_samples_leaf=2,
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "ExtraTrees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("variance", VarianceThreshold(threshold=0.0)),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=500,
                        max_depth=None,
                        min_samples_leaf=2,
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "GradientBoosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("variance", VarianceThreshold(threshold=0.0)),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=300,
                        learning_rate=0.03,
                        max_depth=2,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),

        "HistGradientBoosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("variance", VarianceThreshold(threshold=0.0)),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=300,
                        learning_rate=0.03,
                        max_leaf_nodes=15,
                        l2_regularization=0.1,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
    }

    return models


# =============================================================================
# Metrics
# =============================================================================

def compute_ranking_metrics(actual_value, predicted_value):
    actual_value = np.asarray(actual_value, dtype=float)
    predicted_value = np.asarray(predicted_value, dtype=float)

    n = len(actual_value)

    if n < 2:
        return {}

    actual_rank = rank_values(actual_value, higher_is_better=True)
    predicted_rank = rank_values(predicted_value, higher_is_better=True)

    spearman_r, spearman_p = spearmanr(actual_rank, predicted_rank)
    kendall_r, kendall_p = kendalltau(actual_rank, predicted_rank)

    top3_fraction, top3_count = topk_overlap_fraction(
        actual_value,
        predicted_value,
        k=3,
    )

    top5_fraction, top5_count = topk_overlap_fraction(
        actual_value,
        predicted_value,
        k=5,
    )

    abs_rank_error = np.abs(actual_rank - predicted_rank)

    value_spearman_r = np.nan
    value_spearman_p = np.nan

    if (
        n >= 3
        and pd.Series(actual_value).nunique() > 1
        and pd.Series(predicted_value).nunique() > 1
    ):
        value_spearman_r, value_spearman_p = spearmanr(actual_value, predicted_value)

    value_rmse = np.sqrt(mean_squared_error(actual_value, predicted_value))

    return {
        "n_test_gwas": n,
        "spearman_rank_r": spearman_r,
        "spearman_rank_p_parametric": spearman_p,
        "kendall_tau": kendall_r,
        "kendall_tau_p_parametric": kendall_p,
        "top1_correct": top1_correct(actual_value, predicted_value),
        "top3_overlap_count": top3_count,
        "top3_overlap_fraction": top3_fraction,
        "top5_overlap_count": top5_count,
        "top5_overlap_fraction": top5_fraction,
        "mean_absolute_rank_error": float(np.nanmean(abs_rank_error)),
        "median_absolute_rank_error": float(np.nanmedian(abs_rank_error)),
        "max_absolute_rank_error": float(np.nanmax(abs_rank_error)),
        "ndcg_at_3": ndcg_at_k(actual_value, predicted_value, k=3),
        "ndcg_at_5": ndcg_at_k(actual_value, predicted_value, k=5),
        "value_spearman_r": value_spearman_r,
        "value_spearman_p_parametric": value_spearman_p,
        "value_mae": mean_absolute_error(actual_value, predicted_value),
        "value_rmse": value_rmse,
        "value_r2": r2_score(actual_value, predicted_value) if n > 1 else np.nan,
    }


def permutation_test_ranking(actual_value, predicted_value, n_permutations, seed):
    """
    Phenotype-wise random ranking test.

    Null:
        Predicted scores are exchangeable within the held-out phenotype.

    For high-is-better metrics:
        empirical p = probability random metric >= observed metric

    For rank error:
        empirical p = probability random error <= observed error
    """

    rng = np.random.default_rng(seed)

    actual_value = np.asarray(actual_value, dtype=float)
    predicted_value = np.asarray(predicted_value, dtype=float)

    observed = compute_ranking_metrics(actual_value, predicted_value)

    if len(actual_value) < 3:
        return {
            "random_spearman_mean": np.nan,
            "random_spearman_sd": np.nan,
            "random_spearman_empirical_p": np.nan,
            "random_kendall_mean": np.nan,
            "random_kendall_sd": np.nan,
            "random_kendall_empirical_p": np.nan,
            "random_top1_mean": np.nan,
            "random_top1_sd": np.nan,
            "random_top1_empirical_p": np.nan,
            "random_top3_mean": np.nan,
            "random_top3_sd": np.nan,
            "random_top3_empirical_p": np.nan,
            "random_ndcg3_mean": np.nan,
            "random_ndcg3_sd": np.nan,
            "random_ndcg3_empirical_p": np.nan,
            "random_mare_mean": np.nan,
            "random_mare_sd": np.nan,
            "random_mare_empirical_p": np.nan,
        }

    random_rows = []

    for _ in range(n_permutations):
        perm_pred = rng.permutation(predicted_value)
        m = compute_ranking_metrics(actual_value, perm_pred)

        random_rows.append(
            {
                "spearman_rank_r": m.get("spearman_rank_r", np.nan),
                "kendall_tau": m.get("kendall_tau", np.nan),
                "top1_correct": m.get("top1_correct", np.nan),
                "top3_overlap_fraction": m.get("top3_overlap_fraction", np.nan),
                "ndcg_at_3": m.get("ndcg_at_3", np.nan),
                "mean_absolute_rank_error": m.get("mean_absolute_rank_error", np.nan),
            }
        )

    random_df = pd.DataFrame(random_rows)

    def one_sided_high(metric_name, observed_value):
        vals = random_df[metric_name].dropna().to_numpy()

        if len(vals) == 0 or pd.isna(observed_value):
            return np.nan

        return float((np.sum(vals >= observed_value) + 1) / (len(vals) + 1))

    def one_sided_low(metric_name, observed_value):
        vals = random_df[metric_name].dropna().to_numpy()

        if len(vals) == 0 or pd.isna(observed_value):
            return np.nan

        return float((np.sum(vals <= observed_value) + 1) / (len(vals) + 1))

    return {
        "random_spearman_mean": random_df["spearman_rank_r"].mean(),
        "random_spearman_sd": random_df["spearman_rank_r"].std(ddof=1),
        "random_spearman_empirical_p": one_sided_high(
            "spearman_rank_r",
            observed.get("spearman_rank_r", np.nan),
        ),
        "random_kendall_mean": random_df["kendall_tau"].mean(),
        "random_kendall_sd": random_df["kendall_tau"].std(ddof=1),
        "random_kendall_empirical_p": one_sided_high(
            "kendall_tau",
            observed.get("kendall_tau", np.nan),
        ),
        "random_top1_mean": random_df["top1_correct"].mean(),
        "random_top1_sd": random_df["top1_correct"].std(ddof=1),
        "random_top1_empirical_p": one_sided_high(
            "top1_correct",
            observed.get("top1_correct", np.nan),
        ),
        "random_top3_mean": random_df["top3_overlap_fraction"].mean(),
        "random_top3_sd": random_df["top3_overlap_fraction"].std(ddof=1),
        "random_top3_empirical_p": one_sided_high(
            "top3_overlap_fraction",
            observed.get("top3_overlap_fraction", np.nan),
        ),
        "random_ndcg3_mean": random_df["ndcg_at_3"].mean(),
        "random_ndcg3_sd": random_df["ndcg_at_3"].std(ddof=1),
        "random_ndcg3_empirical_p": one_sided_high(
            "ndcg_at_3",
            observed.get("ndcg_at_3", np.nan),
        ),
        "random_mare_mean": random_df["mean_absolute_rank_error"].mean(),
        "random_mare_sd": random_df["mean_absolute_rank_error"].std(ddof=1),
        "random_mare_empirical_p": one_sided_low(
            "mean_absolute_rank_error",
            observed.get("mean_absolute_rank_error", np.nan),
        ),
    }


# =============================================================================
# Statistics
# =============================================================================

def bootstrap_mean_ci(values, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED):
    values = pd.Series(values).dropna().to_numpy(dtype=float)

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    if len(values) == 1:
        return float(values[0]), float(values[0]), float(values[0])

    rng = np.random.default_rng(seed)

    boot_means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.asarray(boot_means, dtype=float)

    return (
        float(np.mean(values)),
        float(np.percentile(boot_means, 2.5)),
        float(np.percentile(boot_means, 97.5)),
    )


def safe_ttest_greater_than_zero(values):
    values = pd.Series(values).dropna().to_numpy(dtype=float)

    if len(values) < 2:
        return np.nan, np.nan

    if np.nanstd(values, ddof=1) == 0:
        if np.nanmean(values) > 0:
            return np.inf, 0.0
        return 0.0, 1.0

    result = ttest_1samp(values, popmean=0.0, alternative="greater")

    return float(result.statistic), float(result.pvalue)


def safe_wilcoxon_greater_than_zero(values):
    values = pd.Series(values).dropna().to_numpy(dtype=float)

    if len(values) < 2:
        return np.nan, np.nan

    if np.all(values == 0):
        return 0.0, 1.0

    try:
        result = wilcoxon(values, alternative="greater", zero_method="wilcox")
        return float(result.statistic), float(result.pvalue)
    except Exception:
        return np.nan, np.nan


def fisher_combine_pvalues(pvalues):
    pvalues = pd.Series(pvalues).dropna()
    pvalues = pvalues[(pvalues > 0) & (pvalues <= 1)]

    if len(pvalues) == 0:
        return np.nan

    try:
        _, p = combine_pvalues(pvalues, method="fisher")
        return float(p)
    except Exception:
        return np.nan


def significance_label(p):
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def summarise_metric(group, metric, random_metric=None, higher_is_better=True):
    values = group[metric].dropna()

    out = {
        f"{metric}_mean": values.mean(),
        f"{metric}_sd": values.std(ddof=1),
        f"{metric}_se": values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan,
        f"{metric}_median": values.median(),
        f"{metric}_min": values.min(),
        f"{metric}_max": values.max(),
    }

    mean_value, ci_low, ci_high = bootstrap_mean_ci(values)

    out[f"{metric}_bootstrap_mean"] = mean_value
    out[f"{metric}_ci95_low"] = ci_low
    out[f"{metric}_ci95_high"] = ci_high

    if random_metric is not None and random_metric in group.columns:
        diff = group[metric] - group[random_metric]

        if not higher_is_better:
            diff = group[random_metric] - group[metric]

        diff = diff.dropna()

        diff_mean, diff_ci_low, diff_ci_high = bootstrap_mean_ci(diff)
        t_stat, t_p = safe_ttest_greater_than_zero(diff)
        w_stat, w_p = safe_wilcoxon_greater_than_zero(diff)

        out[f"{metric}_minus_random_mean"] = diff_mean
        out[f"{metric}_minus_random_ci95_low"] = diff_ci_low
        out[f"{metric}_minus_random_ci95_high"] = diff_ci_high
        out[f"{metric}_vs_random_t_stat"] = t_stat
        out[f"{metric}_vs_random_t_p"] = t_p
        out[f"{metric}_vs_random_wilcoxon_stat"] = w_stat
        out[f"{metric}_vs_random_wilcoxon_p"] = w_p
        out[f"{metric}_vs_random_significance"] = significance_label(t_p)

    return out


def summarise_model_comparison(evaluations):
    summary_rows = []

    metric_specs = [
        ("spearman_rank_r", "random_spearman_mean", True),
        ("kendall_tau", "random_kendall_mean", True),
        ("top1_correct", "random_top1_mean", True),
        ("top3_overlap_fraction", "random_top3_mean", True),
        ("top5_overlap_fraction", None, True),
        ("ndcg_at_3", "random_ndcg3_mean", True),
        ("ndcg_at_5", None, True),
        ("mean_absolute_rank_error", "random_mare_mean", False),
        ("median_absolute_rank_error", None, False),
        ("value_mae", None, False),
        ("value_rmse", None, False),
        ("value_r2", None, True),
    ]

    for (target, model), group in evaluations.groupby(["target", "model"]):
        row = {
            "target": target,
            "model": model,
            "n_heldout_phenotypes": group["heldout_phenotype"].nunique(),
            "total_test_gwas": group["n_test_gwas"].sum(),
            "n_successful_folds": len(group),
        }

        for metric, random_metric, higher_is_better in metric_specs:
            if metric in group.columns:
                row.update(
                    summarise_metric(
                        group=group,
                        metric=metric,
                        random_metric=random_metric,
                        higher_is_better=higher_is_better,
                    )
                )

        row["combined_permutation_p_spearman"] = fisher_combine_pvalues(
            group["random_spearman_empirical_p"]
            if "random_spearman_empirical_p" in group.columns
            else []
        )

        row["combined_permutation_p_kendall"] = fisher_combine_pvalues(
            group["random_kendall_empirical_p"]
            if "random_kendall_empirical_p" in group.columns
            else []
        )

        row["combined_permutation_p_top3"] = fisher_combine_pvalues(
            group["random_top3_empirical_p"]
            if "random_top3_empirical_p" in group.columns
            else []
        )

        row["combined_permutation_p_ndcg3"] = fisher_combine_pvalues(
            group["random_ndcg3_empirical_p"]
            if "random_ndcg3_empirical_p" in group.columns
            else []
        )

        row["combined_permutation_p_mare"] = fisher_combine_pvalues(
            group["random_mare_empirical_p"]
            if "random_mare_empirical_p" in group.columns
            else []
        )

        row["combined_permutation_significance_spearman"] = significance_label(
            row["combined_permutation_p_spearman"]
        )

        spearman_t_p = row.get("spearman_rank_r_vs_random_t_p", np.nan)
        spearman_ci_low = row.get("spearman_rank_r_ci95_low", np.nan)

        row["ranking_statistically_significant"] = bool(
            not pd.isna(spearman_t_p)
            and spearman_t_p < 0.05
            and not pd.isna(spearman_ci_low)
            and spearman_ci_low > 0
        )

        row["ranking_significance_label"] = (
            "Significant vs random"
            if row["ranking_statistically_significant"]
            else "Not significant vs random"
        )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    sort_cols = [
        "target",
        "ranking_statistically_significant",
        "spearman_rank_r_mean",
        "top3_overlap_fraction_mean",
        "ndcg_at_3_mean",
        "mean_absolute_rank_error_mean",
    ]

    existing_sort_cols = [c for c in sort_cols if c in summary.columns]

    ascending = []

    for col in existing_sort_cols:
        if col == "target":
            ascending.append(True)
        elif col == "mean_absolute_rank_error_mean":
            ascending.append(True)
        else:
            ascending.append(False)

    summary = summary.sort_values(
        existing_sort_cols,
        ascending=ascending,
    ).reset_index(drop=True)

    summary["rank_within_target"] = (
        summary
        .groupby("target")
        .cumcount()
        + 1
    )

    summary["overall_rank"] = np.arange(1, len(summary) + 1)

    return summary


# =============================================================================
# Feature importance
# =============================================================================

def get_selected_feature_names_after_variance(model, original_feature_cols):
    """
    Because VarianceThreshold can remove constant columns inside each fold,
    feature importance length may be shorter than original feature count.

    This function returns the feature names that survive VarianceThreshold.
    """

    if "variance" not in model.named_steps:
        return list(original_feature_cols)

    selector = model.named_steps["variance"]

    try:
        mask = selector.get_support()
        selected = np.asarray(original_feature_cols)[mask].tolist()
        return selected
    except Exception:
        return list(original_feature_cols)


def get_model_native_importance(model, original_feature_cols):
    fitted_model = model.named_steps["model"]
    selected_features = get_selected_feature_names_after_variance(model, original_feature_cols)

    if hasattr(fitted_model, "coef_"):
        values = np.asarray(fitted_model.coef_, dtype=float)

        if values.ndim > 1:
            values = values.ravel()

        importance_type = "coefficient"

    elif hasattr(fitted_model, "feature_importances_"):
        values = np.asarray(fitted_model.feature_importances_, dtype=float)
        importance_type = "native_feature_importance"

    else:
        return None

    if len(values) != len(selected_features):
        return None

    out = pd.DataFrame(
        {
            "feature": selected_features,
            "importance_value": values,
        }
    )

    out["abs_importance_value"] = out["importance_value"].abs()
    out["importance_type"] = importance_type

    out = out.sort_values(
        "abs_importance_value",
        ascending=False,
    ).reset_index(drop=True)

    out["feature_rank_within_fold"] = np.arange(1, len(out) + 1)

    return out


def summarise_feature_importance(feature_importance_all):
    if feature_importance_all is None or len(feature_importance_all) == 0:
        return pd.DataFrame()

    rows = []

    for (target, model, feature), group in feature_importance_all.groupby(
        ["target", "model", "feature"]
    ):
        values = group["importance_value"].dropna()
        abs_values = group["abs_importance_value"].dropna()

        _, ci_low, ci_high = bootstrap_mean_ci(values)
        _, abs_ci_low, abs_ci_high = bootstrap_mean_ci(abs_values)

        rows.append(
            {
                "target": target,
                "model": model,
                "feature": feature,
                "importance_type": group["importance_type"].iloc[0],
                "mean_importance_value": values.mean(),
                "sd_importance_value": values.std(ddof=1),
                "median_importance_value": values.median(),
                "importance_value_ci95_low": ci_low,
                "importance_value_ci95_high": ci_high,
                "mean_abs_importance_value": abs_values.mean(),
                "sd_abs_importance_value": abs_values.std(ddof=1),
                "median_abs_importance_value": abs_values.median(),
                "abs_importance_value_ci95_low": abs_ci_low,
                "abs_importance_value_ci95_high": abs_ci_high,
                "max_abs_importance_value": abs_values.max(),
                "n_folds_with_importance": group["heldout_phenotype"].nunique(),
            }
        )

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        ["target", "model", "mean_abs_importance_value"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    summary["feature_rank_within_target_model"] = (
        summary
        .groupby(["target", "model"])["mean_abs_importance_value"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    return summary


def select_top_features_for_best_models(feature_summary, best_model_per_target):
    if feature_summary is None or len(feature_summary) == 0:
        return pd.DataFrame()

    all_rows = []

    for _, row in best_model_per_target.iterrows():
        target = row["target"]
        model = row["model"]

        sub = feature_summary[
            (feature_summary["target"] == target)
            & (feature_summary["model"] == model)
        ].copy()

        sub = sub.sort_values(
            "mean_abs_importance_value",
            ascending=False,
        ).head(TOP_N_FEATURES_PER_MODEL)

        all_rows.append(sub)

    if len(all_rows) == 0:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


# =============================================================================
# Training
# =============================================================================

def prepare_xy(df, feature_cols, target_col):
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    return X, y


def run_lopo_for_target_and_model(df, feature_cols, target_col, model_name):
    all_predictions = []
    all_evaluations = []
    all_feature_importances = []

    usable_df = df.dropna(subset=[target_col]).copy()

    usable_df = usable_df.drop_duplicates(
        subset=["phenotype", "accessionId"],
        keep="first",
    ).copy()

    phenotypes = sorted(usable_df["phenotype"].dropna().unique())

    if len(phenotypes) < MIN_PHENOTYPES_FOR_TARGET:
        return None, None, None

    model_dir = OUTPUT_DIR / "lopo_models" / safe_name(target_col) / safe_name(model_name)

    if SAVE_LOPO_MODELS:
        model_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, heldout_phenotype in enumerate(phenotypes, start=1):
        train_df = usable_df[usable_df["phenotype"] != heldout_phenotype].copy()
        test_df = usable_df[usable_df["phenotype"] == heldout_phenotype].copy()

        if len(test_df) < MIN_GWAS_IN_HELDOUT_PHENOTYPE:
            continue

        if len(train_df) < MIN_TRAIN_ROWS:
            continue

        if train_df[target_col].nunique(dropna=True) < 2:
            continue

        X_train, y_train = prepare_xy(train_df, feature_cols, target_col)
        X_test, y_test = prepare_xy(test_df, feature_cols, target_col)

        model = build_models(n_train_rows=len(train_df))[model_name]

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        except Exception as e:
            print(
                f"[FAILED] target={target_col}, model={model_name}, "
                f"heldout={heldout_phenotype}, error={repr(e)}"
            )
            continue

        fold_pred = test_df[["phenotype", "accessionId"]].copy()
        fold_pred["target"] = target_col
        fold_pred["model"] = model_name
        fold_pred["heldout_phenotype"] = heldout_phenotype
        fold_pred["actual_value"] = y_test.to_numpy()
        fold_pred["predicted_value"] = y_pred

        fold_pred["actual_rank"] = rank_values(
            fold_pred["actual_value"],
            higher_is_better=True,
        ).astype(int)

        fold_pred["predicted_rank"] = rank_values(
            fold_pred["predicted_value"],
            higher_is_better=True,
        ).astype(int)

        fold_pred["absolute_rank_error"] = (
            fold_pred["actual_rank"] - fold_pred["predicted_rank"]
        ).abs()

        fold_metrics = compute_ranking_metrics(
            actual_value=fold_pred["actual_value"].to_numpy(),
            predicted_value=fold_pred["predicted_value"].to_numpy(),
        )

        permutation_metrics = permutation_test_ranking(
            actual_value=fold_pred["actual_value"].to_numpy(),
            predicted_value=fold_pred["predicted_value"].to_numpy(),
            n_permutations=N_PERMUTATIONS,
            seed=RANDOM_SEED + fold_idx,
        )

        fold_eval = {
            "target": target_col,
            "model": model_name,
            "heldout_phenotype": heldout_phenotype,
            "fold_index": fold_idx,
            "n_train_rows": len(train_df),
            "n_test_rows": len(test_df),
        }

        fold_eval.update(fold_metrics)
        fold_eval.update(permutation_metrics)

        all_predictions.append(fold_pred)
        all_evaluations.append(fold_eval)

        importance_df = get_model_native_importance(model, feature_cols)

        if importance_df is not None:
            importance_df["target"] = target_col
            importance_df["model"] = model_name
            importance_df["heldout_phenotype"] = heldout_phenotype
            importance_df["fold_index"] = fold_idx
            all_feature_importances.append(importance_df)

        if SAVE_LOPO_MODELS and joblib is not None:
            model_file = model_dir / f"fold_{fold_idx:02d}_{safe_name(heldout_phenotype)}.joblib"
            joblib.dump(model, model_file)

        if PRINT_FOLD_RESULTS:
            print(
                f"[FOLD {fold_idx:02d}/{len(phenotypes):02d}] "
                f"heldout={heldout_phenotype:45s} "
                f"n={fold_eval.get('n_test_gwas', np.nan):3} "
                f"spearman={fold_eval.get('spearman_rank_r', np.nan): .4f} "
                f"perm_p={fold_eval.get('random_spearman_empirical_p', np.nan): .4g} "
                f"kendall={fold_eval.get('kendall_tau', np.nan): .4f} "
                f"top1={fold_eval.get('top1_correct', np.nan)} "
                f"top3={fold_eval.get('top3_overlap_fraction', np.nan): .4f} "
                f"ndcg3={fold_eval.get('ndcg_at_3', np.nan): .4f} "
                f"MARE={fold_eval.get('mean_absolute_rank_error', np.nan): .4f}"
            )

    predictions = (
        pd.concat(all_predictions, ignore_index=True)
        if len(all_predictions) > 0
        else None
    )

    evaluations = (
        pd.DataFrame(all_evaluations)
        if len(all_evaluations) > 0
        else None
    )

    feature_importances = (
        pd.concat(all_feature_importances, ignore_index=True)
        if len(all_feature_importances) > 0
        else None
    )

    return predictions, evaluations, feature_importances


def fit_and_save_final_models(df, feature_cols, target_cols, best_model_per_target):
    if not SAVE_FINAL_MODELS:
        return pd.DataFrame()

    if joblib is None:
        print("[WARNING] joblib is not installed. Final models will not be saved.")
        return pd.DataFrame()

    final_model_dir = OUTPUT_DIR / "models"
    final_model_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for _, row in best_model_per_target.iterrows():
        target_col = row["target"]
        model_name = row["model"]

        usable_df = df.dropna(subset=[target_col]).copy()

        if len(usable_df) < MIN_ROWS_FOR_TARGET:
            continue

        if usable_df["phenotype"].nunique() < MIN_PHENOTYPES_FOR_TARGET:
            continue

        if usable_df[target_col].nunique(dropna=True) < 2:
            continue

        X, y = prepare_xy(usable_df, feature_cols, target_col)

        model = build_models(n_train_rows=len(usable_df))[model_name]

        try:
            model.fit(X, y)
        except Exception as e:
            print(
                f"[FINAL MODEL FAILED] target={target_col}, "
                f"model={model_name}, error={repr(e)}"
            )
            continue

        model_file = final_model_dir / f"{safe_name(target_col)}__{safe_name(model_name)}.joblib"
        joblib.dump(model, model_file)

        manifest_rows.append(
            {
                "target": target_col,
                "model": model_name,
                "n_training_rows": len(usable_df),
                "n_phenotypes": usable_df["phenotype"].nunique(),
                "model_file": str(model_file),
            }
        )

    manifest = pd.DataFrame(manifest_rows)

    return manifest


# =============================================================================
# Publication outputs
# =============================================================================

def make_publication_tables(model_comparison, best_model_per_target, top_features):
    main_cols = [
        "overall_rank",
        "rank_within_target",
        "target",
        "model",
        "n_heldout_phenotypes",
        "total_test_gwas",
        "spearman_rank_r_mean",
        "spearman_rank_r_sd",
        "spearman_rank_r_ci95_low",
        "spearman_rank_r_ci95_high",
        "spearman_rank_r_vs_random_t_p",
        "combined_permutation_p_spearman",
        "kendall_tau_mean",
        "top1_correct_mean",
        "top3_overlap_fraction_mean",
        "ndcg_at_3_mean",
        "mean_absolute_rank_error_mean",
        "ranking_significance_label",
    ]

    best_cols = [
        "target",
        "model",
        "rank_within_target",
        "n_heldout_phenotypes",
        "total_test_gwas",
        "spearman_rank_r_mean",
        "spearman_rank_r_sd",
        "spearman_rank_r_ci95_low",
        "spearman_rank_r_ci95_high",
        "spearman_rank_r_vs_random_t_p",
        "combined_permutation_p_spearman",
        "top3_overlap_fraction_mean",
        "ndcg_at_3_mean",
        "mean_absolute_rank_error_mean",
        "ranking_significance_label",
    ]

    feature_cols = [
        "target",
        "model",
        "feature_rank_within_target_model",
        "feature",
        "importance_type",
        "mean_importance_value",
        "sd_importance_value",
        "importance_value_ci95_low",
        "importance_value_ci95_high",
        "mean_abs_importance_value",
        "sd_abs_importance_value",
        "abs_importance_value_ci95_low",
        "abs_importance_value_ci95_high",
    ]

    main_table = model_comparison[[c for c in main_cols if c in model_comparison.columns]].copy()
    best_table = best_model_per_target[[c for c in best_cols if c in best_model_per_target.columns]].copy()

    if top_features is not None and len(top_features) > 0:
        feature_table = top_features[[c for c in feature_cols if c in top_features.columns]].copy()
    else:
        feature_table = pd.DataFrame()

    return main_table, best_table, feature_table


def write_publication_summary(main_table, best_table, feature_table):
    md_path = OUTPUT_DIR / "12_publication_summary.md"
    tex_path = OUTPUT_DIR / "13_publication_summary.tex"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {MODEL_LABEL} training summary\n\n")

        f.write("## Table 1. All target-model comparisons\n\n")
        f.write(main_table.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Table 2. Best model per target\n\n")
        f.write(best_table.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Table 3. Top features for best models\n\n")
        if feature_table is not None and len(feature_table) > 0:
            f.write(feature_table.to_markdown(index=False))
        else:
            f.write("No native feature importance was available.")
        f.write("\n")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(f"% {MODEL_LABEL} training summary\n\n")

        f.write("% Table 1. All target-model comparisons\n")
        f.write(main_table.to_latex(index=False, escape=True, longtable=True))
        f.write("\n\n")

        f.write("% Table 2. Best model per target\n")
        f.write(best_table.to_latex(index=False, escape=True, longtable=True))
        f.write("\n\n")

        f.write("% Table 3. Top features for best models\n")
        if feature_table is not None and len(feature_table) > 0:
            f.write(feature_table.to_latex(index=False, escape=True, longtable=True))
        else:
            f.write("% No native feature importance was available.\n")


# =============================================================================
# Main
# =============================================================================

def main():
    np.random.seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_section(f"ANALYSIS20: TRAIN {MODEL_LABEL.upper()} GWAS RANKING MODELS")

    df, target_cols = load_model_data()

    feature_cols, leakage_audit = get_feature_columns(df, target_cols)

    df = force_numeric_features(df, feature_cols, target_cols)

    feature_columns_df = pd.DataFrame({"feature": feature_cols})
    target_columns_df = pd.DataFrame({"target": target_cols})

    print_section("DATA SUMMARY AFTER NUMERIC CONVERSION")

    print(f"[ROWS]              {len(df)}")
    print(f"[FEATURES]          {len(feature_cols)}")
    print(f"[TARGETS]           {len(target_cols)}")
    print(f"[PHENOTYPES]        {df['phenotype'].nunique()}")
    print(f"[GWAS FILES]        {df['accessionId'].nunique()}")

    print()
    print("[TARGET COMPLETENESS]")
    for target in target_cols:
        print(
            f"{target:40s} "
            f"non_missing={df[target].notna().sum():6d} "
            f"unique={df[target].dropna().nunique():6d} "
            f"mean={df[target].mean(): .6f} "
            f"std={df[target].std(ddof=1): .6f}"
        )

    phenotype_counts = (
        df.groupby("phenotype")
        .size()
        .reset_index(name="n_gwas")
        .sort_values(["n_gwas", "phenotype"], ascending=[False, True])
    )

    print()
    print("[PHENOTYPE COUNTS]")
    print(phenotype_counts.to_string(index=False))

    model_names = list(build_models(n_train_rows=len(df)).keys())

    print_section("MODELS TO TRAIN")

    for model_name in model_names:
        print(f"- {model_name}")

    all_predictions = []
    all_evaluations = []
    all_feature_importances = []

    print_section("RUNNING LEAVE-ONE-PHENOTYPE-OUT CROSS-VALIDATION")

    for target_col in target_cols:
        usable_df = df.dropna(subset=[target_col]).copy()
        usable_rows = len(usable_df)
        usable_phenotypes = usable_df["phenotype"].nunique()

        print_section(f"TARGET: {target_col}")
        print(f"[USABLE ROWS]       {usable_rows}")
        print(f"[USABLE PHENOTYPES] {usable_phenotypes}")

        if usable_rows < MIN_ROWS_FOR_TARGET:
            print("[SKIPPED] Not enough usable rows.")
            continue

        if usable_phenotypes < MIN_PHENOTYPES_FOR_TARGET:
            print("[SKIPPED] Not enough phenotypes.")
            continue

        for model_name in model_names:
            print_subsection(f"MODEL: {model_name}")

            predictions, evaluations, feature_importances = run_lopo_for_target_and_model(
                df=df,
                feature_cols=feature_cols,
                target_col=target_col,
                model_name=model_name,
            )

            if predictions is not None:
                all_predictions.append(predictions)

            if evaluations is not None:
                all_evaluations.append(evaluations)

            if feature_importances is not None:
                all_feature_importances.append(feature_importances)

    if len(all_evaluations) == 0:
        raise RuntimeError("No LOPO model results were generated.")

    predictions_all = pd.concat(all_predictions, ignore_index=True)
    evaluations_all = pd.concat(all_evaluations, ignore_index=True)

    feature_importance_all = (
        pd.concat(all_feature_importances, ignore_index=True)
        if len(all_feature_importances) > 0
        else pd.DataFrame()
    )

    print_section("SUMMARISING MODEL PERFORMANCE")

    model_comparison = summarise_model_comparison(evaluations_all)

    best_model_per_target = (
        model_comparison
        .sort_values(
            [
                "target",
                "ranking_statistically_significant",
                "spearman_rank_r_mean",
                "top3_overlap_fraction_mean",
                "ndcg_at_3_mean",
                "mean_absolute_rank_error_mean",
            ],
            ascending=[True, False, False, False, False, True],
        )
        .groupby("target", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    feature_importance_summary = summarise_feature_importance(feature_importance_all)

    top_features_best_models = select_top_features_for_best_models(
        feature_summary=feature_importance_summary,
        best_model_per_target=best_model_per_target,
    )

    target_correlation_matrix = df[target_cols].corr(method="spearman")

    print_section("BEST MODEL PER TARGET")

    best_display_cols = [
        "target",
        "model",
        "rank_within_target",
        "n_heldout_phenotypes",
        "total_test_gwas",
        "spearman_rank_r_mean",
        "spearman_rank_r_sd",
        "spearman_rank_r_ci95_low",
        "spearman_rank_r_ci95_high",
        "spearman_rank_r_vs_random_t_p",
        "combined_permutation_p_spearman",
        "top3_overlap_fraction_mean",
        "ndcg_at_3_mean",
        "mean_absolute_rank_error_mean",
        "ranking_significance_label",
    ]

    print(best_model_per_target[[c for c in best_display_cols if c in best_model_per_target.columns]].to_string(index=False))

    print_section("TOP FEATURES FOR BEST MODELS")

    if top_features_best_models is not None and len(top_features_best_models) > 0:
        top_display_cols = [
            "target",
            "model",
            "feature_rank_within_target_model",
            "feature",
            "importance_type",
            "mean_importance_value",
            "sd_importance_value",
            "mean_abs_importance_value",
            "sd_abs_importance_value",
        ]

        print(
            top_features_best_models[
                [c for c in top_display_cols if c in top_features_best_models.columns]
            ].head(TOP_N_FEATURES_PRINT).to_string(index=False)
        )
    else:
        print("[INFO] No native feature importance available.")

    print_section("FITTING FINAL BEST MODELS ON ALL AVAILABLE DATA")

    final_model_manifest = fit_and_save_final_models(
        df=df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        best_model_per_target=best_model_per_target,
    )

    print_section("WRITING OUTPUT FILES")

    metadata = {
        "script": Path(__file__).name,
        "model_label": MODEL_LABEL,
        "model_tag": MODEL_TAG,
        "feature_range_text": FEATURE_RANGE_TEXT,
        "run_datetime": datetime.now().isoformat(),
        "feature_file": str(FEATURE_FILE),
        "target_file": str(TARGET_FILE),
        "output_dir": str(OUTPUT_DIR),
        "random_seed": RANDOM_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "n_permutations": N_PERMUTATIONS,
        "validation_design": "leave-one-phenotype-out cross-validation",
        "n_rows": int(len(df)),
        "n_feature_columns": int(len(feature_cols)),
        "n_target_columns": int(len(target_cols)),
        "target_columns": target_cols,
        "n_phenotypes": int(df["phenotype"].nunique()),
        "n_gwas_files": int(df["accessionId"].nunique()),
        "models": model_names,
        "save_final_models": SAVE_FINAL_MODELS,
        "save_lopo_models": SAVE_LOPO_MODELS,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    with open(OUTPUT_DIR / "00_training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    feature_columns_df.to_csv(OUTPUT_DIR / "01_feature_columns_used.csv", index=False)
    target_columns_df.to_csv(OUTPUT_DIR / "02_target_columns_used.csv", index=False)
    predictions_all.to_csv(OUTPUT_DIR / "03_lopo_all_predictions.csv", index=False)
    evaluations_all.to_csv(OUTPUT_DIR / "04_lopo_fold_metrics.csv", index=False)
    model_comparison.to_csv(OUTPUT_DIR / "05_model_comparison_summary.csv", index=False)
    best_model_per_target.to_csv(OUTPUT_DIR / "06_best_model_per_target.csv", index=False)
    feature_importance_all.to_csv(OUTPUT_DIR / "07_feature_importance_all_folds.csv", index=False)
    feature_importance_summary.to_csv(OUTPUT_DIR / "08_feature_importance_summary.csv", index=False)
    top_features_best_models.to_csv(OUTPUT_DIR / "09_top_features_best_models.csv", index=False)
    final_model_manifest.to_csv(OUTPUT_DIR / "10_final_model_manifest.csv", index=False)
    target_correlation_matrix.to_csv(OUTPUT_DIR / "11_target_correlation_matrix.csv")
    leakage_audit.to_csv(OUTPUT_DIR / "14_leakage_audit_training.csv", index=False)

    main_table, best_table, feature_table = make_publication_tables(
        model_comparison=model_comparison,
        best_model_per_target=best_model_per_target,
        top_features=top_features_best_models,
    )

    write_publication_summary(
        main_table=main_table,
        best_table=best_table,
        feature_table=feature_table,
    )

    saved_files = [
        "00_training_metadata.json",
        "01_feature_columns_used.csv",
        "02_target_columns_used.csv",
        "03_lopo_all_predictions.csv",
        "04_lopo_fold_metrics.csv",
        "05_model_comparison_summary.csv",
        "06_best_model_per_target.csv",
        "07_feature_importance_all_folds.csv",
        "08_feature_importance_summary.csv",
        "09_top_features_best_models.csv",
        "10_final_model_manifest.csv",
        "11_target_correlation_matrix.csv",
        "12_publication_summary.md",
        "13_publication_summary.tex",
        "14_leakage_audit_training.csv",
    ]

    print()
    for file_name in saved_files:
        path = OUTPUT_DIR / file_name
        if path.exists():
            print(f"[SAVED] {path}")

    if SAVE_FINAL_MODELS:
        print(f"[SAVED] {OUTPUT_DIR / 'models'}")

    print_section("DONE")

    print(
        f"""
Main files to inspect:

1. Best model per target:
   {MODEL_LABEL}/Training/06_best_model_per_target.csv

2. Full model comparison:
   {MODEL_LABEL}/Training/05_model_comparison_summary.csv

3. All held-out phenotype predictions:
   {MODEL_LABEL}/Training/03_lopo_all_predictions.csv

4. Fold-level statistics:
   {MODEL_LABEL}/Training/04_lopo_fold_metrics.csv

5. Feature importance:
   {MODEL_LABEL}/Training/08_feature_importance_summary.csv
   {MODEL_LABEL}/Training/09_top_features_best_models.csv

6. Final saved models:
   {MODEL_LABEL}/Training/models/

Important:
   The final saved models are trained on all available data for the best model
   selected per target. They are for prediction/use after validation, not for
   reporting cross-validation performance.
"""
    )


if __name__ == "__main__":
    main()
