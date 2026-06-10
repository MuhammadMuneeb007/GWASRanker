#!/usr/bin/env python3

"""
Analysis21-FinalRankerModel1.py

Final configured-model analysis after LOPO training.

This script does NOT redo LOPO cross-validation.

It reads model-specific files such as:
    Model1/model1_features.csv
    Model1/model1_target.csv
    Model1/Training/05_model_comparison_summary.csv
    Model1/Training/06_best_model_per_target.csv
    Model1/Training/07_feature_importance_all_folds.csv
    Model1/Training/08_feature_importance_summary.csv
    Model1/Training/03_lopo_all_predictions.csv

It performs:
    Step 3:
        Select best model per target using LOPO performance.

    Step 4:
        Summarise feature stability across folds.

    Step 5:
        Train final selected model on all available data.

    Step 6:
        Use final all-data model to rank all GWAS files within phenotype.

    Step 7:
        Save separate outputs for:
            A. LOPO validation performance
            B. Final all-data model ranking and final feature weights

Important interpretation
------------------------
LOPO predictions are the honest validation result.

Final all-data predictions are useful for final ranking/deployment, but they are
in-sample because the final model is trained on all available data.

Therefore, manuscript performance should be based mainly on LOPO outputs.
Final all-data weights/rankings should be reported as the final selected model.
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

MODEL_DIR = Path(MODEL_LABEL)
TRAINING_DIR = MODEL_DIR / "Training"
OUTPUT_DIR = MODEL_DIR / "Final"

FEATURE_FILE = MODEL_DIR / f"{MODEL_TAG}_features.csv"
TARGET_FILE = MODEL_DIR / f"{MODEL_TAG}_target.csv"

MODEL_COMPARISON_FILE = TRAINING_DIR / "05_model_comparison_summary.csv"
BEST_MODEL_FILE = TRAINING_DIR / "06_best_model_per_target.csv"
LOPO_PREDICTIONS_FILE = TRAINING_DIR / "03_lopo_all_predictions.csv"
FOLD_IMPORTANCE_FILE = TRAINING_DIR / "07_feature_importance_all_folds.csv"
FOLD_IMPORTANCE_SUMMARY_FILE = TRAINING_DIR / "08_feature_importance_summary.csv"

ID_COLUMNS = ["phenotype", "accessionId"]

EXPECTED_TARGETS = [
    "target_plink_train_pure_prs",
    "target_plink_test_pure_prs",
    "target_plink_pure_prs_gap_quality",
]

RANDOM_SEED = 42
N_BOOTSTRAP = 2000

TOP_N_FEATURES_TO_SAVE = 200
TOP_N_FEATURES_TO_PRINT = 40

SAVE_FINAL_MODELS = True


# =============================================================================
# Helper functions
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
    global MODEL_DIR, TRAINING_DIR, OUTPUT_DIR
    global FEATURE_FILE, TARGET_FILE
    global MODEL_COMPARISON_FILE, BEST_MODEL_FILE, LOPO_PREDICTIONS_FILE
    global FOLD_IMPORTANCE_FILE, FOLD_IMPORTANCE_SUMMARY_FILE

    MODEL_NUMBER = int(model_number)
    FEATURE_END = int(feature_end)
    MODEL_LABEL = f"Model{MODEL_NUMBER}"
    MODEL_TAG = f"model{MODEL_NUMBER}"
    FEATURE_RANGE_TEXT = f"Feature{FEATURE_START}-Feature{FEATURE_END}"

    MODEL_DIR = Path(MODEL_LABEL)
    TRAINING_DIR = MODEL_DIR / "Training"
    OUTPUT_DIR = Path(output_dir) if output_dir is not None else MODEL_DIR / "Final"

    FEATURE_FILE = MODEL_DIR / f"{MODEL_TAG}_features.csv"
    TARGET_FILE = MODEL_DIR / f"{MODEL_TAG}_target.csv"
    MODEL_COMPARISON_FILE = TRAINING_DIR / "05_model_comparison_summary.csv"
    BEST_MODEL_FILE = TRAINING_DIR / "06_best_model_per_target.csv"
    LOPO_PREDICTIONS_FILE = TRAINING_DIR / "03_lopo_all_predictions.csv"
    FOLD_IMPORTANCE_FILE = TRAINING_DIR / "07_feature_importance_all_folds.csv"
    FOLD_IMPORTANCE_SUMMARY_FILE = TRAINING_DIR / "08_feature_importance_summary.csv"


def safe_name(x) -> str:
    x = str(x)

    for bad in [" ", "/", "\\", ":", ";", ",", "(", ")", "[", "]", "{", "}", "|", "@", "#"]:
        x = x.replace(bad, "_")

    while "__" in x:
        x = x.replace("__", "_")

    return x.strip("_")


def clean_numeric(series: pd.Series) -> pd.Series:
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


def safe_ttest_against_zero(values):
    values = pd.Series(values).dropna().to_numpy(dtype=float)

    if len(values) < 2:
        return np.nan, np.nan

    if np.nanstd(values, ddof=1) == 0:
        if np.nanmean(values) == 0:
            return 0.0, 1.0
        return np.inf, 0.0

    result = ttest_1samp(values, popmean=0.0, alternative="two-sided")

    return float(result.statistic), float(result.pvalue)


def safe_wilcoxon_against_zero(values):
    values = pd.Series(values).dropna().to_numpy(dtype=float)

    if len(values) < 2:
        return np.nan, np.nan

    if np.all(values == 0):
        return 0.0, 1.0

    try:
        result = wilcoxon(values, alternative="two-sided", zero_method="wilcox")
        return float(result.statistic), float(result.pvalue)
    except Exception:
        return np.nan, np.nan


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


def rank_values(values, higher_is_better=True):
    values = pd.Series(values)

    return values.rank(
        method="min",
        ascending=not higher_is_better,
    ).to_numpy()


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


def compute_ranking_metrics(actual_value, predicted_value):
    actual_value = np.asarray(actual_value, dtype=float)
    predicted_value = np.asarray(predicted_value, dtype=float)

    n = len(actual_value)

    if n < 2:
        return {
            "n_gwas": n,
            "spearman_rank_r": np.nan,
            "spearman_rank_p": np.nan,
            "kendall_tau": np.nan,
            "kendall_tau_p": np.nan,
            "top1_correct": np.nan,
            "top3_overlap_fraction": np.nan,
            "top3_overlap_count": np.nan,
            "top5_overlap_fraction": np.nan,
            "top5_overlap_count": np.nan,
            "ndcg_at_3": np.nan,
            "ndcg_at_5": np.nan,
            "mean_absolute_rank_error": np.nan,
            "median_absolute_rank_error": np.nan,
            "max_absolute_rank_error": np.nan,
            "sd_absolute_rank_error": np.nan,
            "value_mae": np.nan,
            "value_rmse": np.nan,
            "value_r2": np.nan,
        }

    actual_rank = rank_values(actual_value, higher_is_better=True)
    predicted_rank = rank_values(predicted_value, higher_is_better=True)

    spearman_r, spearman_p = spearmanr(actual_rank, predicted_rank)
    kendall_r, kendall_p = kendalltau(actual_rank, predicted_rank)

    top3_fraction, top3_count = topk_overlap_fraction(actual_value, predicted_value, k=3)
    top5_fraction, top5_count = topk_overlap_fraction(actual_value, predicted_value, k=5)

    abs_rank_error = np.abs(actual_rank - predicted_rank)

    rmse = np.sqrt(mean_squared_error(actual_value, predicted_value))

    return {
        "n_gwas": n,
        "spearman_rank_r": spearman_r,
        "spearman_rank_p": spearman_p,
        "kendall_tau": kendall_r,
        "kendall_tau_p": kendall_p,
        "top1_correct": top1_correct(actual_value, predicted_value),
        "top3_overlap_fraction": top3_fraction,
        "top3_overlap_count": top3_count,
        "top5_overlap_fraction": top5_fraction,
        "top5_overlap_count": top5_count,
        "ndcg_at_3": ndcg_at_k(actual_value, predicted_value, k=3),
        "ndcg_at_5": ndcg_at_k(actual_value, predicted_value, k=5),
        "mean_absolute_rank_error": float(np.nanmean(abs_rank_error)),
        "median_absolute_rank_error": float(np.nanmedian(abs_rank_error)),
        "max_absolute_rank_error": float(np.nanmax(abs_rank_error)),
        "sd_absolute_rank_error": float(np.nanstd(abs_rank_error, ddof=1)) if n > 1 else np.nan,
        "value_mae": mean_absolute_error(actual_value, predicted_value),
        "value_rmse": rmse,
        "value_r2": r2_score(actual_value, predicted_value) if n > 1 else np.nan,
    }


# =============================================================================
# Models
# =============================================================================

def build_models(n_train_rows):
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


def get_selected_features_after_variance(model, feature_cols):
    if "variance" not in model.named_steps:
        return list(feature_cols)

    try:
        mask = model.named_steps["variance"].get_support()
        return np.asarray(feature_cols)[mask].tolist()
    except Exception:
        return list(feature_cols)


def extract_final_model_weights(model, feature_cols):
    """
    Extract final model weights or native feature importance.

    Linear models:
        coefficient

    Tree models:
        native_feature_importance

    HistGradientBoosting:
        no native feature_importances_ in sklearn, so returns empty table.
    """

    fitted_model = model.named_steps["model"]
    selected_features = get_selected_features_after_variance(model, feature_cols)

    if hasattr(fitted_model, "coef_"):
        values = np.asarray(fitted_model.coef_, dtype=float)

        if values.ndim > 1:
            values = values.ravel()

        importance_type = "coefficient"

    elif hasattr(fitted_model, "feature_importances_"):
        values = np.asarray(fitted_model.feature_importances_, dtype=float)
        importance_type = "native_feature_importance"

    else:
        return pd.DataFrame()

    if len(values) != len(selected_features):
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "feature": selected_features,
            "final_weight": values,
        }
    )

    out["abs_final_weight"] = out["final_weight"].abs()
    out["importance_type"] = importance_type

    out = out.sort_values(
        "abs_final_weight",
        ascending=False,
    ).reset_index(drop=True)

    out["final_weight_rank"] = np.arange(1, len(out) + 1)

    return out


# =============================================================================
# Load data
# =============================================================================

def load_required_data():
    required_files = [
        FEATURE_FILE,
        TARGET_FILE,
        MODEL_COMPARISON_FILE,
        BEST_MODEL_FILE,
        LOPO_PREDICTIONS_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print_section("READING INPUT FILES")

    features = pd.read_csv(FEATURE_FILE, low_memory=False)
    targets = pd.read_csv(TARGET_FILE, low_memory=False)
    model_comparison = pd.read_csv(MODEL_COMPARISON_FILE, low_memory=False)
    best_models = pd.read_csv(BEST_MODEL_FILE, low_memory=False)
    lopo_predictions = pd.read_csv(LOPO_PREDICTIONS_FILE, low_memory=False)

    if FOLD_IMPORTANCE_FILE.exists():
        fold_importance = pd.read_csv(FOLD_IMPORTANCE_FILE, low_memory=False)
    else:
        fold_importance = pd.DataFrame()

    if FOLD_IMPORTANCE_SUMMARY_FILE.exists():
        fold_importance_summary = pd.read_csv(FOLD_IMPORTANCE_SUMMARY_FILE, low_memory=False)
    else:
        fold_importance_summary = pd.DataFrame()

    print(f"[FEATURES]                 {FEATURE_FILE}")
    print(f"[TARGETS]                  {TARGET_FILE}")
    print(f"[MODEL COMPARISON]         {MODEL_COMPARISON_FILE}")
    print(f"[BEST MODELS]              {BEST_MODEL_FILE}")
    print(f"[LOPO PREDICTIONS]         {LOPO_PREDICTIONS_FILE}")

    if len(fold_importance) > 0:
        print(f"[FOLD IMPORTANCE]          {FOLD_IMPORTANCE_FILE}")
    else:
        print("[FOLD IMPORTANCE]          not available")

    target_cols = [c for c in EXPECTED_TARGETS if c in targets.columns]

    if len(target_cols) == 0:
        raise ValueError(f"No expected target columns found in {TARGET_FILE.name}")

    df = features.merge(
        targets[ID_COLUMNS + target_cols],
        on=ID_COLUMNS,
        how="inner",
        validate="one_to_one",
    )

    feature_cols = [
        c for c in features.columns
        if c not in ID_COLUMNS
    ]

    for col in feature_cols:
        df[col] = clean_numeric(df[col])

    for col in target_cols:
        df[col] = clean_numeric(df[col])

    print()
    print(f"[MERGED ROWS]              {len(df)}")
    print(f"[FEATURE COLUMNS]          {len(feature_cols)}")
    print(f"[TARGET COLUMNS]           {len(target_cols)}")
    print(f"[PHENOTYPES]               {df['phenotype'].nunique()}")
    print(f"[GWAS FILES]               {df['accessionId'].nunique()}")

    return (
        df,
        feature_cols,
        target_cols,
        model_comparison,
        best_models,
        lopo_predictions,
        fold_importance,
        fold_importance_summary,
    )


# =============================================================================
# Best model selection
# =============================================================================

def select_best_models_for_targets(best_models, target_cols):
    """
    Use 06_best_model_per_target.csv.

    Ensures train, test, and gap targets are represented if available.
    """

    required_cols = ["target", "model"]

    for col in required_cols:
        if col not in best_models.columns:
            raise ValueError(f"Missing column in best model file: {col}")

    selected = best_models[best_models["target"].isin(target_cols)].copy()

    selected = selected.drop_duplicates(
        subset=["target"],
        keep="first",
    ).copy()

    missing_targets = sorted(set(target_cols) - set(selected["target"]))

    if len(missing_targets) > 0:
        raise RuntimeError(
            "Some target columns do not have a selected best model in "
            f"{BEST_MODEL_FILE}: {missing_targets}"
        )

    target_order = {target: i for i, target in enumerate(target_cols)}
    selected["target_order"] = selected["target"].map(target_order)
    selected = selected.sort_values("target_order").drop(columns=["target_order"])

    return selected


# =============================================================================
# Feature stability across folds
# =============================================================================

def summarise_fold_feature_stability(fold_importance, selected_best_models):
    """
    Summarise feature stability across LOPO folds.

    For linear models:
        importance_value = fold-specific coefficient.
        We test coefficient against zero across folds.

    For tree models:
        importance_value = native feature importance.
        This is non-negative, so p-values against zero are descriptive only.
        Stability is better interpreted using selection frequency and CI.

    Output columns include:
        mean importance
        SD
        bootstrap 95% CI
        t-test p-value against zero
        Wilcoxon p-value against zero
        sign consistency
        selection frequency
    """

    if fold_importance is None or len(fold_importance) == 0:
        return pd.DataFrame(), pd.DataFrame()

    required_cols = [
        "target",
        "model",
        "feature",
        "importance_value",
        "abs_importance_value",
        "heldout_phenotype",
    ]

    missing = [c for c in required_cols if c not in fold_importance.columns]

    if len(missing) > 0:
        raise ValueError(f"Fold importance file is missing required columns: {missing}")

    selected_pairs = selected_best_models[["target", "model"]].drop_duplicates()

    fi = fold_importance.merge(
        selected_pairs,
        on=["target", "model"],
        how="inner",
    ).copy()

    if len(fi) == 0:
        return pd.DataFrame(), pd.DataFrame()

    rows = []

    for (target, model, feature), group in fi.groupby(["target", "model", "feature"]):
        values = clean_numeric(group["importance_value"]).dropna()
        abs_values = clean_numeric(group["abs_importance_value"]).dropna()

        if len(values) == 0:
            continue

        mean_value, ci_low, ci_high = bootstrap_mean_ci(values)
        mean_abs, abs_ci_low, abs_ci_high = bootstrap_mean_ci(abs_values)

        t_stat, t_p = safe_ttest_against_zero(values)
        w_stat, w_p = safe_wilcoxon_against_zero(values)

        n_folds = group["heldout_phenotype"].nunique()
        positive_count = int((values > 0).sum())
        negative_count = int((values < 0).sum())
        zero_count = int((values == 0).sum())

        nonzero_count = positive_count + negative_count
        selection_frequency = nonzero_count / max(1, len(values))

        dominant_sign_count = max(positive_count, negative_count)
        sign_consistency = dominant_sign_count / max(1, nonzero_count) if nonzero_count > 0 else 0.0

        if positive_count > negative_count:
            dominant_sign = "positive"
        elif negative_count > positive_count:
            dominant_sign = "negative"
        else:
            dominant_sign = "mixed_or_zero"

        importance_type = group["importance_type"].iloc[0] if "importance_type" in group.columns else "unknown"

        stable_feature = bool(
            selection_frequency >= 0.50
            and sign_consistency >= 0.70
            and not pd.isna(ci_low)
            and not pd.isna(ci_high)
            and (
                ci_low > 0
                or ci_high < 0
                or abs_ci_low > 0
            )
        )

        rows.append(
            {
                "target": target,
                "model": model,
                "feature": feature,
                "importance_type": importance_type,
                "n_folds_observed": n_folds,
                "n_importance_values": len(values),
                "mean_importance_value": mean_value,
                "sd_importance_value": values.std(ddof=1) if len(values) > 1 else np.nan,
                "median_importance_value": values.median(),
                "importance_ci95_low": ci_low,
                "importance_ci95_high": ci_high,
                "mean_abs_importance_value": mean_abs,
                "sd_abs_importance_value": abs_values.std(ddof=1) if len(abs_values) > 1 else np.nan,
                "median_abs_importance_value": abs_values.median(),
                "abs_importance_ci95_low": abs_ci_low,
                "abs_importance_ci95_high": abs_ci_high,
                "t_stat_vs_zero": t_stat,
                "t_p_vs_zero": t_p,
                "wilcoxon_stat_vs_zero": w_stat,
                "wilcoxon_p_vs_zero": w_p,
                "t_significance": significance_label(t_p),
                "wilcoxon_significance": significance_label(w_p),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "zero_count": zero_count,
                "selection_frequency": selection_frequency,
                "dominant_sign": dominant_sign,
                "sign_consistency": sign_consistency,
                "stable_feature": stable_feature,
            }
        )

    stability = pd.DataFrame(rows)

    if len(stability) == 0:
        return pd.DataFrame(), pd.DataFrame()

    stability = stability.sort_values(
        [
            "target",
            "stable_feature",
            "mean_abs_importance_value",
            "selection_frequency",
            "sign_consistency",
        ],
        ascending=[True, False, False, False, False],
    ).reset_index(drop=True)

    stability["stability_rank_within_target"] = (
        stability
        .groupby("target")["mean_abs_importance_value"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    top_stability = (
        stability
        .sort_values(
            ["target", "stability_rank_within_target"],
            ascending=[True, True],
        )
        .groupby("target", as_index=False)
        .head(TOP_N_FEATURES_TO_SAVE)
        .reset_index(drop=True)
    )

    return stability, top_stability


# =============================================================================
# Train final all-data models and generate rankings
# =============================================================================

def train_final_models_and_rank_all_data(df, feature_cols, target_cols, selected_best_models):
    final_model_manifest_rows = []
    final_weights_all = []
    final_predictions_all = []
    final_rankings_all = []
    final_metrics_rows = []

    final_model_dir = OUTPUT_DIR / "models"

    if SAVE_FINAL_MODELS:
        final_model_dir.mkdir(parents=True, exist_ok=True)

    for _, best_row in selected_best_models.iterrows():
        target = best_row["target"]
        model_name = best_row["model"]

        print_section(f"FINAL ALL-DATA MODEL: {target} | {model_name}")

        usable_df = df.dropna(subset=[target]).copy()

        if len(usable_df) < 2:
            print("[SKIPPED] Not enough rows.")
            continue

        if usable_df[target].nunique(dropna=True) < 2:
            print("[SKIPPED] Target has fewer than 2 unique values.")
            continue

        X = usable_df[feature_cols].copy()
        y = usable_df[target].copy()

        models = build_models(n_train_rows=len(usable_df))

        if model_name not in models:
            raise ValueError(f"Unknown model selected: {model_name}")

        model = models[model_name]

        model.fit(X, y)

        final_score = model.predict(X)

        model_file = ""

        if SAVE_FINAL_MODELS and joblib is not None:
            model_file_path = final_model_dir / f"{safe_name(target)}__{safe_name(model_name)}.joblib"
            joblib.dump(model, model_file_path)
            model_file = str(model_file_path)

        weights = extract_final_model_weights(model, feature_cols)

        if len(weights) > 0:
            weights["target"] = target
            weights["model"] = model_name
            weights["n_training_rows"] = len(usable_df)
            weights["n_training_phenotypes"] = usable_df["phenotype"].nunique()
            weights["model_file"] = model_file
            final_weights_all.append(weights)

        pred = usable_df[ID_COLUMNS].copy()
        pred["target"] = target
        pred["model"] = model_name
        pred["actual_value"] = y.to_numpy()
        pred["final_model_score"] = final_score

        final_predictions_all.append(pred)

        ranking_rows = []

        for phenotype, group in pred.groupby("phenotype"):
            g = group.copy()

            g["actual_rank"] = rank_values(g["actual_value"], higher_is_better=True).astype(int)
            g["final_model_rank"] = rank_values(g["final_model_score"], higher_is_better=True).astype(int)
            g["rank_difference"] = g["final_model_rank"] - g["actual_rank"]
            g["absolute_rank_error"] = g["rank_difference"].abs()

            g = g.sort_values(
                ["phenotype", "final_model_rank", "actual_rank", "accessionId"],
                ascending=[True, True, True, True],
            )

            ranking_rows.append(g)

            metrics = compute_ranking_metrics(
                actual_value=g["actual_value"].to_numpy(),
                predicted_value=g["final_model_score"].to_numpy(),
            )

            metrics.update(
                {
                    "target": target,
                    "model": model_name,
                    "phenotype": phenotype,
                    "ranking_type": "final_all_data_in_sample",
                }
            )

            final_metrics_rows.append(metrics)

        if len(ranking_rows) > 0:
            final_rankings_all.append(pd.concat(ranking_rows, ignore_index=True))

        final_model_manifest_rows.append(
            {
                "target": target,
                "model": model_name,
                "n_training_rows": len(usable_df),
                "n_training_phenotypes": usable_df["phenotype"].nunique(),
                "n_features": len(feature_cols),
                "model_file": model_file,
                "note": "Final model trained on all available data for this target. Rankings are in-sample.",
            }
        )

        print(f"[TRAINING ROWS]       {len(usable_df)}")
        print(f"[PHENOTYPES]          {usable_df['phenotype'].nunique()}")
        print(f"[FEATURES]            {len(feature_cols)}")
        print(f"[MODEL FILE]          {model_file if model_file else 'not saved'}")

    final_manifest = pd.DataFrame(final_model_manifest_rows)

    final_weights = (
        pd.concat(final_weights_all, ignore_index=True)
        if len(final_weights_all) > 0
        else pd.DataFrame()
    )

    final_predictions = (
        pd.concat(final_predictions_all, ignore_index=True)
        if len(final_predictions_all) > 0
        else pd.DataFrame()
    )

    final_rankings = (
        pd.concat(final_rankings_all, ignore_index=True)
        if len(final_rankings_all) > 0
        else pd.DataFrame()
    )

    final_metrics_by_phenotype = pd.DataFrame(final_metrics_rows)

    return (
        final_manifest,
        final_weights,
        final_predictions,
        final_rankings,
        final_metrics_by_phenotype,
    )


def summarise_final_ranking_metrics(final_metrics_by_phenotype):
    if final_metrics_by_phenotype is None or len(final_metrics_by_phenotype) == 0:
        return pd.DataFrame()

    metric_cols = [
        "spearman_rank_r",
        "kendall_tau",
        "top1_correct",
        "top3_overlap_fraction",
        "top5_overlap_fraction",
        "ndcg_at_3",
        "ndcg_at_5",
        "mean_absolute_rank_error",
        "median_absolute_rank_error",
        "max_absolute_rank_error",
        "sd_absolute_rank_error",
        "value_mae",
        "value_rmse",
        "value_r2",
    ]

    rows = []

    for (target, model), group in final_metrics_by_phenotype.groupby(["target", "model"]):
        row = {
            "target": target,
            "model": model,
            "ranking_type": "final_all_data_in_sample",
            "n_phenotypes": group["phenotype"].nunique(),
            "total_gwas": group["n_gwas"].sum(),
        }

        for metric in metric_cols:
            if metric not in group.columns:
                continue

            values = group[metric].dropna()

            mean_value, ci_low, ci_high = bootstrap_mean_ci(values)

            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_sd"] = values.std(ddof=1) if len(values) > 1 else np.nan
            row[f"{metric}_median"] = values.median()
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Compare LOPO validation with final all-data ranking
# =============================================================================

def make_lopo_validation_report(model_comparison, selected_best_models, lopo_predictions):
    selected_pairs = selected_best_models[["target", "model"]].drop_duplicates()

    lopo_best_summary = model_comparison.merge(
        selected_pairs,
        on=["target", "model"],
        how="inner",
    ).copy()

    lopo_best_predictions = lopo_predictions.merge(
        selected_pairs,
        on=["target", "model"],
        how="inner",
    ).copy()

    return lopo_best_summary, lopo_best_predictions


# =============================================================================
# Markdown and LaTeX summaries
# =============================================================================

def write_summary_files(
    selected_best_models,
    lopo_best_summary,
    final_metric_summary,
    top_feature_stability,
    final_weights,
):
    md_path = OUTPUT_DIR / "12_final_report.md"
    tex_path = OUTPUT_DIR / "13_final_report.tex"

    lopo_cols = [
        "target",
        "model",
        "n_heldout_phenotypes",
        "total_test_gwas",
        "spearman_rank_r_mean",
        "spearman_rank_r_sd",
        "spearman_rank_r_ci95_low",
        "spearman_rank_r_ci95_high",
        "combined_permutation_p_spearman",
        "top3_overlap_fraction_mean",
        "ndcg_at_3_mean",
        "mean_absolute_rank_error_mean",
        "ranking_significance_label",
    ]

    final_cols = [
        "target",
        "model",
        "ranking_type",
        "n_phenotypes",
        "total_gwas",
        "spearman_rank_r_mean",
        "spearman_rank_r_sd",
        "spearman_rank_r_ci95_low",
        "spearman_rank_r_ci95_high",
        "top3_overlap_fraction_mean",
        "ndcg_at_3_mean",
        "mean_absolute_rank_error_mean",
    ]

    stability_cols = [
        "target",
        "model",
        "stability_rank_within_target",
        "feature",
        "importance_type",
        "mean_importance_value",
        "sd_importance_value",
        "importance_ci95_low",
        "importance_ci95_high",
        "t_p_vs_zero",
        "wilcoxon_p_vs_zero",
        "selection_frequency",
        "dominant_sign",
        "sign_consistency",
        "stable_feature",
    ]

    weight_cols = [
        "target",
        "model",
        "final_weight_rank",
        "feature",
        "importance_type",
        "final_weight",
        "abs_final_weight",
    ]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {MODEL_LABEL} final model and ranking report\n\n")

        f.write("## Interpretation note\n\n")
        f.write(
            "LOPO validation performance is the honest held-out phenotype estimate. "
            "Final all-data rankings are in-sample and should be interpreted as the final deployed ranking, "
            "not as independent validation performance.\n\n"
        )

        f.write("## Selected best model per target\n\n")
        f.write(selected_best_models.to_markdown(index=False))
        f.write("\n\n")

        f.write("## A. LOPO validation performance for selected models\n\n")
        f.write(lopo_best_summary[[c for c in lopo_cols if c in lopo_best_summary.columns]].to_markdown(index=False))
        f.write("\n\n")

        f.write("## B. Final all-data ranking metrics\n\n")
        if final_metric_summary is not None and len(final_metric_summary) > 0:
            f.write(final_metric_summary[[c for c in final_cols if c in final_metric_summary.columns]].to_markdown(index=False))
        else:
            f.write("No final all-data ranking metrics available.")
        f.write("\n\n")

        f.write("## C. Top stable features across LOPO folds\n\n")
        if top_feature_stability is not None and len(top_feature_stability) > 0:
            f.write(
                top_feature_stability[
                    [c for c in stability_cols if c in top_feature_stability.columns]
                ].head(TOP_N_FEATURES_TO_SAVE).to_markdown(index=False)
            )
        else:
            f.write("No fold feature-stability results available.")
        f.write("\n\n")

        f.write("## D. Final all-data model weights\n\n")
        if final_weights is not None and len(final_weights) > 0:
            f.write(
                final_weights[
                    [c for c in weight_cols if c in final_weights.columns]
                ].head(TOP_N_FEATURES_TO_SAVE).to_markdown(index=False)
            )
        else:
            f.write("No final model weights available for selected models.")
        f.write("\n")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(f"% {MODEL_LABEL} final model and ranking report\n\n")

        f.write("% A. LOPO validation performance\n")
        f.write(
            lopo_best_summary[
                [c for c in lopo_cols if c in lopo_best_summary.columns]
            ].to_latex(index=False, escape=True, longtable=True)
        )
        f.write("\n\n")

        f.write("% B. Final all-data ranking metrics\n")
        if final_metric_summary is not None and len(final_metric_summary) > 0:
            f.write(
                final_metric_summary[
                    [c for c in final_cols if c in final_metric_summary.columns]
                ].to_latex(index=False, escape=True, longtable=True)
            )
        else:
            f.write("% No final all-data ranking metrics available.\n")
        f.write("\n\n")

        f.write("% C. Top stable features across LOPO folds\n")
        if top_feature_stability is not None and len(top_feature_stability) > 0:
            f.write(
                top_feature_stability[
                    [c for c in stability_cols if c in top_feature_stability.columns]
                ].head(TOP_N_FEATURES_TO_SAVE).to_latex(index=False, escape=True, longtable=True)
            )
        else:
            f.write("% No feature stability results available.\n")
        f.write("\n\n")

        f.write("% D. Final all-data model weights\n")
        if final_weights is not None and len(final_weights) > 0:
            f.write(
                final_weights[
                    [c for c in weight_cols if c in final_weights.columns]
                ].head(TOP_N_FEATURES_TO_SAVE).to_latex(index=False, escape=True, longtable=True)
            )
        else:
            f.write("% No final model weights available.\n")


# =============================================================================
# Main
# =============================================================================

def main():
    np.random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_section(f"ANALYSIS21: FINAL {MODEL_LABEL.upper()} MODEL, FEATURE STABILITY, AND GWAS RANKING")

    (
        df,
        feature_cols,
        target_cols,
        model_comparison,
        best_models,
        lopo_predictions,
        fold_importance,
        fold_importance_summary,
    ) = load_required_data()

    print_section("STEP 3: SELECT BEST MODEL PER TARGET")

    selected_best_models = select_best_models_for_targets(
        best_models=best_models,
        target_cols=target_cols,
    )

    print(selected_best_models[["target", "model"]].to_string(index=False))

    print_section("STEP 4: SUMMARISE FEATURE STABILITY ACROSS LOPO FOLDS")

    feature_stability, top_feature_stability = summarise_fold_feature_stability(
        fold_importance=fold_importance,
        selected_best_models=selected_best_models,
    )

    if len(top_feature_stability) > 0:
        display_cols = [
            "target",
            "model",
            "stability_rank_within_target",
            "feature",
            "importance_type",
            "mean_importance_value",
            "importance_ci95_low",
            "importance_ci95_high",
            "t_p_vs_zero",
            "selection_frequency",
            "dominant_sign",
            "sign_consistency",
            "stable_feature",
        ]

        print(
            top_feature_stability[
                [c for c in display_cols if c in top_feature_stability.columns]
            ].head(TOP_N_FEATURES_TO_PRINT).to_string(index=False)
        )
    else:
        print("[INFO] No fold feature importance found. Stability table will be empty.")

    print_section("STEP 5 AND 6: TRAIN FINAL ALL-DATA MODELS AND RANK ALL GWAS FILES")

    (
        final_model_manifest,
        final_weights,
        final_predictions,
        final_rankings,
        final_metrics_by_phenotype,
    ) = train_final_models_and_rank_all_data(
        df=df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        selected_best_models=selected_best_models,
    )

    final_metric_summary = summarise_final_ranking_metrics(
        final_metrics_by_phenotype=final_metrics_by_phenotype,
    )

    print_section("STEP 7A: LOPO VALIDATION PERFORMANCE FOR SELECTED MODELS")

    lopo_best_summary, lopo_best_predictions = make_lopo_validation_report(
        model_comparison=model_comparison,
        selected_best_models=selected_best_models,
        lopo_predictions=lopo_predictions,
    )

    lopo_display_cols = [
        "target",
        "model",
        "n_heldout_phenotypes",
        "total_test_gwas",
        "spearman_rank_r_mean",
        "spearman_rank_r_sd",
        "spearman_rank_r_ci95_low",
        "spearman_rank_r_ci95_high",
        "combined_permutation_p_spearman",
        "top3_overlap_fraction_mean",
        "ndcg_at_3_mean",
        "mean_absolute_rank_error_mean",
        "ranking_significance_label",
    ]

    print(
        lopo_best_summary[
            [c for c in lopo_display_cols if c in lopo_best_summary.columns]
        ].to_string(index=False)
    )

    print_section("STEP 7B: FINAL ALL-DATA RANKING METRICS")

    final_display_cols = [
        "target",
        "model",
        "ranking_type",
        "n_phenotypes",
        "total_gwas",
        "spearman_rank_r_mean",
        "spearman_rank_r_sd",
        "spearman_rank_r_ci95_low",
        "spearman_rank_r_ci95_high",
        "top3_overlap_fraction_mean",
        "ndcg_at_3_mean",
        "mean_absolute_rank_error_mean",
    ]

    if len(final_metric_summary) > 0:
        print(
            final_metric_summary[
                [c for c in final_display_cols if c in final_metric_summary.columns]
            ].to_string(index=False)
        )
    else:
        print("[INFO] No final metric summary available.")

    print_section("SAVING OUTPUTS")

    metadata = {
        "script": Path(__file__).name,
        "model_label": MODEL_LABEL,
        "model_tag": MODEL_TAG,
        "feature_range_text": FEATURE_RANGE_TEXT,
        "run_datetime": datetime.now().isoformat(),
        "feature_file": str(FEATURE_FILE),
        "target_file": str(TARGET_FILE),
        "model_comparison_file": str(MODEL_COMPARISON_FILE),
        "best_model_file": str(BEST_MODEL_FILE),
        "lopo_predictions_file": str(LOPO_PREDICTIONS_FILE),
        "fold_importance_file": str(FOLD_IMPORTANCE_FILE),
        "output_dir": str(OUTPUT_DIR),
        "random_seed": RANDOM_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "targets": target_cols,
        "n_targets": int(len(target_cols)),
        "n_phenotypes": int(df["phenotype"].nunique()),
        "n_gwas_files": int(df["accessionId"].nunique()),
        "selected_models": selected_best_models[["target", "model"]].to_dict(orient="records"),
        "interpretation": {
            "lopo_validation": "Honest held-out phenotype validation.",
            "final_all_data_ranking": "In-sample final ranking from models trained on all available data.",
            "feature_stability": "Stability of fold-specific feature importance across LOPO folds.",
            "final_weights": "Weights or native importance from final models trained on all data.",
        },
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    with open(OUTPUT_DIR / "00_final_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    selected_best_models.to_csv(OUTPUT_DIR / "01_selected_best_model_per_target.csv", index=False)
    lopo_best_summary.to_csv(OUTPUT_DIR / "02_lopo_validation_summary_selected_models.csv", index=False)
    lopo_best_predictions.to_csv(OUTPUT_DIR / "03_lopo_validation_predictions_selected_models.csv", index=False)

    feature_stability.to_csv(OUTPUT_DIR / "04_feature_stability_all.csv", index=False)
    top_feature_stability.to_csv(OUTPUT_DIR / "05_top_feature_stability.csv", index=False)

    final_model_manifest.to_csv(OUTPUT_DIR / "06_final_model_manifest.csv", index=False)
    final_weights.to_csv(OUTPUT_DIR / "07_final_model_weights.csv", index=False)
    final_predictions.to_csv(OUTPUT_DIR / "08_final_all_data_predictions.csv", index=False)
    final_rankings.to_csv(OUTPUT_DIR / "09_final_all_data_rankings_by_phenotype.csv", index=False)

    final_metrics_by_phenotype.to_csv(OUTPUT_DIR / "10_final_ranking_metrics_by_phenotype.csv", index=False)
    final_metric_summary.to_csv(OUTPUT_DIR / "11_final_ranking_metrics_summary.csv", index=False)

    write_summary_files(
        selected_best_models=selected_best_models,
        lopo_best_summary=lopo_best_summary,
        final_metric_summary=final_metric_summary,
        top_feature_stability=top_feature_stability,
        final_weights=final_weights,
    )

    saved_files = [
        "00_final_metadata.json",
        "01_selected_best_model_per_target.csv",
        "02_lopo_validation_summary_selected_models.csv",
        "03_lopo_validation_predictions_selected_models.csv",
        "04_feature_stability_all.csv",
        "05_top_feature_stability.csv",
        "06_final_model_manifest.csv",
        "07_final_model_weights.csv",
        "08_final_all_data_predictions.csv",
        "09_final_all_data_rankings_by_phenotype.csv",
        "10_final_ranking_metrics_by_phenotype.csv",
        "11_final_ranking_metrics_summary.csv",
        "12_final_report.md",
        "13_final_report.tex",
    ]

    for file_name in saved_files:
        path = OUTPUT_DIR / file_name
        if path.exists():
            print(f"[SAVED] {path}")

    if SAVE_FINAL_MODELS:
        print(f"[SAVED] {OUTPUT_DIR / 'models'}")

    print_section("DONE")

    print(
        f"""
Main outputs to inspect:

A. LOPO validation performance:
   {MODEL_LABEL}/Final/02_lopo_validation_summary_selected_models.csv
   {MODEL_LABEL}/Final/03_lopo_validation_predictions_selected_models.csv

B. Feature stability across folds:
   {MODEL_LABEL}/Final/04_feature_stability_all.csv
   {MODEL_LABEL}/Final/05_top_feature_stability.csv

C. Final all-data model:
   {MODEL_LABEL}/Final/06_final_model_manifest.csv
   {MODEL_LABEL}/Final/07_final_model_weights.csv
   {MODEL_LABEL}/Final/models/

D. Final all-data rankings:
   {MODEL_LABEL}/Final/08_final_all_data_predictions.csv
   {MODEL_LABEL}/Final/09_final_all_data_rankings_by_phenotype.csv
   {MODEL_LABEL}/Final/10_final_ranking_metrics_by_phenotype.csv
   {MODEL_LABEL}/Final/11_final_ranking_metrics_summary.csv

Important interpretation:
   Use LOPO validation outputs for manuscript performance.
   Use final all-data weights and rankings as the final deployed model/ranking.
"""
    )


if __name__ == "__main__":
    main()
