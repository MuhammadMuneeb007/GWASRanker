#!/usr/bin/env python3

"""
Analysis19-PublicationModel1-Fixed.py

Publication-ready multi-target, multi-model GWAS ranking analysis.

IMPORTANT FIX
-------------
This version fixes leakage from derived target columns.

All columns starting with:
    target_

are excluded from predictors.

This prevents leakage such as:
    target_plink_pure_prs_gap_abs
being used to predict:
    target_plink_pure_prs_gap_quality

Validation
----------
Leave-one-phenotype-out cross-validation.

Outputs
-------
All outputs are saved in:

    Model1/

Main files:
    00_Run_Metadata.json
    01_Target_Source_Columns.csv
    02_Target_Definitions.csv
    03_Target_Overview.csv
    04_Phenotype_Counts_By_Target.csv
    05_Feature_Screening.csv
    06_Numeric_Features_Used.csv
    07_LOPO_All_Predictions.csv
    08_LOPO_Fold_Metrics_All.csv
    09_LOPO_Model_Comparison_With_Statistics.csv
    10_Final_Target_Model_Ranking.csv
    11_Best_Model_Per_Target.csv
    12_Feature_Importance_All_Folds.csv
    13_Feature_Importance_Summary.csv
    14_Top_Features_For_Best_Models.csv
    15_Target_Correlation_Matrix.csv
    16_Publication_Tables.md
    17_Publication_Tables.tex
    18_Final_Model_Manifest.csv
    19_Leakage_Audit.csv
    README_Model1.txt
"""

from pathlib import Path
import json
import platform
import warnings
from datetime import datetime

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

INPUT_FILE = "AllFeatures_PRS_Merged.csv"
OUTPUT_DIR = Path("Model1")

RANDOM_SEED = 42

MIN_COMPLETE_ROWS_FOR_FEATURE = 10
MIN_GWAS_IN_TEST_PHENOTYPE = 2
MIN_ROWS_FOR_TARGET = 20
MIN_PHENOTYPES_FOR_TARGET = 2

# Keep this False if you want to include Feature9 as a genuine upstream feature.
# Set True if you want a strict model without target-fisher/clumping information.
EXCLUDE_FEATURE9 = False

# Keep True to avoid leakage from downstream PRS outputs.
EXCLUDE_PLINK_COLUMNS = True
EXCLUDE_PRSICE2_COLUMNS = True

SAVE_FINAL_MODELS = True
SAVE_LOPO_MODELS = False

# Statistics
N_BOOTSTRAP = 2000
N_PERMUTATIONS = 1000

# Output display
TOP_N_FEATURES_TO_SAVE_PER_BEST_MODEL = 100
TOP_N_FEATURES_TO_PRINT = 30
PRINT_FOLD_RESULTS = True


# Previous baseline from earlier Ridge-only PLINK test pure PRS result.
# This is used only for a comparison file, not for training.
PREVIOUS_BASELINE = {
    "baseline_name": "Previous_Ridge_only_plink_Test_pure_prs_mean",
    "mean_spearman_rank_r": 0.508165021767694,
    "top1_accuracy": 0.16666666666666666,
    "mean_top3_overlap_fraction": 0.6944444444444445,
    "mean_ndcg_at_3": 0.8015594034341406,
    "mean_ndcg_at_5": 0.8217038082655934,
    "mean_absolute_rank_error": 3.6516705369966243,
}


# =============================================================================
# General helpers
# =============================================================================

def safe_name(x):
    x = str(x)
    for bad in [" ", "/", "\\", ":", ";", ",", "(", ")", "[", "]", "{", "}", "|", "@", "#"]:
        x = x.replace(bad, "_")
    while "__" in x:
        x = x.replace("__", "_")
    return x.strip("_")


def print_section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def print_subsection(title):
    print()
    print("-" * 100)
    print(title)
    print("-" * 100)


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalise_column_name(x):
    return clean_text(x).lower().replace("-", "_").replace(" ", "_")


def find_column(df, candidates):
    lookup = {normalise_column_name(c): c for c in df.columns}

    for cand in candidates:
        key = normalise_column_name(cand)
        if key in lookup:
            return lookup[key]

    return None


def clean_numeric(series):
    """
    Convert numeric-looking columns safely.
    """

    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    x = series.astype(str).str.strip()

    x = x.replace(
        {
            "": np.nan,
            "nan": np.nan,
            "NaN": np.nan,
            "None": np.nan,
            "none": np.nan,
            "NA": np.nan,
            "N/A": np.nan,
            "na": np.nan,
            "n/a": np.nan,
            ".": np.nan,
            "inf": np.nan,
            "-inf": np.nan,
            "Infinity": np.nan,
            "-Infinity": np.nan,
            "True": "1",
            "False": "0",
            "true": "1",
            "false": "0",
            "TRUE": "1",
            "FALSE": "0",
            "YES": "1",
            "NO": "0",
            "yes": "1",
            "no": "0",
            "Yes": "1",
            "No": "0",
        }
    )

    x = x.astype(str).str.strip()
    x = x.str.replace(",", "", regex=False)
    x = x.str.replace("%", "", regex=False)

    return pd.to_numeric(x, errors="coerce")


def standardise_key_columns(df):
    phenotype_col = find_column(
        df,
        [
            "phenotype",
            "phenotype_id",
            "trait",
            "GWAS1Pheno",
            "GWASPheno",
        ],
    )

    accession_col = find_column(
        df,
        [
            "accessionId",
            "accessionID",
            "AccessionID",
            "accession_id",
            "gwas_id",
            "GWAS_ID",
            "GCST",
            "gwas_accession",
        ],
    )

    if phenotype_col is None:
        raise ValueError(
            "Could not find phenotype column.\n"
            f"Available columns:\n{list(df.columns)}"
        )

    if accession_col is None:
        raise ValueError(
            "Could not find accessionId or GWAS ID column.\n"
            f"Available columns:\n{list(df.columns)}"
        )

    df = df.rename(
        columns={
            phenotype_col: "phenotype",
            accession_col: "accessionId",
        }
    )

    df["phenotype"] = df["phenotype"].astype(str).str.strip()
    df["accessionId"] = df["accessionId"].astype(str).str.strip()

    return df


# =============================================================================
# Target construction
# =============================================================================

def find_existing_target_column(df, candidates):
    lookup = {normalise_column_name(c): c for c in df.columns}

    for cand in candidates:
        key = normalise_column_name(cand)
        if key in lookup:
            return lookup[key]

    return None


def make_target_table(df):
    """
    Detect source PRS performance columns and create derived target columns.

    For every target, higher is better.

    Gap quality:
        -abs(train - test)

    Stability score:
        test - abs(train - test)
    """

    df = df.copy()

    target_candidates = {
        "plink_train_pure_prs": [
            "plink_Train_pure_prs_mean",
            "plink_train_pure_prs_mean",
            "Train_pure_prs_mean",
            "train_pure_prs_mean",
        ],
        "plink_test_pure_prs": [
            "plink_Test_pure_prs_mean",
            "plink_test_pure_prs_mean",
            "Test_pure_prs_mean",
            "test_pure_prs_mean",
        ],
        "plink_train_best_model": [
            "plink_Train_best_model_mean",
            "plink_train_best_model_mean",
            "Train_best_model_mean",
            "train_best_model_mean",
        ],
        "plink_test_best_model": [
            "plink_Test_best_model_mean",
            "plink_test_best_model_mean",
            "Test_best_model_mean",
            "test_best_model_mean",
        ],
        "prsice2_train_pure_prs": [
            "prsice2_Train_pure_prs_mean",
            "prsice2_train_pure_prs_mean",
        ],
        "prsice2_test_pure_prs": [
            "prsice2_Test_pure_prs_mean",
            "prsice2_test_pure_prs_mean",
        ],
        "prsice2_train_best_model": [
            "prsice2_Train_best_model_mean",
            "prsice2_train_best_model_mean",
        ],
        "prsice2_test_best_model": [
            "prsice2_Test_best_model_mean",
            "prsice2_test_best_model_mean",
        ],
    }

    found = {}

    print_section("TARGET COLUMN DETECTION")

    for target_name, candidates in target_candidates.items():
        col = find_existing_target_column(df, candidates)
        found[target_name] = col

        if col is not None:
            print(f"[FOUND]   {target_name:32s} -> {col}")
        else:
            print(f"[MISSING] {target_name:32s}")

    created_targets = []
    all_derived_target_columns = []
    target_definitions = []

    def add_raw_target(target_name, source_col, description):
        if source_col is None:
            return

        df[target_name] = clean_numeric(df[source_col])
        created_targets.append(target_name)

        target_definitions.append(
            {
                "target": target_name,
                "target_type": "raw_performance",
                "source_columns": source_col,
                "formula": source_col,
                "higher_is_better": True,
                "used_for_modelling": True,
                "description": description,
            }
        )

    add_raw_target(
        "target_plink_train_pure_prs",
        found["plink_train_pure_prs"],
        "PLINK train pure PRS performance.",
    )
    add_raw_target(
        "target_plink_test_pure_prs",
        found["plink_test_pure_prs"],
        "PLINK test pure PRS performance.",
    )
    add_raw_target(
        "target_plink_train_best_model",
        found["plink_train_best_model"],
        "PLINK train best model performance.",
    )
    add_raw_target(
        "target_plink_test_best_model",
        found["plink_test_best_model"],
        "PLINK test best model performance.",
    )

    add_raw_target(
        "target_prsice2_train_pure_prs",
        found["prsice2_train_pure_prs"],
        "PRSice2 train pure PRS performance.",
    )
    add_raw_target(
        "target_prsice2_test_pure_prs",
        found["prsice2_test_pure_prs"],
        "PRSice2 test pure PRS performance.",
    )
    add_raw_target(
        "target_prsice2_train_best_model",
        found["prsice2_train_best_model"],
        "PRSice2 train best model performance.",
    )
    add_raw_target(
        "target_prsice2_test_best_model",
        found["prsice2_test_best_model"],
        "PRSice2 test best model performance.",
    )

    def add_gap_and_stability(prefix, train_col, test_col, label):
        if train_col is None or test_col is None:
            return

        train = clean_numeric(df[train_col])
        test = clean_numeric(df[test_col])
        gap_abs = (train - test).abs()

        gap_abs_col = f"target_{prefix}_gap_abs"
        gap_quality_col = f"target_{prefix}_gap_quality"
        stability_col = f"target_{prefix}_stability_score"

        df[gap_abs_col] = gap_abs
        df[gap_quality_col] = -gap_abs
        df[stability_col] = test - gap_abs

        # gap_abs is stored for audit/interpretation but NOT used as target.
        # It is also excluded from predictors because all target_* columns are excluded.
        all_derived_target_columns.extend([gap_abs_col, gap_quality_col, stability_col])
        created_targets.extend([gap_quality_col, stability_col])

        target_definitions.append(
            {
                "target": gap_abs_col,
                "target_type": "gap_abs_audit_only",
                "source_columns": f"{train_col}; {test_col}",
                "formula": "abs(train_performance - test_performance)",
                "higher_is_better": False,
                "used_for_modelling": False,
                "description": (
                    f"{label} absolute train-test gap. Audit only. "
                    "Not used as a modelling target and excluded from predictors."
                ),
            }
        )

        target_definitions.append(
            {
                "target": gap_quality_col,
                "target_type": "gap_quality",
                "source_columns": f"{train_col}; {test_col}",
                "formula": "-abs(train_performance - test_performance)",
                "higher_is_better": True,
                "used_for_modelling": True,
                "description": (
                    f"{label} train-test gap quality. Higher values mean smaller "
                    "absolute train-test gap."
                ),
            }
        )

        target_definitions.append(
            {
                "target": stability_col,
                "target_type": "stability_score",
                "source_columns": f"{train_col}; {test_col}",
                "formula": "test_performance - abs(train_performance - test_performance)",
                "higher_is_better": True,
                "used_for_modelling": True,
                "description": (
                    f"{label} stability score. Higher values mean higher test "
                    "performance with smaller train-test gap."
                ),
            }
        )

    add_gap_and_stability(
        prefix="plink_pure_prs",
        train_col=found["plink_train_pure_prs"],
        test_col=found["plink_test_pure_prs"],
        label="PLINK pure PRS",
    )

    add_gap_and_stability(
        prefix="plink_best_model",
        train_col=found["plink_train_best_model"],
        test_col=found["plink_test_best_model"],
        label="PLINK best model",
    )

    add_gap_and_stability(
        prefix="prsice2_pure_prs",
        train_col=found["prsice2_train_pure_prs"],
        test_col=found["prsice2_test_pure_prs"],
        label="PRSice2 pure PRS",
    )

    add_gap_and_stability(
        prefix="prsice2_best_model",
        train_col=found["prsice2_train_best_model"],
        test_col=found["prsice2_test_best_model"],
        label="PRSice2 best model",
    )

    created_targets = list(dict.fromkeys(created_targets))
    all_derived_target_columns = list(dict.fromkeys(all_derived_target_columns))

    if len(created_targets) == 0:
        raise RuntimeError(
            "No target columns were created. Please check target column names."
        )

    source_cols_df = pd.DataFrame(
        [
            {
                "target_source_name": k,
                "detected_column": v,
            }
            for k, v in found.items()
        ]
    )

    target_definitions_df = pd.DataFrame(target_definitions)

    target_overview_rows = []

    print_section("CREATED MODELLING TARGETS")

    for target in created_targets:
        non_missing = int(df[target].notna().sum())
        unique_values = int(df[target].dropna().nunique())
        n_phenotypes = int(df.dropna(subset=[target])["phenotype"].nunique())

        target_overview_rows.append(
            {
                "target": target,
                "non_missing_rows": non_missing,
                "unique_values": unique_values,
                "n_phenotypes": n_phenotypes,
                "mean": df[target].mean(),
                "std": df[target].std(ddof=1),
                "median": df[target].median(),
                "min": df[target].min(),
                "max": df[target].max(),
            }
        )

        print(
            f"{target:45s} "
            f"non_missing={non_missing:5d} "
            f"unique={unique_values:5d} "
            f"phenotypes={n_phenotypes:3d}"
        )

    target_overview_df = pd.DataFrame(target_overview_rows)

    return (
        df,
        created_targets,
        all_derived_target_columns,
        source_cols_df,
        target_definitions_df,
        target_overview_df,
    )


# =============================================================================
# Feature selection and leakage protection
# =============================================================================

def should_exclude_predictor(col, all_source_target_cols):
    """
    Leakage-safe predictor exclusion.

    Critical rule:
        Exclude every target_* column.

    This blocks:
        target_*_gap_abs
        target_*_gap_quality
        target_*_stability_score
        raw target_* columns

    from being used as predictors.
    """

    if col in ["phenotype", "accessionId"]:
        return True

    if col == "job_index":
        return True

    # Critical leakage fix.
    if col.startswith("target_"):
        return True

    # Source downstream target columns.
    if col in all_source_target_cols:
        return True

    # Downstream PRS outputs.
    if EXCLUDE_PLINK_COLUMNS and col.startswith("plink_"):
        return True

    if EXCLUDE_PRSICE2_COLUMNS and col.startswith("prsice2_"):
        return True

    if EXCLUDE_FEATURE9 and col.startswith("feature9_"):
        return True

    return False


def get_numeric_feature_columns(df, all_source_target_cols):
    numeric_cols = []
    feature_rows = []

    for col in df.columns:
        excluded = should_exclude_predictor(col, all_source_target_cols)

        numeric_values = clean_numeric(df[col])
        non_missing = int(numeric_values.notna().sum())
        unique_values = int(numeric_values.dropna().nunique())
        missing_fraction = 1.0 - (non_missing / max(1, len(df)))

        keep = (
            not excluded
            and non_missing >= MIN_COMPLETE_ROWS_FOR_FEATURE
            and unique_values > 1
        )

        if keep:
            numeric_cols.append(col)

        feature_rows.append(
            {
                "feature": col,
                "excluded_by_rule": excluded,
                "non_missing_rows": non_missing,
                "unique_values": unique_values,
                "missing_fraction": missing_fraction,
                "used_as_predictor": keep,
            }
        )

    feature_screening = pd.DataFrame(feature_rows)

    return numeric_cols, feature_screening


def make_leakage_audit(feature_cols):
    rows = []

    for feature in feature_cols:
        rows.append(
            {
                "feature": feature,
                "starts_with_target": feature.startswith("target_"),
                "starts_with_plink": feature.startswith("plink_"),
                "starts_with_prsice2": feature.startswith("prsice2_"),
                "starts_with_feature9": feature.startswith("feature9_"),
                "contains_gap": "gap" in feature.lower(),
                "possible_leakage_flag": (
                    feature.startswith("target_")
                    or feature.startswith("plink_")
                    or feature.startswith("prsice2_")
                ),
            }
        )

    audit = pd.DataFrame(rows)

    return audit


def prepare_xy(df, feature_cols, target_col):
    X = df[feature_cols].copy()

    for col in feature_cols:
        X[col] = clean_numeric(X[col])

    y = clean_numeric(df[target_col])

    return X, y


# =============================================================================
# Models
# =============================================================================

def build_models(n_train_rows):
    cv_folds = min(5, max(2, n_train_rows // 20))

    models = {
        "RidgeCV": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
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
                ("scaler", StandardScaler()),
                (
                    "model",
                    LassoCV(
                        alphas=np.logspace(-4, 1, 60),
                        cv=cv_folds,
                        max_iter=20000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "ElasticNetCV": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNetCV(
                        alphas=np.logspace(-4, 1, 60),
                        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                        cv=cv_folds,
                        max_iter=20000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
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
# Ranking metrics
# =============================================================================

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
    fraction = overlap / max(1, k_eff)

    return float(fraction), int(overlap)


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

    rmse = np.sqrt(mean_squared_error(actual_value, predicted_value))

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
        "value_rmse": rmse,
        "value_r2": r2_score(actual_value, predicted_value) if n > 1 else np.nan,
    }


def permutation_test_ranking(actual_value, predicted_value, n_permutations, seed):
    """
    Test whether observed ranking is better than random ranking.

    Null:
        predicted scores are exchangeable within phenotype.

    One-sided empirical p-values:
        Higher-is-better metrics: random >= observed
        MARE: random <= observed
    """

    rng = np.random.default_rng(seed)

    actual_value = np.asarray(actual_value, dtype=float)
    predicted_value = np.asarray(predicted_value, dtype=float)

    n = len(actual_value)

    observed = compute_ranking_metrics(actual_value, predicted_value)

    if n < 3:
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
            "random_ndcg5_mean": np.nan,
            "random_ndcg5_sd": np.nan,
            "random_ndcg5_empirical_p": np.nan,
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
                "ndcg_at_5": m.get("ndcg_at_5", np.nan),
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
        "random_ndcg5_mean": random_df["ndcg_at_5"].mean(),
        "random_ndcg5_sd": random_df["ndcg_at_5"].std(ddof=1),
        "random_ndcg5_empirical_p": one_sided_high(
            "ndcg_at_5",
            observed.get("ndcg_at_5", np.nan),
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


def summarise_evaluations_with_statistics(evaluations):
    summary_rows = []

    ranking_metrics = [
        ("spearman_rank_r", "random_spearman_mean", True),
        ("kendall_tau", "random_kendall_mean", True),
        ("top1_correct", "random_top1_mean", True),
        ("top3_overlap_fraction", "random_top3_mean", True),
        ("top5_overlap_fraction", None, True),
        ("ndcg_at_3", "random_ndcg3_mean", True),
        ("ndcg_at_5", "random_ndcg5_mean", True),
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
        }

        for metric, random_metric, higher_is_better in ranking_metrics:
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
        row["combined_permutation_significance_top3"] = significance_label(
            row["combined_permutation_p_top3"]
        )
        row["combined_permutation_significance_ndcg3"] = significance_label(
            row["combined_permutation_p_ndcg3"]
        )

        spearman_p = row.get("spearman_rank_r_vs_random_t_p", np.nan)
        spearman_ci_low = row.get("spearman_rank_r_ci95_low", np.nan)

        row["ranking_statistically_significant"] = bool(
            (not pd.isna(spearman_p))
            and (spearman_p < 0.05)
            and (not pd.isna(spearman_ci_low))
            and (spearman_ci_low > 0)
        )

        row["ranking_significance_label"] = (
            "Significant vs random"
            if row["ranking_statistically_significant"]
            else "Not significant vs random"
        )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    sort_cols = [
        "ranking_statistically_significant",
        "spearman_rank_r_mean",
        "top3_overlap_fraction_mean",
        "ndcg_at_3_mean",
        "mean_absolute_rank_error_mean",
    ]

    existing_sort_cols = [c for c in sort_cols if c in summary.columns]

    ascending = []
    for col in existing_sort_cols:
        if col == "mean_absolute_rank_error_mean":
            ascending.append(True)
        else:
            ascending.append(False)

    summary = summary.sort_values(
        existing_sort_cols,
        ascending=ascending,
    ).reset_index(drop=True)

    summary["final_model_rank"] = np.arange(1, len(summary) + 1)

    return summary


# =============================================================================
# Feature importance
# =============================================================================

def get_model_native_importance(model, feature_cols):
    fitted_model = model.named_steps["model"]

    if hasattr(fitted_model, "coef_"):
        values = np.asarray(fitted_model.coef_, dtype=float)
        importance_type = "coefficient"

    elif hasattr(fitted_model, "feature_importances_"):
        values = np.asarray(fitted_model.feature_importances_, dtype=float)
        importance_type = "native_feature_importance"

    else:
        return None

    out = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_value": values,
        }
    )

    out["abs_importance_value"] = out["importance_value"].abs()
    out["importance_type"] = importance_type

    out = out.sort_values(
        "abs_importance_value",
        ascending=False,
    ).reset_index(drop=True)

    out["feature_rank_within_model"] = np.arange(1, len(out) + 1)

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
                "n_folds": group["heldout_phenotype"].nunique(),
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

    out = []

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
        ).head(TOP_N_FEATURES_TO_SAVE_PER_BEST_MODEL)

        out.append(sub)

    if len(out) == 0:
        return pd.DataFrame()

    return pd.concat(out, ignore_index=True)


# =============================================================================
# LOPO training
# =============================================================================

def run_lopo_for_target_and_model(df, feature_cols, target_col, model_name, output_dir):
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

    lopo_model_dir = output_dir / "lopo_models" / safe_name(target_col) / safe_name(model_name)

    if SAVE_LOPO_MODELS:
        lopo_model_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, heldout_phenotype in enumerate(phenotypes, start=1):
        train_df = usable_df[usable_df["phenotype"] != heldout_phenotype].copy()
        test_df = usable_df[usable_df["phenotype"] == heldout_phenotype].copy()

        if len(test_df) < MIN_GWAS_IN_TEST_PHENOTYPE:
            continue

        X_train, y_train = prepare_xy(train_df, feature_cols, target_col)
        X_test, y_test = prepare_xy(test_df, feature_cols, target_col)

        if y_train.notna().sum() < 10:
            continue

        if y_train.nunique() < 2:
            continue

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

        all_evaluations.append(fold_eval)
        all_predictions.append(fold_pred)

        importance_df = get_model_native_importance(model, feature_cols)

        if importance_df is not None:
            importance_df["target"] = target_col
            importance_df["model"] = model_name
            importance_df["heldout_phenotype"] = heldout_phenotype
            importance_df["fold_index"] = fold_idx
            all_feature_importances.append(importance_df)

        if SAVE_LOPO_MODELS and joblib is not None:
            model_file = lopo_model_dir / f"fold_{fold_idx:02d}_{safe_name(heldout_phenotype)}.joblib"
            joblib.dump(model, model_file)

        if PRINT_FOLD_RESULTS:
            print(
                f"[FOLD {fold_idx:02d}/{len(phenotypes):02d}] "
                f"heldout={heldout_phenotype:45s} "
                f"n={fold_eval['n_test_gwas']:3d} "
                f"spearman={fold_eval['spearman_rank_r']: .4f} "
                f"p_perm={fold_eval['random_spearman_empirical_p']: .4g} "
                f"kendall={fold_eval['kendall_tau']: .4f} "
                f"top1={fold_eval['top1_correct']} "
                f"top3={fold_eval['top3_overlap_fraction']: .4f} "
                f"ndcg3={fold_eval['ndcg_at_3']: .4f} "
                f"MARE={fold_eval['mean_absolute_rank_error']: .4f}"
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


# =============================================================================
# Final models
# =============================================================================

def fit_and_save_final_models(df, feature_cols, target_cols, output_dir):
    if not SAVE_FINAL_MODELS:
        return

    if joblib is None:
        print("[WARNING] joblib is not installed. Final models will not be saved.")
        return

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_manifest = []

    for target_col in target_cols:
        usable_df = df.dropna(subset=[target_col]).copy()

        if len(usable_df) < MIN_ROWS_FOR_TARGET:
            continue

        if usable_df["phenotype"].nunique() < MIN_PHENOTYPES_FOR_TARGET:
            continue

        X, y = prepare_xy(usable_df, feature_cols, target_col)

        if y.nunique() < 2:
            continue

        model_names = list(build_models(n_train_rows=len(usable_df)).keys())

        for model_name in model_names:
            model = build_models(n_train_rows=len(usable_df))[model_name]

            try:
                model.fit(X, y)
            except Exception as e:
                print(
                    f"[FINAL MODEL FAILED] target={target_col}, "
                    f"model={model_name}, error={repr(e)}"
                )
                continue

            model_file = model_dir / f"{safe_name(target_col)}__{safe_name(model_name)}.joblib"
            joblib.dump(model, model_file)

            model_manifest.append(
                {
                    "target": target_col,
                    "model": model_name,
                    "n_training_rows": len(usable_df),
                    "n_phenotypes": usable_df["phenotype"].nunique(),
                    "model_file": str(model_file),
                }
            )

    manifest_df = pd.DataFrame(model_manifest)
    manifest_df.to_csv(output_dir / "18_Final_Model_Manifest.csv", index=False)


# =============================================================================
# Publication tables
# =============================================================================

def make_publication_tables(final_ranking, best_model_per_target, top_features):
    main_cols = [
        "final_model_rank",
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

    existing_main_cols = [c for c in main_cols if c in final_ranking.columns]
    main_table = final_ranking[existing_main_cols].copy()

    best_cols = [
        "target",
        "model",
        "final_model_rank",
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

    existing_best_cols = [c for c in best_cols if c in best_model_per_target.columns]
    best_table = best_model_per_target[existing_best_cols].copy()

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

    existing_feature_cols = [c for c in feature_cols if c in top_features.columns]
    feature_table = top_features[existing_feature_cols].copy()

    return main_table, best_table, feature_table


def write_markdown_and_latex_tables(output_dir, main_table, best_table, feature_table):
    md_path = output_dir / "16_Publication_Tables.md"
    tex_path = output_dir / "17_Publication_Tables.tex"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Publication tables for Model1\n\n")

        f.write("## Table 1. Overall target-model comparison\n\n")
        f.write(main_table.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Table 2. Best model per target\n\n")
        f.write(best_table.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Table 3. Top features for best models\n\n")
        f.write(feature_table.to_markdown(index=False))
        f.write("\n")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Publication tables for Model1\n\n")

        f.write("% Table 1. Overall target-model comparison\n")
        f.write(main_table.to_latex(index=False, escape=True, longtable=True))
        f.write("\n\n")

        f.write("% Table 2. Best model per target\n")
        f.write(best_table.to_latex(index=False, escape=True, longtable=True))
        f.write("\n\n")

        f.write("% Table 3. Top features for best models\n")
        f.write(feature_table.to_latex(index=False, escape=True, longtable=True))
        f.write("\n")


def make_baseline_comparison(final_ranking):
    rows = []

    for _, row in final_ranking.iterrows():
        rows.append(
            {
                "final_model_rank": row.get("final_model_rank"),
                "target": row.get("target"),
                "model": row.get("model"),
                "current_spearman": row.get("spearman_rank_r_mean"),
                "previous_spearman": PREVIOUS_BASELINE["mean_spearman_rank_r"],
                "delta_spearman": row.get("spearman_rank_r_mean") - PREVIOUS_BASELINE["mean_spearman_rank_r"],
                "current_top1": row.get("top1_correct_mean"),
                "previous_top1": PREVIOUS_BASELINE["top1_accuracy"],
                "delta_top1": row.get("top1_correct_mean") - PREVIOUS_BASELINE["top1_accuracy"],
                "current_top3": row.get("top3_overlap_fraction_mean"),
                "previous_top3": PREVIOUS_BASELINE["mean_top3_overlap_fraction"],
                "delta_top3": row.get("top3_overlap_fraction_mean") - PREVIOUS_BASELINE["mean_top3_overlap_fraction"],
                "current_ndcg3": row.get("ndcg_at_3_mean"),
                "previous_ndcg3": PREVIOUS_BASELINE["mean_ndcg_at_3"],
                "delta_ndcg3": row.get("ndcg_at_3_mean") - PREVIOUS_BASELINE["mean_ndcg_at_3"],
                "current_mare": row.get("mean_absolute_rank_error_mean"),
                "previous_mare": PREVIOUS_BASELINE["mean_absolute_rank_error"],
                "delta_mare_lower_is_better": row.get("mean_absolute_rank_error_mean") - PREVIOUS_BASELINE["mean_absolute_rank_error"],
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# README
# =============================================================================

def write_readme(output_dir):
    text = f"""
Model1 output directory
=======================

Generated by:
    Analysis19-PublicationModel1-Fixed.py

Generated on:
    {datetime.now().isoformat()}

Critical leakage fix
--------------------
All columns starting with target_ are excluded from predictors.

This prevents derived target audit columns such as:

    target_*_gap_abs

from being used to predict:

    target_*_gap_quality
    target_*_stability_score

This is important because otherwise gap-quality targets can show artificially
perfect performance.

Validation design
-----------------
Leave-one-phenotype-out cross-validation.

Targets
-------
Raw performance:
    target_*_train_pure_prs
    target_*_test_pure_prs
    target_*_train_best_model
    target_*_test_best_model

Derived:
    target_*_gap_quality
        -abs(train performance - test performance)

    target_*_stability_score
        test performance - abs(train performance - test performance)

Main metrics
------------
spearman_rank_r
kendall_tau
top1_correct
top3_overlap_fraction
ndcg_at_3
ndcg_at_5
mean_absolute_rank_error

Statistical testing
-------------------
Each fold is compared with a random-ranking baseline using phenotype-wise
permutation testing. Summary tables include mean, SD, SE, bootstrap 95 percent
CI, paired t-test versus random, Wilcoxon test versus random, and Fisher
combined permutation p-values.

Most important files
--------------------
09_LOPO_Model_Comparison_With_Statistics.csv
10_Final_Target_Model_Ranking.csv
11_Best_Model_Per_Target.csv
13_Feature_Importance_Summary.csv
14_Top_Features_For_Best_Models.csv
19_Leakage_Audit.csv
20_Comparison_To_Previous_Baseline.csv
"""
    with open(output_dir / "README_Model1.txt", "w", encoding="utf-8") as f:
        f.write(text)


# =============================================================================
# Main
# =============================================================================

def main():
    np.random.seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_section("MODEL1 FIXED: PUBLICATION-READY MULTI-TARGET MULTI-MODEL GWAS RANKING ANALYSIS")

    input_path = Path(INPUT_FILE)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"[READING] {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    df = standardise_key_columns(df)

    print(f"[ROWS RAW]        {len(df)}")
    print(f"[COLUMNS RAW]     {len(df.columns)}")
    print(f"[PHENOTYPES RAW]  {df['phenotype'].nunique()}")
    print(f"[GWAS RAW]        {df['accessionId'].nunique()}")

    (
        df,
        target_cols,
        all_derived_target_columns,
        target_source_cols_df,
        target_definitions_df,
        target_overview_df,
    ) = make_target_table(df)

    all_source_target_cols = [
        c for c in target_source_cols_df["detected_column"].dropna().tolist()
    ]

    feature_cols, feature_screening_df = get_numeric_feature_columns(
        df=df,
        all_source_target_cols=all_source_target_cols,
    )

    print_section("FEATURE SELECTION")
    print(f"[NUMERIC FEATURES USED] {len(feature_cols)}")

    if len(feature_cols) == 0:
        raise RuntimeError("No usable numeric feature columns found.")

    leakage_audit_df = make_leakage_audit(feature_cols)
    leakage_problem_count = int(leakage_audit_df["possible_leakage_flag"].sum())

    print_section("LEAKAGE AUDIT")
    print(f"[POSSIBLE LEAKAGE FEATURES USED] {leakage_problem_count}")

    if leakage_problem_count > 0:
        print("[WARNING] Possible leakage features remain in feature_cols:")
        print(
            leakage_audit_df[
                leakage_audit_df["possible_leakage_flag"]
            ].to_string(index=False)
        )
        raise RuntimeError("Leakage audit failed. Stop and inspect feature selection.")
    else:
        print("[PASSED] No target_, plink_, or prsice2_ columns are used as predictors.")

    numeric_features_df = pd.DataFrame({"feature": feature_cols})

    print()
    print("First 50 selected features:")
    for i, feature in enumerate(feature_cols[:50], start=1):
        print(f"{i:03d}. {feature}")

    if len(feature_cols) > 50:
        print(f"... plus {len(feature_cols) - 50} more features")

    phenotype_target_rows = []

    for target_col in target_cols:
        sub = df.dropna(subset=[target_col]).copy()

        counts = (
            sub.groupby("phenotype")
            .size()
            .reset_index(name="n_gwas")
        )

        counts["target"] = target_col
        phenotype_target_rows.append(counts)

    phenotype_counts_by_target = pd.concat(
        phenotype_target_rows,
        ignore_index=True,
    )

    print_section("PHENOTYPE COUNTS BY TARGET")
    print(phenotype_counts_by_target.head(100).to_string(index=False))

    model_names = list(build_models(n_train_rows=len(df)).keys())

    print_section("MODELS TO TEST")
    for model_name in model_names:
        print(f"- {model_name}")

    all_predictions = []
    all_evaluations = []
    all_feature_importances = []

    print_section("RUNNING LOPO EXPERIMENTS")

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
            print("[SKIPPED] Not enough usable phenotypes.")
            continue

        for model_name in model_names:
            print_subsection(f"MODEL: {model_name}")

            predictions, evaluations, feature_importances = run_lopo_for_target_and_model(
                df=df,
                feature_cols=feature_cols,
                target_col=target_col,
                model_name=model_name,
                output_dir=OUTPUT_DIR,
            )

            if predictions is not None:
                all_predictions.append(predictions)

            if evaluations is not None:
                all_evaluations.append(evaluations)

            if feature_importances is not None:
                all_feature_importances.append(feature_importances)

    if len(all_evaluations) == 0:
        raise RuntimeError("No usable LOPO model results were generated.")

    predictions_all = pd.concat(all_predictions, ignore_index=True)
    evaluations_all = pd.concat(all_evaluations, ignore_index=True)

    feature_importance_all = (
        pd.concat(all_feature_importances, ignore_index=True)
        if len(all_feature_importances) > 0
        else pd.DataFrame()
    )

    print_section("SUMMARISING MODEL PERFORMANCE WITH STATISTICS")

    model_comparison = summarise_evaluations_with_statistics(evaluations_all)
    final_ranking = model_comparison.copy()

    best_model_per_target = (
        final_ranking
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

    feature_summary = summarise_feature_importance(feature_importance_all)

    top_features_for_best_models = select_top_features_for_best_models(
        feature_summary,
        best_model_per_target,
    )

    target_correlation_matrix = df[target_cols].corr(method="spearman")

    baseline_comparison = make_baseline_comparison(final_ranking)

    print_section("FINAL TARGET-MODEL RANKING")

    display_cols = [
        "final_model_rank",
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
        "top3_overlap_fraction_mean",
        "ndcg_at_3_mean",
        "mean_absolute_rank_error_mean",
        "ranking_significance_label",
    ]

    existing_display_cols = [c for c in display_cols if c in final_ranking.columns]
    print(final_ranking[existing_display_cols].head(30).to_string(index=False))

    print_section("BEST MODEL PER TARGET")

    existing_best_display_cols = [c for c in display_cols if c in best_model_per_target.columns]
    print(best_model_per_target[existing_best_display_cols].to_string(index=False))

    print_section("TOP FEATURES FOR BEST MODELS")

    if len(top_features_for_best_models) > 0:
        top_feature_display_cols = [
            "target",
            "model",
            "feature_rank_within_target_model",
            "feature",
            "importance_type",
            "mean_importance_value",
            "sd_importance_value",
            "mean_abs_importance_value",
            "sd_abs_importance_value",
            "abs_importance_value_ci95_low",
            "abs_importance_value_ci95_high",
        ]

        existing_top_feature_display_cols = [
            c for c in top_feature_display_cols
            if c in top_features_for_best_models.columns
        ]

        print(
            top_features_for_best_models[
                existing_top_feature_display_cols
            ].head(TOP_N_FEATURES_TO_PRINT).to_string(index=False)
        )
    else:
        print("[NO FEATURE IMPORTANCE AVAILABLE]")

    print_section("SAVING OUTPUTS")

    metadata = {
        "script": "Analysis19-PublicationModel1-Fixed.py",
        "run_datetime": datetime.now().isoformat(),
        "input_file": INPUT_FILE,
        "output_dir": str(OUTPUT_DIR),
        "random_seed": RANDOM_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "n_permutations": N_PERMUTATIONS,
        "min_complete_rows_for_feature": MIN_COMPLETE_ROWS_FOR_FEATURE,
        "min_gwas_in_test_phenotype": MIN_GWAS_IN_TEST_PHENOTYPE,
        "exclude_feature9": EXCLUDE_FEATURE9,
        "exclude_plink_columns": EXCLUDE_PLINK_COLUMNS,
        "exclude_prsice2_columns": EXCLUDE_PRSICE2_COLUMNS,
        "critical_leakage_fix": "all columns starting with target_ excluded from predictors",
        "n_rows_after_target_creation": len(df),
        "n_columns_after_target_creation": len(df.columns),
        "n_phenotypes": int(df["phenotype"].nunique()),
        "n_gwas": int(df["accessionId"].nunique()),
        "n_targets": len(target_cols),
        "targets": target_cols,
        "n_numeric_features_used": len(feature_cols),
        "models": model_names,
        "previous_baseline": PREVIOUS_BASELINE,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    with open(OUTPUT_DIR / "00_Run_Metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    target_source_cols_df.to_csv(OUTPUT_DIR / "01_Target_Source_Columns.csv", index=False)
    target_definitions_df.to_csv(OUTPUT_DIR / "02_Target_Definitions.csv", index=False)
    target_overview_df.to_csv(OUTPUT_DIR / "03_Target_Overview.csv", index=False)
    phenotype_counts_by_target.to_csv(OUTPUT_DIR / "04_Phenotype_Counts_By_Target.csv", index=False)
    feature_screening_df.to_csv(OUTPUT_DIR / "05_Feature_Screening.csv", index=False)
    numeric_features_df.to_csv(OUTPUT_DIR / "06_Numeric_Features_Used.csv", index=False)
    predictions_all.to_csv(OUTPUT_DIR / "07_LOPO_All_Predictions.csv", index=False)
    evaluations_all.to_csv(OUTPUT_DIR / "08_LOPO_Fold_Metrics_All.csv", index=False)
    model_comparison.to_csv(OUTPUT_DIR / "09_LOPO_Model_Comparison_With_Statistics.csv", index=False)
    final_ranking.to_csv(OUTPUT_DIR / "10_Final_Target_Model_Ranking.csv", index=False)
    best_model_per_target.to_csv(OUTPUT_DIR / "11_Best_Model_Per_Target.csv", index=False)
    feature_importance_all.to_csv(OUTPUT_DIR / "12_Feature_Importance_All_Folds.csv", index=False)
    feature_summary.to_csv(OUTPUT_DIR / "13_Feature_Importance_Summary.csv", index=False)
    top_features_for_best_models.to_csv(OUTPUT_DIR / "14_Top_Features_For_Best_Models.csv", index=False)
    target_correlation_matrix.to_csv(OUTPUT_DIR / "15_Target_Correlation_Matrix.csv")
    leakage_audit_df.to_csv(OUTPUT_DIR / "19_Leakage_Audit.csv", index=False)
    baseline_comparison.to_csv(OUTPUT_DIR / "20_Comparison_To_Previous_Baseline.csv", index=False)

    main_table, best_table, feature_table = make_publication_tables(
        final_ranking=final_ranking,
        best_model_per_target=best_model_per_target,
        top_features=top_features_for_best_models,
    )

    write_markdown_and_latex_tables(
        output_dir=OUTPUT_DIR,
        main_table=main_table,
        best_table=best_table,
        feature_table=feature_table,
    )

    fit_and_save_final_models(
        df=df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        output_dir=OUTPUT_DIR,
    )

    write_readme(OUTPUT_DIR)

    print_section("SAVED FILES")

    saved_files = [
        "00_Run_Metadata.json",
        "01_Target_Source_Columns.csv",
        "02_Target_Definitions.csv",
        "03_Target_Overview.csv",
        "04_Phenotype_Counts_By_Target.csv",
        "05_Feature_Screening.csv",
        "06_Numeric_Features_Used.csv",
        "07_LOPO_All_Predictions.csv",
        "08_LOPO_Fold_Metrics_All.csv",
        "09_LOPO_Model_Comparison_With_Statistics.csv",
        "10_Final_Target_Model_Ranking.csv",
        "11_Best_Model_Per_Target.csv",
        "12_Feature_Importance_All_Folds.csv",
        "13_Feature_Importance_Summary.csv",
        "14_Top_Features_For_Best_Models.csv",
        "15_Target_Correlation_Matrix.csv",
        "16_Publication_Tables.md",
        "17_Publication_Tables.tex",
        "18_Final_Model_Manifest.csv",
        "19_Leakage_Audit.csv",
        "20_Comparison_To_Previous_Baseline.csv",
        "README_Model1.txt",
    ]

    for file_name in saved_files:
        file_path = OUTPUT_DIR / file_name
        if file_path.exists():
            print(f"[SAVED] {file_path}")

    if SAVE_FINAL_MODELS:
        print(f"[SAVED] {OUTPUT_DIR / 'models'}")

    print_section("INTERPRETATION GUIDE")

    print(
        """
Use these files for interpretation:

1. Main model comparison:
   Model1/09_LOPO_Model_Comparison_With_Statistics.csv

2. Final sorted ranking:
   Model1/10_Final_Target_Model_Ranking.csv

3. Best model per target:
   Model1/11_Best_Model_Per_Target.csv

4. Feature importance:
   Model1/13_Feature_Importance_Summary.csv
   Model1/14_Top_Features_For_Best_Models.csv

5. Leakage audit:
   Model1/19_Leakage_Audit.csv

6. Comparison to previous Ridge-only baseline:
   Model1/20_Comparison_To_Previous_Baseline.csv

Important:
   If gap-quality models are no longer perfect after this fix, that confirms
   the previous perfect result was leakage.

Recommended manuscript metrics:
   Spearman rank correlation with 95% CI
   Kendall tau
   Top1 accuracy
   Top3 overlap
   NDCG@3
   Mean absolute rank error
   p-value versus random ranking
"""
    )

    print_section("DONE")


if __name__ == "__main__":
    main()