#!/usr/bin/env python3

"""
Analysis17-ModelDevelopment.py

Purpose:
    Build a phenotype-general GWAS ranking model using AllFeatures_PRS_Merged.csv.

Main idea:
    We are NOT training one model per phenotype.
    We train one general model across phenotypes.

Validation design:
    Leave-one-phenotype-out cross-validation.

For each fold:
    1. Hold out one phenotype completely.
    2. Train model on all other phenotypes.
    3. Predict PRS performance for all GWAS files in held-out phenotype.
    4. Rank held-out GWAS files by predicted performance.
    5. Rank held-out GWAS files by actual PRS performance.
    6. Compare actual rank vs predicted rank.

Input:
    AllFeatures_PRS_Merged.csv

Primary target:
    plink_Test_pure_prs_mean

Outputs:
    GWAS_Ranking_Model_LOPO_Output/
        LOPO_All_Predictions.csv
        LOPO_Ranking_Evaluation_By_Phenotype.csv
        LOPO_Overall_Ranking_Summary.csv
        LOPO_Feature_Weights_All_Folds.csv
        LOPO_Feature_Weights_Mean.csv
        LOPO_Numeric_Features_Used.csv
        LOPO_Model_Input_Table.csv
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, kendalltau

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


warnings.filterwarnings("ignore")


# =============================================================================
# User settings
# =============================================================================

INPUT_FILE = "AllFeatures_PRS_Merged.csv"

OUTPUT_DIR = Path("GWAS_Ranking_Model_LOPO_Output")

TARGET_COLUMN = "plink_Test_pure_prs_mean"

MIN_COMPLETE_ROWS_FOR_FEATURE = 10
MIN_GWAS_IN_TEST_PHENOTYPE = 2

RANDOM_SEED = 42

EXCLUDE_FEATURE9 = False

# Keep this True to avoid leakage.
EXCLUDE_PLINK_COLUMNS = True
EXCLUDE_PRSICE2_COLUMNS = True


# =============================================================================
# Helper functions
# =============================================================================

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

    Handles:
        - numeric values
        - boolean values
        - numeric strings
        - percentages
        - comma-separated numbers
        - yes/no and true/false values
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
    """
    Standardise phenotype and accession ID columns.
    """

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
            "Could not find accessionId / GWAS ID column.\n"
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


def should_exclude_predictor(col):
    """
    Exclude leakage columns and non-feature columns.
    """

    if col in ["phenotype", "accessionId", "actual_performance"]:
        return True

    if col == TARGET_COLUMN:
        return True

    if col == "job_index":
        return True

    if EXCLUDE_PLINK_COLUMNS and col.startswith("plink_"):
        return True

    if EXCLUDE_PRSICE2_COLUMNS and col.startswith("prsice2_"):
        return True

    if EXCLUDE_FEATURE9 and col.startswith("feature9_"):
        return True

    return False


def load_model_table():
    path = Path(INPUT_FILE)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    print(f"[READING] {path}")

    df = pd.read_csv(path, low_memory=False)
    df = standardise_key_columns(df)

    if TARGET_COLUMN not in df.columns:
        plink_cols = [c for c in df.columns if c.startswith("plink_")]
        raise ValueError(
            f"Target column not found: {TARGET_COLUMN}\n\n"
            f"Available plink columns:\n" + "\n".join(plink_cols)
        )

    df["actual_performance"] = clean_numeric(df[TARGET_COLUMN])

    before = len(df)
    df = df.dropna(subset=["actual_performance"]).copy()
    after = len(df)

    print(f"[ROWS BEFORE TARGET FILTER] {before}")
    print(f"[ROWS AFTER TARGET FILTER]  {after}")

    df = df.drop_duplicates(subset=["phenotype", "accessionId"], keep="first").copy()

    return df


def get_numeric_feature_columns(df):
    """
    Keep numeric upstream features only.
    """

    numeric_cols = []

    for col in df.columns:
        if should_exclude_predictor(col):
            continue

        numeric_values = clean_numeric(df[col])

        non_missing = int(numeric_values.notna().sum())
        unique_values = int(numeric_values.dropna().nunique())

        if non_missing >= MIN_COMPLETE_ROWS_FOR_FEATURE and unique_values > 1:
            numeric_cols.append(col)

    return numeric_cols


def prepare_xy(df, feature_cols):
    X = df[feature_cols].copy()

    for col in feature_cols:
        X[col] = clean_numeric(X[col])

    y = clean_numeric(df["actual_performance"])

    return X, y


def train_ridge_model(train_df, feature_cols):
    X_train, y_train = prepare_xy(train_df, feature_cols)

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "ridge",
                RidgeCV(
                    alphas=np.logspace(-4, 4, 100),
                    cv=5,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    return model


def predict_for_test(model, test_df, feature_cols, heldout_phenotype):
    X_test, _ = prepare_xy(test_df, feature_cols)

    out = test_df[["phenotype", "accessionId", "actual_performance"]].copy()
    out["predicted_performance"] = model.predict(X_test)
    out["heldout_phenotype"] = heldout_phenotype

    return out


def assign_ranks_within_phenotype(df):
    """
    Highest actual and predicted performance gets rank 1.
    """

    df = df.copy()

    df["actual_rank"] = (
        df.groupby("phenotype")["actual_performance"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    df["predicted_rank"] = (
        df.groupby("phenotype")["predicted_performance"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    df["absolute_rank_error"] = (
        df["actual_rank"] - df["predicted_rank"]
    ).abs()

    return df


def dcg_at_k(relevance, k):
    relevance = np.asarray(relevance, dtype=float)[:k]

    if len(relevance) == 0:
        return np.nan

    discounts = np.log2(np.arange(2, len(relevance) + 2))
    return np.sum(relevance / discounts)


def ndcg_at_k(actual_performance, predicted_score, k=3):
    """
    NDCG@k using actual performance as relevance.
    """

    actual_performance = np.asarray(actual_performance, dtype=float)
    predicted_score = np.asarray(predicted_score, dtype=float)

    if len(actual_performance) < 2:
        return np.nan

    # Shift relevance to non-negative values.
    min_val = np.nanmin(actual_performance)
    relevance = actual_performance - min_val

    # If all relevance values are equal, NDCG is not informative.
    if np.nanmax(relevance) == 0:
        return np.nan

    predicted_order = np.argsort(-predicted_score)
    ideal_order = np.argsort(-actual_performance)

    dcg = dcg_at_k(relevance[predicted_order], k)
    idcg = dcg_at_k(relevance[ideal_order], k)

    if idcg == 0 or pd.isna(idcg):
        return np.nan

    return dcg / idcg


def evaluate_heldout_phenotype(pred_df):
    """
    Evaluate ranking for one held-out phenotype.
    """

    sub = pred_df.copy()

    if len(sub) < MIN_GWAS_IN_TEST_PHENOTYPE:
        return None

    actual_rank = sub["actual_rank"].to_numpy()
    predicted_rank = sub["predicted_rank"].to_numpy()

    spearman_r, spearman_p = spearmanr(actual_rank, predicted_rank)
    kendall_r, kendall_p = kendalltau(actual_rank, predicted_rank)

    true_best = set(sub.loc[sub["actual_rank"] == 1, "accessionId"])
    pred_best = set(sub.loc[sub["predicted_rank"] == 1, "accessionId"])

    top1_correct = int(len(true_best.intersection(pred_best)) > 0)

    true_top3 = set(sub.sort_values("actual_rank").head(3)["accessionId"])
    pred_top3 = set(sub.sort_values("predicted_rank").head(3)["accessionId"])

    top3_overlap_count = len(true_top3.intersection(pred_top3))
    top3_overlap_fraction = top3_overlap_count / max(1, len(true_top3))

    y_true = sub["actual_performance"]
    y_pred = sub["predicted_performance"]

    regression_r = np.nan
    regression_p = np.nan
    if len(sub) >= 3 and y_true.nunique() > 1 and pd.Series(y_pred).nunique() > 1:
        regression_r, regression_p = spearmanr(y_true, y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        "heldout_phenotype": sub["phenotype"].iloc[0],
        "n_test_gwas": len(sub),
        "spearman_rank_r": spearman_r,
        "spearman_rank_p": spearman_p,
        "kendall_tau": kendall_r,
        "kendall_tau_p": kendall_p,
        "top1_correct": top1_correct,
        "top3_overlap_count": top3_overlap_count,
        "top3_overlap_fraction": top3_overlap_fraction,
        "mean_absolute_rank_error": sub["absolute_rank_error"].mean(),
        "median_absolute_rank_error": sub["absolute_rank_error"].median(),
        "max_absolute_rank_error": sub["absolute_rank_error"].max(),
        "ndcg_at_3": ndcg_at_k(y_true, y_pred, k=3),
        "ndcg_at_5": ndcg_at_k(y_true, y_pred, k=5),
        "performance_spearman_r": regression_r,
        "performance_spearman_p": regression_p,
        "performance_mae": mean_absolute_error(y_true, y_pred),
        "performance_rmse": rmse,
        "performance_r2": r2_score(y_true, y_pred) if len(sub) > 1 else np.nan,
    }


def extract_fold_weights(model, feature_cols, heldout_phenotype):
    ridge = model.named_steps["ridge"]
    alpha = ridge.alpha_

    weights = pd.DataFrame(
        {
            "heldout_phenotype": heldout_phenotype,
            "feature": feature_cols,
            "weight": ridge.coef_,
        }
    )

    weights["abs_weight"] = weights["weight"].abs()
    weights["ridge_alpha"] = alpha

    return weights.sort_values("abs_weight", ascending=False)


def summarise_weights(all_weights):
    summary = (
        all_weights
        .groupby("feature", as_index=False)
        .agg(
            mean_weight=("weight", "mean"),
            median_weight=("weight", "median"),
            mean_abs_weight=("abs_weight", "mean"),
            max_abs_weight=("abs_weight", "max"),
            n_folds=("heldout_phenotype", "nunique"),
        )
    )

    summary = summary.sort_values("mean_abs_weight", ascending=False).reset_index(drop=True)

    return summary


def print_section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_section("LEAVE-ONE-PHENOTYPE-OUT GWAS RANKING MODEL")

    df = load_model_table()

    print(f"[ROWS USED]    {len(df)}")
    print(f"[PHENOTYPES]   {df['phenotype'].nunique()}")
    print(f"[GWAS FILES]   {df['accessionId'].nunique()}")
    print(f"[TARGET]       {TARGET_COLUMN}")

    phenotype_counts = (
        df.groupby("phenotype")
        .size()
        .reset_index(name="n_gwas")
        .sort_values("phenotype")
    )

    print_section("GWAS FILES PER PHENOTYPE")
    print(phenotype_counts.to_string(index=False))

    feature_cols = get_numeric_feature_columns(df)

    print_section("FEATURE SELECTION")
    print(f"[NUMERIC FEATURES USED] {len(feature_cols)}")

    if len(feature_cols) == 0:
        raise RuntimeError("No usable numeric feature columns found.")

    all_predictions = []
    all_evaluations = []
    all_weights = []

    phenotypes = sorted(df["phenotype"].dropna().unique())

    print_section("LOPO TRAINING")

    for fold_idx, heldout_phenotype in enumerate(phenotypes, start=1):
        train_df = df[df["phenotype"] != heldout_phenotype].copy()
        test_df = df[df["phenotype"] == heldout_phenotype].copy()

        print()
        print("-" * 100)
        print(f"[FOLD {fold_idx}/{len(phenotypes)}] Held-out phenotype: {heldout_phenotype}")
        print(f"[TRAIN ROWS] {len(train_df)}")
        print(f"[TEST ROWS]  {len(test_df)}")

        if len(test_df) < MIN_GWAS_IN_TEST_PHENOTYPE:
            print("[SKIPPED] Not enough GWAS files in held-out phenotype.")
            continue

        model = train_ridge_model(train_df, feature_cols)

        fold_pred = predict_for_test(
            model=model,
            test_df=test_df,
            feature_cols=feature_cols,
            heldout_phenotype=heldout_phenotype,
        )

        fold_pred = assign_ranks_within_phenotype(fold_pred)

        fold_eval = evaluate_heldout_phenotype(fold_pred)

        fold_weights = extract_fold_weights(
            model=model,
            feature_cols=feature_cols,
            heldout_phenotype=heldout_phenotype,
        )

        all_predictions.append(fold_pred)
        all_weights.append(fold_weights)

        if fold_eval is not None:
            all_evaluations.append(fold_eval)

            print(
                f"[RESULT] Spearman rank r={fold_eval['spearman_rank_r']:.4f}, "
                f"Kendall tau={fold_eval['kendall_tau']:.4f}, "
                f"Top1={fold_eval['top1_correct']}, "
                f"Top3 overlap={fold_eval['top3_overlap_fraction']:.4f}, "
                f"MARE={fold_eval['mean_absolute_rank_error']:.4f}"
            )

    if not all_predictions:
        raise RuntimeError("No LOPO predictions were generated.")

    predictions = pd.concat(all_predictions, ignore_index=True)
    evaluations = pd.DataFrame(all_evaluations)
    weights_all = pd.concat(all_weights, ignore_index=True)
    weights_mean = summarise_weights(weights_all)

    print_section("LOPO RANKING EVALUATION BY PHENOTYPE")
    print(evaluations.to_string(index=False))

    print_section("OVERALL LOPO SUMMARY")

    overall_summary = {
        "n_heldout_phenotypes": evaluations["heldout_phenotype"].nunique(),
        "total_test_gwas": evaluations["n_test_gwas"].sum(),
        "mean_spearman_rank_r": evaluations["spearman_rank_r"].mean(),
        "median_spearman_rank_r": evaluations["spearman_rank_r"].median(),
        "mean_kendall_tau": evaluations["kendall_tau"].mean(),
        "median_kendall_tau": evaluations["kendall_tau"].median(),
        "top1_accuracy": evaluations["top1_correct"].mean(),
        "mean_top3_overlap_fraction": evaluations["top3_overlap_fraction"].mean(),
        "mean_absolute_rank_error": evaluations["mean_absolute_rank_error"].mean(),
        "median_absolute_rank_error": evaluations["median_absolute_rank_error"].median(),
        "mean_ndcg_at_3": evaluations["ndcg_at_3"].mean(),
        "mean_ndcg_at_5": evaluations["ndcg_at_5"].mean(),
        "mean_performance_mae": evaluations["performance_mae"].mean(),
        "mean_performance_rmse": evaluations["performance_rmse"].mean(),
    }

    overall_summary_df = pd.DataFrame([overall_summary])

    for key, value in overall_summary.items():
        print(f"{key}: {value}")

    print_section("TOP 100 MEAN FEATURE WEIGHTS")
    print(weights_mean.head(100).to_string(index=False))

    # Save model input table
    model_input = df[["phenotype", "accessionId", "actual_performance"] + feature_cols].copy()

    model_input.to_csv(
        OUTPUT_DIR / "LOPO_Model_Input_Table.csv",
        index=False,
    )

    pd.DataFrame({"feature": feature_cols}).to_csv(
        OUTPUT_DIR / "LOPO_Numeric_Features_Used.csv",
        index=False,
    )

    predictions.to_csv(
        OUTPUT_DIR / "LOPO_All_Predictions.csv",
        index=False,
    )

    evaluations.to_csv(
        OUTPUT_DIR / "LOPO_Ranking_Evaluation_By_Phenotype.csv",
        index=False,
    )

    overall_summary_df.to_csv(
        OUTPUT_DIR / "LOPO_Overall_Ranking_Summary.csv",
        index=False,
    )

    weights_all.to_csv(
        OUTPUT_DIR / "LOPO_Feature_Weights_All_Folds.csv",
        index=False,
    )

    weights_mean.to_csv(
        OUTPUT_DIR / "LOPO_Feature_Weights_Mean.csv",
        index=False,
    )

    print_section("SAVED FILES")
    print(f"[MODEL INPUT]        {OUTPUT_DIR / 'LOPO_Model_Input_Table.csv'}")
    print(f"[FEATURES USED]      {OUTPUT_DIR / 'LOPO_Numeric_Features_Used.csv'}")
    print(f"[ALL PREDICTIONS]    {OUTPUT_DIR / 'LOPO_All_Predictions.csv'}")
    print(f"[EVALUATION]         {OUTPUT_DIR / 'LOPO_Ranking_Evaluation_By_Phenotype.csv'}")
    print(f"[OVERALL SUMMARY]    {OUTPUT_DIR / 'LOPO_Overall_Ranking_Summary.csv'}")
    print(f"[ALL WEIGHTS]        {OUTPUT_DIR / 'LOPO_Feature_Weights_All_Folds.csv'}")
    print(f"[MEAN WEIGHTS]       {OUTPUT_DIR / 'LOPO_Feature_Weights_Mean.csv'}")

    print_section("DONE")


if __name__ == "__main__":
    main()