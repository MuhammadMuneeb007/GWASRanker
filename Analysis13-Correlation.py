#!/usr/bin/env python3

"""
Analysis13-Correlation.py

Purpose:
    Save Spearman correlation values for all valid GWAS-level features
    against three PLINK pure PRS outcomes.

Outputs:
    1. Plink_Correlation_plink_Train_pure_prs_mean.csv
    2. Plink_Correlation_plink_Test_pure_prs_mean.csv
    3. Plink_Correlation_plink_generalisation_gap.csv

Important:
    - Saves all valid correlations, not only significant ones.
    - Excludes all plink_* columns as candidate features.
    - Excludes all prsice2_* columns as candidate features.
    - Uses only pure train PRS, pure test PRS, and generalisation gap as targets.
    - Applies Benjamini-Hochberg FDR separately within each target file.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


INPUT_FILE = "AllFeatures_PRS_Merged.csv"
OUTPUT_DIR = Path("Plink_Pure_PRS_Correlation_Results")

MIN_COMPLETE_ROWS = 10
FDR_CUTOFF = 0.05

PLINK_TARGET_COLUMNS = [
    "plink_Train_pure_prs_mean",
    "plink_Test_pure_prs_mean",
    "plink_generalisation_gap",
]


def clean_numeric(series):
    """
    Convert a column to numeric safely.
    Non-numeric strings become NaN.
    """

    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    x = series.copy()
    x = x.astype(str).str.strip()

    x = x.replace({
        "": np.nan,
        "nan": np.nan,
        "NaN": np.nan,
        "None": np.nan,
        "none": np.nan,
        "NA": np.nan,
        "N/A": np.nan,
        "na": np.nan,
        "n/a": np.nan,

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

        "inf": np.nan,
        "-inf": np.nan,
        "Infinity": np.nan,
        "-Infinity": np.nan,
    })

    x = x.astype(str).str.strip()
    x = x.str.replace(",", "", regex=False)
    x = x.str.replace("%", "", regex=False)

    return pd.to_numeric(x, errors="coerce")


def benjamini_hochberg_fdr(p_values):
    """
    Benjamini-Hochberg FDR correction.
    Returns q-values in original order.
    """

    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)

    q_values = np.full(n, np.nan)

    valid_mask = ~np.isnan(p_values)
    valid_p = p_values[valid_mask]

    if len(valid_p) == 0:
        return q_values

    order = np.argsort(valid_p)
    ranked_p = valid_p[order]

    m = len(ranked_p)
    ranked_q = ranked_p * m / np.arange(1, m + 1)

    ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
    ranked_q = np.minimum(ranked_q, 1.0)

    valid_q = np.empty_like(valid_p)
    valid_q[order] = ranked_q

    q_values[valid_mask] = valid_q

    return q_values


def direction_label(rho):
    if pd.isna(rho):
        return "NA"
    if rho > 0:
        return "positive"
    if rho < 0:
        return "negative"
    return "zero"


def should_exclude_candidate_column(column_name, target_columns):
    """
    Exclude columns that should not be used as candidate explanatory features.
    """

    if column_name in target_columns:
        return True

    if column_name == "job_index":
        return True

    if column_name.startswith("plink_"):
        return True

    if column_name.startswith("prsice2_"):
        return True

    return False


def classify_candidate_columns(df, target_columns):
    """
    Identify all usable numeric/numeric-convertible candidate features.
    """

    excluded_columns = []
    candidate_columns = []

    native_numeric_columns = []
    native_string_columns = []

    transformed_to_numeric_columns = []
    ignored_string_columns = []

    final_numeric_candidate_columns = []
    numeric_cache = {}

    for col in df.columns:
        if should_exclude_candidate_column(col, target_columns):
            excluded_columns.append(col)
            continue

        candidate_columns.append(col)

        original_series = df[col]

        is_native_numeric = (
            pd.api.types.is_numeric_dtype(original_series)
            or pd.api.types.is_bool_dtype(original_series)
        )

        if is_native_numeric:
            native_numeric_columns.append(col)
        else:
            native_string_columns.append(col)

        numeric_values = clean_numeric(original_series)
        numeric_cache[col] = numeric_values

        non_missing_count = int(numeric_values.notna().sum())
        unique_count = int(numeric_values.dropna().nunique())

        if not is_native_numeric and non_missing_count > 0:
            transformed_to_numeric_columns.append(col)

        if non_missing_count < MIN_COMPLETE_ROWS or unique_count <= 1:
            if not is_native_numeric:
                ignored_string_columns.append(col)
            continue

        final_numeric_candidate_columns.append(col)

    return {
        "excluded_columns": excluded_columns,
        "candidate_columns": candidate_columns,
        "native_numeric_columns": native_numeric_columns,
        "native_string_columns": native_string_columns,
        "transformed_to_numeric_columns": transformed_to_numeric_columns,
        "ignored_string_columns": ignored_string_columns,
        "final_numeric_candidate_columns": final_numeric_candidate_columns,
        "numeric_cache": numeric_cache,
    }


def calculate_correlations_for_target(df, target, feature_columns, numeric_cache):
    """
    Calculate Spearman correlation between one target and all candidate features.
    """

    y = clean_numeric(df[target])

    results = []

    for feature in feature_columns:
        x = numeric_cache[feature]

        valid = x.notna() & y.notna()
        n = int(valid.sum())

        if n < MIN_COMPLETE_ROWS:
            continue

        x_valid = x[valid]
        y_valid = y[valid]

        if x_valid.nunique() <= 1:
            continue

        if y_valid.nunique() <= 1:
            continue

        rho, p_value = spearmanr(x_valid, y_valid)

        if pd.isna(rho) or pd.isna(p_value):
            continue

        results.append({
            "target": target,
            "feature_column": feature,
            "n": n,
            "rho": float(rho),
            "abs_rho": abs(float(rho)),
            "p_value": float(p_value),
            "direction": direction_label(float(rho)),
        })

    result_df = pd.DataFrame(results)

    if result_df.empty:
        return result_df

    result_df["fdr_q_value"] = benjamini_hochberg_fdr(result_df["p_value"].values)
    result_df["significant_fdr"] = result_df["fdr_q_value"] < FDR_CUTOFF

    result_df = result_df.sort_values(
        by="abs_rho",
        ascending=False
    ).reset_index(drop=True)

    result_df.insert(0, "rank_by_abs_rho", np.arange(1, len(result_df) + 1))

    return result_df


def print_summary(df, info, available_targets):
    print("=" * 100)
    print("SAVE SPEARMAN CORRELATIONS: FEATURES vs PLINK PURE PRS TARGETS")
    print("=" * 100)
    print(f"[INPUT FILE] {INPUT_FILE}")
    print(f"[ROWS]       {df.shape[0]}")
    print(f"[COLUMNS]    {df.shape[1]}")

    print()
    print("[TARGET COLUMNS]")
    for target in available_targets:
        print(f"  - {target}")

    print()
    print("=" * 100)
    print("COLUMN SUMMARY")
    print("=" * 100)
    print(f"[TOTAL COLUMNS IN FILE]                    {df.shape[1]}")
    print(f"[PLINK TARGET COLUMNS]                      {len(available_targets)}")
    print(f"[EXCLUDED COLUMNS]                          {len(info['excluded_columns'])}")
    print(f"[CANDIDATE COLUMNS AFTER EXCLUSION]         {len(info['candidate_columns'])}")
    print(f"[NATIVE NUMERIC CANDIDATE COLUMNS]          {len(info['native_numeric_columns'])}")
    print(f"[NATIVE STRING / OBJECT CANDIDATE COLUMNS]  {len(info['native_string_columns'])}")
    print(f"[STRING COLUMNS TRANSFORMED TO NUMERIC]     {len(info['transformed_to_numeric_columns'])}")
    print(f"[STRING COLUMNS IGNORED AS NON-NUMERIC]     {len(info['ignored_string_columns'])}")
    print(f"[FINAL NUMERIC CANDIDATE COLUMNS USED]      {len(info['final_numeric_candidate_columns'])}")
    print("=" * 100)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    available_targets = [c for c in PLINK_TARGET_COLUMNS if c in df.columns]
    missing_targets = [c for c in PLINK_TARGET_COLUMNS if c not in df.columns]

    if missing_targets:
        print("[WARNING] Missing target columns:")
        for col in missing_targets:
            print(f"  - {col}")

    if not available_targets:
        raise ValueError("No PLINK target columns were found.")

    info = classify_candidate_columns(df, available_targets)
    print_summary(df, info, available_targets)

    feature_columns = info["final_numeric_candidate_columns"]
    numeric_cache = info["numeric_cache"]

    total_correlations = 0
    total_fdr_significant = 0

    print()
    print("=" * 100)
    print("WRITING TARGET-SPECIFIC CORRELATION FILES")
    print("=" * 100)

    for target in available_targets:
        result_df = calculate_correlations_for_target(
            df=df,
            target=target,
            feature_columns=feature_columns,
            numeric_cache=numeric_cache,
        )

        safe_target = target.replace("/", "_").replace("\\", "_").replace(" ", "_")
        output_file = OUTPUT_DIR / f"Plink_Correlation_{safe_target}.csv"

        result_df.to_csv(output_file, index=False)

        n_valid = len(result_df)
        n_fdr = int(result_df["significant_fdr"].sum()) if not result_df.empty else 0

        total_correlations += n_valid
        total_fdr_significant += n_fdr

        print()
        print(f"[TARGET] {target}")
        print(f"[VALID CORRELATIONS]     {n_valid}")
        print(f"[FDR SIGNIFICANT q<0.05] {n_fdr}")
        print(f"[SAVED] {output_file}")

        if not result_df.empty:
            print("[TOP 10 BY ABSOLUTE RHO]")
            preview_cols = [
                "rank_by_abs_rho",
                "feature_column",
                "rho",
                "p_value",
                "fdr_q_value",
                "n",
                "direction",
                "significant_fdr",
            ]
            print(result_df[preview_cols].head(10).to_string(index=False))

    print()
    print("=" * 100)
    print("OVERALL SUMMARY")
    print("=" * 100)
    print(f"[TARGETS ANALYSED]              {len(available_targets)}")
    print(f"[NUMERIC FEATURES TESTED]       {len(feature_columns)}")
    print(f"[TOTAL VALID CORRELATIONS]      {total_correlations}")
    print(f"[TOTAL FDR SIGNIFICANT]         {total_fdr_significant}")
    print(f"[OUTPUT DIRECTORY]              {OUTPUT_DIR}")
    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()