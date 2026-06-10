#!/usr/bin/env python3

"""
Analysis12-Correlation.py

Purpose:
    Correlate upstream feature columns with PLINK pure PRS performance.

Targets used:
    - plink_Train_pure_prs_mean
    - plink_Test_pure_prs_mean
    - plink_generalisation_gap

Important:
    - Nothing is saved.
    - Only prints to screen.
    - Excludes all plink_* columns from candidate features.
    - Excludes all prsice2_* columns from candidate features.
    - Excludes null model and best model PLINK performance.
    - Converts numeric-looking strings to numeric.
    - Ignores true string columns.
    - Calculates Spearman correlation.
    - Applies Benjamini-Hochberg FDR correction.
    - Prints complete FDR-significant results.
"""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


INPUT_FILE = "AllFeatures_PRS_Merged.csv"

MIN_COMPLETE_ROWS = 10

RAW_P_CUTOFF = 0.05
FDR_CUTOFF = 0.05

# Set this to False if you want to print all correlations, including non-significant.
PRINT_ONLY_FDR_SIGNIFICANT = True

# Set to None to print complete significant list.
TOP_N = None

# Set this to True if you want to remove feature9 PRS-readiness/clumping features.
# For discovery, I am keeping it False.
EXCLUDE_FEATURE9 = False


PLINK_TARGET_COLUMNS = [
    "plink_Train_pure_prs_mean",
    "plink_Test_pure_prs_mean",
    "plink_generalisation_gap",
]


def clean_numeric(series):
    """
    Convert a column to numeric safely.

    Handles:
        - native numeric columns
        - boolean columns
        - numeric strings
        - percentages
        - comma-separated numbers
        - True/False and Yes/No values

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

    Returns q-values in the original order.
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

    # Enforce monotonicity from largest to smallest p-value
    ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
    ranked_q = np.minimum(ranked_q, 1.0)

    valid_q = np.empty_like(valid_p)
    valid_q[order] = ranked_q

    q_values[valid_mask] = valid_q

    return q_values


def significance_label(p_value, q_value):
    if pd.isna(q_value):
        return "NA"

    if q_value < 0.001:
        return "FDR SIGNIFICANT ***"

    if q_value < 0.01:
        return "FDR SIGNIFICANT **"

    if q_value < 0.05:
        return "FDR SIGNIFICANT *"

    if p_value < 0.05:
        return "raw p significant only"

    return "not significant"


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
    Exclude columns that should not be candidate explanatory features.
    """

    if column_name in target_columns:
        return True

    if column_name == "job_index":
        return True

    # Critical:
    # Do not use any PLINK result/performance columns as predictors.
    if column_name.startswith("plink_"):
        return True

    # Critical:
    # Do not use PRSice-2 performance/result columns as predictors.
    if column_name.startswith("prsice2_"):
        return True

    # Optional:
    # Remove feature9 if you want only upstream QC/features.
    if EXCLUDE_FEATURE9 and column_name.startswith("feature9_"):
        return True

    return False


def classify_candidate_columns(df, target_columns):
    """
    Classify columns into:
        - total columns
        - target columns
        - excluded candidate columns
        - native numeric columns
        - string/object columns
        - string columns transformed to numeric
        - string columns ignored
        - final numeric candidate columns
    """

    target_set = set(target_columns)

    excluded_columns = []
    candidate_columns = []

    native_numeric_columns = []
    native_string_columns = []

    transformed_to_numeric_columns = []
    ignored_string_columns = []

    final_numeric_candidate_columns = []

    numeric_cache = {}

    for col in df.columns:
        if col in target_set:
            continue

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
        "total_columns": list(df.columns),
        "target_columns": target_columns,
        "excluded_columns": excluded_columns,
        "candidate_columns": candidate_columns,
        "native_numeric_columns": native_numeric_columns,
        "native_string_columns": native_string_columns,
        "transformed_to_numeric_columns": transformed_to_numeric_columns,
        "ignored_string_columns": ignored_string_columns,
        "final_numeric_candidate_columns": final_numeric_candidate_columns,
        "numeric_cache": numeric_cache,
    }


def print_column_summary(info):
    print()
    print("=" * 100)
    print("COLUMN SUMMARY BEFORE CORRELATION")
    print("=" * 100)
    print(f"[TOTAL COLUMNS IN FILE]                    {len(info['total_columns'])}")
    print(f"[PLINK PURE PRS TARGET COLUMNS]             {len(info['target_columns'])}")
    print(f"[EXCLUDED COLUMNS]                          {len(info['excluded_columns'])}")
    print(f"[CANDIDATE COLUMNS AFTER EXCLUSION]         {len(info['candidate_columns'])}")
    print(f"[NATIVE NUMERIC CANDIDATE COLUMNS]          {len(info['native_numeric_columns'])}")
    print(f"[NATIVE STRING / OBJECT CANDIDATE COLUMNS]  {len(info['native_string_columns'])}")
    print(f"[STRING COLUMNS TRANSFORMED TO NUMERIC]     {len(info['transformed_to_numeric_columns'])}")
    print(f"[STRING COLUMNS IGNORED AS NON-NUMERIC]     {len(info['ignored_string_columns'])}")
    print(f"[FINAL NUMERIC CANDIDATE COLUMNS USED]      {len(info['final_numeric_candidate_columns'])}")

    print()
    print("[PLINK PURE PRS TARGET COLUMNS]")
    for col in info["target_columns"]:
        print(f"  - {col}")

    print()
    print("[IMPORTANT EXCLUSIONS]")
    print("  - All plink_* columns are excluded as candidate features.")
    print("  - All prsice2_* columns are excluded as candidate features.")
    print("  - Null-model and best-model PLINK columns are not used as targets.")

    print()
    print("[STRING COLUMNS TRANSFORMED TO NUMERIC]")
    if len(info["transformed_to_numeric_columns"]) == 0:
        print("  None")
    else:
        for col in info["transformed_to_numeric_columns"]:
            print(f"  - {col}")

    print()
    print("[STRING / NON-NUMERIC COLUMNS IGNORED]")
    if len(info["ignored_string_columns"]) == 0:
        print("  None")
    else:
        for col in info["ignored_string_columns"]:
            print(f"  - {col}")

    print("=" * 100)


def run_correlations(df, info):
    target_columns = info["target_columns"]
    feature_columns = info["final_numeric_candidate_columns"]

    numeric_cache = info["numeric_cache"].copy()

    for target in target_columns:
        numeric_cache[target] = clean_numeric(df[target])

    all_results = []

    for target in target_columns:
        y = numeric_cache[target]

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

            all_results.append({
                "target": target,
                "feature_column": feature,
                "n": n,
                "rho": float(rho),
                "abs_rho": abs(float(rho)),
                "p_value": float(p_value),
                "direction": direction_label(float(rho)),
            })

    results_df = pd.DataFrame(all_results)

    if results_df.empty:
        return results_df

    # FDR correction across ALL tests together.
    # This is stricter and better for discovery across multiple targets.
    results_df["q_value_global_fdr"] = benjamini_hochberg_fdr(
        results_df["p_value"].values
    )

    # FDR correction within each target as well.
    results_df["q_value_target_fdr"] = np.nan

    for target in results_df["target"].unique():
        mask = results_df["target"] == target
        results_df.loc[mask, "q_value_target_fdr"] = benjamini_hochberg_fdr(
            results_df.loc[mask, "p_value"].values
        )

    results_df["significance"] = results_df.apply(
        lambda row: significance_label(
            row["p_value"],
            row["q_value_global_fdr"],
        ),
        axis=1,
    )

    results_df = results_df.sort_values(
        by=["target", "abs_rho"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return results_df


def print_results(results_df):
    if results_df.empty:
        print()
        print("No valid correlations were calculated.")
        return

    print()
    print("=" * 100)
    print("OVERALL CORRELATION SUMMARY")
    print("=" * 100)
    print(f"[TOTAL VALID CORRELATIONS]              {len(results_df)}")
    print(f"[RAW p < {RAW_P_CUTOFF}]                         {(results_df['p_value'] < RAW_P_CUTOFF).sum()}")
    print(f"[GLOBAL FDR q < {FDR_CUTOFF}]                    {(results_df['q_value_global_fdr'] < FDR_CUTOFF).sum()}")
    print(f"[TARGET-WISE FDR q < {FDR_CUTOFF}]               {(results_df['q_value_target_fdr'] < FDR_CUTOFF).sum()}")
    print("=" * 100)

    if PRINT_ONLY_FDR_SIGNIFICANT:
        print_df = results_df[
            results_df["q_value_global_fdr"] < FDR_CUTOFF
        ].copy()
    else:
        print_df = results_df.copy()

    if print_df.empty:
        print()
        print("No globally FDR-significant correlations found.")
        print("Try checking target-wise FDR or increasing sample size.")
        return

    for target in print_df["target"].unique():
        target_df = print_df[print_df["target"] == target].copy()

        target_df = target_df.sort_values(
            by="abs_rho",
            ascending=False
        ).reset_index(drop=True)

        if TOP_N is not None:
            target_df = target_df.head(TOP_N)

        print()
        print("=" * 100)
        print(f"PLINK TARGET: {target}")
        print("=" * 100)

        total_for_target = len(results_df[results_df["target"] == target])
        raw_sig_for_target = int(
            (
                (results_df["target"] == target)
                & (results_df["p_value"] < RAW_P_CUTOFF)
            ).sum()
        )
        global_fdr_sig_for_target = int(
            (
                (results_df["target"] == target)
                & (results_df["q_value_global_fdr"] < FDR_CUTOFF)
            ).sum()
        )
        target_fdr_sig_for_target = int(
            (
                (results_df["target"] == target)
                & (results_df["q_value_target_fdr"] < FDR_CUTOFF)
            ).sum()
        )

        print(f"[VALID CORRELATIONS FOR TARGET]       {total_for_target}")
        print(f"[RAW SIGNIFICANT FOR TARGET]          {raw_sig_for_target}")
        print(f"[GLOBAL FDR SIGNIFICANT FOR TARGET]   {global_fdr_sig_for_target}")
        print(f"[TARGET FDR SIGNIFICANT FOR TARGET]   {target_fdr_sig_for_target}")

        print()
        print("Complete ranked FDR-significant list:")
        print()

        for i, row in target_df.iterrows():
            rank = i + 1

            print(
                f"Rank {rank:03d} | "
                f"{row['feature_column']} | "
                f"rho={row['rho']:.4f} | "
                f"p={row['p_value']:.4g} | "
                f"global_FDR_q={row['q_value_global_fdr']:.4g} | "
                f"target_FDR_q={row['q_value_target_fdr']:.4g} | "
                f"n={int(row['n'])} | "
                f"{row['direction']} | "
                f"{row['significance']}"
            )


def print_interpretation_guide():
    print()
    print("=" * 100)
    print("INTERPRETATION GUIDE")
    print("=" * 100)
    print("rho > 0:")
    print("  Higher feature value is associated with higher PLINK pure PRS performance.")
    print()
    print("rho < 0:")
    print("  Higher feature value is associated with lower PLINK pure PRS performance.")
    print()
    print("global_FDR_q < 0.05:")
    print("  Significant after correcting across all feature-target tests.")
    print()
    print("target_FDR_q < 0.05:")
    print("  Significant after correcting within that PLINK target only.")
    print()
    print("Recommended interpretation:")
    print("  Use global_FDR_q for strict discovery.")
    print("  Use target_FDR_q for target-specific exploration.")
    print("=" * 100)


def main():
    print("=" * 100)
    print("SPEARMAN CORRELATION WITH FDR: FEATURES vs PLINK PURE PRS PERFORMANCE")
    print("=" * 100)

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    print(f"[INPUT FILE] {INPUT_FILE}")
    print(f"[ROWS]       {df.shape[0]}")
    print(f"[COLUMNS]    {df.shape[1]}")

    available_targets = [
        col for col in PLINK_TARGET_COLUMNS
        if col in df.columns
    ]

    missing_targets = [
        col for col in PLINK_TARGET_COLUMNS
        if col not in df.columns
    ]

    if missing_targets:
        print()
        print("[WARNING] Missing expected PLINK target columns:")
        for col in missing_targets:
            print(f"  - {col}")

    if len(available_targets) == 0:
        raise ValueError("No PLINK pure PRS target columns were found.")

    info = classify_candidate_columns(df, available_targets)
    print_column_summary(info)

    print()
    print("=" * 100)
    print("CORRELATION SETTINGS")
    print("=" * 100)
    print(f"[METHOD]                         Spearman correlation")
    print(f"[MIN COMPLETE ROWS REQUIRED]     {MIN_COMPLETE_ROWS}")
    print(f"[RAW P CUTOFF]                   p < {RAW_P_CUTOFF}")
    print(f"[FDR CUTOFF]                     q < {FDR_CUTOFF}")
    print(f"[FDR METHOD]                     Benjamini-Hochberg")
    print(f"[FDR SCOPE]                      Global across all feature-target tests")
    print(f"[PRINT ONLY FDR SIGNIFICANT]     {PRINT_ONLY_FDR_SIGNIFICANT}")
    print(f"[EXCLUDE FEATURE9]               {EXCLUDE_FEATURE9}")
    print("=" * 100)

    results_df = run_correlations(df, info)

    print_results(results_df)
    print_interpretation_guide()

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()