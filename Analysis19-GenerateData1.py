#!/usr/bin/env python3

"""
Analysis19-GenerateData1.py

Generate leakage-safe model input data for GWAS ranking.

Purpose
-------
This script reads:

    AllFeatures_PRS_Merged.csv

and saves model-specific files such as:

    Model1/model1_features.csv
    Model1/model1_target.csv

Design
------
Predictors:
    Only the configured feature range is allowed.

Targets:
    1. target_plink_train_pure_prs
    2. target_plink_test_pure_prs
    3. target_plink_pure_prs_gap_quality

Important leakage protection
----------------------------
The model features must NOT include any downstream PRS/performance columns.

Therefore this script excludes:
    - target_* columns
    - plink_* columns
    - prsice2_* columns
    - later feature groups beyond the configured upper bound
    - any non-configured feature columns

The gap target is defined as:

    target_plink_pure_prs_gap_quality = -abs(train_pure_prs - test_pure_prs)

Higher is better:
    - Higher train pure PRS is better.
    - Higher test pure PRS is better.
    - Higher gap quality is better because it means smaller train-test gap.

Rows are retained only if the PLINK train and test pure PRS targets are available.
Features with too much missingness are discarded.
Remaining feature values are saved as numeric columns with NaN retained.
Imputation should be done later inside the modelling cross-validation pipeline.
"""

from pathlib import Path
from datetime import datetime
import json
import platform
import re

import numpy as np
import pandas as pd


# =============================================================================
# User settings
# =============================================================================

INPUT_FILE = "AllFeatures_PRS_Merged.csv"
MODEL_NUMBER = 1
FEATURE_START = 1
FEATURE_END = 6
MAX_FEATURE_NUMBER_KNOWN = 12
OUTPUT_DIR = Path(f"Model{MODEL_NUMBER}")

MODEL_LABEL = f"Model{MODEL_NUMBER}"
MODEL_TAG = f"model{MODEL_NUMBER}"
FEATURE_RANGE_TEXT = f"Feature{FEATURE_START}-Feature{FEATURE_END}"
FEATURE_PREFIXES_ALLOWED = tuple([f"feature{i}_" for i in range(FEATURE_START, FEATURE_END + 1)])
FORBIDDEN_FEATURE_PREFIXES = tuple(
    [f"feature{i}_" for i in range(FEATURE_END + 1, MAX_FEATURE_NUMBER_KNOWN + 1)]
)
ALLOWED_FEATURE_FLAG_COL = f"allowed_prefix_feature{FEATURE_START}_to_feature{FEATURE_END}"
USED_FEATURE_FLAG_COL = f"used_in_{MODEL_TAG}_features"

# Discard features with more than this fraction missing.
# Example: 0.50 means remove columns with >50% missing values.
MAX_FEATURE_MISSING_FRACTION = 0.50

# Minimum number of non-missing rows required for a feature.
MIN_NON_MISSING_ROWS = 10

# Minimum number of unique numeric values required.
MIN_UNIQUE_VALUES = 2

# Keep only rows where all three targets can be constructed.
REQUIRE_COMPLETE_TARGETS = True


# =============================================================================
# Helpers
# =============================================================================

def print_section(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def configure_model(model_number: int, feature_end: int, output_dir: str | Path | None = None) -> None:
    global MODEL_NUMBER, FEATURE_END, OUTPUT_DIR
    global MODEL_LABEL, MODEL_TAG, FEATURE_RANGE_TEXT
    global FEATURE_PREFIXES_ALLOWED, FORBIDDEN_FEATURE_PREFIXES
    global ALLOWED_FEATURE_FLAG_COL, USED_FEATURE_FLAG_COL

    MODEL_NUMBER = int(model_number)
    FEATURE_END = int(feature_end)
    OUTPUT_DIR = Path(output_dir) if output_dir is not None else Path(f"Model{MODEL_NUMBER}")

    MODEL_LABEL = f"Model{MODEL_NUMBER}"
    MODEL_TAG = f"model{MODEL_NUMBER}"
    FEATURE_RANGE_TEXT = f"Feature{FEATURE_START}-Feature{FEATURE_END}"
    FEATURE_PREFIXES_ALLOWED = tuple([f"feature{i}_" for i in range(FEATURE_START, FEATURE_END + 1)])
    FORBIDDEN_FEATURE_PREFIXES = tuple(
        [f"feature{i}_" for i in range(FEATURE_END + 1, MAX_FEATURE_NUMBER_KNOWN + 1)]
    )
    ALLOWED_FEATURE_FLAG_COL = f"allowed_prefix_feature{FEATURE_START}_to_feature{FEATURE_END}"
    USED_FEATURE_FLAG_COL = f"used_in_{MODEL_TAG}_features"


def normalise_column_name(x: str) -> str:
    return str(x).strip().lower().replace("-", "_").replace(" ", "_")


def find_column(df: pd.DataFrame, candidates) -> str | None:
    """
    Find a column in df using case/spacing/hyphen insensitive matching.
    """
    lookup = {normalise_column_name(c): c for c in df.columns}

    for cand in candidates:
        key = normalise_column_name(cand)
        if key in lookup:
            return lookup[key]

    return None


def clean_numeric(series: pd.Series) -> pd.Series:
    """
    Convert numeric-looking values to float safely.

    Handles:
        - commas
        - percentages
        - booleans
        - common missing strings
        - inf values
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)

    if pd.api.types.is_numeric_dtype(series):
        out = pd.to_numeric(series, errors="coerce")
        out = out.replace([np.inf, -np.inf], np.nan)
        return out

    x = series.astype(str).str.strip()

    missing_values = {
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
        "Inf": np.nan,
        "inf": np.nan,
        "-Inf": np.nan,
        "-inf": np.nan,
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

    x = x.replace(missing_values)
    x = x.astype(str).str.strip()
    x = x.str.replace(",", "", regex=False)
    x = x.str.replace("%", "", regex=False)

    out = pd.to_numeric(x, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)

    return out


def standardise_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise phenotype and GWAS accession columns.
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
            "Could not find phenotype column. "
            "Expected one of: phenotype, phenotype_id, trait, GWAS1Pheno, GWASPheno."
        )

    if accession_col is None:
        raise ValueError(
            "Could not find accession/GWAS ID column. "
            "Expected one of: accessionId, accessionID, AccessionID, accession_id, gwas_id, GWAS_ID, GCST."
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


def detect_plink_pure_prs_columns(df: pd.DataFrame) -> tuple[str, str]:
    """
    Detect PLINK train/test pure PRS performance columns.
    """

    train_col = find_column(
        df,
        [
            "plink_Train_pure_prs_mean",
            "plink_train_pure_prs_mean",
            "Train_pure_prs_mean",
            "train_pure_prs_mean",
            "plink_Train_pure_prs",
            "plink_train_pure_prs",
            "Train_pure_prs",
            "train_pure_prs",
        ],
    )

    test_col = find_column(
        df,
        [
            "plink_Test_pure_prs_mean",
            "plink_test_pure_prs_mean",
            "Test_pure_prs_mean",
            "test_pure_prs_mean",
            "plink_Test_pure_prs",
            "plink_test_pure_prs",
            "Test_pure_prs",
            "test_pure_prs",
        ],
    )

    if train_col is None:
        raise ValueError(
            "Could not detect PLINK train pure PRS column. "
            "Please check whether plink_Train_pure_prs_mean exists."
        )

    if test_col is None:
        raise ValueError(
            "Could not detect PLINK test pure PRS column. "
            "Please check whether plink_Test_pure_prs_mean exists."
        )

    return train_col, test_col


def is_allowed_feature_column(col: str) -> bool:
    """
    Only allow the configured feature range columns.

    This prevents leakage from downstream PRS outputs and excludes later features.
    """

    col_lower = col.lower()

    if not col_lower.startswith(FEATURE_PREFIXES_ALLOWED):
        return False

    forbidden_prefixes = (
        "target_",
        "plink_",
        "prsice2_",
        *FORBIDDEN_FEATURE_PREFIXES,
    )

    if col_lower.startswith(forbidden_prefixes):
        return False

    forbidden_terms = [
        "pure_prs",
        "best_model",
        "train_null_model",
        "test_null_model",
        "train_best_model",
        "test_best_model",
        "train_pure_prs",
        "test_pure_prs",
        "prs_performance",
        "performance",
        "auc",
        "r2",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "precision",
        "recall",
        "gap",
    ]

    # These terms are excluded only if they appear inside Feature1-6 columns.
    # This is an extra safety net in case any downstream metric was accidentally
    # saved with a feature prefix.
    if any(term in col_lower for term in forbidden_terms):
        return False

    return True


def make_feature_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert the configured feature-range columns to numeric and remove bad/missing-heavy columns.
    """

    candidate_features = [c for c in df.columns if is_allowed_feature_column(c)]

    feature_rows = []
    used_features = []

    numeric_data = {}

    for col in candidate_features:
        numeric_col = clean_numeric(df[col])

        non_missing = int(numeric_col.notna().sum())
        missing_fraction = float(1.0 - non_missing / max(1, len(df)))
        unique_values = int(numeric_col.dropna().nunique())

        used = (
            non_missing >= MIN_NON_MISSING_ROWS
            and missing_fraction <= MAX_FEATURE_MISSING_FRACTION
            and unique_values >= MIN_UNIQUE_VALUES
        )

        feature_rows.append(
            {
                "feature": col,
                ALLOWED_FEATURE_FLAG_COL: col.lower().startswith(FEATURE_PREFIXES_ALLOWED),
                "non_missing_rows": non_missing,
                "missing_fraction": missing_fraction,
                "unique_values": unique_values,
                USED_FEATURE_FLAG_COL: used,
                "reason_if_removed": (
                    ""
                    if used
                    else (
                        "too_few_non_missing"
                        if non_missing < MIN_NON_MISSING_ROWS
                        else "too_much_missing"
                        if missing_fraction > MAX_FEATURE_MISSING_FRACTION
                        else "too_few_unique_values"
                    )
                ),
            }
        )

        if used:
            used_features.append(col)
            numeric_data[col] = numeric_col

    if len(used_features) == 0:
        raise RuntimeError(
            f"No usable {FEATURE_RANGE_TEXT} numeric features were found after filtering. "
            "Try increasing MAX_FEATURE_MISSING_FRACTION or check column names."
        )

    features_df = pd.DataFrame(numeric_data)
    screening_df = pd.DataFrame(feature_rows).sort_values(
        [USED_FEATURE_FLAG_COL, "missing_fraction", "feature"],
        ascending=[False, True, True],
    )

    return features_df, screening_df


def make_leakage_audit(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Audit selected features for possible leakage.
    """

    rows = []

    for col in features_df.columns:
        col_lower = col.lower()

        starts_with_allowed_feature = col_lower.startswith(FEATURE_PREFIXES_ALLOWED)
        starts_with_forbidden_prefix = col_lower.startswith(
            (
                "target_",
                "plink_",
                "prsice2_",
                *FORBIDDEN_FEATURE_PREFIXES,
            )
        )

        contains_forbidden_metric = any(
            term in col_lower
            for term in [
                "pure_prs",
                "best_model",
                "train_pure_prs",
                "test_pure_prs",
                "performance",
                "gap",
                "auc",
                "r2",
                "accuracy",
            ]
        )

        rows.append(
            {
                "feature": col,
                f"starts_with_allowed_feature{FEATURE_START}_to_feature{FEATURE_END}": starts_with_allowed_feature,
                "starts_with_forbidden_prefix": starts_with_forbidden_prefix,
                "contains_forbidden_metric_term": contains_forbidden_metric,
                "possible_leakage": (
                    not starts_with_allowed_feature
                    or starts_with_forbidden_prefix
                    or contains_forbidden_metric
                ),
            }
        )

    audit_df = pd.DataFrame(rows)

    return audit_df


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_section(f"ANALYSIS19: GENERATE {MODEL_LABEL.upper()} FEATURES AND TARGETS")

    input_path = Path(INPUT_FILE)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"[READING] {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    df = standardise_key_columns(df)

    print(f"[RAW ROWS]        {len(df)}")
    print(f"[RAW COLUMNS]     {len(df.columns)}")
    print(f"[RAW PHENOTYPES]  {df['phenotype'].nunique()}")
    print(f"[RAW GWAS FILES]  {df['accessionId'].nunique()}")

    print_section("DETECTING PLINK PURE PRS TARGET COLUMNS")

    plink_train_col, plink_test_col = detect_plink_pure_prs_columns(df)

    print(f"[PLINK TRAIN PURE PRS] {plink_train_col}")
    print(f"[PLINK TEST PURE PRS]  {plink_test_col}")

    target_df = df[["phenotype", "accessionId"]].copy()

    target_df["target_plink_train_pure_prs"] = clean_numeric(df[plink_train_col])
    target_df["target_plink_test_pure_prs"] = clean_numeric(df[plink_test_col])

    gap_abs = (
        target_df["target_plink_train_pure_prs"]
        - target_df["target_plink_test_pure_prs"]
    ).abs()

    # Higher is better because smaller train-test gap gives a value closer to 0.
    target_df["target_plink_pure_prs_gap_quality"] = -gap_abs

    target_definitions = pd.DataFrame(
        [
            {
                "target": "target_plink_train_pure_prs",
                "source_column": plink_train_col,
                "formula": plink_train_col,
                "higher_is_better": True,
                "description": "PLINK train pure PRS performance.",
            },
            {
                "target": "target_plink_test_pure_prs",
                "source_column": plink_test_col,
                "formula": plink_test_col,
                "higher_is_better": True,
                "description": "PLINK test pure PRS performance.",
            },
            {
                "target": "target_plink_pure_prs_gap_quality",
                "source_column": f"{plink_train_col}; {plink_test_col}",
                "formula": "-abs(target_plink_train_pure_prs - target_plink_test_pure_prs)",
                "higher_is_better": True,
                "description": "PLINK pure PRS train-test gap quality. Higher means smaller train-test gap.",
            },
        ]
    )

    print_section("TARGET SUMMARY BEFORE ROW FILTERING")

    target_cols = [
        "target_plink_train_pure_prs",
        "target_plink_test_pure_prs",
        "target_plink_pure_prs_gap_quality",
    ]

    for col in target_cols:
        print(
            f"{col:40s} "
            f"non_missing={target_df[col].notna().sum():6d} "
            f"unique={target_df[col].dropna().nunique():6d} "
            f"mean={target_df[col].mean(): .6f} "
            f"std={target_df[col].std(ddof=1): .6f}"
        )

    print_section("GENERATING NUMERIC FEATURE TABLE")

    features_df, feature_screening_df = make_feature_table(df)

    # Add identifiers to features.
    features_df.insert(0, "accessionId", df["accessionId"].values)
    features_df.insert(0, "phenotype", df["phenotype"].values)

    print(f"[CANDIDATE {FEATURE_RANGE_TEXT.upper()} COLUMNS] {len(feature_screening_df)}")
    print(f"[FEATURES KEPT]                     {features_df.shape[1] - 2}")
    print(f"[FEATURES REMOVED]                  {(~feature_screening_df[USED_FEATURE_FLAG_COL]).sum()}")

    print_section("ALIGNING FEATURES AND TARGETS")

    combined = pd.concat(
        [
            features_df.reset_index(drop=True),
            target_df[target_cols].reset_index(drop=True),
        ],
        axis=1,
    )

    before_rows = len(combined)

    if REQUIRE_COMPLETE_TARGETS:
        combined = combined.dropna(subset=target_cols).copy()

    after_rows = len(combined)

    print(f"[ROWS BEFORE TARGET FILTER] {before_rows}")
    print(f"[ROWS AFTER TARGET FILTER]  {after_rows}")
    print(f"[ROWS REMOVED]              {before_rows - after_rows}")

    if after_rows == 0:
        raise RuntimeError(
            "No rows remain after target filtering. "
            "Check PLINK train/test pure PRS columns."
        )

    model_features = combined[["phenotype", "accessionId"] + list(features_df.columns[2:])].copy()
    model_target = combined[["phenotype", "accessionId"] + target_cols].copy()

    # Final numeric enforcement for all feature columns except identifiers.
    for col in model_features.columns:
        if col not in ["phenotype", "accessionId"]:
            model_features[col] = clean_numeric(model_features[col])

    for col in target_cols:
        model_target[col] = clean_numeric(model_target[col])

    print_section("LEAKAGE AUDIT")

    leakage_audit_df = make_leakage_audit(model_features.drop(columns=["phenotype", "accessionId"]))

    leakage_count = int(leakage_audit_df["possible_leakage"].sum())

    print(f"[POSSIBLE LEAKAGE FEATURES] {leakage_count}")

    if leakage_count > 0:
        print()
        print("[FAILED] Leakage audit found suspicious features:")
        print(
            leakage_audit_df[
                leakage_audit_df["possible_leakage"]
            ].to_string(index=False)
        )
        raise RuntimeError(f"Leakage audit failed. Inspect {MODEL_TAG}_leakage_audit.csv.")

    print(f"[PASSED] Only {FEATURE_RANGE_TEXT} predictors are included.")
    print("[PASSED] No target_, plink_, prsice2_, or later feature predictors are included.")

    print_section("FINAL OUTPUT SUMMARY")

    print(f"[FINAL ROWS]             {len(model_features)}")
    print(f"[FINAL FEATURE COLUMNS]  {model_features.shape[1] - 2}")
    print(f"[FINAL TARGET COLUMNS]   {len(target_cols)}")
    print(f"[PHENOTYPES]             {model_features['phenotype'].nunique()}")
    print(f"[GWAS FILES]             {model_features['accessionId'].nunique()}")

    phenotype_counts = (
        model_target
        .groupby("phenotype")
        .size()
        .reset_index(name="n_gwas")
        .sort_values(["n_gwas", "phenotype"], ascending=[False, True])
    )

    print()
    print("[PHENOTYPE COUNTS]")
    print(phenotype_counts.to_string(index=False))

    print_section("SAVING FILES")

    model_features_path = OUTPUT_DIR / f"{MODEL_TAG}_features.csv"
    model_target_path = OUTPUT_DIR / f"{MODEL_TAG}_target.csv"
    feature_screening_path = OUTPUT_DIR / f"{MODEL_TAG}_feature_screening.csv"
    target_definitions_path = OUTPUT_DIR / f"{MODEL_TAG}_target_definitions.csv"
    leakage_audit_path = OUTPUT_DIR / f"{MODEL_TAG}_leakage_audit.csv"
    phenotype_counts_path = OUTPUT_DIR / f"{MODEL_TAG}_phenotype_counts.csv"
    metadata_path = OUTPUT_DIR / f"{MODEL_TAG}_generate_data_metadata.json"

    model_features.to_csv(model_features_path, index=False)
    model_target.to_csv(model_target_path, index=False)
    feature_screening_df.to_csv(feature_screening_path, index=False)
    target_definitions.to_csv(target_definitions_path, index=False)
    leakage_audit_df.to_csv(leakage_audit_path, index=False)
    phenotype_counts.to_csv(phenotype_counts_path, index=False)

    metadata = {
        "script": "Analysis19-GenerateData.py",
        "run_datetime": datetime.now().isoformat(),
        "input_file": str(input_path),
        "model_label": MODEL_LABEL,
        "model_tag": MODEL_TAG,
        "feature_range_text": FEATURE_RANGE_TEXT,
        "output_dir": str(OUTPUT_DIR),
        "plink_train_pure_prs_source_column": plink_train_col,
        "plink_test_pure_prs_source_column": plink_test_col,
        "allowed_feature_prefixes": list(FEATURE_PREFIXES_ALLOWED),
        "max_feature_missing_fraction": MAX_FEATURE_MISSING_FRACTION,
        "min_non_missing_rows": MIN_NON_MISSING_ROWS,
        "min_unique_values": MIN_UNIQUE_VALUES,
        "require_complete_targets": REQUIRE_COMPLETE_TARGETS,
        "raw_rows": int(len(df)),
        "final_rows": int(len(model_features)),
        "raw_columns": int(len(df.columns)),
        "final_feature_columns_excluding_ids": int(model_features.shape[1] - 2),
        "target_columns": target_cols,
        "n_phenotypes": int(model_features["phenotype"].nunique()),
        "n_gwas_files": int(model_features["accessionId"].nunique()),
        "leakage_audit_possible_leakage_features": leakage_count,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    for path in [
        model_features_path,
        model_target_path,
        feature_screening_path,
        target_definitions_path,
        leakage_audit_path,
        phenotype_counts_path,
        metadata_path,
    ]:
        print(f"[SAVED] {path}")

    print_section("DONE")

    print(
        f"""
Use these two files for the next modelling script:

    {MODEL_LABEL}/{MODEL_TAG}_features.csv
    {MODEL_LABEL}/{MODEL_TAG}_target.csv

Important:
    Do not merge PLINK/PRS columns back into the feature matrix.
    Do not use target_* columns as predictors.
    Imputation/scaling should be done inside cross-validation later, not here.
"""
    )


if __name__ == "__main__":
    main()
