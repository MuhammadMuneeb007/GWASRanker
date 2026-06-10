#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


ROOT = Path(".")
FEATURE_NAME = "Feature2.csv"


def print_section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def safe_read_csv(path):
    try:
        df = pd.read_csv(path)
        df["source_feature_file"] = str(path)
        return df
    except Exception as e:
        print(f"[READ FAILED] {path} | {type(e).__name__}: {e}")
        return None


def print_value_counts(df, col, top_n=40):
    if col not in df.columns:
        return

    print_section(col)

    s = df[col].fillna("").astype(str).str.strip()
    counts = s.value_counts(dropna=False)

    print(f"[UNIQUE VALUES] {s.nunique(dropna=False)}")
    print(f"[TOP {top_n}]")
    print(counts.head(top_n).to_string())


def print_numeric_summary(df, col):
    if col not in df.columns:
        return

    print_section(f"NUMERIC SUMMARY: {col}")

    x = pd.to_numeric(df[col], errors="coerce")

    print(f"[COUNT]  {x.notna().sum()}")
    print(f"[MIN]    {x.min()}")
    print(f"[MAX]    {x.max()}")
    print(f"[MEAN]   {round(x.mean(), 4)}")
    print(f"[MEDIAN] {x.median()}")

    print()
    print("[MOST COMMON VALUES]")
    print(x.value_counts(dropna=False).head(40).to_string())


def print_group_summary(df, group_cols, title, top_n=80):
    existing = [c for c in group_cols if c in df.columns]

    if not existing:
        return

    print_section(title)

    summary = (
        df.groupby(existing, dropna=False)
        .size()
        .reset_index(name="n_files")
        .sort_values("n_files", ascending=False)
    )

    print(summary.head(top_n).to_string(index=False))


def print_bool_summary(df, bool_cols):
    print_section("BOOLEAN / FLAG SUMMARY")

    rows = []

    for col in bool_cols:
        if col not in df.columns:
            continue

        s = df[col].fillna("").astype(str).str.strip().str.lower()

        true_count = int(s.isin(["true", "1", "yes", "y"]).sum())
        false_count = int(s.isin(["false", "0", "no", "n"]).sum())
        empty_count = int((s == "").sum())
        other_count = len(s) - true_count - false_count - empty_count

        rows.append({
            "feature": col,
            "true": true_count,
            "false": false_count,
            "empty": empty_count,
            "other": other_count,
            "true_pct": round((true_count / len(s)) * 100, 3) if len(s) else 0,
            "false_pct": round((false_count / len(s)) * 100, 3) if len(s) else 0,
        })

    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        print("[INFO] No boolean columns found.")


def print_column_uniqueness(df):
    print_section("COLUMN UNIQUENESS SUMMARY")

    rows = []

    for col in df.columns:
        s = df[col].fillna("").astype(str).str.strip()
        vc = s.value_counts(dropna=False)

        rows.append({
            "column": col,
            "non_empty": int((s != "").sum()),
            "empty": int((s == "").sum()),
            "unique_values": int(s.nunique(dropna=False)),
            "most_common_value": vc.index[0] if len(vc) else "",
            "most_common_count": int(vc.iloc[0]) if len(vc) else 0,
        })

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))


def print_problem_files(df):
    print_section("POTENTIAL PROBLEM FILES")

    mask = pd.Series(False, index=df.index)

    if "cannot_normalise_safely" in df.columns:
        mask |= df["cannot_normalise_safely"].fillna("").astype(str).str.lower().eq("true")

    if "prs_minimum_columns_present" in df.columns:
        mask |= df["prs_minimum_columns_present"].fillna("").astype(str).str.lower().ne("true")

    if "header_detected" in df.columns:
        mask |= df["header_detected"].fillna("").astype(str).str.lower().ne("true")

    if "delimiter" in df.columns:
        mask |= df["delimiter"].fillna("").astype(str).str.lower().isin(["unknown", ""])

    if "n_rows_loaded" in df.columns:
        nrows = pd.to_numeric(df["n_rows_loaded"], errors="coerce").fillna(0)
        mask |= nrows.eq(0)

    bad = df[mask].copy()

    print(f"[POTENTIAL PROBLEM FILES] {len(bad)}")

    if bad.empty:
        return

    show_cols = [
        "phenotype",
        "accessionId",
        "file_name",
        "file_extension",
        "compression_type",
        "delimiter",
        "n_columns",
        "n_rows_loaded",
        "effect_column_type",
        "prs_minimum_columns_present",
        "cannot_normalise_safely",
        "dataframe_issue",
        "read_issue",
        "file_path",
    ]

    show_cols = [c for c in show_cols if c in bad.columns]

    print(bad[show_cols].head(150).to_string(index=False))


def print_missingness_extremes(df):
    print_section("HIGH MISSINGNESS FILES")

    missing_cols = [
        "missing_snp_percentage",
        "missing_chr_percentage",
        "missing_bp_percentage",
        "missing_chr_bp_percentage",
        "missing_a1_percentage",
        "missing_a2_percentage",
        "missing_a1_a2_percentage",
        "missing_effect_beta_or_z_percentage",
        "missing_p_percentage",
    ]

    existing = [c for c in missing_cols if c in df.columns]

    if not existing:
        print("[INFO] No missingness columns found.")
        return

    for col in existing:
        x = pd.to_numeric(df[col], errors="coerce")

        high = df[x >= 50].copy()

        print()
        print("-" * 100)
        print(f"{col} >= 50% : {len(high)} files")
        print("-" * 100)

        if high.empty:
            continue

        show_cols = [
            "phenotype",
            "accessionId",
            "file_name",
            col,
            "prs_minimum_columns_present",
            "cannot_normalise_safely",
            "file_path",
        ]

        show_cols = [c for c in show_cols if c in high.columns]
        print(high[show_cols].head(50).to_string(index=False))


def print_top_problem_rates(df):
    print_section("ALLELE / SNP WARNING RATES")

    rate_cols = [
        "duplicated_snp_percentage",
        "ambiguous_allele_percentage",
        "invalid_allele_percentage",
    ]

    for col in rate_cols:
        if col not in df.columns:
            continue

        x = pd.to_numeric(df[col], errors="coerce")

        print()
        print("-" * 100)
        print(col)
        print("-" * 100)
        print(f"[MIN]    {x.min()}")
        print(f"[MAX]    {x.max()}")
        print(f"[MEAN]   {round(x.mean(), 4)}")
        print(f"[MEDIAN] {x.median()}")

        high = df[x >= 10].copy()

        print(f"[FILES >= 10%] {len(high)}")

        if not high.empty:
            show_cols = [
                "phenotype",
                "accessionId",
                "file_name",
                col,
                "effect_column_type",
                "prs_minimum_columns_present",
                "file_path",
            ]

            show_cols = [c for c in show_cols if c in high.columns]
            print(high[show_cols].head(50).to_string(index=False))


def main():
    print_section("FINDING Feature2.csv FILES")

    feature_files = sorted(ROOT.rglob(FEATURE_NAME))

    print(f"[FOUND Feature2.csv FILES] {len(feature_files)}")

    if not feature_files:
        print("[STOP] No Feature2.csv files found.")
        return

    dfs = []

    for path in feature_files:
        df = safe_read_csv(path)
        if df is not None and not df.empty:
            dfs.append(df)

    if not dfs:
        print("[STOP] No readable Feature2.csv files.")
        return

    merged = pd.concat(dfs, ignore_index=True)

    print_section("BASIC OVERVIEW")
    print(f"Files found:      {len(feature_files)}")
    print(f"Files processed:  {len(dfs)}")
    print(f"Merged rows:      {len(merged)}")
    print(f"Merged columns:   {len(merged.columns)}")

    print_group_summary(
        merged,
        ["phenotype"],
        "FILES PROCESSED BY PHENOTYPE"
    )

    print_group_summary(
        merged,
        ["file_extension", "compression_type", "delimiter"],
        "STORAGE FORMAT GROUPS"
    )

    print_group_summary(
        merged,
        ["effect_column_type"],
        "EFFECT COLUMN TYPE GROUPS"
    )

    print_group_summary(
        merged,
        ["chromosome_format"],
        "CHROMOSOME FORMAT GROUPS"
    )

    print_group_summary(
        merged,
        ["snp_format"],
        "SNP FORMAT GROUPS"
    )

    print_group_summary(
        merged,
        ["bp_format"],
        "BP FORMAT GROUPS"
    )

    print_group_summary(
        merged,
        ["a1_allele_format", "a2_allele_format"],
        "ALLELE FORMAT GROUPS"
    )

    print_group_summary(
        merged,
        ["pvalue_format"],
        "P-VALUE FORMAT GROUPS"
    )

    print_group_summary(
        merged,
        ["eaf_range", "maf_range", "info_range"],
        "EAF / MAF / INFO RANGE GROUPS"
    )

    print_group_summary(
        merged,
        [
            "prs_minimum_columns_present",
            "close_to_gwas_ssf",
            "requires_column_renaming",
            "requires_effect_conversion_or_to_log_or",
            "cannot_normalise_safely",
        ],
        "USABILITY PATTERN GROUPS"
    )

    bool_cols = [
        "has_SNP", "has_CHR", "has_BP", "has_A1", "has_A2",
        "has_BETA", "has_OR", "has_Z", "has_SE", "has_P",
        "has_N", "has_N_CASES", "has_N_CONTROLS", "has_EAF",
        "has_MAF", "has_INFO", "has_DIRECTION",
        "prs_minimum_columns_present",
        "close_to_gwas_ssf",
        "requires_column_renaming",
        "requires_delimiter_fix",
        "requires_decompression_or_extraction",
        "requires_effect_conversion_or_to_log_or",
        "cannot_normalise_safely",
    ]

    print_bool_summary(merged, bool_cols)

    for col in [
        "file_extension",
        "compression_type",
        "delimiter",
        "effect_column_type",
        "chromosome_format",
        "snp_format",
        "bp_format",
        "a1_allele_format",
        "a2_allele_format",
        "pvalue_format",
        "eaf_range",
        "maf_range",
        "info_range",
        "prs_minimum_columns_present",
        "close_to_gwas_ssf",
        "requires_column_renaming",
        "requires_delimiter_fix",
        "requires_decompression_or_extraction",
        "requires_effect_conversion_or_to_log_or",
        "cannot_normalise_safely",
    ]:
        print_value_counts(merged, col)

    for col in [
        "raw_file_size_mb",
        "n_lines_read",
        "n_rows_loaded",
        "n_columns",
        "n_unique_raw_column_names_in_file",
        "n_unique_normalised_column_names_in_file",
        "n_unique_mapped_standard_columns_in_file",
        "n_mapped_standard_headers",
        "n_unmapped_standard_headers",
        "n_unmapped_columns",
        "missing_snp_percentage",
        "missing_chr_percentage",
        "missing_bp_percentage",
        "missing_chr_bp_percentage",
        "missing_a1_percentage",
        "missing_a2_percentage",
        "missing_a1_a2_percentage",
        "missing_effect_beta_or_z_percentage",
        "missing_p_percentage",
        "duplicated_snp_percentage",
        "ambiguous_allele_percentage",
        "invalid_allele_percentage",
    ]:
        print_numeric_summary(merged, col)

    print_missingness_extremes(merged)
    print_top_problem_rates(merged)
    print_problem_files(merged)
    print_column_uniqueness(merged)

    print_section("DONE")
    print("Nothing was saved. Feature2 summaries were printed only on screen.")


if __name__ == "__main__":
    main()