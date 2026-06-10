#!/usr/bin/env python3

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats


# =============================================================================
# Helpers
# =============================================================================

COLUMN_ALIASES = {
    "SNPID": ["SNPID", "snpid", "SNP", "snp", "rsID", "rsid", "rs_id", "ID", "id"],
    "CHR": ["CHR", "chr", "chrom", "chromosome", "CHROM"],
    "POS": ["POS", "pos", "BP", "bp", "position", "base_pair_location"],
    "EA": ["EA", "ea", "A1", "a1", "effect_allele", "tested_allele", "ALT", "alt"],
    "NEA": ["NEA", "nea", "A2", "a2", "other_allele", "non_effect_allele", "REF", "ref"],
    "BETA": ["BETA", "beta", "Effect", "effect", "logOR", "log_or", "b"],
    "P": ["P", "p", "PVAL", "pval", "p_value", "p-value", "PVALUE", "pvalue"],
}


def clean_name(x):
    return str(x).strip()


def find_matching_column(columns, standard_name):
    lookup = {clean_name(c).lower(): c for c in columns}

    for alias in COLUMN_ALIASES[standard_name]:
        key = alias.lower()
        if key in lookup:
            return lookup[key]

    return None


def get_accession_from_path(path):
    path = Path(path)
    # Expected: phenotype/allgwas/GCST.../FinalGWAS.csv
    try:
        return path.parent.name
    except Exception:
        return path.stem


def read_sumstats_minimal(path):
    """
    Read only the columns required for correlation:
    SNPID, CHR, POS, EA, NEA, BETA, P

    Supports FinalGWAS.csv and also target Fisher files if they use A1/A2.
    """
    path = Path(path)

    header_df = pl.read_csv(
        path,
        n_rows=0,
        infer_schema_length=1000,
        ignore_errors=True,
    )
    columns = header_df.columns

    selected_exprs = []

    missing = []
    for standard in ["SNPID", "CHR", "POS", "EA", "NEA", "BETA", "P"]:
        actual = find_matching_column(columns, standard)
        if actual is None:
            missing.append(standard)
        else:
            selected_exprs.append(pl.col(actual).alias(standard))

    if missing:
        raise ValueError(f"{path} is missing required columns after alias matching: {missing}")

    df = (
        pl.scan_csv(
            path,
            infer_schema_length=10000,
            ignore_errors=True,
            null_values=["", "NA", "NaN", "nan", "NULL", "None", "."],
        )
        .select(selected_exprs)
        .with_columns(
            [
                pl.col("SNPID").cast(pl.Utf8, strict=False).str.strip_chars(),
                pl.col("CHR")
                .cast(pl.Utf8, strict=False)
                .str.strip_chars()
                .str.replace(r"(?i)^chr", "")
                .alias("CHR"),
                pl.col("POS").cast(pl.Int64, strict=False),
                pl.col("EA")
                .cast(pl.Utf8, strict=False)
                .str.strip_chars()
                .str.to_uppercase()
                .alias("EA"),
                pl.col("NEA")
                .cast(pl.Utf8, strict=False)
                .str.strip_chars()
                .str.to_uppercase()
                .alias("NEA"),
                pl.col("BETA").cast(pl.Float64, strict=False),
                pl.col("P").cast(pl.Float64, strict=False),
            ]
        )
        .filter(
            pl.col("CHR").is_not_null()
            & pl.col("POS").is_not_null()
            & pl.col("EA").is_not_null()
            & pl.col("NEA").is_not_null()
            & pl.col("BETA").is_not_null()
            & pl.col("P").is_not_null()
            & (pl.col("EA") != "")
            & (pl.col("NEA") != "")
            & (pl.col("P") > 0)
            & (pl.col("P") <= 1)
            & pl.col("BETA").is_finite()
            & pl.col("P").is_finite()
        )
        .with_columns(
            [
                (-pl.col("P").log10()).alias("LOG10P"),
            ]
        )
        .unique(subset=["CHR", "POS", "EA", "NEA"], keep="first")
    )

    return df


def harmonised_join(df1, df2):
    """
    Join common SNPs between two GWAS files.

    Direct allele match:
        EA1 == EA2 and NEA1 == NEA2
        BETA2 unchanged

    Reversed allele match:
        EA1 == NEA2 and NEA1 == EA2
        BETA2 multiplied by -1
    """
    key = ["CHR", "POS", "EA", "NEA"]

    a = df1.select(
        [
            pl.col("CHR"),
            pl.col("POS"),
            pl.col("EA"),
            pl.col("NEA"),
            pl.col("SNPID").alias("SNPID_1"),
            pl.col("BETA").alias("BETA_1"),
            pl.col("P").alias("P_1"),
            pl.col("LOG10P").alias("LOG10P_1"),
        ]
    )

    b_direct = df2.select(
        [
            pl.col("CHR"),
            pl.col("POS"),
            pl.col("EA"),
            pl.col("NEA"),
            pl.col("SNPID").alias("SNPID_2"),
            pl.col("BETA").alias("BETA_2"),
            pl.col("P").alias("P_2"),
            pl.col("LOG10P").alias("LOG10P_2"),
        ]
    )

    direct = (
        a.join(b_direct, on=key, how="inner")
        .with_columns(pl.lit("direct").alias("allele_match"))
    )

    b_flip = df2.select(
        [
            pl.col("CHR"),
            pl.col("POS"),
            pl.col("NEA").alias("EA"),
            pl.col("EA").alias("NEA"),
            pl.col("SNPID").alias("SNPID_2"),
            (-pl.col("BETA")).alias("BETA_2"),
            pl.col("P").alias("P_2"),
            pl.col("LOG10P").alias("LOG10P_2"),
        ]
    )

    flipped = (
        a.join(b_flip, on=key, how="inner")
        .with_columns(pl.lit("flipped").alias("allele_match"))
    )

    joined = (
        pl.concat([direct, flipped], how="vertical")
        .unique(subset=["CHR", "POS", "EA", "NEA"], keep="first")
    )

    return joined


def safe_corr(x, y, method):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)

    if n < 3:
        return n, np.nan, np.nan

    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return n, np.nan, np.nan

    try:
        if method == "pearson":
            r, p = stats.pearsonr(x, y)
        elif method == "spearman":
            r, p = stats.spearmanr(x, y)
        else:
            raise ValueError(f"Unknown method: {method}")

        return n, float(r), float(p)

    except Exception:
        return n, np.nan, np.nan


def summarise_joined(joined_df):
    """
    joined_df is a collected pandas dataframe.
    """
    result = {}

    result["n_common_snps"] = len(joined_df)
    result["n_direct_allele_match"] = int((joined_df["allele_match"] == "direct").sum())
    result["n_flipped_allele_match"] = int((joined_df["allele_match"] == "flipped").sum())

    comparisons = {
        "beta_vs_beta": ("BETA_1", "BETA_2"),
        "p_vs_p": ("P_1", "P_2"),
        "log10p_vs_log10p": ("LOG10P_1", "LOG10P_2"),
        "beta1_vs_p2": ("BETA_1", "P_2"),
        "p1_vs_beta2": ("P_1", "BETA_2"),
        "beta1_vs_log10p2": ("BETA_1", "LOG10P_2"),
        "log10p1_vs_beta2": ("LOG10P_1", "BETA_2"),
    }

    for label, (xcol, ycol) in comparisons.items():
        for method in ["pearson", "spearman"]:
            n, r, p = safe_corr(joined_df[xcol], joined_df[ycol], method)
            result[f"{label}_{method}_n"] = n
            result[f"{label}_{method}_r"] = r
            result[f"{label}_{method}_p"] = p

    return result


def compare_two_files(path1, path2, label1=None, label2=None):
    path1 = Path(path1)
    path2 = Path(path2)

    label1 = label1 or get_accession_from_path(path1)
    label2 = label2 or get_accession_from_path(path2)

    df1 = read_sumstats_minimal(path1)
    df2 = read_sumstats_minimal(path2)

    joined = harmonised_join(df1, df2).collect()

    if joined.height == 0:
        row = {
            "file1": str(path1),
            "file2": str(path2),
            "label1": label1,
            "label2": label2,
            "n_common_snps": 0,
            "n_direct_allele_match": 0,
            "n_flipped_allele_match": 0,
        }
        return row

    joined_pd = joined.to_pandas()
    row = summarise_joined(joined_pd)

    row.update(
        {
            "file1": str(path1),
            "file2": str(path2),
            "label1": label1,
            "label2": label2,
        }
    )

    return row


def make_pairwise_summary(df, alpha=0.05):
    rows = []

    corr_cols = [c for c in df.columns if c.endswith("_r")]
    for r_col in corr_cols:
        p_col = r_col.replace("_r", "_p")

        values = pd.to_numeric(df[r_col], errors="coerce")
        pvals = pd.to_numeric(df[p_col], errors="coerce") if p_col in df.columns else pd.Series(np.nan, index=df.index)

        valid = values.notna()
        sig = valid & pvals.notna() & (pvals < alpha)

        rows.append(
            {
                "correlation_metric": r_col,
                "n_pairs_with_value": int(valid.sum()),
                "mean_r": float(values.mean()) if valid.any() else np.nan,
                "median_r": float(values.median()) if valid.any() else np.nan,
                "min_r": float(values.min()) if valid.any() else np.nan,
                "max_r": float(values.max()) if valid.any() else np.nan,
                "n_positive": int((values > 0).sum()),
                "n_negative": int((values < 0).sum()),
                "n_abs_r_ge_0_1": int((values.abs() >= 0.10).sum()),
                "n_abs_r_ge_0_2": int((values.abs() >= 0.20).sum()),
                "n_abs_r_ge_0_5": int((values.abs() >= 0.50).sum()),
                "n_significant": int(sig.sum()),
                "alpha": alpha,
            }
        )

    return pd.DataFrame(rows)


def find_finalgwas_files(phenotype_dir):
    phenotype_dir = Path(phenotype_dir)

    files = sorted(phenotype_dir.glob("allgwas/*/FinalGWAS.csv"))

    if not files:
        files = sorted(phenotype_dir.glob("*/FinalGWAS.csv"))

    return files


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute allele-harmonised GWAS correlation between FinalGWAS.csv files using Polars."
    )

    parser.add_argument(
        "--phenotype-dir",
        required=True,
        help="Phenotype directory, e.g. migraine, asthma, osteoarthritis."
    )

    parser.add_argument(
        "--out-dir",
        default="GWAS_Correlation_Output",
        help="Output directory."
    )

    parser.add_argument(
        "--target-fisher",
        default="",
        help="Optional target Fisher GWAS file to compare against every FinalGWAS.csv."
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for correlation p-values."
    )

    args = parser.parse_args()

    phenotype_dir = Path(args.phenotype_dir)
    phenotype_name = phenotype_dir.name

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("GWAS CORRELATION ANALYSIS")
    print("=" * 100)
    print(f"[PHENOTYPE DIR] {phenotype_dir}")
    print(f"[OUTPUT DIR]    {out_dir}")
    print("=" * 100)

    finalgwas_files = find_finalgwas_files(phenotype_dir)

    if len(finalgwas_files) < 2:
        raise ValueError(f"Need at least 2 FinalGWAS.csv files. Found: {len(finalgwas_files)}")

    print(f"[FINALGWAS FILES FOUND] {len(finalgwas_files)}")
    for f in finalgwas_files:
        print(f"  - {f}")

    # -------------------------------------------------------------------------
    # Pairwise GWAS-vs-GWAS correlations
    # -------------------------------------------------------------------------
    pair_rows = []

    all_pairs = list(itertools.combinations(finalgwas_files, 2))
    print(f"\n[PAIRWISE COMPARISONS] {len(all_pairs)}")

    for i, (file1, file2) in enumerate(all_pairs, start=1):
        label1 = get_accession_from_path(file1)
        label2 = get_accession_from_path(file2)

        print(f"[{i}/{len(all_pairs)}] {label1} vs {label2}")

        try:
            row = compare_two_files(file1, file2, label1=label1, label2=label2)
            row["comparison_type"] = "gwas_vs_gwas"
            row["phenotype"] = phenotype_name
            pair_rows.append(row)
        except Exception as e:
            pair_rows.append(
                {
                    "phenotype": phenotype_name,
                    "comparison_type": "gwas_vs_gwas",
                    "file1": str(file1),
                    "file2": str(file2),
                    "label1": label1,
                    "label2": label2,
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    pairwise_df = pd.DataFrame(pair_rows)

    pairwise_out = out_dir / f"{phenotype_name}_gwas_pairwise_correlations.csv"
    pairwise_df.to_csv(pairwise_out, index=False)

    summary_df = make_pairwise_summary(pairwise_df, alpha=args.alpha)
    summary_out = out_dir / f"{phenotype_name}_gwas_pairwise_correlation_summary.csv"
    summary_df.to_csv(summary_out, index=False)

    print(f"\n[SAVED] {pairwise_out}")
    print(f"[SAVED] {summary_out}")

    # -------------------------------------------------------------------------
    # Optional: GWAS-vs-target Fisher correlations
    # -------------------------------------------------------------------------
    if args.target_fisher:
        target_path = Path(args.target_fisher)

        if not target_path.exists():
            raise FileNotFoundError(f"Target Fisher file does not exist: {target_path}")

        target_rows = []

        print("\n" + "=" * 100)
        print("GWAS VS TARGET FISHER CORRELATION")
        print("=" * 100)
        print(f"[TARGET FISHER] {target_path}")

        for i, file1 in enumerate(finalgwas_files, start=1):
            label1 = get_accession_from_path(file1)
            label2 = "target_fisher"

            print(f"[{i}/{len(finalgwas_files)}] {label1} vs target_fisher")

            try:
                row = compare_two_files(file1, target_path, label1=label1, label2=label2)
                row["comparison_type"] = "gwas_vs_target_fisher"
                row["phenotype"] = phenotype_name
                target_rows.append(row)
            except Exception as e:
                target_rows.append(
                    {
                        "phenotype": phenotype_name,
                        "comparison_type": "gwas_vs_target_fisher",
                        "file1": str(file1),
                        "file2": str(target_path),
                        "label1": label1,
                        "label2": label2,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )

        target_df = pd.DataFrame(target_rows)

        target_out = out_dir / f"{phenotype_name}_gwas_vs_target_fisher_correlations.csv"
        target_df.to_csv(target_out, index=False)

        target_summary_df = make_pairwise_summary(target_df, alpha=args.alpha)
        target_summary_out = out_dir / f"{phenotype_name}_gwas_vs_target_fisher_correlation_summary.csv"
        target_summary_df.to_csv(target_summary_out, index=False)

        print(f"\n[SAVED] {target_out}")
        print(f"[SAVED] {target_summary_out}")

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()