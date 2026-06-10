#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd


# =============================================================================
# Phenotypes
# =============================================================================

PHENOTYPES = [
    "asthma",
    "blood_pressure_medication",
    "body_mass_index_bmi",
    "cholesterol_lowering_medication",
    "depression",
    "gastro_oesophageal_reflux_gord_gastric_reflux",
    "hayfever_allergic_rhinitis",
    "high_cholesterol",
    "hypertension",
    "hypothyroidism_myxoedema",
    "irritable_bowel_syndrome",
    "migraine",
    "osteoarthritis",
]

CORR_DIR_NAME = "GWAS_Correlation_Output"
PAIRWISE_FILE_SUFFIX = "_gwas_pairwise_correlations.csv"

OUTPUT_MAIN_CSV = "GWAS_Correlation_Main_Manuscript_Table.csv"
OUTPUT_MAIN_LATEX = "GWAS_Correlation_Main_Manuscript_Table.tex"
OUTPUT_SUPP_EXCEL = "Supplementary_Material_2_GWAS_Correlation.xlsx"
OUTPUT_SUPP_ALL_CSV = "Supplementary_Material_2_All_Pairwise_GWAS_Correlations.csv"


# =============================================================================
# Helper functions
# =============================================================================

def clean_phenotype_name(name):
    manual = {
        "asthma": "Asthma",
        "blood_pressure_medication": "Blood pressure medication",
        "body_mass_index_bmi": "Body mass index BMI",
        "cholesterol_lowering_medication": "Cholesterol-lowering medication",
        "depression": "Depression",
        "gastro_oesophageal_reflux_gord_gastric_reflux": "Gastro-oesophageal reflux",
        "hayfever_allergic_rhinitis": "Hayfever / allergic rhinitis",
        "high_cholesterol": "High cholesterol",
        "hypertension": "Hypertension",
        "hypothyroidism_myxoedema": "Hypothyroidism / myxoedema",
        "irritable_bowel_syndrome": "Irritable bowel syndrome",
        "migraine": "Migraine",
        "osteoarthritis": "Osteoarthritis",
    }
    return manual.get(name, name.replace("_", " ").title())


def find_pairwise_file(phenotype):
    corr_dir = Path(phenotype) / CORR_DIR_NAME

    expected = corr_dir / f"{phenotype}{PAIRWISE_FILE_SUFFIX}"
    if expected.exists():
        return expected

    matches = sorted(corr_dir.glob(f"*{PAIRWISE_FILE_SUFFIX}"))
    if matches:
        return matches[0]

    return None


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def get_col(df, preferred, contains=None):
    if preferred in df.columns:
        return preferred

    if contains:
        matches = []
        for c in df.columns:
            ok = True
            for term in contains:
                if term not in c:
                    ok = False
                    break
            if ok:
                matches.append(c)

        if matches:
            return matches[0]

    return ""


def median_iqr_range(values):
    values = pd.to_numeric(values, errors="coerce").dropna()

    if values.empty:
        return {
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "min": np.nan,
            "max": np.nan,
            "iqr_text": "NA",
            "range_text": "NA",
        }

    q1 = values.quantile(0.25)
    med = values.median()
    q3 = values.quantile(0.75)

    return {
        "median": med,
        "q1": q1,
        "q3": q3,
        "min": values.min(),
        "max": values.max(),
        "iqr_text": f"{q1:.3f}--{q3:.3f}",
        "range_text": f"{values.min():.3f}--{values.max():.3f}",
    }


def format_int(x):
    if pd.isna(x):
        return "NA"
    return f"{int(round(x)):,}"


def format_float(x, digits=3):
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def count_significant(df, p_col, alpha=0.05):
    if not p_col or p_col not in df.columns:
        return 0

    pvals = safe_numeric(df[p_col])
    return int((pvals < alpha).sum())


def count_threshold(values, threshold):
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return 0
    return int((values.abs() >= threshold).sum())


def add_descriptive_columns(df, phenotype):
    df = df.copy()

    df.insert(0, "phenotype_id", phenotype)
    df.insert(1, "phenotype_name", clean_phenotype_name(phenotype))

    beta_col = get_col(
        df,
        "beta_vs_beta_spearman_r",
        contains=["beta_vs_beta", "spearman", "_r"],
    )

    beta_p_col = get_col(
        df,
        "beta_vs_beta_spearman_p",
        contains=["beta_vs_beta", "spearman", "_p"],
    )

    logp_col = get_col(
        df,
        "log10p_vs_log10p_spearman_r",
        contains=["log10p_vs_log10p", "spearman", "_r"],
    )

    logp_p_col = get_col(
        df,
        "log10p_vs_log10p_spearman_p",
        contains=["log10p_vs_log10p", "spearman", "_p"],
    )

    if beta_col:
        beta_values = safe_numeric(df[beta_col])
        df["beta_spearman_direction"] = np.where(
            beta_values > 0,
            "positive",
            np.where(beta_values < 0, "negative", "zero_or_missing"),
        )
        df["abs_beta_spearman_r"] = beta_values.abs()
        df["beta_r_ge_0_2"] = beta_values.abs() >= 0.20
        df["beta_r_ge_0_5"] = beta_values.abs() >= 0.50

    if beta_p_col:
        df["beta_spearman_significant_p_lt_0_05"] = safe_numeric(df[beta_p_col]) < 0.05

    if logp_col:
        logp_values = safe_numeric(df[logp_col])
        df["log10p_spearman_direction"] = np.where(
            logp_values > 0,
            "positive",
            np.where(logp_values < 0, "negative", "zero_or_missing"),
        )
        df["abs_log10p_spearman_r"] = logp_values.abs()

    if logp_p_col:
        df["log10p_spearman_significant_p_lt_0_05"] = safe_numeric(df[logp_p_col]) < 0.05

    return df


def read_all_pairwise_files():
    all_rows = []
    missing_rows = []

    for phenotype in PHENOTYPES:
        pairwise_file = find_pairwise_file(phenotype)

        if pairwise_file is None:
            missing_rows.append(
                {
                    "phenotype_id": phenotype,
                    "phenotype_name": clean_phenotype_name(phenotype),
                    "expected_location": str(Path(phenotype) / CORR_DIR_NAME),
                    "status": "missing_pairwise_correlation_file",
                }
            )
            continue

        try:
            df = pd.read_csv(pairwise_file)

            if "error" in df.columns:
                df = df[df["error"].isna() | (df["error"].astype(str).str.strip() == "")].copy()

            df = add_descriptive_columns(df, phenotype)
            df["source_correlation_file"] = str(pairwise_file)

            all_rows.append(df)

        except Exception as e:
            missing_rows.append(
                {
                    "phenotype_id": phenotype,
                    "phenotype_name": clean_phenotype_name(phenotype),
                    "expected_location": str(pairwise_file),
                    "status": f"read_error: {type(e).__name__}: {e}",
                }
            )

    if all_rows:
        all_pairwise = pd.concat(all_rows, ignore_index=True, sort=False)
    else:
        all_pairwise = pd.DataFrame()

    missing = pd.DataFrame(missing_rows)

    return all_pairwise, missing


def make_main_table(all_pairwise):
    rows = []

    for phenotype in PHENOTYPES:
        sub = all_pairwise[all_pairwise["phenotype_id"] == phenotype].copy()

        if sub.empty:
            rows.append(
                {
                    "Phenotype": clean_phenotype_name(phenotype),
                    "GWAS files": 0,
                    "Pairwise comparisons": 0,
                    "Median common SNPs": "NA",
                    "Median beta rho": "NA",
                    "IQR beta rho": "NA",
                    "Range beta rho": "NA",
                    "Median -log10(P) rho": "NA",
                    "IQR -log10(P) rho": "NA",
                    "Significant beta pairs": "NA",
                    "Pairs |rho_beta| >= 0.2": "NA",
                    "Pairs |rho_beta| >= 0.5": "NA",
                }
            )
            continue

        beta_col = get_col(
            sub,
            "beta_vs_beta_spearman_r",
            contains=["beta_vs_beta", "spearman", "_r"],
        )

        beta_p_col = get_col(
            sub,
            "beta_vs_beta_spearman_p",
            contains=["beta_vs_beta", "spearman", "_p"],
        )

        logp_col = get_col(
            sub,
            "log10p_vs_log10p_spearman_r",
            contains=["log10p_vs_log10p", "spearman", "_r"],
        )

        common_col = "n_common_snps" if "n_common_snps" in sub.columns else ""

        labels = set()
        if "label1" in sub.columns:
            labels.update(sub["label1"].dropna().astype(str).tolist())
        if "label2" in sub.columns:
            labels.update(sub["label2"].dropna().astype(str).tolist())

        common_stats = median_iqr_range(sub[common_col]) if common_col else median_iqr_range(pd.Series(dtype=float))
        beta_stats = median_iqr_range(sub[beta_col]) if beta_col else median_iqr_range(pd.Series(dtype=float))
        logp_stats = median_iqr_range(sub[logp_col]) if logp_col else median_iqr_range(pd.Series(dtype=float))

        beta_values = safe_numeric(sub[beta_col]) if beta_col else pd.Series(dtype=float)

        n_pairwise = len(sub)
        sig_beta = count_significant(sub, beta_p_col, alpha=0.05)

        rows.append(
            {
                "Phenotype": clean_phenotype_name(phenotype),
                "GWAS files": len(labels),
                "Pairwise comparisons": n_pairwise,
                "Median common SNPs": format_int(common_stats["median"]),
                "Median beta rho": format_float(beta_stats["median"]),
                "IQR beta rho": beta_stats["iqr_text"],
                "Range beta rho": beta_stats["range_text"],
                "Median -log10(P) rho": format_float(logp_stats["median"]),
                "IQR -log10(P) rho": logp_stats["iqr_text"],
                "Significant beta pairs": f"{sig_beta}/{n_pairwise}",
                "Pairs |rho_beta| >= 0.2": count_threshold(beta_values, 0.20),
                "Pairs |rho_beta| >= 0.5": count_threshold(beta_values, 0.50),
            }
        )

    return pd.DataFrame(rows)


def make_supplementary_summary(all_pairwise):
    rows = []

    for phenotype in PHENOTYPES:
        sub = all_pairwise[all_pairwise["phenotype_id"] == phenotype].copy()

        if sub.empty:
            continue

        beta_col = get_col(
            sub,
            "beta_vs_beta_spearman_r",
            contains=["beta_vs_beta", "spearman", "_r"],
        )
        beta_p_col = get_col(
            sub,
            "beta_vs_beta_spearman_p",
            contains=["beta_vs_beta", "spearman", "_p"],
        )
        logp_col = get_col(
            sub,
            "log10p_vs_log10p_spearman_r",
            contains=["log10p_vs_log10p", "spearman", "_r"],
        )
        logp_p_col = get_col(
            sub,
            "log10p_vs_log10p_spearman_p",
            contains=["log10p_vs_log10p", "spearman", "_p"],
        )

        common_col = "n_common_snps" if "n_common_snps" in sub.columns else ""

        labels = set()
        if "label1" in sub.columns:
            labels.update(sub["label1"].dropna().astype(str).tolist())
        if "label2" in sub.columns:
            labels.update(sub["label2"].dropna().astype(str).tolist())

        beta_values = safe_numeric(sub[beta_col]) if beta_col else pd.Series(dtype=float)
        logp_values = safe_numeric(sub[logp_col]) if logp_col else pd.Series(dtype=float)
        common_values = safe_numeric(sub[common_col]) if common_col else pd.Series(dtype=float)

        rows.append(
            {
                "phenotype_id": phenotype,
                "phenotype_name": clean_phenotype_name(phenotype),
                "n_gwas_files": len(labels),
                "n_pairwise_comparisons": len(sub),
                "median_common_snps": common_values.median(),
                "min_common_snps": common_values.min(),
                "max_common_snps": common_values.max(),
                "median_beta_spearman_r": beta_values.median(),
                "q1_beta_spearman_r": beta_values.quantile(0.25),
                "q3_beta_spearman_r": beta_values.quantile(0.75),
                "min_beta_spearman_r": beta_values.min(),
                "max_beta_spearman_r": beta_values.max(),
                "n_positive_beta_correlations": int((beta_values > 0).sum()),
                "n_negative_beta_correlations": int((beta_values < 0).sum()),
                "n_beta_p_lt_0_05": count_significant(sub, beta_p_col, alpha=0.05),
                "n_abs_beta_r_ge_0_2": count_threshold(beta_values, 0.20),
                "n_abs_beta_r_ge_0_5": count_threshold(beta_values, 0.50),
                "median_log10p_spearman_r": logp_values.median(),
                "q1_log10p_spearman_r": logp_values.quantile(0.25),
                "q3_log10p_spearman_r": logp_values.quantile(0.75),
                "min_log10p_spearman_r": logp_values.min(),
                "max_log10p_spearman_r": logp_values.max(),
                "n_log10p_p_lt_0_05": count_significant(sub, logp_p_col, alpha=0.05),
            }
        )

    return pd.DataFrame(rows)


def dataframe_to_latex_table(df):
    latex = df.to_latex(
        index=False,
        escape=False,
        column_format="lrrrrllllrrr",
        caption=(
            "Within-phenotype GWAS concordance based on pairwise comparisons "
            "of harmonised common SNPs."
        ),
        label="tab:gwas_intrinsic_correlation_summary",
    )

    latex = latex.replace("Median beta rho", "Median $\\rho_{\\beta}$")
    latex = latex.replace("IQR beta rho", "IQR $\\rho_{\\beta}$")
    latex = latex.replace("Range beta rho", "Range $\\rho_{\\beta}$")
    latex = latex.replace("Median -log10(P) rho", "Median $\\rho_{-\\log_{10}(P)}$")
    latex = latex.replace("IQR -log10(P) rho", "IQR $\\rho_{-\\log_{10}(P)}$")
    latex = latex.replace("Pairs |rho_beta| >= 0.2", "Pairs $|\\rho_{\\beta}| \\geq 0.2$")
    latex = latex.replace("Pairs |rho_beta| >= 0.5", "Pairs $|\\rho_{\\beta}| \\geq 0.5$")

    return latex


def write_excel(main_table, supp_summary, all_pairwise, missing):
    with pd.ExcelWriter(OUTPUT_SUPP_EXCEL, engine="openpyxl") as writer:
        main_table.to_excel(writer, sheet_name="Main_Table", index=False)
        supp_summary.to_excel(writer, sheet_name="Phenotype_Summary", index=False)
        all_pairwise.to_excel(writer, sheet_name="All_Pairwise_Correlations", index=False)

        if not missing.empty:
            missing.to_excel(writer, sheet_name="Missing_or_Failed", index=False)

        workbook = writer.book

        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            ws.freeze_panes = "A2"

            for col_cells in ws.columns:
                max_length = 0
                col_letter = col_cells[0].column_letter

                for cell in col_cells:
                    value = cell.value
                    if value is not None:
                        max_length = max(max_length, len(str(value)))

                ws.column_dimensions[col_letter].width = min(max(max_length + 2, 10), 45)

            for cell in ws[1]:
                cell.style = "Headline 3"


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 100)
    print("MERGING GWAS CORRELATION RESULTS")
    print("=" * 100)

    all_pairwise, missing = read_all_pairwise_files()

    print(f"[PAIRWISE ROWS MERGED] {len(all_pairwise)}")
    print(f"[MISSING/FAILED]       {len(missing)}")

    main_table = make_main_table(all_pairwise)
    supp_summary = make_supplementary_summary(all_pairwise)

    main_table.to_csv(OUTPUT_MAIN_CSV, index=False)
    all_pairwise.to_csv(OUTPUT_SUPP_ALL_CSV, index=False)

    latex = dataframe_to_latex_table(main_table)
    with open(OUTPUT_MAIN_LATEX, "w") as f:
        f.write(latex)

    write_excel(main_table, supp_summary, all_pairwise, missing)

    print("\n" + "=" * 100)
    print("MAIN MANUSCRIPT TABLE")
    print("=" * 100)
    print(main_table.to_string(index=False))

    print("\n" + "=" * 100)
    print("SAVED FILES")
    print("=" * 100)
    print(f"[MAIN CSV]       {OUTPUT_MAIN_CSV}")
    print(f"[MAIN LATEX]     {OUTPUT_MAIN_LATEX}")
    print(f"[SUPP ALL CSV]   {OUTPUT_SUPP_ALL_CSV}")
    print(f"[SUPP EXCEL]     {OUTPUT_SUPP_EXCEL}")

    if not missing.empty:
        print("\n" + "=" * 100)
        print("WARNING: Missing or failed phenotype files")
        print("=" * 100)
        print(missing.to_string(index=False))


if __name__ == "__main__":
    main()