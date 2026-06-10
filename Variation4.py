#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


# =============================================================================
# Helpers
# =============================================================================

def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def percent(n, d):
    if d == 0:
        return 0.0
    return round((n / d) * 100, 4)


def is_missing(series):
    return (
        series.isna()
        | (series.astype(str).str.strip() == "")
        | (series.astype(str).str.lower().isin(["nan", "na", "n/a", "none", "null", "."]))
    )


def missing_pct(series):
    if series.empty:
        return 100.0
    return percent(is_missing(series).sum(), len(series))


def get_series(df, col):
    if col and col in df.columns:
        return df[col].astype(str).map(clean_text)
    return pd.Series(dtype=str)


def first_mapped(mapped, standard_col):
    vals = mapped.get(standard_col, [])
    return vals[0] if vals else ""


def standard_mapped_columns(df):
    mapped = {}

    standard_to_final_gwas = {
        "SNP": "SNPID",
        "CHR": "CHR",
        "BP": "POS",
        "A1": "EA",
        "A2": "NEA",
        "BETA": "BETA",
        "OR": "OR",
        "Z": "Z",
        "SE": "SE",
        "P": "P",
        "N": "N",
        "N_CASES": "N_CASES",
        "N_CONTROLS": "N_CONTROLS",
        "EAF": "EAF",
        "MAF": "MAF",
        "INFO": "INFO",
        "DIRECTION": "DIRECTION",
    }

    for standard_col, final_gwas_col in standard_to_final_gwas.items():
        if final_gwas_col in df.columns:
            mapped[standard_col] = [final_gwas_col]

    return mapped

def load_final_gwas_dataframe(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [clean_text(c) for c in df.columns]
    return df, "FinalGWAS.csv"


# =============================================================================
# Group 2: Sample size reporting
# =============================================================================

def analyse_sample_size(df, mapped):
    col_n = first_mapped(mapped, "N")
    col_n_cases = first_mapped(mapped, "N_CASES")
    col_n_controls = first_mapped(mapped, "N_CONTROLS")

    has_n = bool(col_n)
    has_n_cases = bool(col_n_cases)
    has_n_controls = bool(col_n_controls)

    result = {
        "has_total_N": has_n,
        "has_variant_level_N": False,
        "has_N_cases": has_n_cases,
        "has_N_controls": has_n_controls,
        "missing_N_percentage": 100.0,
        "N_constant_or_variable": "not_available",
        "median_N": "",
        "min_N": "",
        "max_N": "",
        "N_variability_ratio": "",
    }

    if not has_n or df.empty:
        return result

    n_series = pd.to_numeric(get_series(df, col_n), errors="coerce")
    n_valid = n_series.dropna()

    result["missing_N_percentage"] = percent(n_series.isna().sum(), len(n_series))
    result["has_variant_level_N"] = len(n_valid) > 0

    if not n_valid.empty:
        result["median_N"] = round(n_valid.median(), 0)
        result["min_N"] = int(n_valid.min())
        result["max_N"] = int(n_valid.max())

        if n_valid.max() > 0:
            ratio = (n_valid.max() - n_valid.min()) / n_valid.max()
            result["N_variability_ratio"] = round(ratio, 6)
            result["N_constant_or_variable"] = "constant" if ratio < 0.001 else "variable"

    return result


# =============================================================================
# Group 4: Allele orientation
# =============================================================================

def analyse_alleles(df, mapped):
    col_a1 = first_mapped(mapped, "A1")
    col_a2 = first_mapped(mapped, "A2")

    a1 = get_series(df, col_a1)
    a2 = get_series(df, col_a2)

    result = {
        "palindromic_snp_percentage": 0.0,
        "non_acgt_allele_percentage": 0.0,
        "indel_percentage": 0.0,
        "multi_character_allele_percentage": 0.0,
        "allele_case_mixed": False,
        "a1_has_lowercase": False,
        "a2_has_lowercase": False,
    }

    if a1.empty and a2.empty:
        return result

    all_alleles = pd.concat([a1, a2]).dropna()
    all_alleles = all_alleles[~all_alleles.str.lower().isin(["", "nan", "na", "n/a", "none", "null", "."])]

    if all_alleles.empty:
        return result

    n = len(all_alleles)
    result["non_acgt_allele_percentage"] = percent(
        (~all_alleles.str.upper().str.match(r"^[ACGT]+$", na=False)).sum(), n
    )
    result["multi_character_allele_percentage"] = percent(
        (all_alleles.str.len() > 1).sum(), n
    )

    has_lower = all_alleles.str.contains(r"[a-z]", regex=True, na=False).any()
    has_upper = all_alleles.str.contains(r"[A-Z]", regex=True, na=False).any()
    result["allele_case_mixed"] = bool(has_lower and has_upper)
    result["a1_has_lowercase"] = bool(a1.str.contains(r"[a-z]", regex=True, na=False).any()) if not a1.empty else False
    result["a2_has_lowercase"] = bool(a2.str.contains(r"[a-z]", regex=True, na=False).any()) if not a2.empty else False

    if not a1.empty and not a2.empty:
        single_base_alleles = pd.DataFrame()
        df_alleles = pd.DataFrame({
            "a1": a1.str.upper().map(clean_text),
            "a2": a2.str.upper().map(clean_text),
        })
        df_alleles = df_alleles[
            (~df_alleles["a1"].isin(["", "NAN", "NA", "N/A", "NONE", "NULL", "."]))
            & (~df_alleles["a2"].isin(["", "NAN", "NA", "N/A", "NONE", "NULL", "."]))
        ]
        if not df_alleles.empty:
            has_symbolic_indel = (
                df_alleles["a1"].str.contains(r"INS|DEL|INSERTION|DELETION|-", regex=True, na=False)
                | df_alleles["a2"].str.contains(r"INS|DEL|INSERTION|DELETION|-", regex=True, na=False)
            )
            has_length_indel = df_alleles["a1"].str.len() != df_alleles["a2"].str.len()
            result["indel_percentage"] = percent(
                (has_symbolic_indel | has_length_indel).sum(),
                len(df_alleles),
            )

            single_base = (
                df_alleles["a1"].str.match(r"^[ACGT]$", na=False)
                & df_alleles["a2"].str.match(r"^[ACGT]$", na=False)
            )
            single_base_alleles = df_alleles[single_base]

        if not single_base_alleles.empty:
            pairs = single_base_alleles["a1"] + "/" + single_base_alleles["a2"]
            result["palindromic_snp_percentage"] = percent(
                pairs.isin(["A/T", "T/A", "C/G", "G/C"]).sum(),
                len(single_base_alleles),
            )

    return result


# =============================================================================
# Group 5: Variant identifier quality
# =============================================================================

def analyse_variant_ids(df, mapped):
    col_snp = first_mapped(mapped, "SNP")
    col_chr = first_mapped(mapped, "CHR")
    col_bp = first_mapped(mapped, "BP")
    col_a1 = first_mapped(mapped, "A1")
    col_a2 = first_mapped(mapped, "A2")

    has_rsid_col = bool(col_snp)
    has_chr_col = bool(col_chr)
    has_bp_col = bool(col_bp)

    result = {
        "has_rsid_column": has_rsid_col,
        "has_chr_bp_columns": bool(has_chr_col and has_bp_col),
        "has_chr_bp_alleles": bool(has_chr_col and has_bp_col and col_a1 and col_a2),
        "rsid_percentage": 0.0,
        "chrpos_percentage": 0.0,
        "chrpos_allele_percentage": 0.0,
        "missing_variant_id_percentage": 100.0,
        "duplicated_variant_id_percentage": 0.0,
        "unique_variant_count": 0,
        "snp_id_format": "not_available",
    }

    if df.empty or not col_snp:
        return result

    snp = get_series(df, col_snp)
    non_missing = snp[~is_missing(snp)]

    result["missing_variant_id_percentage"] = missing_pct(snp)
    result["unique_variant_count"] = int(non_missing.nunique())

    if non_missing.empty:
        return result

    rsid_frac = non_missing.str.match(r"^rs\d+$", case=False, na=False).mean()
    chrpos_frac = non_missing.str.match(r"^(chr)?[0-9XYMTxy]+[:_][0-9]+$", na=False).mean()
    chrpos_allele_frac = non_missing.str.match(
        r"^(chr)?[0-9XYMTxy]+[:_][0-9]+[:_][ACGTacgt]+[:_][ACGTacgt]+$", na=False
    ).mean()

    result["rsid_percentage"] = round(rsid_frac * 100, 2)
    result["chrpos_percentage"] = round(chrpos_frac * 100, 2)
    result["chrpos_allele_percentage"] = round(chrpos_allele_frac * 100, 2)

    if rsid_frac >= 0.8:
        result["snp_id_format"] = "rsID"
    elif chrpos_allele_frac >= 0.8:
        result["snp_id_format"] = "chr_pos_alleles"
    elif chrpos_frac >= 0.8:
        result["snp_id_format"] = "chr_pos"
    else:
        result["snp_id_format"] = "mixed_or_other"

    dup = snp[~is_missing(snp)].duplicated(keep=False).sum()
    result["duplicated_variant_id_percentage"] = percent(dup, len(non_missing))

    return result


# =============================================================================
# Group 6: P-value quality
# =============================================================================

def analyse_pvalue(df, mapped):
    col_p = first_mapped(mapped, "P")

    result = {
        "has_p_column": bool(col_p),
        "p_numeric_percentage": 0.0,
        "p_out_of_range_percentage": 0.0,
        "p_zero_count": 0,
        "p_equal_one_count": 0,
        "min_p": "",
        "median_p": "",
        "genome_wide_significant_count": 0,
        "genome_wide_significant_percentage": 0.0,
        "suggestive_significant_count": 0,
        "suggestive_significant_percentage": 0.0,
        "p_scientific_notation_percentage": 0.0,
    }

    if not col_p or df.empty:
        return result

    p_raw = get_series(df, col_p)
    p_numeric = pd.to_numeric(p_raw, errors="coerce")

    total = len(p_raw)
    n_numeric = p_numeric.notna().sum()

    result["p_numeric_percentage"] = percent(n_numeric, total)

    if n_numeric == 0:
        return result

    p_valid = p_numeric.dropna()
    result["p_scientific_notation_percentage"] = percent(
        p_raw.dropna().str.contains(r"e[-+]?\d+", case=False, regex=True, na=False).sum(),
        len(p_raw.dropna()),
    )

    result["p_out_of_range_percentage"] = percent(
        ((p_valid < 0) | (p_valid > 1)).sum(), len(p_valid)
    )
    p_in_range = p_valid[(p_valid >= 0) & (p_valid <= 1)]

    if p_in_range.empty:
        return result

    result["p_zero_count"] = int((p_in_range == 0).sum())
    result["p_equal_one_count"] = int((p_in_range == 1).sum())
    result["min_p"] = float(p_in_range.min())
    result["median_p"] = float(p_in_range.median())

    gws = (p_in_range < 5e-8).sum()
    sug = (p_in_range < 1e-5).sum()

    result["genome_wide_significant_count"] = int(gws)
    result["genome_wide_significant_percentage"] = percent(gws, len(p_in_range))
    result["suggestive_significant_count"] = int(sug)
    result["suggestive_significant_percentage"] = percent(sug, len(p_in_range))

    return result


# =============================================================================
# Group 7: INFO/imputation quality
# =============================================================================

def analyse_info(df, mapped):
    col_info = first_mapped(mapped, "INFO")

    result = {
        "has_info_column": bool(col_info),
        "info_numeric_percentage": 0.0,
        "info_missing_percentage": 100.0,
        "info_range_type": "not_available",
        "info_out_of_range_percentage": 0.0,
        "info_below_0_3_percentage": 0.0,
        "info_below_0_8_percentage": 0.0,
        "median_info": "",
    }

    if not col_info or df.empty:
        return result

    info_raw = get_series(df, col_info)
    info_numeric = pd.to_numeric(info_raw, errors="coerce")

    total = len(info_raw)
    n_numeric = info_numeric.notna().sum()

    result["info_numeric_percentage"] = percent(n_numeric, total)
    result["info_missing_percentage"] = missing_pct(info_raw)

    if n_numeric == 0:
        return result

    info_valid = info_numeric.dropna()

    result["info_out_of_range_percentage"] = percent(
        ((info_valid < 0) | (info_valid > 1)).sum(), len(info_valid)
    )
    info_in_range = info_valid[(info_valid >= 0) & (info_valid <= 1)]

    if info_in_range.empty:
        result["info_range_type"] = "outside_expected_range"
        return result

    result["info_range_type"] = "range_0_1" if len(info_in_range) == len(info_valid) else "mixed_with_out_of_range"
    result["median_info"] = round(float(info_in_range.median()), 4)
    result["info_below_0_3_percentage"] = percent((info_in_range < 0.3).sum(), len(info_in_range))
    result["info_below_0_8_percentage"] = percent((info_in_range < 0.8).sum(), len(info_in_range))

    return result


# =============================================================================
# Group 8: Frequency quality
# =============================================================================

def analyse_frequency(df, mapped):
    col_eaf = first_mapped(mapped, "EAF")
    col_maf = first_mapped(mapped, "MAF")

    result = {
        "has_eaf_column": bool(col_eaf),
        "has_maf_column": bool(col_maf),
        "eaf_missing_percentage": 100.0,
        "maf_missing_percentage": 100.0,
        "eaf_range_type": "not_available",
        "maf_range_type": "not_available",
        "eaf_out_of_range_percentage": 0.0,
        "maf_out_of_range_percentage": 0.0,
        "median_eaf": "",
        "rare_variant_percentage": 0.0,
        "low_freq_variant_percentage": 0.0,
        "common_variant_percentage": 0.0,
    }

    for col_key, col_name, prefix in [("EAF", col_eaf, "eaf"), ("MAF", col_maf, "maf")]:
        if not col_name or df.empty:
            continue

        freq_raw = get_series(df, col_name)
        freq_numeric = pd.to_numeric(freq_raw, errors="coerce")
        freq_valid = freq_numeric.dropna()

        result[f"{prefix}_missing_percentage"] = missing_pct(freq_raw)

        if freq_valid.empty:
            continue

        upper_bound = 0.5 if prefix == "maf" else 1.0
        result[f"{prefix}_out_of_range_percentage"] = percent(
            ((freq_valid < 0) | (freq_valid > upper_bound)).sum(), len(freq_valid)
        )

        freq_in_range = freq_valid[(freq_valid >= 0) & (freq_valid <= upper_bound)]
        if freq_in_range.empty:
            result[f"{prefix}_range_type"] = "outside_expected_range"
            continue

        expected_range_label = "range_0_0_5" if prefix == "maf" else "range_0_1"
        result[f"{prefix}_range_type"] = (
            expected_range_label if len(freq_in_range) == len(freq_valid) else "mixed_with_out_of_range"
        )

        if prefix == "eaf":
            result["median_eaf"] = round(float(freq_in_range.median()), 4)
            derived_maf = freq_in_range.clip(lower=0, upper=1).map(lambda x: min(x, 1 - x))
            result["rare_variant_percentage"] = percent((derived_maf < 0.01).sum(), len(derived_maf))
            result["low_freq_variant_percentage"] = percent(
                ((derived_maf >= 0.01) & (derived_maf < 0.05)).sum(), len(derived_maf)
            )
            result["common_variant_percentage"] = percent(
                (derived_maf >= 0.05).sum(), len(derived_maf)
            )
        elif prefix == "maf":
            result["rare_variant_percentage"] = percent((freq_in_range < 0.01).sum(), len(freq_in_range))
            result["low_freq_variant_percentage"] = percent(
                ((freq_in_range >= 0.01) & (freq_in_range < 0.05)).sum(), len(freq_in_range)
            )
            result["common_variant_percentage"] = percent(
                (freq_in_range >= 0.05).sum(), len(freq_in_range)
            )

    return result


# =============================================================================
# Main profiling function
# =============================================================================

def profile_variation4(job_index, phenotype, accession, file_path, final_gwas_file,
                        output_feature_file):
    file_path = Path(file_path)
    final_gwas_file = Path(final_gwas_file)
    output_feature_file = Path(output_feature_file)

    try:
        df, dataframe_source = load_final_gwas_dataframe(final_gwas_file)
        mapped_for_analysis = standard_mapped_columns(df)
        analysis_ready = True
        analysis_error = ""
    except Exception as e:
        df = pd.DataFrame()
        mapped_for_analysis = {}
        dataframe_source = ""
        analysis_ready = False
        analysis_error = f"{type(e).__name__}: {e}"

    sample_size = analyse_sample_size(df, mapped_for_analysis)
    alleles = analyse_alleles(df, mapped_for_analysis)
    variant_ids = analyse_variant_ids(df, mapped_for_analysis)
    pvalue = analyse_pvalue(df, mapped_for_analysis)
    info = analyse_info(df, mapped_for_analysis)
    frequency = analyse_frequency(df, mapped_for_analysis)

    row = {
        "job_index": job_index,
        "phenotype": phenotype,
        "accessionId": accession,
        "file_path": str(file_path),
        "file_name": file_path.name,
        "final_gwas_file": str(final_gwas_file),
        "final_gwas_name": final_gwas_file.name,
        "variation4_analysis_ready": analysis_ready,
        "variation4_analysis_error": analysis_error,
        "variation4_dataframe_source": dataframe_source,
    }

    row.update(sample_size)
    row.update(alleles)
    row.update(variant_ids)
    row.update(pvalue)
    row.update(info)
    row.update(frequency)

    output_feature_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(output_feature_file, index=False)

    return row


# =============================================================================
# Job execution
# =============================================================================

def run_one_job(index, jobs_file, force):
    jobs_path = Path(jobs_file)

    if not jobs_path.exists():
        raise FileNotFoundError(f"Missing jobs file: {jobs_path}")

    jobs = pd.read_csv(jobs_path)

    required = ["job_index", "phenotype", "accessionId", "file_path", "output_feature_file"]
    missing = [c for c in required if c not in jobs.columns]
    if missing:
        raise ValueError(f"Missing columns in {jobs_path}: {missing}")

    sub = jobs[jobs["job_index"].astype(int) == int(index)]
    if sub.empty:
        raise ValueError(f"No job found for index: {index}")

    job = sub.iloc[0]

    phenotype = clean_text(job["phenotype"])
    accession = clean_text(job["accessionId"])
    file_path = Path(clean_text(job["file_path"]))

    base_output = Path(clean_text(job["output_feature_file"]))
    final_gwas_file = base_output.parent / "FinalGWAS.csv"
    output_feature_file = base_output.parent / "Feature4.csv"

    print("=" * 100)
    print("VARIATION 4: FINALGWAS SAMPLE SIZE, P-VALUE, INFO, EAF, ALLELES")
    print("=" * 100)
    print(f"[JOB INDEX]  {index}")
    print(f"[PHENOTYPE]  {phenotype}")
    print(f"[ACCESSION]  {accession}")
    print(f"[GWAS FILE]  {file_path}")
    print(f"[FINAL GWAS] {final_gwas_file}")
    print(f"[OUTPUT]     {output_feature_file}")
    print("=" * 100)

    if not file_path.exists():
        raise FileNotFoundError(f"Missing GWAS file: {file_path}")
    if not final_gwas_file.exists():
        raise FileNotFoundError(f"Missing FinalGWAS.csv: {final_gwas_file}")

    if output_feature_file.exists() and not force:
        print(f"[SKIP] Feature4.csv already exists: {output_feature_file}")
        return

    row = profile_variation4(
        job_index=index,
        phenotype=phenotype,
        accession=accession,
        file_path=file_path,
        final_gwas_file=final_gwas_file,
        output_feature_file=output_feature_file,
    )

    print()
    print("[SAVED]")
    print(output_feature_file)
    print()
    print("[KEY RESULTS]")
    print(f"genome_wide_significant_count:{row['genome_wide_significant_count']}")
    print(f"palindromic_snp_percentage:   {row['palindromic_snp_percentage']}")
    print(f"has_total_N:                  {row['has_total_N']}")
    print(f"p_out_of_range_percentage:    {row['p_out_of_range_percentage']}")
    print(f"snp_id_format:                {row['snp_id_format']}")
    print("=" * 100)


def combine_feature4(jobs_file, output_file):
    jobs = pd.read_csv(jobs_file)
    rows = []

    for _, job in jobs.iterrows():
        base_output = Path(clean_text(job["output_feature_file"]))
        feature4 = base_output.parent / "Feature4.csv"
        if feature4.exists():
            rows.append(pd.read_csv(feature4))

    out_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out_df.to_csv(output_file, index=False)

    print("=" * 100)
    print("[COMBINED Feature4.csv FILES]")
    print(f"[FILES COMBINED] {len(rows)}")
    print(f"[OUTPUT] {output_file}")
    print("=" * 100)

    if not out_df.empty:
        print()
        print("[FEATURE DISTRIBUTION]")

        for col in ["has_total_N", "has_eaf_column", "has_info_column", "p_out_of_range_percentage", "snp_id_format"]:
            if col in out_df.columns:
                print()
                print(col)
                print(out_df[col].value_counts(dropna=False).to_string())


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Variation4: pandas-based features computed from saved FinalGWAS.csv for sample size, p-value quality, INFO, EAF and allele features."
    )

    parser.add_argument(
        "index",
        nargs="?",
        type=int,
        help="GWAS job index from GWAS_jobs.csv"
    )

    parser.add_argument(
        "--jobs-file",
        default="GWAS_jobs.csv",
        help="GWAS job manifest. Default: GWAS_jobs.csv"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing Feature4.csv"
    )

    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine all per-GWAS Feature4.csv into one CSV."
    )

    parser.add_argument(
        "--combine-output",
        default="Variation4_allGWAS.csv",
        help="Output CSV for --combine. Default: Variation4_allGWAS.csv"
    )

    args = parser.parse_args()

    if args.combine:
        combine_feature4(
            jobs_file=args.jobs_file,
            output_file=args.combine_output,
        )
        return

    if args.index is None:
        raise ValueError("Please provide a job index, e.g. python Variation4.py 1")

    run_one_job(
        index=args.index,
        jobs_file=args.jobs_file,
        force=args.force,
    )


if __name__ == "__main__":
    main()
