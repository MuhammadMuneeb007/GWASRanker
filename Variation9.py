#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_BASE_DIR = "/data/ascher01/uqmmune1/finalheritability"


# =============================================================================
# BASIC HELPERS
# =============================================================================
def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def percent(numerator, denominator):
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 6)


def normalise_id(x):
    return clean_text(x).lower()


def normalise_chr(x):
    x = clean_text(x)
    x = x.replace("chr", "").replace("CHR", "").strip()

    if x == "23":
        return "X"
    if x == "24":
        return "Y"
    if x in ["25", "M", "MT", "Mt", "mt"]:
        return "MT"

    return x


def normalise_pos(x):
    try:
        if pd.isna(x):
            return ""
        value = int(float(str(x).replace(",", "").strip()))
        if value > 0:
            return str(value)
    except Exception:
        pass
    return ""


def normalise_allele(x):
    x = clean_text(x).upper()
    if x in ["", "NAN", "NA", "N/A", "NONE", "NULL", "."]:
        return ""
    return x


def natural_fold_number(path):
    name = Path(path).name
    if name.startswith("Fold_"):
        try:
            return int(name.split("_", 1)[1])
        except Exception:
            return 10**9
    return 10**9


def is_palindromic_pair(a1, a2):
    a1 = normalise_allele(a1)
    a2 = normalise_allele(a2)
    return len(a1) == 1 and len(a2) == 1 and {a1, a2} in [{"A", "T"}, {"C", "G"}]


def complement_allele(allele):
    allele = normalise_allele(allele)
    if len(allele) != 1:
        return ""
    return {"A": "T", "T": "A", "C": "G", "G": "C"}.get(allele, "")


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def is_valid_file(path):
    if path is None:
        return False

    path = str(path).strip()
    if path == "" or path == "." or path.lower() in {"nan", "none", "null"}:
        return False

    p = Path(path)
    return p.exists() and p.is_file()


def deduplicate_best_p_rows(df, id_col, p_col):
    if df.empty:
        return df.copy()

    working = df.reset_index(drop=True).copy()
    working[p_col] = pd.to_numeric(working[p_col], errors="coerce")
    working = working.loc[
        working[id_col].notna()
        & working[p_col].notna()
        & np.isfinite(working[p_col])
    ].copy()

    if working.empty:
        return working

    best_idx = working.groupby(id_col, sort=False)[p_col].idxmin()
    working = working.loc[best_idx].copy()
    order = np.argsort(working[p_col].to_numpy(dtype=np.float64, copy=False), kind="stable")
    working = working.iloc[order].reset_index(drop=True)
    return working


def finite_series(series):
    return safe_numeric(series).replace([np.inf, -np.inf], np.nan).dropna()


def correlation_or_blank(x, y):
    valid = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()

    if len(valid) < 3:
        return ""

    return round(float(valid["x"].corr(valid["y"])), 6)


def quantile_or_blank(series, q):
    values = finite_series(series)
    if values.empty:
        return ""
    return round(float(values.quantile(q)), 8)


def mean_or_blank(series):
    values = finite_series(series)
    if values.empty:
        return ""
    return round(float(values.mean()), 8)


def median_or_blank(series):
    values = finite_series(series)
    if values.empty:
        return ""
    return round(float(values.median()), 8)


def sd_or_blank(series):
    values = finite_series(series)
    if len(values) < 2:
        return ""
    return round(float(values.std()), 8)


def min_or_blank(series):
    values = finite_series(series)
    if values.empty:
        return ""
    return round(float(values.min()), 12)


def max_or_blank(series):
    values = finite_series(series)
    if values.empty:
        return ""
    return round(float(values.max()), 12)


def skew_or_blank(series):
    values = finite_series(series)
    if len(values) < 3:
        return ""
    return round(float(values.skew()), 8)


def kurtosis_or_blank(series):
    values = finite_series(series)
    if len(values) < 4:
        return ""
    return round(float(values.kurtosis()), 8)


def run_command(cmd, log_path):
    completed = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )

    log_text = ""
    log_text += "[COMMAND]\n"
    log_text += " ".join(str(x) for x in cmd) + "\n\n"
    log_text += "[STDOUT]\n"
    log_text += completed.stdout or ""
    log_text += "\n\n[STDERR]\n"
    log_text += completed.stderr or ""

    Path(log_path).write_text(log_text, encoding="utf-8", errors="replace")

    return completed.returncode, log_text


# =============================================================================
# JOB RESOLUTION
# =============================================================================

def resolve_job_from_manifest(index, jobs_file):
    jobs_path = Path(jobs_file)

    if not jobs_path.exists():
        raise FileNotFoundError(f"Missing jobs file: {jobs_path}")

    jobs = pd.read_csv(jobs_path)

    required = ["job_index", "phenotype", "accessionId", "file_path"]
    missing = [c for c in required if c not in jobs.columns]

    if missing:
        raise ValueError(f"Missing required columns in {jobs_path}: {missing}")

    jobs["job_index"] = pd.to_numeric(jobs["job_index"], errors="coerce").astype("Int64")
    sub = jobs[jobs["job_index"] == int(index)]

    if sub.empty:
        raise ValueError(f"No job found for job_index={index} in {jobs_path}")

    row = sub.iloc[0]

    return {
        "job_index": int(row["job_index"]),
        "phenotype": clean_text(row["phenotype"]),
        "accessionId": clean_text(row["accessionId"]),
        "file_path": clean_text(row["file_path"]),
        "output_feature_file": clean_text(row.get("output_feature_file", "")),
    }


def resolve_path(path_text, base_dir):
    path = Path(path_text)

    if path.is_absolute():
        return path

    candidate = Path(base_dir) / path
    if candidate.exists():
        return candidate

    return path


# =============================================================================
# FIND INPUT FILES
# =============================================================================

def find_final_gwas(job, base_dir):
    file_path = resolve_path(job["file_path"], base_dir)
    accession_dir = file_path.parent

    candidates = [
        accession_dir / "FinalGWAS.csv",
        accession_dir / "finalgwas.csv",
        accession_dir / "FinalGWAS.tsv",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find FinalGWAS.csv for this job.\n"
        f"Expected location: {accession_dir / 'FinalGWAS.csv'}\n"
        "Run your FinalGWAS creation step first."
    )


def find_phenotype_dir(phenotype, base_dir):
    base_dir = Path(base_dir)

    candidates = [
        base_dir / phenotype,
        Path.cwd() / phenotype,
        Path.cwd(),
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    return base_dir / phenotype


def find_target_bim(phenotype, base_dir, explicit_bim=""):
    if explicit_bim:
        explicit = Path(explicit_bim)
        if explicit.exists():
            return explicit
        raise FileNotFoundError(f"Explicit BIM file does not exist: {explicit}")

    phenotype_dir = find_phenotype_dir(phenotype, base_dir)

    candidate_names = [
        f"{phenotype}_QC.bim",
        f"{phenotype}.bim",
        f"{phenotype.lower()}_QC.bim",
        f"{phenotype.lower()}.bim",
        f"{phenotype.upper()}_QC.bim",
        f"{phenotype.upper()}.bim",
    ]

    candidates = []

    fold_dirs = []
    if phenotype_dir.exists():
        fold_dirs = sorted(
            [p for p in phenotype_dir.glob("Fold_*") if p.is_dir()],
            key=natural_fold_number,
        )

    for fold_dir in fold_dirs:
        candidates.extend([
            fold_dir / "train_data.QC.bim",
            fold_dir / "train_data.bim",
            fold_dir / "test_data.QC.bim",
            fold_dir / "test_data.bim",
        ])

    for name in candidate_names:
        candidates.append(phenotype_dir / name)
        candidates.append(Path.cwd() / name)
        candidates.append(Path(base_dir) / name)

    if phenotype_dir.exists():
        for fold_dir in fold_dirs:
            candidates.extend(sorted(fold_dir.glob("*.bim")))
        candidates.extend(sorted(phenotype_dir.glob("*.bim")))

    seen = set()
    unique_candidates = []

    for c in candidates:
        key = str(c)
        if key not in seen:
            unique_candidates.append(c)
            seen.add(key)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find target .bim file for phenotype={phenotype}.\n"
        "Expected something like finalheritability/<phenotype>/Fold_*/train_data.QC.bim,\n"
        "finalheritability/<phenotype>/Fold_*/test_data.bim, asthma_QC.bim, or asthma.bim.\n"
        "Use --target-bim if needed."
    )


def infer_bfile_prefix_from_bim(bim_path):
    return Path(bim_path).with_suffix("")


def bfile_component_path(bfile_prefix, suffix):
    return Path(f"{bfile_prefix}{suffix}")


def find_plink(plink_arg=""):
    if plink_arg:
        p = Path(plink_arg)
        if p.exists():
            return str(p)
        return plink_arg

    cwd_plink = Path.cwd() / "plink"
    if cwd_plink.exists():
        return str(cwd_plink)

    cwd_plink2 = Path.cwd() / "plink2"
    if cwd_plink2.exists():
        return str(cwd_plink2)

    return "plink"


# =============================================================================
# LOAD DATA
# =============================================================================

def load_final_gwas(path):
    path = Path(path)

    if path.suffix.lower() == ".tsv":
        df = pd.read_csv(path, sep="\t", low_memory=False)
    else:
        df = pd.read_csv(path, low_memory=False)

    df.columns = [clean_text(c) for c in df.columns]
    return df


def load_bim(path):
    bim = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["BIM_CHR", "BIM_SNPID", "BIM_CM", "BIM_POS", "BIM_A1", "BIM_A2"],
        dtype=str,
        engine="python",
    )

    bim["BIM_ID_NORM"] = bim["BIM_SNPID"].map(normalise_id)
    bim["BIM_CHR_NORM"] = bim["BIM_CHR"].map(normalise_chr)
    bim["BIM_POS_NORM"] = bim["BIM_POS"].map(normalise_pos)
    bim["BIM_CHRPOS"] = bim["BIM_CHR_NORM"] + ":" + bim["BIM_POS_NORM"]

    bim.loc[
        (bim["BIM_CHR_NORM"] == "") | (bim["BIM_POS_NORM"] == ""),
        "BIM_CHRPOS",
    ] = ""

    bim["BIM_A1_NORM"] = bim["BIM_A1"].map(normalise_allele)
    bim["BIM_A2_NORM"] = bim["BIM_A2"].map(normalise_allele)

    return bim


def infer_target_trait_type_from_fam(fam_path):
    fam_path = Path(fam_path)
    if not fam_path.exists():
        return "unknown"

    fam = pd.read_csv(
        fam_path,
        sep=r"\s+",
        header=None,
        names=["FID", "IID", "PID", "MID", "SEX", "PHENO"],
        dtype=str,
        engine="python",
    )
    pheno = pd.to_numeric(fam["PHENO"], errors="coerce")
    pheno = pheno.dropna()
    pheno = pheno[~pheno.isin([-9])]
    values = sorted(pheno.unique().tolist())

    if not values:
        return "unknown"
    if values == [1, 2] or values == [0, 1] or len(values) == 2:
        return "binary"
    return "continuous"


def prepare_gwas(gwas):
    gwas = gwas.copy()

    for col in [
        "SNPID", "CHR", "POS", "EA", "NEA", "BETA", "OR",
        "SE", "Z", "P", "N", "EAF", "MAF", "INFO"
    ]:
        if col not in gwas.columns:
            gwas[col] = ""

    gwas["GWAS_ROW_ID"] = np.arange(len(gwas))
    gwas["GWAS_ID_RAW"] = gwas["SNPID"].astype(str).map(clean_text)
    gwas["GWAS_ID_NORM"] = gwas["GWAS_ID_RAW"].map(normalise_id)

    gwas["GWAS_CHR_NORM"] = gwas["CHR"].map(normalise_chr)
    gwas["GWAS_POS_NORM"] = gwas["POS"].map(normalise_pos)
    gwas["GWAS_CHRPOS"] = gwas["GWAS_CHR_NORM"] + ":" + gwas["GWAS_POS_NORM"]

    gwas.loc[
        (gwas["GWAS_CHR_NORM"] == "") | (gwas["GWAS_POS_NORM"] == ""),
        "GWAS_CHRPOS",
    ] = ""

    gwas["GWAS_EA_NORM"] = gwas["EA"].map(normalise_allele)
    gwas["GWAS_NEA_NORM"] = gwas["NEA"].map(normalise_allele)

    gwas["BETA_NUM"] = safe_numeric(gwas["BETA"])
    gwas["OR_NUM"] = safe_numeric(gwas["OR"])
    gwas["SE_NUM"] = safe_numeric(gwas["SE"])
    gwas["Z_NUM"] = safe_numeric(gwas["Z"])
    gwas["P_NUM"] = safe_numeric(gwas["P"])
    gwas["N_NUM"] = safe_numeric(gwas["N"])
    gwas["EAF_NUM"] = safe_numeric(gwas["EAF"])
    gwas["MAF_NUM"] = safe_numeric(gwas["MAF"])
    gwas["INFO_NUM"] = safe_numeric(gwas["INFO"])

    valid_or = gwas["OR_NUM"].notna() & (gwas["OR_NUM"] > 0)
    gwas["BETA_FOR_DIST"] = gwas["BETA_NUM"]
    fill_beta_from_or = gwas["BETA_FOR_DIST"].isna() & valid_or
    gwas.loc[fill_beta_from_or, "BETA_FOR_DIST"] = np.log(gwas.loc[fill_beta_from_or, "OR_NUM"])

    valid_se = gwas["SE_NUM"].notna() & (gwas["SE_NUM"] > 0)
    gwas["Z_FOR_DIST"] = gwas["Z_NUM"]
    fill_z = gwas["Z_FOR_DIST"].isna() & gwas["BETA_FOR_DIST"].notna() & valid_se
    gwas.loc[fill_z, "Z_FOR_DIST"] = (
        gwas.loc[fill_z, "BETA_FOR_DIST"] / gwas.loc[fill_z, "SE_NUM"]
    )

    gwas["ABS_BETA_FOR_DIST"] = gwas["BETA_FOR_DIST"].abs()
    gwas["ABS_Z_FOR_DIST"] = gwas["Z_FOR_DIST"].abs()

    valid_p = gwas["P_NUM"].notna() & (gwas["P_NUM"] > 0) & (gwas["P_NUM"] <= 1)
    gwas["NEG_LOG10_P"] = np.nan
    gwas.loc[valid_p, "NEG_LOG10_P"] = -np.log10(gwas.loc[valid_p, "P_NUM"])

    return gwas


# =============================================================================
# EFFECT-SIZE AND P-VALUE FEATURES
# =============================================================================

def calculate_effect_size_distribution_features(gwas):
    result = {}
    n = len(gwas)

    beta = gwas["BETA_FOR_DIST"]
    abs_beta = gwas["ABS_BETA_FOR_DIST"]
    beta_valid = finite_series(beta)
    abs_beta_valid = finite_series(abs_beta)

    result["effect_beta_available_count"] = int(beta_valid.shape[0])
    result["effect_beta_available_pct"] = percent(result["effect_beta_available_count"], n)
    result["effect_beta_from_or_count"] = int(
        (gwas["BETA_NUM"].isna() & gwas["OR_NUM"].notna() & (gwas["OR_NUM"] > 0)).sum()
    )

    result["beta_mean"] = mean_or_blank(beta)
    result["beta_sd"] = sd_or_blank(beta)
    result["beta_min"] = min_or_blank(beta)
    result["beta_max"] = max_or_blank(beta)
    result["beta_median"] = median_or_blank(beta)
    result["beta_median_abs"] = median_or_blank(abs_beta)
    result["beta_abs_q90"] = quantile_or_blank(abs_beta, 0.90)
    result["beta_abs_q95"] = quantile_or_blank(abs_beta, 0.95)
    result["beta_abs_q99"] = quantile_or_blank(abs_beta, 0.99)
    result["beta_abs_q999"] = quantile_or_blank(abs_beta, 0.999)
    result["beta_skewness"] = skew_or_blank(beta)
    result["beta_kurtosis"] = kurtosis_or_blank(beta)

    if not abs_beta_valid.empty:
        result["extreme_beta_abs_gt_1_count"] = int((abs_beta_valid > 1).sum())
        result["extreme_beta_abs_gt_1_pct"] = percent((abs_beta_valid > 1).sum(), len(abs_beta_valid))
        result["extreme_beta_abs_gt_2_count"] = int((abs_beta_valid > 2).sum())
        result["extreme_beta_abs_gt_2_pct"] = percent((abs_beta_valid > 2).sum(), len(abs_beta_valid))
        result["extreme_beta_abs_gt_5_count"] = int((abs_beta_valid > 5).sum())
        result["extreme_beta_abs_gt_5_pct"] = percent((abs_beta_valid > 5).sum(), len(abs_beta_valid))
    else:
        result["extreme_beta_abs_gt_1_count"] = 0
        result["extreme_beta_abs_gt_1_pct"] = 0.0
        result["extreme_beta_abs_gt_2_count"] = 0
        result["extreme_beta_abs_gt_2_pct"] = 0.0
        result["extreme_beta_abs_gt_5_count"] = 0
        result["extreme_beta_abs_gt_5_pct"] = 0.0

    valid_or = finite_series(gwas["OR_NUM"])
    valid_or = valid_or[valid_or > 0]

    result["or_available_count"] = int(valid_or.shape[0])
    result["or_available_pct"] = percent(result["or_available_count"], n)
    result["or_median"] = median_or_blank(valid_or)
    result["or_q95"] = quantile_or_blank(valid_or, 0.95)
    result["or_q99"] = quantile_or_blank(valid_or, 0.99)

    if not valid_or.empty:
        result["extreme_or_gt_10_count"] = int((valid_or > 10).sum())
        result["extreme_or_gt_10_pct"] = percent((valid_or > 10).sum(), len(valid_or))
        result["extreme_or_lt_0_1_count"] = int((valid_or < 0.1).sum())
        result["extreme_or_lt_0_1_pct"] = percent((valid_or < 0.1).sum(), len(valid_or))
    else:
        result["extreme_or_gt_10_count"] = 0
        result["extreme_or_gt_10_pct"] = 0.0
        result["extreme_or_lt_0_1_count"] = 0
        result["extreme_or_lt_0_1_pct"] = 0.0

    valid_se = finite_series(gwas["SE_NUM"])
    valid_se = valid_se[valid_se > 0]

    result["se_available_count"] = int(valid_se.shape[0])
    result["se_available_pct"] = percent(result["se_available_count"], n)
    result["se_median"] = median_or_blank(valid_se)
    result["se_q95"] = quantile_or_blank(valid_se, 0.95)
    result["se_q99"] = quantile_or_blank(valid_se, 0.99)
    result["se_zero_or_negative_count"] = int((safe_numeric(gwas["SE_NUM"]) <= 0).sum())

    z = gwas["Z_FOR_DIST"]
    abs_z = gwas["ABS_Z_FOR_DIST"]

    result["z_available_count"] = int(finite_series(z).shape[0])
    result["z_available_pct"] = percent(result["z_available_count"], n)
    result["z_abs_median"] = median_or_blank(abs_z)
    result["z_abs_q95"] = quantile_or_blank(abs_z, 0.95)
    result["z_abs_q99"] = quantile_or_blank(abs_z, 0.99)
    result["z_abs_max"] = max_or_blank(abs_z)

    return result


def calculate_pvalue_distribution_features(gwas):
    result = {}
    n = len(gwas)

    p = gwas["P_NUM"]
    valid_p = p[p.notna() & (p > 0) & (p <= 1)]
    invalid_p = p[p.notna() & ((p <= 0) | (p > 1))]

    result["p_available_count"] = int(p.notna().sum())
    result["p_available_pct"] = percent(result["p_available_count"], n)
    result["p_valid_count"] = int(valid_p.shape[0])
    result["p_valid_pct"] = percent(result["p_valid_count"], n)
    result["p_missing_count"] = int(p.isna().sum())
    result["p_missing_pct"] = percent(result["p_missing_count"], n)
    result["p_invalid_count"] = int(invalid_p.shape[0])
    result["p_invalid_pct"] = percent(result["p_invalid_count"], n)
    result["p_zero_count"] = int((p == 0).sum())
    result["p_zero_pct"] = percent(result["p_zero_count"], n)

    result["p_min"] = min_or_blank(valid_p)
    result["p_median"] = median_or_blank(valid_p)
    result["p_q01"] = quantile_or_blank(valid_p, 0.01)
    result["p_q05"] = quantile_or_blank(valid_p, 0.05)

    neglog = gwas["NEG_LOG10_P"]
    result["neglog10p_max"] = max_or_blank(neglog)
    result["neglog10p_median"] = median_or_blank(neglog)
    result["neglog10p_q95"] = quantile_or_blank(neglog, 0.95)
    result["neglog10p_q99"] = quantile_or_blank(neglog, 0.99)

    thresholds = [
        ("5e8", 5e-8),
        ("1e6", 1e-6),
        ("1e5", 1e-5),
        ("1e4", 1e-4),
        ("1e3", 1e-3),
        ("1e2", 1e-2),
        ("0_05", 0.05),
        ("0_1", 0.1),
    ]

    for label, threshold in thresholds:
        count = int((valid_p <= threshold).sum())
        result[f"p_le_{label}_count"] = count
        result[f"p_le_{label}_pct"] = percent(count, len(valid_p))

    return result


def beta_orientation_status(gwas_ea, gwas_nea, target_a1, target_a2):
    gwas_ea = normalise_allele(gwas_ea)
    gwas_nea = normalise_allele(gwas_nea)
    target_a1 = normalise_allele(target_a1)
    target_a2 = normalise_allele(target_a2)

    if not gwas_ea or not gwas_nea or not target_a1 or not target_a2:
        return "missing_allele"

    if is_palindromic_pair(gwas_ea, gwas_nea) and {gwas_ea, gwas_nea} == {target_a1, target_a2}:
        return "palindromic_ambiguous"

    if gwas_ea == target_a1 and gwas_nea == target_a2:
        return "direct_match"
    if gwas_ea == target_a2 and gwas_nea == target_a1:
        return "flip_match"

    comp_ea = complement_allele(gwas_ea)
    comp_nea = complement_allele(gwas_nea)
    if comp_ea and comp_nea:
        if comp_ea == target_a1 and comp_nea == target_a2:
            return "strand_match"
        if comp_ea == target_a2 and comp_nea == target_a1:
            return "strand_flip_match"

    return "mismatch"


def build_gwas_target_mapping(gwas, bim):
    compatible_statuses = {"direct_match", "flip_match", "strand_match", "strand_flip_match"}
    target = bim[
        ["BIM_ID_NORM", "BIM_SNPID", "BIM_CHRPOS", "BIM_A1_NORM", "BIM_A2_NORM"]
    ].drop_duplicates("BIM_ID_NORM", keep="first")

    by_id = gwas[gwas["GWAS_ID_NORM"] != ""].merge(
        target[target["BIM_ID_NORM"] != ""],
        left_on="GWAS_ID_NORM",
        right_on="BIM_ID_NORM",
        how="inner",
    )

    coord_target = bim[
        ["BIM_ID_NORM", "BIM_SNPID", "BIM_CHRPOS", "BIM_A1_NORM", "BIM_A2_NORM"]
    ].drop_duplicates("BIM_CHRPOS", keep="first")
    unmatched = gwas[~gwas["GWAS_ROW_ID"].isin(by_id["GWAS_ROW_ID"]) & (gwas["GWAS_CHRPOS"] != "")]
    by_coord = unmatched.merge(
        coord_target[coord_target["BIM_CHRPOS"] != ""],
        left_on="GWAS_CHRPOS",
        right_on="BIM_CHRPOS",
        how="inner",
    )

    mapped = pd.concat([by_id, by_coord], ignore_index=True, sort=False)
    if mapped.empty:
        mapped["TARGET_ALLELE_STATUS"] = pd.Series(dtype="object")
        return mapped

    mapped["TARGET_ALLELE_STATUS"] = [
        beta_orientation_status(g_ea, g_nea, b_a1, b_a2)
        for g_ea, g_nea, b_a1, b_a2 in zip(
            mapped["GWAS_EA_NORM"],
            mapped["GWAS_NEA_NORM"],
            mapped["BIM_A1_NORM"],
            mapped["BIM_A2_NORM"],
        )
    ]
    mapped = mapped[mapped["TARGET_ALLELE_STATUS"].isin(compatible_statuses)]
    return deduplicate_best_p_rows(mapped, id_col="GWAS_ROW_ID", p_col="P_NUM")


def calculate_prs_input_readiness_features(gwas, bim):
    result = {}
    n = len(gwas)

    has_snp = gwas["GWAS_ID_NORM"] != ""
    has_p = gwas["P_NUM"].notna() & (gwas["P_NUM"] > 0) & (gwas["P_NUM"] <= 1)
    has_effect = gwas["BETA_FOR_DIST"].notna()
    has_alleles = (gwas["GWAS_EA_NORM"] != "") & (gwas["GWAS_NEA_NORM"] != "")
    target_mapping = build_gwas_target_mapping(gwas, bim)
    in_target_compatible = gwas["GWAS_ROW_ID"].isin(target_mapping["GWAS_ROW_ID"])

    clump_ready = has_snp & has_p & in_target_compatible
    score_ready = has_snp & has_effect & has_alleles & in_target_compatible
    prs_ready = has_snp & has_p & has_effect & has_alleles & in_target_compatible
    target_prs_ready = prs_ready

    result["prs_has_snpid_count"] = int(has_snp.sum())
    result["prs_has_snpid_pct"] = percent(has_snp.sum(), n)

    result["prs_has_valid_p_count"] = int(has_p.sum())
    result["prs_has_valid_p_pct"] = percent(has_p.sum(), n)

    result["prs_has_effect_count"] = int(has_effect.sum())
    result["prs_has_effect_pct"] = percent(has_effect.sum(), n)
    result["prs_has_alleles_count"] = int(has_alleles.sum())
    result["prs_has_alleles_pct"] = percent(has_alleles.sum(), n)

    result["prs_clump_ready_count"] = int(clump_ready.sum())
    result["prs_clump_ready_pct"] = percent(clump_ready.sum(), n)

    result["prs_score_ready_count"] = int(score_ready.sum())
    result["prs_score_ready_pct"] = percent(score_ready.sum(), n)

    result["prs_full_ready_count"] = int(prs_ready.sum())
    result["prs_full_ready_pct"] = percent(prs_ready.sum(), n)

    result["prs_full_ready_and_in_target_count"] = int(target_prs_ready.sum())
    result["prs_full_ready_and_in_target_pct_of_gwas"] = percent(target_prs_ready.sum(), n)
    result["prs_full_ready_and_in_target_pct_of_ready"] = percent(target_prs_ready.sum(), prs_ready.sum())

    duplicated_snp_count = int(gwas.loc[has_snp, "GWAS_ID_NORM"].duplicated().sum())
    result["prs_duplicate_snpid_count"] = duplicated_snp_count
    result["prs_duplicate_snpid_pct_of_snpid_available"] = percent(duplicated_snp_count, has_snp.sum())

    return result


# =============================================================================
# PLINK INPUT CREATION
# =============================================================================

def make_plink_clump_input(gwas, bim, output_dir):
    output_dir = Path(output_dir)
    output_path = output_dir / "Feature9_clump_input.tsv"

    clump = build_gwas_target_mapping(gwas, bim)
    clump = clump[
        clump["P_NUM"].notna()
        & (clump["P_NUM"] > 0)
        & (clump["P_NUM"] <= 1)
    ][["BIM_SNPID", "P_NUM"]].copy()

    if clump.empty:
        pd.DataFrame(columns=["SNP", "P"]).to_csv(output_path, sep="\t", index=False)
        return output_path, 0

    clump = clump.rename(columns={"BIM_SNPID": "SNP", "P_NUM": "P"})
    clump["P"] = pd.to_numeric(clump["P"], errors="coerce")
    clump = clump.dropna(subset=["P"])
    clump = deduplicate_best_p_rows(clump, id_col="SNP", p_col="P")
    clump = clump[["SNP", "P"]]
    clump.to_csv(output_path, sep="\t", index=False)

    return output_path, int(len(clump))


def make_plink_extract_file(gwas, bim, output_dir):
    output_dir = Path(output_dir)
    output_path = output_dir / "Feature9_extract_snps.txt"

    ids = build_gwas_target_mapping(gwas, bim)[["BIM_SNPID"]].drop_duplicates()

    if ids.empty:
        output_path.write_text("", encoding="utf-8")
        return output_path, 0

    ids["BIM_SNPID"].drop_duplicates().to_csv(output_path, index=False, header=False)

    return output_path, int(ids["BIM_SNPID"].nunique())


# =============================================================================
# TARGET-DERIVED GWAS USING PLINK --FISHER
# =============================================================================

def run_plink_target_assoc(plink, bfile_prefix, extract_file, output_dir, trait_type):
    output_dir = Path(output_dir)
    out_prefix = output_dir / "Feature9_target_assoc"
    log_path = output_dir / "Feature9_target_assoc.log"

    result = {
        "target_fisher_status": "",
        "target_fisher_returncode": "",
        "target_fisher_log_file": str(log_path),
        "target_fisher_assoc_file": "",
        "target_fisher_input_snp_count": 0,
        "target_assoc_method": "",
    }

    if not Path(extract_file).exists() or Path(extract_file).stat().st_size == 0:
        result["target_fisher_status"] = "skipped_empty_extract"
        return result

    input_count = sum(1 for _ in open(extract_file, "rt", encoding="utf-8", errors="replace"))
    result["target_fisher_input_snp_count"] = int(input_count)

    if trait_type == "continuous":
        assoc_method = "linear"
        cmd = [
            plink,
            "--bfile", str(bfile_prefix),
            "--extract", str(extract_file),
            "--allow-no-sex",
            "--linear",
            "--out", str(out_prefix),
        ]
    else:
        assoc_method = "fisher"
        cmd = [
            plink,
            "--bfile", str(bfile_prefix),
            "--extract", str(extract_file),
            "--allow-no-sex",
            "--fisher",
            "--out", str(out_prefix),
        ]

    returncode, _ = run_command(cmd, log_path)

    assoc_fisher = out_prefix.with_suffix(".assoc.fisher")
    assoc_linear = out_prefix.with_suffix(".assoc.linear")
    assoc = out_prefix.with_suffix(".assoc")

    if assoc_fisher.exists():
        assoc_file = assoc_fisher
    elif assoc_linear.exists():
        assoc_file = assoc_linear
    elif assoc.exists():
        assoc_file = assoc
    else:
        assoc_file = ""

    result["target_fisher_returncode"] = returncode
    result["target_fisher_status"] = "success" if returncode == 0 and assoc_file else "failed_or_missing_output"
    result["target_fisher_assoc_file"] = str(assoc_file) if assoc_file else ""
    result["target_assoc_method"] = assoc_method

    return result


def choose_column(columns, candidates):
    lower_to_original = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return ""


def load_plink_fisher_assoc(path, bim=None):
    if not is_valid_file(path):
        print(f"[WARNING] Invalid or missing PLINK Fisher assoc file: {path}")
        return pd.DataFrame()

    path = Path(path)

    try:
        assoc = pd.read_csv(path, sep=r"\s+", engine="python", dtype=str)
    except Exception as e:
        print(f"[WARNING] Could not read PLINK Fisher assoc file: {path}")
        print(f"[WARNING] {type(e).__name__}: {e}")
        return pd.DataFrame()

    if assoc.empty:
        print(f"[WARNING] PLINK Fisher assoc file is empty: {path}")
        return pd.DataFrame()

    assoc.columns = [clean_text(c) for c in assoc.columns]

    snp_col = choose_column(assoc.columns, ["SNP", "ID", "SNPID", "rsID"])
    chr_col = choose_column(assoc.columns, ["CHR", "CHROM", "CHROMOSOME"])
    bp_col = choose_column(assoc.columns, ["BP", "POS", "POSITION"])
    a1_col = choose_column(assoc.columns, ["A1", "ALLELE1"])
    a2_col = choose_column(assoc.columns, ["A2", "ALLELE2"])
    p_col = choose_column(assoc.columns, ["P", "P_FISHER", "P_VALUE", "PVAL"])
    or_col = choose_column(assoc.columns, ["OR", "ODDS_RATIO"])
    beta_col = choose_column(assoc.columns, ["BETA", "EFF", "EFFECT"])

    if not snp_col:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["TARGET_SNPID"] = assoc[snp_col].astype(str).map(clean_text)
    out["TARGET_ID_NORM"] = out["TARGET_SNPID"].map(normalise_id)

    out["TARGET_CHR"] = assoc[chr_col].map(normalise_chr) if chr_col else ""
    out["TARGET_POS"] = assoc[bp_col].map(normalise_pos) if bp_col else ""
    out["TARGET_CHRPOS"] = out["TARGET_CHR"].astype(str) + ":" + out["TARGET_POS"].astype(str)

    out.loc[
        (out["TARGET_CHR"].astype(str) == "") | (out["TARGET_POS"].astype(str) == ""),
        "TARGET_CHRPOS",
    ] = ""

    out["TARGET_A1"] = assoc[a1_col].map(normalise_allele) if a1_col else ""
    out["TARGET_A2"] = assoc[a2_col].map(normalise_allele) if a2_col else ""

    out["TARGET_P"] = safe_numeric(assoc[p_col]) if p_col else np.nan
    out["TARGET_OR"] = safe_numeric(assoc[or_col]) if or_col else np.nan
    out["TARGET_BETA"] = safe_numeric(assoc[beta_col]) if beta_col else np.nan

    valid_or = out["TARGET_OR"].notna() & (out["TARGET_OR"] > 0)
    out["TARGET_BETA_LOG_OR"] = np.nan
    out.loc[valid_or, "TARGET_BETA_LOG_OR"] = np.log(out.loc[valid_or, "TARGET_OR"])
    out.loc[out["TARGET_BETA_LOG_OR"].isna() & out["TARGET_BETA"].notna(), "TARGET_BETA_LOG_OR"] = out.loc[
        out["TARGET_BETA_LOG_OR"].isna() & out["TARGET_BETA"].notna(),
        "TARGET_BETA",
    ]

    if bim is not None and not bim.empty:
        bim_lookup = bim[
            ["BIM_ID_NORM", "BIM_CHRPOS", "BIM_A1_NORM", "BIM_A2_NORM"]
        ].drop_duplicates("BIM_ID_NORM", keep="first").copy()

        out = out.merge(
            bim_lookup,
            left_on="TARGET_ID_NORM",
            right_on="BIM_ID_NORM",
            how="left",
        )

        # Recover missing coordinate information from BIM when association output is sparse.
        missing_chrpos = out["TARGET_CHRPOS"].astype(str).eq("")
        out.loc[missing_chrpos, "TARGET_CHRPOS"] = out.loc[missing_chrpos, "BIM_CHRPOS"].fillna("")

        chrpos_parts = out["TARGET_CHRPOS"].astype(str).str.split(":", n=1, expand=True)
        if chrpos_parts.shape[1] >= 2:
            missing_chr = out["TARGET_CHR"].astype(str).eq("")
            missing_pos = out["TARGET_POS"].astype(str).eq("")
            out.loc[missing_chr, "TARGET_CHR"] = chrpos_parts[0].fillna("")
            out.loc[missing_pos, "TARGET_POS"] = chrpos_parts[1].fillna("")

        # Recover missing alleles for PLINK --linear outputs, which often only carry A1.
        target_a1 = out["TARGET_A1"].astype(str)
        target_a2 = out["TARGET_A2"].astype(str)
        bim_a1 = out["BIM_A1_NORM"].fillna("").astype(str)
        bim_a2 = out["BIM_A2_NORM"].fillna("").astype(str)

        missing_a1 = target_a1.eq("")
        out.loc[missing_a1, "TARGET_A1"] = bim_a1[missing_a1]

        missing_a2 = target_a2.eq("")
        a1_matches_bim_a1 = out["TARGET_A1"].astype(str).eq(bim_a1) & missing_a2
        a1_matches_bim_a2 = out["TARGET_A1"].astype(str).eq(bim_a2) & missing_a2
        out.loc[a1_matches_bim_a1, "TARGET_A2"] = bim_a2[a1_matches_bim_a1]
        out.loc[a1_matches_bim_a2, "TARGET_A2"] = bim_a1[a1_matches_bim_a2]

        still_missing_a2 = out["TARGET_A2"].astype(str).eq("") & missing_a2
        out.loc[still_missing_a2, "TARGET_A2"] = bim_a2[still_missing_a2]

        out = out.drop(columns=["BIM_ID_NORM", "BIM_CHRPOS", "BIM_A1_NORM", "BIM_A2_NORM"], errors="ignore")

    out = out.drop_duplicates("TARGET_ID_NORM", keep="first")

    return out


def calculate_target_fisher_comparison_features(gwas, fisher_assoc_file, bim=None):
    result = {
        "target_fisher_available": 0,
        "target_fisher_assoc_file_checked": str(fisher_assoc_file) if fisher_assoc_file is not None else "",
        "target_fisher_assoc_variant_count": "",
        "target_fisher_beta_available_count": "",
        "target_fisher_p_available_count": "",

        "target_fisher_common_snp_count": 0,
        "target_fisher_common_beta_count": 0,
        "target_fisher_common_p_count": 0,

        "target_fisher_allele_direct_match_count": 0,
        "target_fisher_allele_flip_match_count": 0,
        "target_fisher_allele_strand_match_count": 0,
        "target_fisher_allele_strand_flip_match_count": 0,
        "target_fisher_palindromic_ambiguous_count": 0,
        "target_fisher_allele_mismatch_count": 0,
        "target_fisher_allele_missing_count": 0,
        "target_fisher_allele_compatible_count": 0,
        "target_fisher_allele_compatible_pct": 0.0,

        "target_fisher_beta_corr_raw_all_common": "",
        "target_fisher_beta_corr_signed_compatible": "",
        "target_fisher_abs_beta_corr_compatible": "",
        "target_fisher_beta_abs_diff_mean_compatible": "",
        "target_fisher_beta_abs_diff_median_compatible": "",

        "target_fisher_pvalue_corr_all_common": "",
        "target_fisher_neglog10p_corr_all_common": "",
        "target_fisher_pvalue_corr_compatible": "",
        "target_fisher_neglog10p_corr_compatible": "",
    }

    fisher = load_plink_fisher_assoc(fisher_assoc_file, bim=bim)

    if fisher.empty:
        print("[WARNING] Fisher comparison skipped because Fisher assoc file is unavailable.")
        return result

    result["target_fisher_available"] = 1
    result["target_fisher_assoc_variant_count"] = int(len(fisher))
    result["target_fisher_beta_available_count"] = int(fisher["TARGET_BETA_LOG_OR"].notna().sum())
    result["target_fisher_p_available_count"] = int(fisher["TARGET_P"].notna().sum())

    if bim is not None:
        gwas_sub = build_gwas_target_mapping(gwas, bim)
        merged = gwas_sub.merge(
            fisher,
            left_on="BIM_ID_NORM",
            right_on="TARGET_ID_NORM",
            how="inner",
        )
    else:
        gwas_sub = gwas[
            (gwas["GWAS_ID_NORM"] != "")
        ].drop_duplicates("GWAS_ID_NORM", keep="first").copy()
        merged = gwas_sub.merge(
            fisher,
            left_on="GWAS_ID_NORM",
            right_on="TARGET_ID_NORM",
            how="inner",
        )

    if merged.empty:
        return result

    result["target_fisher_common_snp_count"] = int(len(merged))

    merged["TARGET_FISHER_ALLELE_STATUS"] = [
        beta_orientation_status(g_ea, g_nea, t_a1, t_a2)
        for g_ea, g_nea, t_a1, t_a2 in zip(
            merged["GWAS_EA_NORM"],
            merged["GWAS_NEA_NORM"],
            merged["TARGET_A1"],
            merged["TARGET_A2"],
        )
    ]

    direct = int((merged["TARGET_FISHER_ALLELE_STATUS"] == "direct_match").sum())
    flip = int((merged["TARGET_FISHER_ALLELE_STATUS"] == "flip_match").sum())
    strand = int((merged["TARGET_FISHER_ALLELE_STATUS"] == "strand_match").sum())
    strand_flip = int((merged["TARGET_FISHER_ALLELE_STATUS"] == "strand_flip_match").sum())
    palindromic_ambiguous = int(
        (merged["TARGET_FISHER_ALLELE_STATUS"] == "palindromic_ambiguous").sum()
    )
    mismatch = int((merged["TARGET_FISHER_ALLELE_STATUS"] == "mismatch").sum())
    missing = int((merged["TARGET_FISHER_ALLELE_STATUS"] == "missing_allele").sum())
    compatible = direct + flip + strand + strand_flip

    result["target_fisher_allele_direct_match_count"] = direct
    result["target_fisher_allele_flip_match_count"] = flip
    result["target_fisher_allele_strand_match_count"] = strand
    result["target_fisher_allele_strand_flip_match_count"] = strand_flip
    result["target_fisher_palindromic_ambiguous_count"] = palindromic_ambiguous
    result["target_fisher_allele_mismatch_count"] = mismatch
    result["target_fisher_allele_missing_count"] = missing
    result["target_fisher_allele_compatible_count"] = compatible
    result["target_fisher_allele_compatible_pct"] = percent(compatible, len(merged))

    gwas_beta = safe_numeric(merged["BETA_FOR_DIST"])
    target_beta_raw = safe_numeric(merged["TARGET_BETA_LOG_OR"])
    target_beta_signed = target_beta_raw.copy()

    flip_mask = merged["TARGET_FISHER_ALLELE_STATUS"].isin(["flip_match", "strand_flip_match"])
    target_beta_signed.loc[flip_mask] = -target_beta_signed.loc[flip_mask]

    beta_common = pd.DataFrame({
        "gwas_beta": gwas_beta,
        "target_beta_raw": target_beta_raw,
        "target_beta_signed": target_beta_signed,
        "status": merged["TARGET_FISHER_ALLELE_STATUS"],
    }).replace([np.inf, -np.inf], np.nan).dropna(subset=["gwas_beta", "target_beta_raw"])

    result["target_fisher_common_beta_count"] = int(len(beta_common))

    if len(beta_common) >= 3:
        result["target_fisher_beta_corr_raw_all_common"] = correlation_or_blank(
            beta_common["gwas_beta"],
            beta_common["target_beta_raw"],
        )

    beta_compatible = beta_common[
        beta_common["status"].isin(["direct_match", "flip_match", "strand_match", "strand_flip_match"])
    ].copy()

    if len(beta_compatible) >= 3:
        result["target_fisher_beta_corr_signed_compatible"] = correlation_or_blank(
            beta_compatible["gwas_beta"],
            beta_compatible["target_beta_signed"],
        )
        result["target_fisher_abs_beta_corr_compatible"] = correlation_or_blank(
            beta_compatible["gwas_beta"].abs(),
            beta_compatible["target_beta_signed"].abs(),
        )

        diff = (beta_compatible["gwas_beta"] - beta_compatible["target_beta_signed"]).abs()
        result["target_fisher_beta_abs_diff_mean_compatible"] = round(float(diff.mean()), 8)
        result["target_fisher_beta_abs_diff_median_compatible"] = round(float(diff.median()), 8)

    gwas_p = safe_numeric(merged["P_NUM"])
    target_p = safe_numeric(merged["TARGET_P"])

    p_common = pd.DataFrame({
        "gwas_p": gwas_p,
        "target_p": target_p,
        "status": merged["TARGET_FISHER_ALLELE_STATUS"],
    }).replace([np.inf, -np.inf], np.nan).dropna()

    p_common = p_common[
        (p_common["gwas_p"] > 0)
        & (p_common["gwas_p"] <= 1)
        & (p_common["target_p"] > 0)
        & (p_common["target_p"] <= 1)
    ]

    result["target_fisher_common_p_count"] = int(len(p_common))

    if len(p_common) >= 3:
        result["target_fisher_pvalue_corr_all_common"] = correlation_or_blank(
            p_common["gwas_p"],
            p_common["target_p"],
        )
        result["target_fisher_neglog10p_corr_all_common"] = correlation_or_blank(
            -np.log10(p_common["gwas_p"]),
            -np.log10(p_common["target_p"]),
        )

    p_compatible = p_common[
        p_common["status"].isin(["direct_match", "flip_match", "strand_match", "strand_flip_match"])
    ].copy()

    if len(p_compatible) >= 3:
        result["target_fisher_pvalue_corr_compatible"] = correlation_or_blank(
            p_compatible["gwas_p"],
            p_compatible["target_p"],
        )
        result["target_fisher_neglog10p_corr_compatible"] = correlation_or_blank(
            -np.log10(p_compatible["gwas_p"]),
            -np.log10(p_compatible["target_p"]),
        )

    return result


# =============================================================================
# PLINK CLUMPING AND PRUNING
# =============================================================================

def parse_clumped_file(clumped_path):
    clumped_path = Path(clumped_path)

    result = {
        "clump_file_exists": False,
        "clump_count": 0,
        "clump_min_p": "",
        "clump_median_p": "",
        "clump_chromosome_count": "",
        "clump_mean_sp2_count": "",
        "clump_max_sp2_count": "",
    }

    if not clumped_path.exists() or clumped_path.stat().st_size == 0:
        return result

    try:
        clumped = pd.read_csv(clumped_path, sep=r"\s+", engine="python", dtype=str, comment="#")
    except Exception:
        return result

    if clumped.empty:
        return result

    if "CHR" in clumped.columns:
        clumped = clumped[clumped["CHR"].astype(str).str.match(r"^[0-9XYMT]+$", case=False, na=False)]

    if clumped.empty:
        return result

    result["clump_file_exists"] = True
    result["clump_count"] = int(len(clumped))

    if "P" in clumped.columns:
        p = safe_numeric(clumped["P"])
        result["clump_min_p"] = min_or_blank(p)
        result["clump_median_p"] = median_or_blank(p)

    if "CHR" in clumped.columns:
        result["clump_chromosome_count"] = int(clumped["CHR"].nunique())

    if "SP2" in clumped.columns:
        sp2_counts = []
        for value in clumped["SP2"].astype(str):
            value = clean_text(value)
            if value in ["", "NONE", "NA", "nan"]:
                sp2_counts.append(0)
            else:
                sp2_counts.append(len([x for x in value.split(",") if clean_text(x)]))

        if sp2_counts:
            result["clump_mean_sp2_count"] = round(float(np.mean(sp2_counts)), 6)
            result["clump_max_sp2_count"] = int(np.max(sp2_counts))

    return result


def run_plink_clumping(plink, bfile_prefix, clump_input, output_dir, p1_thresholds, clump_r2, clump_kb):
    output_dir = Path(output_dir)
    features = {}

    if not Path(clump_input).exists() or Path(clump_input).stat().st_size == 0:
        for label, _ in p1_thresholds:
            prefix = f"clump_{label}"
            features[f"{prefix}_status"] = "skipped_empty_input"
            features[f"{prefix}_count"] = 0
        return features

    for label, threshold in p1_thresholds:
        out_prefix = output_dir / f"Feature9_clump_{label}"
        log_path = output_dir / f"Feature9_clump_{label}.log"

        cmd = [
            plink,
            "--bfile", str(bfile_prefix),
            "--clump", str(clump_input),
            "--clump-snp-field", "SNP",
            "--clump-field", "P",
            "--clump-p1", str(threshold),
            "--clump-p2", "1",
            "--clump-r2", str(clump_r2),
            "--clump-kb", str(clump_kb),
            "--out", str(out_prefix),
        ]

        returncode, _ = run_command(cmd, log_path)

        clumped_path = out_prefix.with_suffix(".clumped")
        parsed = parse_clumped_file(clumped_path)

        prefix = f"clump_{label}"
        features[f"{prefix}_returncode"] = returncode
        features[f"{prefix}_status"] = "success" if returncode == 0 else "failed"
        features[f"{prefix}_file"] = str(clumped_path)
        features[f"{prefix}_log_file"] = str(log_path)
        features[f"{prefix}_count"] = parsed["clump_count"]
        features[f"{prefix}_min_p"] = parsed["clump_min_p"]
        features[f"{prefix}_median_p"] = parsed["clump_median_p"]
        features[f"{prefix}_chromosome_count"] = parsed["clump_chromosome_count"]
        features[f"{prefix}_mean_sp2_count"] = parsed["clump_mean_sp2_count"]
        features[f"{prefix}_max_sp2_count"] = parsed["clump_max_sp2_count"]

    return features


def parse_prune_files(out_prefix):
    out_prefix = Path(out_prefix)
    prune_in = out_prefix.with_suffix(".prune.in")
    prune_out = out_prefix.with_suffix(".prune.out")

    result = {
        "prune_in_file": str(prune_in),
        "prune_out_file": str(prune_out),
        "prune_in_count": 0,
        "prune_out_count": 0,
    }

    if prune_in.exists():
        result["prune_in_count"] = sum(1 for _ in open(prune_in, "rt", encoding="utf-8", errors="replace"))

    if prune_out.exists():
        result["prune_out_count"] = sum(1 for _ in open(prune_out, "rt", encoding="utf-8", errors="replace"))

    return result


def run_plink_pruning(plink, bfile_prefix, extract_file, output_dir, window_kb, step_size, prune_r2):
    output_dir = Path(output_dir)
    out_prefix = output_dir / "Feature9_prune_gwas_overlap"
    log_path = output_dir / "Feature9_prune_gwas_overlap.log"

    features = {
        "prune_status": "",
        "prune_returncode": "",
        "prune_log_file": str(log_path),
        "prune_window_kb": window_kb,
        "prune_step_size": step_size,
        "prune_r2": prune_r2,
        "prune_in_file": "",
        "prune_out_file": "",
        "prune_in_count": 0,
        "prune_out_count": 0,
        "prune_total_input_count": 0,
        "prune_in_pct_of_input": 0.0,
        "prune_out_pct_of_input": 0.0,
    }

    if not Path(extract_file).exists() or Path(extract_file).stat().st_size == 0:
        features["prune_status"] = "skipped_empty_extract"
        return features

    input_count = sum(1 for _ in open(extract_file, "rt", encoding="utf-8", errors="replace"))
    features["prune_total_input_count"] = int(input_count)

    cmd = [
        plink,
        "--bfile", str(bfile_prefix),
        "--extract", str(extract_file),
        "--indep-pairwise", f"{window_kb}kb", str(step_size), str(prune_r2),
        "--out", str(out_prefix),
    ]

    returncode, _ = run_command(cmd, log_path)
    parsed = parse_prune_files(out_prefix)

    features["prune_status"] = "success" if returncode == 0 else "failed"
    features["prune_returncode"] = returncode
    features.update(parsed)

    features["prune_in_pct_of_input"] = percent(features["prune_in_count"], input_count)
    features["prune_out_pct_of_input"] = percent(features["prune_out_count"], input_count)

    return features


# =============================================================================
# FEATURE 9 MAIN
# =============================================================================

def calculate_feature9(
    job,
    base_dir,
    target_bim="",
    plink_path="",
    clump_r2=0.1,
    clump_kb=250,
    prune_window_kb=200,
    prune_step_size=50,
    prune_r2=0.25,
    skip_fisher=False,
):
    final_gwas_path = find_final_gwas(job, base_dir)
    bim_path = find_target_bim(job["phenotype"], base_dir, explicit_bim=target_bim)

    bfile_prefix = infer_bfile_prefix_from_bim(bim_path)
    bed_path = bfile_component_path(bfile_prefix, ".bed")
    fam_path = bfile_component_path(bfile_prefix, ".fam")
    plink = find_plink(plink_path)
    target_trait_type = infer_target_trait_type_from_fam(fam_path)

    output_dir = final_gwas_path.parent

    print("=" * 100)
    print("FEATURE 9: EFFECT-SIZE + P-VALUE + TARGET PLINK --FISHER GWAS + CLUMPING/PRUNING")
    print("=" * 100)
    print(f"[JOB INDEX]        {job['job_index']}")
    print(f"[PHENOTYPE]        {job['phenotype']}")
    print(f"[ACCESSION]        {job['accessionId']}")
    print(f"[GWAS FILE]        {job['file_path']}")
    print(f"[FINAL GWAS]       {final_gwas_path}")
    print(f"[TARGET BIM]       {bim_path}")
    print(f"[TARGET BED]       {bed_path}")
    print(f"[TARGET FAM]       {fam_path}")
    print(f"[PLINK]            {plink}")
    print("=" * 100)

    gwas_raw = load_final_gwas(final_gwas_path)
    gwas = prepare_gwas(gwas_raw)
    bim = load_bim(bim_path)

    features = {
        "job_index": job["job_index"],
        "phenotype": job["phenotype"],
        "accessionId": job["accessionId"],
        "gwas_file": job["file_path"],
        "final_gwas_file": str(final_gwas_path),
        "target_bim_file": str(bim_path),
        "target_bed_file": str(bed_path),
        "target_fam_file": str(fam_path),
        "target_bfile_prefix": str(bfile_prefix),
        "target_trait_type": target_trait_type,
        "plink_path": plink,
        "feature9_status": "started",
        "feature9_error": "",
        "clump_r2": clump_r2,
        "clump_kb": clump_kb,
        "prune_window_kb": prune_window_kb,
        "prune_step_size": prune_step_size,
        "prune_r2": prune_r2,
        "skip_fisher": skip_fisher,
    }

    features["final_gwas_variant_count"] = int(len(gwas))
    features["target_bim_variant_count"] = int(len(bim))

    # 1. Effect-size and P-value distribution
    features.update(calculate_effect_size_distribution_features(gwas))
    features.update(calculate_pvalue_distribution_features(gwas))

    # 2. PRS / clumping input readiness
    features.update(calculate_prs_input_readiness_features(gwas, bim))

    # 3. Create PLINK input files
    clump_input, clump_input_count = make_plink_clump_input(gwas, bim, output_dir)
    extract_file, extract_count = make_plink_extract_file(gwas, bim, output_dir)

    features["plink_clump_input_file"] = str(clump_input)
    features["plink_clump_input_variant_count"] = int(clump_input_count)
    features["plink_clump_input_pct_of_final_gwas"] = percent(clump_input_count, len(gwas))

    features["plink_extract_file"] = str(extract_file)
    features["plink_extract_variant_count"] = int(extract_count)
    features["plink_extract_pct_of_final_gwas"] = percent(extract_count, len(gwas))

    # 4. Run target-derived PLINK association using trait-appropriate method
    if skip_fisher:
        features.update({
            "target_fisher_status": "skipped_by_user",
            "target_fisher_returncode": "",
            "target_fisher_log_file": "",
            "target_fisher_assoc_file": "",
            "target_fisher_input_snp_count": 0,
            "target_assoc_method": "skipped_by_user",
        })
        features.update(calculate_target_fisher_comparison_features(gwas, "", bim=bim))
    else:
        fisher_run_features = run_plink_target_assoc(
            plink=plink,
            bfile_prefix=bfile_prefix,
            extract_file=extract_file,
            output_dir=output_dir,
            trait_type=target_trait_type,
        )
        features.update(fisher_run_features)

        target_fisher_assoc_file = features.get("target_fisher_assoc_file", "")
        if not is_valid_file(target_fisher_assoc_file):
            print(f"[WARNING] target_fisher_assoc_file is invalid: {target_fisher_assoc_file}")
            print("[WARNING] Fisher comparison features will be saved as missing.")

        fisher_comparison_features = calculate_target_fisher_comparison_features(
            gwas=gwas,
            fisher_assoc_file=target_fisher_assoc_file,
            bim=bim,
        )
        features.update(fisher_comparison_features)

    # 5. Run PLINK clumping at multiple P thresholds
    p1_thresholds = [
        ("p5e8", 5e-8),
        ("p1e6", 1e-6),
        ("p1e5", 1e-5),
        ("p1e4", 1e-4),
        ("p1e3", 1e-3),
        ("p0_05", 0.05),
        ("p1", 1.0),
    ]

    clump_features = run_plink_clumping(
        plink=plink,
        bfile_prefix=bfile_prefix,
        clump_input=clump_input,
        output_dir=output_dir,
        p1_thresholds=p1_thresholds,
        clump_r2=clump_r2,
        clump_kb=clump_kb,
    )

    features.update(clump_features)

    # 6. Run LD pruning on the GWAS-overlapping target SNP set
    prune_features = run_plink_pruning(
        plink=plink,
        bfile_prefix=bfile_prefix,
        extract_file=extract_file,
        output_dir=output_dir,
        window_kb=prune_window_kb,
        step_size=prune_step_size,
        prune_r2=prune_r2,
    )

    features.update(prune_features)

    external_statuses = [
        clean_text(features.get("target_fisher_status", "")),
        clean_text(features.get("prune_status", "")),
    ]
    external_statuses.extend(
        clean_text(features.get(f"clump_{label}_status", ""))
        for label in ["p5e8", "p1e6", "p1e5", "p1e4", "p1e3", "p0_05", "p1"]
    )
    features["feature9_status"] = (
        "success_with_external_tool_failures"
        if any(status in {"failed", "failed_or_missing_output"} for status in external_statuses)
        else "success"
    )

    output_path = output_dir / "Feature9.csv"
    pd.DataFrame([features]).to_csv(output_path, index=False)

    print("[DONE] Feature9 saved")
    print(f"[OUTPUT] {output_path}")
    print()
    print("[KEY RESULTS]")
    print(f"Final GWAS variants:                         {features['final_gwas_variant_count']}")
    print(f"Target BIM variants:                         {features['target_bim_variant_count']}")
    print(f"PRS full-ready SNPs:                         {features['prs_full_ready_count']}")
    print(f"PRS full-ready and in target SNPs:           {features['prs_full_ready_and_in_target_count']}")
    print(f"PLINK extract SNPs:                          {features['plink_extract_variant_count']}")
    print(f"PLINK clump input SNPs:                      {features['plink_clump_input_variant_count']}")
    print(f"Target Fisher status:                        {features.get('target_fisher_status', '')}")
    print(f"Target Fisher common SNPs:                   {features.get('target_fisher_common_snp_count', '')}")
    print(f"Target Fisher compatible SNPs:               {features.get('target_fisher_allele_compatible_count', '')}")
    print(f"Target Fisher signed beta corr compatible:   {features.get('target_fisher_beta_corr_signed_compatible', '')}")
    print(f"Target Fisher -log10P corr compatible:       {features.get('target_fisher_neglog10p_corr_compatible', '')}")
    print(f"Clumps p<=5e-8:                              {features.get('clump_p5e8_count', '')}")
    print(f"Clumps p<=1e-5:                              {features.get('clump_p1e5_count', '')}")
    print(f"Clumps p<=0.05:                              {features.get('clump_p0_05_count', '')}")
    print(f"Prune input SNPs:                            {features['prune_total_input_count']}")
    print(f"Prune-in SNPs:                               {features['prune_in_count']}")
    print(f"Prune-out SNPs:                              {features['prune_out_count']}")
    print(f"Beta median abs:                             {features['beta_median_abs']}")
    print(f"P min:                                       {features['p_min']}")
    print(f"-log10(P) max:                               {features['neglog10p_max']}")
    print("=" * 100)

    return output_path, features


def save_failure_feature9(job, base_dir, error_message):
    try:
        file_path = resolve_path(job["file_path"], base_dir)
        out_dir = file_path.parent
    except Exception:
        out_dir = Path.cwd()

    out_dir.mkdir(parents=True, exist_ok=True)

    features = {
        "job_index": job.get("job_index", ""),
        "phenotype": job.get("phenotype", ""),
        "accessionId": job.get("accessionId", ""),
        "gwas_file": job.get("file_path", ""),
        "feature9_status": "failed",
        "feature9_error": error_message,
    }

    output_path = out_dir / "Feature9.csv"
    pd.DataFrame([features]).to_csv(output_path, index=False)

    print("[FAILED] Feature9 saved with error")
    print(f"[OUTPUT] {output_path}")
    print(f"[ERROR] {error_message}")

    return output_path


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate Feature9: effect-size distribution, P-value distribution, "
            "target-derived PLINK --fisher GWAS beta/P concordance, PRS readiness, "
            "PLINK clumping, and LD pruning features."
        )
    )

    parser.add_argument(
        "index",
        type=int,
        help="GWAS job index from GWAS_jobs.csv"
    )

    parser.add_argument(
        "--jobs-file",
        default="GWAS_jobs.csv",
        help="GWAS jobs manifest. Default: GWAS_jobs.csv"
    )

    parser.add_argument(
        "--base-dir",
        default=DEFAULT_BASE_DIR,
        help=f"Base directory containing phenotype folders and GWAS_jobs.csv. Default: {DEFAULT_BASE_DIR}"
    )

    parser.add_argument(
        "--target-bim",
        default="",
        help="Optional explicit target BIM file. If not given, script searches Fold_*/train_data.QC.bim, Fold_*/test_data.bim, then phenotype_QC.bim / phenotype.bim."
    )

    parser.add_argument(
        "--plink",
        default="",
        help="Optional PLINK path. Default: ./plink if present, otherwise plink."
    )

    parser.add_argument(
        "--clump-r2",
        type=float,
        default=0.1,
        help="PLINK clumping LD r2 threshold. Default: 0.1"
    )

    parser.add_argument(
        "--clump-kb",
        type=int,
        default=250,
        help="PLINK clumping window in kb. Default: 250"
    )

    parser.add_argument(
        "--prune-window-kb",
        type=int,
        default=200,
        help="PLINK pruning window. Default: 200"
    )

    parser.add_argument(
        "--prune-step-size",
        type=int,
        default=50,
        help="PLINK pruning step size. Default: 50"
    )

    parser.add_argument(
        "--prune-r2",
        type=float,
        default=0.25,
        help="PLINK pruning r2 threshold. Default: 0.25"
    )

    parser.add_argument(
        "--skip-fisher",
        action="store_true",
        help="Skip target-derived PLINK --fisher GWAS."
    )

    parser.add_argument(
        "--no-failure-file",
        action="store_true",
        help="If set, do not write Feature9.csv when calculation fails."
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    job = resolve_job_from_manifest(args.index, args.jobs_file)

    try:
        calculate_feature9(
            job=job,
            base_dir=base_dir,
            target_bim=args.target_bim,
            plink_path=args.plink,
            clump_r2=args.clump_r2,
            clump_kb=args.clump_kb,
            prune_window_kb=args.prune_window_kb,
            prune_step_size=args.prune_step_size,
            prune_r2=args.prune_r2,
            skip_fisher=args.skip_fisher,
        )
    except Exception as e:
        error_message = f"{type(e).__name__}: {e}"

        if args.no_failure_file:
            raise

        save_failure_feature9(
            job=job,
            base_dir=base_dir,
            error_message=error_message,
        )

        raise


if __name__ == "__main__":
    main()
