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


def normalise_chr(x):
    x = clean_text(x)
    x = x.replace("chr", "").replace("CHR", "").strip()

    if x in ["23"]:
        return "X"
    if x in ["24"]:
        return "Y"
    if x in ["25", "M", "MT", "Mt", "mt"]:
        return "MT"

    return x


def normalise_pos(x):
    try:
        if pd.isna(x):
            return ""
        pos = int(float(str(x).replace(",", "").strip()))
        if pos > 0:
            return str(pos)
    except Exception:
        pass
    return ""


def normalise_allele(x):
    x = clean_text(x).upper()
    if x in ["", "NAN", "NA", "N/A", "NONE", "NULL", "."]:
        return ""
    return x


def normalise_rsid(x):
    x = clean_text(x)
    if x.lower().startswith("rs"):
        return x.lower()
    return ""


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

    if len(a1) != 1 or len(a2) != 1:
        return False

    return f"{a1}/{a2}" in ["A/T", "T/A", "C/G", "G/C"]


def complement_allele(allele):
    allele = normalise_allele(allele)
    if len(allele) != 1:
        return ""
    return {"A": "T", "T": "A", "C": "G", "G": "C"}.get(allele, "")


def canonical_allele_pair(a1, a2):
    a1 = normalise_allele(a1)
    a2 = normalise_allele(a2)
    if not a1 or not a2:
        return ""

    direct = "/".join(sorted([a1, a2]))
    comp_a1 = complement_allele(a1)
    comp_a2 = complement_allele(a2)
    if not comp_a1 or not comp_a2:
        return direct

    complement = "/".join(sorted([comp_a1, comp_a2]))
    return min(direct, complement)


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def correlation_or_blank(x, y):
    valid = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()

    if len(valid) < 3:
        return ""

    return round(float(valid["x"].corr(valid["y"])), 6)


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
        "Run the FinalGWAS creation step first."
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
    bim_path = Path(bim_path)
    return bim_path.with_suffix("")


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

    bim["BIM_CHR_NORM"] = bim["BIM_CHR"].map(normalise_chr)
    bim["BIM_POS_NORM"] = bim["BIM_POS"].map(normalise_pos)
    bim["BIM_SNPID_RAW"] = bim["BIM_SNPID"].astype(str).map(clean_text)
    bim["BIM_SNPID_NORM"] = bim["BIM_SNPID_RAW"].map(normalise_rsid)

    # Keep a general ID too, because not all BIM IDs are rsIDs.
    bim["BIM_ID_NORM"] = bim["BIM_SNPID_RAW"].str.lower()

    bim["BIM_A1_NORM"] = bim["BIM_A1"].map(normalise_allele)
    bim["BIM_A2_NORM"] = bim["BIM_A2"].map(normalise_allele)

    bim["BIM_CHRPOS"] = bim["BIM_CHR_NORM"] + ":" + bim["BIM_POS_NORM"]
    bim.loc[
        (bim["BIM_CHR_NORM"] == "") | (bim["BIM_POS_NORM"] == ""),
        "BIM_CHRPOS",
    ] = ""

    bim["BIM_ALLELE_PAIR_SORTED"] = [
        canonical_allele_pair(a1, a2)
        for a1, a2 in zip(bim["BIM_A1_NORM"], bim["BIM_A2_NORM"])
    ]

    bim["BIM_CHRPOS_ALLELES"] = (
        bim["BIM_CHRPOS"] + ":" + bim["BIM_ALLELE_PAIR_SORTED"]
    )

    bim.loc[
        (bim["BIM_CHRPOS"] == "") | (bim["BIM_ALLELE_PAIR_SORTED"] == ""),
        "BIM_CHRPOS_ALLELES",
    ] = ""

    return bim


def load_fam(path):
    fam_path = Path(path)

    fam = pd.read_csv(
        fam_path,
        sep=r"\s+",
        header=None,
        names=["FID", "IID", "PAT", "MAT", "SEX", "PHENO"],
        dtype=str,
        engine="python",
    )

    return fam


def prepare_gwas_keys(gwas):
    gwas = gwas.copy()

    for col in ["SNPID", "CHR", "POS", "EA", "NEA", "EAF", "MAF", "BETA", "P", "N", "N_CASES", "N_CONTROLS"]:
        if col not in gwas.columns:
            gwas[col] = ""

    gwas["GWAS_ROW_ID"] = np.arange(len(gwas))

    gwas["GWAS_SNPID_RAW"] = gwas["SNPID"].astype(str).map(clean_text)
    gwas["GWAS_RSID"] = gwas["GWAS_SNPID_RAW"].map(normalise_rsid)
    gwas["GWAS_ID_NORM"] = gwas["GWAS_SNPID_RAW"].str.lower()

    gwas["GWAS_CHR_NORM"] = gwas["CHR"].map(normalise_chr)
    gwas["GWAS_POS_NORM"] = gwas["POS"].map(normalise_pos)
    gwas["GWAS_CHRPOS"] = gwas["GWAS_CHR_NORM"] + ":" + gwas["GWAS_POS_NORM"]

    gwas.loc[
        (gwas["GWAS_CHR_NORM"] == "") | (gwas["GWAS_POS_NORM"] == ""),
        "GWAS_CHRPOS",
    ] = ""

    gwas["GWAS_EA_NORM"] = gwas["EA"].map(normalise_allele)
    gwas["GWAS_NEA_NORM"] = gwas["NEA"].map(normalise_allele)

    gwas["GWAS_HAS_ALLELES"] = (
        (gwas["GWAS_EA_NORM"] != "")
        & (gwas["GWAS_NEA_NORM"] != "")
    )

    gwas["GWAS_ALLELE_PAIR_SORTED"] = [
        canonical_allele_pair(a1, a2)
        for a1, a2 in zip(gwas["GWAS_EA_NORM"], gwas["GWAS_NEA_NORM"])
    ]

    gwas["GWAS_CHRPOS_ALLELES"] = (
        gwas["GWAS_CHRPOS"] + ":" + gwas["GWAS_ALLELE_PAIR_SORTED"]
    )

    gwas.loc[
        (gwas["GWAS_CHRPOS"] == "") | (gwas["GWAS_ALLELE_PAIR_SORTED"] == ""),
        "GWAS_CHRPOS_ALLELES",
    ] = ""

    gwas["GWAS_PALINDROMIC"] = [
        is_palindromic_pair(a1, a2)
        for a1, a2 in zip(gwas["GWAS_EA_NORM"], gwas["GWAS_NEA_NORM"])
    ]

    # GWAS MAF:
    # Prefer MAF if present. If only EAF is present, convert EAF to minor allele frequency.
    maf = safe_numeric(gwas["MAF"])
    eaf = safe_numeric(gwas["EAF"])

    maf_from_eaf = eaf.where((eaf >= 0) & (eaf <= 1)).map(
        lambda x: min(x, 1 - x) if pd.notna(x) else np.nan
    )

    gwas["GWAS_MAF_FOR_CORR"] = maf.where((maf >= 0) & (maf <= 0.5), maf_from_eaf)

    # Also keep EAF if future orientation-aware frequency comparison is needed.
    gwas["GWAS_EAF_NUMERIC"] = eaf.where((eaf >= 0) & (eaf <= 1))

    return gwas


# =============================================================================
# TARGET FREQUENCY USING PLINK
# =============================================================================

def run_plink_freq(bfile_prefix, output_dir, plink_path):
    output_dir = Path(output_dir)
    out_prefix = output_dir / "Feature8_target_freq"

    frq_path = out_prefix.with_suffix(".frq")

    if frq_path.exists():
        return frq_path, "existing_frq_used", ""

    cmd = [
        plink_path,
        "--bfile", str(bfile_prefix),
        "--freq",
        "--out", str(out_prefix),
    ]

    completed = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        return "", "plink_freq_failed", completed.stderr + "\n" + completed.stdout

    if not frq_path.exists():
        return "", "plink_freq_output_missing", completed.stderr + "\n" + completed.stdout

    return frq_path, "plink_freq_success", completed.stderr + "\n" + completed.stdout


def load_plink_frq(frq_path):
    if not frq_path or not Path(frq_path).exists():
        return pd.DataFrame()

    frq = pd.read_csv(frq_path, sep=r"\s+", engine="python", dtype=str)
    frq.columns = [clean_text(c) for c in frq.columns]

    # PLINK 1.9 .frq columns usually:
    # CHR SNP A1 A2 MAF NCHROBS
    rename = {}
    for c in frq.columns:
        cl = c.lower()
        if cl == "chr":
            rename[c] = "FRQ_CHR"
        elif cl in ["snp", "id"]:
            rename[c] = "FRQ_SNP"
        elif cl == "a1":
            rename[c] = "FRQ_A1"
        elif cl == "a2":
            rename[c] = "FRQ_A2"
        elif cl == "maf":
            rename[c] = "FRQ_MAF"
        elif cl == "nchrobs":
            rename[c] = "FRQ_NCHROBS"

    frq = frq.rename(columns=rename)

    for col in ["FRQ_CHR", "FRQ_SNP", "FRQ_A1", "FRQ_A2", "FRQ_MAF"]:
        if col not in frq.columns:
            frq[col] = ""

    frq["FRQ_CHR_NORM"] = frq["FRQ_CHR"].map(normalise_chr)
    frq["FRQ_SNP_RAW"] = frq["FRQ_SNP"].astype(str).map(clean_text)
    frq["FRQ_RSID"] = frq["FRQ_SNP_RAW"].map(normalise_rsid)
    frq["FRQ_ID_NORM"] = frq["FRQ_SNP_RAW"].str.lower()
    frq["FRQ_A1_NORM"] = frq["FRQ_A1"].map(normalise_allele)
    frq["FRQ_A2_NORM"] = frq["FRQ_A2"].map(normalise_allele)
    frq["TARGET_MAF"] = safe_numeric(frq["FRQ_MAF"])

    frq["FRQ_ALLELE_PAIR_SORTED"] = [
        "/".join(sorted([a1, a2]))
        if a1 and a2 else ""
        for a1, a2 in zip(frq["FRQ_A1_NORM"], frq["FRQ_A2_NORM"])
    ]

    return frq


# =============================================================================
# PHENOTYPE FEATURES
# =============================================================================

def calculate_fam_phenotype_features(fam_path):
    result = {
        "target_fam_file": str(fam_path),
        "target_fam_exists": False,
        "target_total_samples_fam": "",
        "target_phenotype_nonmissing_count": "",
        "target_phenotype_missing_count": "",
        "target_phenotype_missing_pct": "",
        "target_case_count": "",
        "target_control_count": "",
        "target_case_fraction": "",
        "target_effective_n_case_control": "",
        "target_phenotype_type_guess": "",
        "target_continuous_mean": "",
        "target_continuous_sd": "",
        "target_sex_male_count": "",
        "target_sex_female_count": "",
        "target_sex_missing_count": "",
    }

    fam_path = Path(fam_path)

    if not fam_path.exists():
        return result

    fam = load_fam(fam_path)

    result["target_fam_exists"] = True
    result["target_total_samples_fam"] = int(len(fam))

    pheno = safe_numeric(fam["PHENO"])
    missing = pheno.isna() | pheno.isin([-9, 0])
    nonmissing = ~missing

    result["target_phenotype_nonmissing_count"] = int(nonmissing.sum())
    result["target_phenotype_missing_count"] = int(missing.sum())
    result["target_phenotype_missing_pct"] = percent(missing.sum(), len(fam))

    nonmissing_values = pheno[nonmissing]
    unique_values = sorted(nonmissing_values.dropna().unique().tolist())

    # PLINK binary phenotype convention: 1 = control, 2 = case
    if set(unique_values).issubset({1.0, 2.0}) and len(unique_values) > 0:
        controls = int((pheno == 1).sum())
        cases = int((pheno == 2).sum())

        result["target_phenotype_type_guess"] = "binary_plink_1_control_2_case"
        result["target_case_count"] = cases
        result["target_control_count"] = controls
        result["target_case_fraction"] = percent(cases, cases + controls)

        if cases > 0 and controls > 0:
            result["target_effective_n_case_control"] = round(
                4 / ((1 / cases) + (1 / controls)),
                0,
            )

    else:
        result["target_phenotype_type_guess"] = "continuous_or_nonstandard"
        if len(nonmissing_values) > 0:
            result["target_continuous_mean"] = round(float(nonmissing_values.mean()), 6)
            result["target_continuous_sd"] = round(float(nonmissing_values.std()), 6)

    sex = safe_numeric(fam["SEX"])
    result["target_sex_male_count"] = int((sex == 1).sum())
    result["target_sex_female_count"] = int((sex == 2).sum())
    result["target_sex_missing_count"] = int(sex.isna().sum() | (sex == 0).sum()) if False else int((sex.isna() | (sex == 0)).sum())

    return result


def calculate_metadata_phenotype_features(job, base_dir):
    result = {
        "phenotype_metadata_file": "",
        "phenotype_metadata_found": False,
        "metadata_sample_n": "",
        "metadata_sample_cases": "",
        "metadata_sample_controls": "",
    }

    phenotype = clean_text(job["phenotype"])
    accession = clean_text(job["accessionId"])

    search_paths = [
        Path(base_dir) / f"{phenotype}.csv",
        Path.cwd() / f"{phenotype}.csv",
        Path(base_dir) / phenotype / f"{phenotype}.csv",
    ]

    metadata_path = ""
    for p in search_paths:
        if p.exists():
            metadata_path = p
            break

    if not metadata_path:
        return result

    result["phenotype_metadata_file"] = str(metadata_path)

    try:
        meta = pd.read_csv(metadata_path, low_memory=False)
    except Exception:
        return result

    if meta.empty or "accessionId" not in meta.columns:
        return result

    matched = meta[
        meta["accessionId"].astype(str).str.strip().str.lower()
        == accession.lower()
    ]

    if matched.empty:
        return result

    row = matched.iloc[0]
    result["phenotype_metadata_found"] = True

    for source_col, out_col in [
        ("SAMPLES", "metadata_sample_n"),
        ("CASES", "metadata_sample_cases"),
        ("CONTROLS", "metadata_sample_controls"),
    ]:
        value = clean_text(row.get(source_col, ""))
        value = value.replace(",", "")
        if value:
            try:
                result[out_col] = int(round(float(value)))
            except Exception:
                result[out_col] = value

    return result


# =============================================================================
# OVERLAP FEATURES
# =============================================================================

def calculate_basic_gwas_features(gwas):
    result = {}
    n = len(gwas)

    result["final_gwas_variant_count"] = n
    result["final_gwas_unique_rsid_count"] = int(gwas.loc[gwas["GWAS_RSID"] != "", "GWAS_RSID"].nunique())
    result["final_gwas_rsid_available_count"] = int((gwas["GWAS_RSID"] != "").sum())
    result["final_gwas_rsid_available_pct"] = percent(result["final_gwas_rsid_available_count"], n)

    result["final_gwas_chrpos_available_count"] = int((gwas["GWAS_CHRPOS"] != "").sum())
    result["final_gwas_chrpos_available_pct"] = percent(result["final_gwas_chrpos_available_count"], n)
    result["final_gwas_unique_chrpos_count"] = int(gwas.loc[gwas["GWAS_CHRPOS"] != "", "GWAS_CHRPOS"].nunique())

    result["final_gwas_chrpos_alleles_available_count"] = int((gwas["GWAS_CHRPOS_ALLELES"] != "").sum())
    result["final_gwas_chrpos_alleles_available_pct"] = percent(result["final_gwas_chrpos_alleles_available_count"], n)
    result["final_gwas_unique_chrpos_alleles_count"] = int(
        gwas.loc[gwas["GWAS_CHRPOS_ALLELES"] != "", "GWAS_CHRPOS_ALLELES"].nunique()
    )

    result["final_gwas_allele_available_count"] = int(gwas["GWAS_HAS_ALLELES"].sum())
    result["final_gwas_allele_available_pct"] = percent(result["final_gwas_allele_available_count"], n)

    result["final_gwas_palindromic_count"] = int(gwas["GWAS_PALINDROMIC"].sum())
    result["final_gwas_palindromic_pct"] = percent(result["final_gwas_palindromic_count"], n)

    result["final_gwas_maf_available_count"] = int(gwas["GWAS_MAF_FOR_CORR"].notna().sum())
    result["final_gwas_maf_available_pct"] = percent(result["final_gwas_maf_available_count"], n)
    result["final_gwas_median_maf"] = round(float(gwas["GWAS_MAF_FOR_CORR"].dropna().median()), 6) if gwas["GWAS_MAF_FOR_CORR"].notna().any() else ""

    result["final_gwas_duplicate_rsid_count"] = int(
        gwas.loc[gwas["GWAS_RSID"] != "", "GWAS_RSID"].duplicated().sum()
    )
    result["final_gwas_duplicate_rsid_pct"] = percent(
        result["final_gwas_duplicate_rsid_count"],
        result["final_gwas_rsid_available_count"],
    )

    result["final_gwas_duplicate_chrpos_count"] = int(
        gwas.loc[gwas["GWAS_CHRPOS"] != "", "GWAS_CHRPOS"].duplicated().sum()
    )
    result["final_gwas_duplicate_chrpos_pct"] = percent(
        result["final_gwas_duplicate_chrpos_count"],
        result["final_gwas_chrpos_available_count"],
    )

    return result


def calculate_basic_target_features(bim, frq):
    result = {}
    n = len(bim)

    result["target_bim_variant_count"] = n
    result["target_bim_unique_rsid_count"] = int(bim.loc[bim["BIM_SNPID_NORM"] != "", "BIM_SNPID_NORM"].nunique())
    result["target_bim_rsid_available_count"] = int((bim["BIM_SNPID_NORM"] != "").sum())
    result["target_bim_rsid_available_pct"] = percent(result["target_bim_rsid_available_count"], n)

    result["target_bim_chrpos_available_count"] = int((bim["BIM_CHRPOS"] != "").sum())
    result["target_bim_unique_chrpos_count"] = int(bim.loc[bim["BIM_CHRPOS"] != "", "BIM_CHRPOS"].nunique())

    result["target_bim_chrpos_alleles_available_count"] = int((bim["BIM_CHRPOS_ALLELES"] != "").sum())
    result["target_bim_unique_chrpos_alleles_count"] = int(
        bim.loc[bim["BIM_CHRPOS_ALLELES"] != "", "BIM_CHRPOS_ALLELES"].nunique()
    )

    result["target_bim_duplicate_rsid_count"] = int(
        bim.loc[bim["BIM_SNPID_NORM"] != "", "BIM_SNPID_NORM"].duplicated().sum()
    )

    result["target_bim_duplicate_chrpos_count"] = int(
        bim.loc[bim["BIM_CHRPOS"] != "", "BIM_CHRPOS"].duplicated().sum()
    )

    if frq is not None and not frq.empty:
        result["target_freq_variant_count"] = int(len(frq))
        result["target_maf_available_count"] = int(frq["TARGET_MAF"].notna().sum())
        result["target_maf_available_pct"] = percent(result["target_maf_available_count"], len(frq))
        result["target_median_maf"] = round(float(frq["TARGET_MAF"].dropna().median()), 6) if frq["TARGET_MAF"].notna().any() else ""
    else:
        result["target_freq_variant_count"] = 0
        result["target_maf_available_count"] = 0
        result["target_maf_available_pct"] = 0.0
        result["target_median_maf"] = ""

    return result


def allele_match_status(gwas_ea, gwas_nea, bim_a1, bim_a2):
    gwas_ea = normalise_allele(gwas_ea)
    gwas_nea = normalise_allele(gwas_nea)
    bim_a1 = normalise_allele(bim_a1)
    bim_a2 = normalise_allele(bim_a2)

    if not gwas_ea or not gwas_nea or not bim_a1 or not bim_a2:
        return "missing_allele"

    if is_palindromic_pair(gwas_ea, gwas_nea):
        if {gwas_ea, gwas_nea} == {bim_a1, bim_a2}:
            return "palindromic_ambiguous"

    if gwas_ea == bim_a1 and gwas_nea == bim_a2:
        return "direct_match"

    if gwas_ea == bim_a2 and gwas_nea == bim_a1:
        return "flip_match"

    comp_ea = complement_allele(gwas_ea)
    comp_nea = complement_allele(gwas_nea)

    if comp_ea and comp_nea:
        if comp_ea == bim_a1 and comp_nea == bim_a2:
            return "strand_match"
        if comp_ea == bim_a2 and comp_nea == bim_a1:
            return "strand_flip_match"

    return "mismatch"


def calculate_set_overlaps(gwas, bim):
    result = {}

    gwas_rsid = set(gwas.loc[gwas["GWAS_RSID"] != "", "GWAS_RSID"])
    bim_rsid = set(bim.loc[bim["BIM_SNPID_NORM"] != "", "BIM_SNPID_NORM"])
    rsid_overlap = gwas_rsid.intersection(bim_rsid)

    result["gwas_target_rsid_overlap_count"] = len(rsid_overlap)
    result["gwas_target_rsid_overlap_pct_of_gwas_unique_rsid"] = percent(len(rsid_overlap), len(gwas_rsid))
    result["gwas_target_rsid_overlap_pct_of_target_unique_rsid"] = percent(len(rsid_overlap), len(bim_rsid))

    gwas_chrpos = set(gwas.loc[gwas["GWAS_CHRPOS"] != "", "GWAS_CHRPOS"])
    bim_chrpos = set(bim.loc[bim["BIM_CHRPOS"] != "", "BIM_CHRPOS"])
    chrpos_overlap = gwas_chrpos.intersection(bim_chrpos)

    result["gwas_target_chrpos_overlap_count"] = len(chrpos_overlap)
    result["gwas_target_chrpos_overlap_pct_of_gwas_unique_chrpos"] = percent(len(chrpos_overlap), len(gwas_chrpos))
    result["gwas_target_chrpos_overlap_pct_of_target_unique_chrpos"] = percent(len(chrpos_overlap), len(bim_chrpos))

    gwas_chrpos_alleles = set(gwas.loc[gwas["GWAS_CHRPOS_ALLELES"] != "", "GWAS_CHRPOS_ALLELES"])
    bim_chrpos_alleles = set(bim.loc[bim["BIM_CHRPOS_ALLELES"] != "", "BIM_CHRPOS_ALLELES"])
    chrpos_alleles_overlap = gwas_chrpos_alleles.intersection(bim_chrpos_alleles)

    result["gwas_target_chrpos_alleles_overlap_count"] = len(chrpos_alleles_overlap)
    result["gwas_target_chrpos_alleles_overlap_pct_of_gwas_unique_chrpos_alleles"] = percent(
        len(chrpos_alleles_overlap),
        len(gwas_chrpos_alleles),
    )
    result["gwas_target_chrpos_alleles_overlap_pct_of_target_unique_chrpos_alleles"] = percent(
        len(chrpos_alleles_overlap),
        len(bim_chrpos_alleles),
    )

    return result


def calculate_allele_aware_overlap_by_rsid(gwas, bim):
    result = {}

    gwas_sub = gwas[gwas["GWAS_RSID"] != ""].copy()
    bim_sub = bim[bim["BIM_SNPID_NORM"] != ""].copy()

    gwas_sub = gwas_sub.drop_duplicates("GWAS_RSID", keep="first")
    bim_sub = bim_sub.drop_duplicates("BIM_SNPID_NORM", keep="first")

    merged = gwas_sub.merge(
        bim_sub,
        left_on="GWAS_RSID",
        right_on="BIM_SNPID_NORM",
        how="inner",
    )

    if merged.empty:
        result.update({
            "rsid_allele_overlap_rows": 0,
            "rsid_direct_allele_match_count": 0,
            "rsid_flip_allele_match_count": 0,
            "rsid_strand_allele_match_count": 0,
            "rsid_strand_flip_allele_match_count": 0,
            "rsid_allele_mismatch_count": 0,
            "rsid_missing_allele_count": 0,
            "rsid_palindromic_ambiguous_count": 0,
            "rsid_allele_compatible_count": 0,
            "rsid_allele_compatible_pct_of_rsid_overlap": 0.0,
            "rsid_palindromic_overlap_count": 0,
        })
        return result

    merged["ALLELE_MATCH_STATUS"] = [
        allele_match_status(g_ea, g_nea, b_a1, b_a2)
        for g_ea, g_nea, b_a1, b_a2 in zip(
            merged["GWAS_EA_NORM"],
            merged["GWAS_NEA_NORM"],
            merged["BIM_A1_NORM"],
            merged["BIM_A2_NORM"],
        )
    ]

    direct = int((merged["ALLELE_MATCH_STATUS"] == "direct_match").sum())
    flip = int((merged["ALLELE_MATCH_STATUS"] == "flip_match").sum())
    strand = int((merged["ALLELE_MATCH_STATUS"] == "strand_match").sum())
    strand_flip = int((merged["ALLELE_MATCH_STATUS"] == "strand_flip_match").sum())
    mismatch = int((merged["ALLELE_MATCH_STATUS"] == "mismatch").sum())
    missing = int((merged["ALLELE_MATCH_STATUS"] == "missing_allele").sum())
    palindromic_ambiguous = int((merged["ALLELE_MATCH_STATUS"] == "palindromic_ambiguous").sum())
    compatible = direct + flip + strand + strand_flip

    result["rsid_allele_overlap_rows"] = int(len(merged))
    result["rsid_direct_allele_match_count"] = direct
    result["rsid_flip_allele_match_count"] = flip
    result["rsid_strand_allele_match_count"] = strand
    result["rsid_strand_flip_allele_match_count"] = strand_flip
    result["rsid_allele_mismatch_count"] = mismatch
    result["rsid_missing_allele_count"] = missing
    result["rsid_palindromic_ambiguous_count"] = palindromic_ambiguous
    result["rsid_allele_compatible_count"] = compatible
    result["rsid_allele_compatible_pct_of_rsid_overlap"] = percent(compatible, len(merged))
    result["rsid_palindromic_overlap_count"] = int(merged["GWAS_PALINDROMIC"].sum())

    return result


def calculate_allele_aware_overlap_by_chrpos(gwas, bim):
    result = {}

    gwas_sub = gwas[gwas["GWAS_CHRPOS"] != ""].copy()
    bim_sub = bim[bim["BIM_CHRPOS"] != ""].copy()

    gwas_sub = gwas_sub.drop_duplicates("GWAS_CHRPOS", keep="first")
    bim_sub = bim_sub.drop_duplicates("BIM_CHRPOS", keep="first")

    merged = gwas_sub.merge(
        bim_sub,
        left_on="GWAS_CHRPOS",
        right_on="BIM_CHRPOS",
        how="inner",
    )

    if merged.empty:
        result.update({
            "chrpos_allele_overlap_rows": 0,
            "chrpos_direct_allele_match_count": 0,
            "chrpos_flip_allele_match_count": 0,
            "chrpos_strand_allele_match_count": 0,
            "chrpos_strand_flip_allele_match_count": 0,
            "chrpos_allele_mismatch_count": 0,
            "chrpos_missing_allele_count": 0,
            "chrpos_palindromic_ambiguous_count": 0,
            "chrpos_allele_compatible_count": 0,
            "chrpos_allele_compatible_pct_of_chrpos_overlap": 0.0,
            "chrpos_palindromic_overlap_count": 0,
        })
        return result

    merged["ALLELE_MATCH_STATUS"] = [
        allele_match_status(g_ea, g_nea, b_a1, b_a2)
        for g_ea, g_nea, b_a1, b_a2 in zip(
            merged["GWAS_EA_NORM"],
            merged["GWAS_NEA_NORM"],
            merged["BIM_A1_NORM"],
            merged["BIM_A2_NORM"],
        )
    ]

    direct = int((merged["ALLELE_MATCH_STATUS"] == "direct_match").sum())
    flip = int((merged["ALLELE_MATCH_STATUS"] == "flip_match").sum())
    strand = int((merged["ALLELE_MATCH_STATUS"] == "strand_match").sum())
    strand_flip = int((merged["ALLELE_MATCH_STATUS"] == "strand_flip_match").sum())
    mismatch = int((merged["ALLELE_MATCH_STATUS"] == "mismatch").sum())
    missing = int((merged["ALLELE_MATCH_STATUS"] == "missing_allele").sum())
    palindromic_ambiguous = int((merged["ALLELE_MATCH_STATUS"] == "palindromic_ambiguous").sum())
    compatible = direct + flip + strand + strand_flip

    result["chrpos_allele_overlap_rows"] = int(len(merged))
    result["chrpos_direct_allele_match_count"] = direct
    result["chrpos_flip_allele_match_count"] = flip
    result["chrpos_strand_allele_match_count"] = strand
    result["chrpos_strand_flip_allele_match_count"] = strand_flip
    result["chrpos_allele_mismatch_count"] = mismatch
    result["chrpos_missing_allele_count"] = missing
    result["chrpos_palindromic_ambiguous_count"] = palindromic_ambiguous
    result["chrpos_allele_compatible_count"] = compatible
    result["chrpos_allele_compatible_pct_of_chrpos_overlap"] = percent(compatible, len(merged))
    result["chrpos_palindromic_overlap_count"] = int(merged["GWAS_PALINDROMIC"].sum())

    return result


def calculate_maf_correlation_by_rsid(gwas, frq):
    result = {
        "maf_corr_rsid_common_count": 0,
        "maf_corr_rsid_common_allele_compatible_count": 0,
        "maf_corr_rsid": "",
        "maf_corr_rsid_abs_diff_mean": "",
        "maf_corr_rsid_abs_diff_median": "",
    }

    if frq is None or frq.empty:
        return result

    gwas_sub = gwas[
        (gwas["GWAS_RSID"] != "")
        & gwas["GWAS_MAF_FOR_CORR"].notna()
    ].drop_duplicates("GWAS_RSID", keep="first").copy()

    frq_sub = frq[
        (frq["FRQ_RSID"] != "")
        & frq["TARGET_MAF"].notna()
    ].drop_duplicates("FRQ_RSID", keep="first").copy()

    merged = gwas_sub.merge(
        frq_sub,
        left_on="GWAS_RSID",
        right_on="FRQ_RSID",
        how="inner",
    )

    result["maf_corr_rsid_common_count"] = int(len(merged))

    if merged.empty:
        return result

    merged["ALLELE_MATCH_STATUS"] = [
        allele_match_status(g_ea, g_nea, f_a1, f_a2)
        for g_ea, g_nea, f_a1, f_a2 in zip(
            merged["GWAS_EA_NORM"],
            merged["GWAS_NEA_NORM"],
            merged["FRQ_A1_NORM"],
            merged["FRQ_A2_NORM"],
        )
    ]

    compatible = merged[
        merged["ALLELE_MATCH_STATUS"].isin(
            ["direct_match", "flip_match", "strand_match", "strand_flip_match"]
        )
    ].copy()

    result["maf_corr_rsid_common_allele_compatible_count"] = int(len(compatible))

    if len(compatible) >= 3:
        result["maf_corr_rsid"] = correlation_or_blank(
            compatible["GWAS_MAF_FOR_CORR"],
            compatible["TARGET_MAF"],
        )
        diff = (compatible["GWAS_MAF_FOR_CORR"] - compatible["TARGET_MAF"]).abs()
        result["maf_corr_rsid_abs_diff_mean"] = round(float(diff.mean()), 6)
        result["maf_corr_rsid_abs_diff_median"] = round(float(diff.median()), 6)

    return result


def calculate_maf_correlation_by_chrpos_alleles(gwas, bim, frq):
    result = {
        "maf_corr_chrpos_alleles_common_count": 0,
        "maf_corr_chrpos_alleles": "",
        "maf_corr_chrpos_alleles_abs_diff_mean": "",
        "maf_corr_chrpos_alleles_abs_diff_median": "",
    }

    if frq is None or frq.empty:
        return result

    # Add CHR:POS from BIM to FRQ because .frq usually has SNP ID but not BP.
    bim_for_frq = bim[
        (bim["BIM_ID_NORM"] != "")
        & (bim["BIM_CHRPOS_ALLELES"] != "")
    ][["BIM_ID_NORM", "BIM_CHRPOS_ALLELES"]].drop_duplicates("BIM_ID_NORM", keep="first")

    frq2 = frq.merge(
        bim_for_frq,
        left_on="FRQ_ID_NORM",
        right_on="BIM_ID_NORM",
        how="left",
    )

    gwas_sub = gwas[
        (gwas["GWAS_CHRPOS_ALLELES"] != "")
        & gwas["GWAS_MAF_FOR_CORR"].notna()
    ].drop_duplicates("GWAS_CHRPOS_ALLELES", keep="first").copy()

    frq_sub = frq2[
        (frq2["BIM_CHRPOS_ALLELES"].notna())
        & (frq2["BIM_CHRPOS_ALLELES"] != "")
        & frq2["TARGET_MAF"].notna()
    ].drop_duplicates("BIM_CHRPOS_ALLELES", keep="first").copy()

    merged = gwas_sub.merge(
        frq_sub,
        left_on="GWAS_CHRPOS_ALLELES",
        right_on="BIM_CHRPOS_ALLELES",
        how="inner",
    )

    result["maf_corr_chrpos_alleles_common_count"] = int(len(merged))

    if len(merged) >= 3:
        result["maf_corr_chrpos_alleles"] = correlation_or_blank(
            merged["GWAS_MAF_FOR_CORR"],
            merged["TARGET_MAF"],
        )
        diff = (merged["GWAS_MAF_FOR_CORR"] - merged["TARGET_MAF"]).abs()
        result["maf_corr_chrpos_alleles_abs_diff_mean"] = round(float(diff.mean()), 6)
        result["maf_corr_chrpos_alleles_abs_diff_median"] = round(float(diff.median()), 6)

    return result


def calculate_final_usable_variant_estimate(gwas, bim, features):
    compatible_statuses = {"direct_match", "flip_match", "strand_match", "strand_flip_match"}
    usable_target_keys = set()

    bim_work = bim.copy()
    bim_work["BIM_TARGET_KEY"] = np.where(
        bim_work["BIM_ID_NORM"] != "",
        "id:" + bim_work["BIM_ID_NORM"],
        "coord:" + bim_work["BIM_CHRPOS_ALLELES"],
    )

    rsid_merged = gwas[gwas["GWAS_RSID"] != ""].drop_duplicates("GWAS_RSID").merge(
        bim_work[bim_work["BIM_SNPID_NORM"] != ""].drop_duplicates("BIM_SNPID_NORM"),
        left_on="GWAS_RSID",
        right_on="BIM_SNPID_NORM",
        how="inner",
    )
    if not rsid_merged.empty:
        rsid_merged["ALLELE_MATCH_STATUS"] = [
            allele_match_status(g_ea, g_nea, b_a1, b_a2)
            for g_ea, g_nea, b_a1, b_a2 in zip(
                rsid_merged["GWAS_EA_NORM"],
                rsid_merged["GWAS_NEA_NORM"],
                rsid_merged["BIM_A1_NORM"],
                rsid_merged["BIM_A2_NORM"],
            )
        ]
        usable_target_keys.update(
            rsid_merged.loc[
                rsid_merged["ALLELE_MATCH_STATUS"].isin(compatible_statuses),
                "BIM_TARGET_KEY",
            ]
        )

    chrpos_merged = gwas[gwas["GWAS_CHRPOS"] != ""].drop_duplicates("GWAS_CHRPOS").merge(
        bim_work[bim_work["BIM_CHRPOS"] != ""].drop_duplicates("BIM_CHRPOS"),
        left_on="GWAS_CHRPOS",
        right_on="BIM_CHRPOS",
        how="inner",
    )
    if not chrpos_merged.empty:
        chrpos_merged["ALLELE_MATCH_STATUS"] = [
            allele_match_status(g_ea, g_nea, b_a1, b_a2)
            for g_ea, g_nea, b_a1, b_a2 in zip(
                chrpos_merged["GWAS_EA_NORM"],
                chrpos_merged["GWAS_NEA_NORM"],
                chrpos_merged["BIM_A1_NORM"],
                chrpos_merged["BIM_A2_NORM"],
            )
        ]
        usable_target_keys.update(
            chrpos_merged.loc[
                chrpos_merged["ALLELE_MATCH_STATUS"].isin(compatible_statuses),
                "BIM_TARGET_KEY",
            ]
        )

    final_usable = len(usable_target_keys)

    features["gwas_target_final_usable_variant_count"] = final_usable
    features["gwas_target_final_usable_variant_count_estimate"] = final_usable
    features["gwas_target_final_usable_variant_pct_of_final_gwas"] = percent(
        final_usable,
        int(features.get("final_gwas_variant_count", 0)),
    )

    features["gwas_target_final_usable_variant_pct_of_target_bim"] = percent(
        final_usable,
        int(features.get("target_bim_variant_count", 0)),
    )

    return features


# =============================================================================
# MAIN FEATURE CALCULATION
# =============================================================================

def calculate_feature8(job, base_dir, target_bim="", plink_path=""):
    final_gwas_path = find_final_gwas(job, base_dir)
    bim_path = find_target_bim(job["phenotype"], base_dir, explicit_bim=target_bim)

    bfile_prefix = infer_bfile_prefix_from_bim(bim_path)
    fam_path = bfile_prefix.with_suffix(".fam")
    bed_path = bfile_prefix.with_suffix(".bed")

    plink = find_plink(plink_path)

    print("=" * 100)
    print("FEATURE 8: GWAS-TO-TARGET GENOTYPE OVERLAP + MAF CORRELATION + PHENOTYPE FEATURES")
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
    gwas = prepare_gwas_keys(gwas_raw)
    bim = load_bim(bim_path)

    freq_path, freq_status, freq_log = run_plink_freq(
        bfile_prefix=bfile_prefix,
        output_dir=final_gwas_path.parent,
        plink_path=plink,
    )

    frq = load_plink_frq(freq_path) if freq_path else pd.DataFrame()

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
        "plink_path": plink,
        "target_freq_file": str(freq_path) if freq_path else "",
        "target_freq_status": freq_status,
        "target_freq_log_file": "",
        "feature8_status": "started",
        "feature8_error": "",
    }

    if freq_log:
        log_path = final_gwas_path.parent / "Feature8_plink_freq.log"
        log_path.write_text(freq_log, encoding="utf-8", errors="replace")
        features["target_freq_log_file"] = str(log_path)

    # 1. GWAS and target basic features
    features.update(calculate_basic_gwas_features(gwas))
    features.update(calculate_basic_target_features(bim, frq))

    # 2. Set overlap by rsID, CHR:POS, and CHR:POS:allele-pair
    features.update(calculate_set_overlaps(gwas, bim))

    # 3. Allele-aware overlap
    features.update(calculate_allele_aware_overlap_by_rsid(gwas, bim))
    features.update(calculate_allele_aware_overlap_by_chrpos(gwas, bim))

    # 4. MAF correlation
    features.update(calculate_maf_correlation_by_rsid(gwas, frq))
    features.update(calculate_maf_correlation_by_chrpos_alleles(gwas, bim, frq))

    # 5. Phenotype features from FAM
    features.update(calculate_fam_phenotype_features(fam_path))

    # 6. GWAS metadata phenotype features, if phenotype.csv exists
    features.update(calculate_metadata_phenotype_features(job, base_dir))

    # 7. Final usable variant estimate
    features = calculate_final_usable_variant_estimate(gwas, bim, features)

    features["feature8_status"] = (
        "success"
        if freq_status in {"plink_freq_success", "existing_frq_used"}
        else "success_without_target_frequency"
    )

    output_path = final_gwas_path.parent / "Feature8.csv"
    pd.DataFrame([features]).to_csv(output_path, index=False)

    print("[DONE] Feature8 saved")
    print(f"[OUTPUT] {output_path}")
    print()
    print("[KEY RESULTS]")
    print(f"Final GWAS variants:                         {features['final_gwas_variant_count']}")
    print(f"Target BIM variants:                         {features['target_bim_variant_count']}")
    print(f"rsID overlap count:                          {features['gwas_target_rsid_overlap_count']}")
    print(f"CHR:POS overlap count:                       {features['gwas_target_chrpos_overlap_count']}")
    print(f"CHR:POS + allele-pair overlap count:         {features['gwas_target_chrpos_alleles_overlap_count']}")
    print(f"rsID allele-compatible count:                {features['rsid_allele_compatible_count']}")
    print(f"CHR:POS allele-compatible count:             {features['chrpos_allele_compatible_count']}")
    print(f"Final usable variant estimate:               {features['gwas_target_final_usable_variant_count_estimate']}")
    print(f"MAF corr by rsID:                            {features['maf_corr_rsid']}")
    print(f"MAF corr by CHR:POS + alleles:               {features['maf_corr_chrpos_alleles']}")
    print(f"Target cases:                                {features['target_case_count']}")
    print(f"Target controls:                             {features['target_control_count']}")
    print(f"Target effective N:                          {features['target_effective_n_case_control']}")
    print("=" * 100)

    return output_path, features


def save_failure_feature8(job, base_dir, error_message):
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
        "final_gwas_file": "",
        "target_bim_file": "",
        "feature8_status": "failed",
        "feature8_error": error_message,
    }

    output_path = out_dir / "Feature8.csv"
    pd.DataFrame([features]).to_csv(output_path, index=False)

    print("[FAILED] Feature8 saved with error")
    print(f"[OUTPUT] {output_path}")
    print(f"[ERROR] {error_message}")

    return output_path


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate Feature8: GWAS-to-target overlap, allele compatibility, "
            "target MAF correlation, and target phenotype features."
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
        "--no-failure-file",
        action="store_true",
        help="If set, do not write Feature8.csv when calculation fails."
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    job = resolve_job_from_manifest(args.index, args.jobs_file)

    try:
        calculate_feature8(
            job=job,
            base_dir=base_dir,
            target_bim=args.target_bim,
            plink_path=args.plink,
        )
    except Exception as e:
        error_message = f"{type(e).__name__}: {e}"

        if args.no_failure_file:
            raise

        save_failure_feature8(
            job=job,
            base_dir=base_dir,
            error_message=error_message,
        )

        raise


if __name__ == "__main__":
    main()
