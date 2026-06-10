#!/usr/bin/env python3

"""
PRS2-Correlation.py / Variation8.py

Feature8:
GWAS-to-target genotype compatibility + MAF correlation metrics.

Run:
    python PRS2-Correlation.py 10 --force

This version:
    - Uses ProcessedGWAS.csv for one GWAS.
    - Uses phenotype-level genotype:
          <phenotype>/<phenotype>.bim
          <phenotype>/<phenotype>.bed
          <phenotype>/<phenotype>.fam
    - Uses existing <phenotype>/<phenotype>.frq if available.
    - If .frq is missing, runs:
          ./plink --bfile <phenotype>/<phenotype> --freq --out <phenotype>/<phenotype>
    - Saves outputs directly in the GWAS directory:
          <phenotype>/allgwas/<accession>/Features8.csv
          <phenotype>/allgwas/<accession>/Features8_common_snp_overlap.csv
          <phenotype>/allgwas/<accession>/Features8_common_chrpos_overlap.csv

It does NOT save inside Feature8/ folder.
It does NOT use Fold_* folders.
It calculates Feature8 once per GWAS.
"""

import argparse
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_JOBS_FILE = "GWAS_jobs.csv"


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
    return round((numerator / denominator) * 100, 4)


def normalise_chr(x):
    x = clean_text(x).upper()
    x = x.replace("CHR", "").strip()

    if x == "23":
        return "X"
    if x == "24":
        return "Y"
    if x in ["25", "M", "MT"]:
        return "MT"

    return x


def normalise_allele(x):
    return clean_text(x).upper()


def make_chrpos(chrom, bp):
    try:
        chrom = normalise_chr(chrom)
        bp = int(float(bp))
        if chrom and bp > 0:
            return f"{chrom}:{bp}"
    except Exception:
        pass
    return ""


def is_palindromic(a1, a2):
    pair = {normalise_allele(a1), normalise_allele(a2)}
    return pair == {"A", "T"} or pair == {"C", "G"}


def safe_corr(x, y, method="pearson"):
    data = pd.DataFrame({"x": x, "y": y})
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if len(data) < 3:
        return ""

    try:
        return round(float(data["x"].corr(data["y"], method=method)), 6)
    except Exception:
        return ""


def safe_mean_abs_diff(x, y):
    data = pd.DataFrame({"x": x, "y": y})
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if len(data) == 0:
        return ""

    return round(float((data["x"] - data["y"]).abs().mean()), 6)


def safe_median(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna()

    if len(x) == 0:
        return ""

    return float(x.median())


def safe_mean(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna()

    if len(x) == 0:
        return ""

    return float(x.mean())


def find_plink():
    local = Path.cwd() / "plink"
    if local.exists():
        return str(local.resolve())

    local_exe = Path.cwd() / "plink.exe"
    if local_exe.exists():
        return str(local_exe.resolve())

    found = shutil.which("plink")
    if found:
        return found

    return ""


def run_command(command, log_file):
    command = [str(x) for x in command]
    log_file = Path(log_file)

    with open(log_file, "a", encoding="utf-8") as log:
        log.write("\n" + "=" * 100 + "\n")
        log.write(" ".join(command) + "\n")
        log.write("=" * 100 + "\n")

        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        log.write("\n[STDOUT]\n")
        log.write(completed.stdout or "")
        log.write("\n[STDERR]\n")
        log.write(completed.stderr or "")
        log.write("\n[RETURN CODE]\n")
        log.write(str(completed.returncode) + "\n")

    return completed


# =============================================================================
# JOB HANDLING
# =============================================================================

def load_job(index, jobs_file):
    jobs_file = Path(jobs_file)

    if not jobs_file.exists():
        raise FileNotFoundError(f"Missing jobs file: {jobs_file}")

    jobs = pd.read_csv(jobs_file)

    required = ["job_index", "file_path"]
    missing = [c for c in required if c not in jobs.columns]

    if missing:
        raise ValueError(f"Missing required columns in {jobs_file}: {missing}")

    sub = jobs[jobs["job_index"].astype(int) == int(index)]

    if sub.empty:
        raise ValueError(f"No job found for index: {index}")

    return sub.iloc[0]


def infer_phenotype_dir(job, file_path):
    phenotype = clean_text(job.get("phenotype", ""))

    if phenotype and Path(phenotype).exists():
        return Path(phenotype), phenotype

    parts = Path(file_path).parts

    if len(parts) >= 1:
        candidate = Path(parts[0])
        if candidate.exists() and candidate.is_dir():
            return candidate, candidate.name

    if phenotype:
        return Path(phenotype), phenotype

    raise ValueError("Could not infer phenotype directory.")


def get_gwas_output_dir(job, file_path):
    if "output_feature_file" in job.index:
        output_feature_file = clean_text(job.get("output_feature_file", ""))
        if output_feature_file:
            return Path(output_feature_file).parent

    return Path(file_path).parent


def find_phenotype_bfile(phenotype_dir, phenotype):
    phenotype_dir = Path(phenotype_dir)

    # Preferred base bfile:
    # migraine/migraine.bed/.bim/.fam
    preferred = phenotype_dir / phenotype
    if all(Path(str(preferred) + ext).exists() for ext in [".bed", ".bim", ".fam"]):
        return preferred

    # Second preference:
    # migraine/migraine_QC.bed/.bim/.fam
    preferred_qc = phenotype_dir / f"{phenotype}_QC"
    if all(Path(str(preferred_qc) + ext).exists() for ext in [".bed", ".bim", ".fam"]):
        return preferred_qc

    # Fallback: any bfile in root
    bim_files = sorted(phenotype_dir.glob("*.bim"))

    for bim in bim_files:
        prefix = bim.with_suffix("")
        if all(Path(str(prefix) + ext).exists() for ext in [".bed", ".bim", ".fam"]):
            return prefix

    raise FileNotFoundError(
        f"No phenotype-level PLINK bfile found in {phenotype_dir}. "
        f"Expected {phenotype}.bed/.bim/.fam or {phenotype}_QC.bed/.bim/.fam"
    )


# =============================================================================
# LOAD PROCESSED GWAS
# =============================================================================

def load_processed_gwas(processed_gwas_file):
    processed_gwas_file = Path(processed_gwas_file)

    if not processed_gwas_file.exists():
        raise FileNotFoundError(f"Missing ProcessedGWAS.csv: {processed_gwas_file}")

    raw = pd.read_csv(processed_gwas_file)

    required = ["CHR", "BP", "SNP", "A1", "A2", "P", "OR"]
    missing = [c for c in required if c not in raw.columns]

    if missing:
        raise ValueError(f"ProcessedGWAS.csv missing required columns: {missing}")

    for col in ["N", "SE", "INFO", "MAF"]:
        if col not in raw.columns:
            raw[col] = np.nan

    gwas = pd.DataFrame()
    gwas["SNP"] = raw["SNP"].astype(str).map(clean_text)
    gwas["CHR"] = raw["CHR"].map(normalise_chr)
    gwas["BP"] = pd.to_numeric(raw["BP"], errors="coerce")
    gwas["A1"] = raw["A1"].map(normalise_allele)
    gwas["A2"] = raw["A2"].map(normalise_allele)
    gwas["P"] = pd.to_numeric(raw["P"], errors="coerce")
    gwas["OR"] = pd.to_numeric(raw["OR"], errors="coerce")
    gwas["BETA"] = np.where(gwas["OR"] > 0, np.log(gwas["OR"]), np.nan)
    gwas["ABS_BETA"] = np.abs(gwas["BETA"])
    gwas["NEG_LOG10_P"] = np.where(
        (gwas["P"] > 0) & (gwas["P"] <= 1),
        -np.log10(gwas["P"]),
        np.nan,
    )
    gwas["MAF"] = pd.to_numeric(raw["MAF"], errors="coerce")
    gwas["INFO"] = pd.to_numeric(raw["INFO"], errors="coerce")
    gwas["N"] = pd.to_numeric(raw["N"], errors="coerce")
    gwas["SE"] = pd.to_numeric(raw["SE"], errors="coerce")

    gwas["CHRPOS"] = [
        make_chrpos(c, b) for c, b in zip(gwas["CHR"], gwas["BP"])
    ]

    gwas["PALINDROMIC"] = [
        is_palindromic(a1, a2) for a1, a2 in zip(gwas["A1"], gwas["A2"])
    ]

    raw_rows = len(gwas)

    gwas = gwas[
        gwas["SNP"].ne("")
        & gwas["CHRPOS"].ne("")
        & gwas["A1"].str.match(r"^[ACGT]$", na=False)
        & gwas["A2"].str.match(r"^[ACGT]$", na=False)
        & gwas["A1"].ne(gwas["A2"])
        & gwas["P"].notna()
        & (gwas["P"] > 0)
        & (gwas["P"] <= 1)
        & gwas["OR"].notna()
        & (gwas["OR"] > 0)
    ].copy()

    valid_rows_before_dedup = len(gwas)

    before_dedup = len(gwas)
    gwas = gwas.sort_values("P").drop_duplicates(subset=["SNP"], keep="first")
    duplicate_snp_removed = before_dedup - len(gwas)

    metric = {
        "processed_gwas_raw_rows": raw_rows,
        "processed_gwas_valid_rows_before_dedup": valid_rows_before_dedup,
        "processed_gwas_duplicate_snp_removed": duplicate_snp_removed,
        "processed_gwas_clean_rows": len(gwas),
        "processed_gwas_clean_retained_pct": percent(len(gwas), raw_rows),
        "processed_gwas_maf_available_count": int(gwas["MAF"].notna().sum()),
        "processed_gwas_maf_available_pct": percent(int(gwas["MAF"].notna().sum()), len(gwas)),
    }

    return gwas, metric


# =============================================================================
# LOAD TARGET BIM AND FREQUENCY
# =============================================================================

def load_bim(bfile_prefix):
    bim_file = Path(str(bfile_prefix) + ".bim")

    if not bim_file.exists():
        raise FileNotFoundError(f"Missing BIM file: {bim_file}")

    bim = pd.read_csv(
        bim_file,
        sep=r"\s+",
        header=None,
        names=["CHR", "SNP", "CM", "BP", "A1", "A2"],
        dtype=str,
    )

    target = pd.DataFrame()
    target["CHR"] = bim["CHR"].map(normalise_chr)
    target["SNP"] = bim["SNP"].astype(str).map(clean_text)
    target["BP"] = pd.to_numeric(bim["BP"], errors="coerce")
    target["A1"] = bim["A1"].map(normalise_allele)
    target["A2"] = bim["A2"].map(normalise_allele)
    target["CHRPOS"] = [
        make_chrpos(c, b) for c, b in zip(target["CHR"], target["BP"])
    ]
    target["PALINDROMIC"] = [
        is_palindromic(a1, a2) for a1, a2 in zip(target["A1"], target["A2"])
    ]

    raw_rows = len(target)

    target = target[
        target["SNP"].ne("")
        & target["CHRPOS"].ne("")
        & target["A1"].str.match(r"^[ACGT]$", na=False)
        & target["A2"].str.match(r"^[ACGT]$", na=False)
        & target["A1"].ne(target["A2"])
    ].copy()

    valid_rows_before_dedup = len(target)

    before_dedup = len(target)
    target = target.drop_duplicates(subset=["SNP"], keep="first")
    duplicate_snp_removed = before_dedup - len(target)

    metric = {
        "target_bim_file": str(bim_file),
        "target_bim_raw_rows": raw_rows,
        "target_bim_valid_rows_before_dedup": valid_rows_before_dedup,
        "target_bim_duplicate_snp_removed": duplicate_snp_removed,
        "target_bim_clean_rows": len(target),
        "target_bim_clean_retained_pct": percent(len(target), raw_rows),
    }

    return target, metric


def ensure_frq_file(bfile_prefix, force=False):
    bfile_prefix = Path(bfile_prefix)
    frq_file = Path(str(bfile_prefix) + ".frq")
    log_file = Path(str(bfile_prefix) + ".Feature8_freq.log")

    if frq_file.exists() and not force:
        return frq_file, True, "existing_frq_file"

    plink = find_plink()

    if not plink:
        return frq_file, False, "plink_not_found_and_frq_missing"

    missing = []
    for ext in [".bed", ".bim", ".fam"]:
        if not Path(str(bfile_prefix) + ext).exists():
            missing.append(str(bfile_prefix) + ext)

    if missing:
        return frq_file, False, "missing_bfile:" + "|".join(missing)

    cmd = [
        plink,
        "--bfile", str(bfile_prefix),
        "--freq",
        "--out", str(bfile_prefix),
    ]

    completed = run_command(cmd, log_file)

    if completed.returncode == 0 and frq_file.exists():
        return frq_file, True, "created_by_plink_freq"

    return frq_file, False, f"plink_freq_failed_returncode_{completed.returncode}"


def load_frq(frq_file):
    frq_file = Path(frq_file)

    if not frq_file.exists():
        return pd.DataFrame(), {
            "target_frq_file": str(frq_file),
            "target_frq_available": False,
            "target_frq_rows": 0,
        }

    frq = pd.read_csv(frq_file, sep=r"\s+")

    required = ["SNP", "A1", "A2", "MAF"]
    missing = [c for c in required if c not in frq.columns]

    if missing:
        return pd.DataFrame(), {
            "target_frq_file": str(frq_file),
            "target_frq_available": False,
            "target_frq_issue": "missing_columns:" + "|".join(missing),
            "target_frq_rows": len(frq),
        }

    target_freq = pd.DataFrame()
    target_freq["SNP"] = frq["SNP"].astype(str).map(clean_text)
    target_freq["FREQ_A1"] = frq["A1"].map(normalise_allele)
    target_freq["FREQ_A2"] = frq["A2"].map(normalise_allele)
    target_freq["TARGET_MAF"] = pd.to_numeric(frq["MAF"], errors="coerce")

    target_freq = target_freq[
        target_freq["SNP"].ne("")
        & target_freq["FREQ_A1"].str.match(r"^[ACGT]$", na=False)
        & target_freq["FREQ_A2"].str.match(r"^[ACGT]$", na=False)
        & target_freq["TARGET_MAF"].notna()
        & (target_freq["TARGET_MAF"] >= 0)
        & (target_freq["TARGET_MAF"] <= 0.5)
    ].copy()

    target_freq = target_freq.drop_duplicates(subset=["SNP"], keep="first")

    metric = {
        "target_frq_file": str(frq_file),
        "target_frq_available": True,
        "target_frq_rows": len(frq),
        "target_frq_clean_rows": len(target_freq),
    }

    return target_freq, metric


# =============================================================================
# METRICS
# =============================================================================

def classify_allele_status(row):
    g1 = row["A1_GWAS"]
    g2 = row["A2_GWAS"]
    t1 = row["A1_TARGET"]
    t2 = row["A2_TARGET"]

    if g1 == t1 and g2 == t2:
        return "same_orientation"

    if g1 == t2 and g2 == t1:
        return "flipped_orientation"

    return "allele_mismatch"


def add_correlation_metrics(prefix, df, result):
    """
    Calculates correlations where possible.
    Requires GWAS_MAF and TARGET_MAF.
    Also calculates correlation of target MAF with ABS_BETA and -log10(P).
    """
    if df.empty:
        result[f"{prefix}_rows_for_maf_correlation"] = 0
        result[f"{prefix}_maf_pearson_correlation"] = ""
        result[f"{prefix}_maf_spearman_correlation"] = ""
        result[f"{prefix}_maf_mean_abs_difference"] = ""
        result[f"{prefix}_abs_beta_vs_target_maf_pearson"] = ""
        result[f"{prefix}_neglog10p_vs_target_maf_pearson"] = ""
        return result

    data = df.copy()

    data["GWAS_MAF"] = pd.to_numeric(data.get("MAF", np.nan), errors="coerce")
    data["TARGET_MAF"] = pd.to_numeric(data.get("TARGET_MAF", np.nan), errors="coerce")
    data["ABS_BETA"] = pd.to_numeric(data.get("ABS_BETA", np.nan), errors="coerce")
    data["NEG_LOG10_P"] = pd.to_numeric(data.get("NEG_LOG10_P", np.nan), errors="coerce")

    maf_data = data[
        data["GWAS_MAF"].notna()
        & data["TARGET_MAF"].notna()
        & (data["GWAS_MAF"] >= 0)
        & (data["GWAS_MAF"] <= 0.5)
        & (data["TARGET_MAF"] >= 0)
        & (data["TARGET_MAF"] <= 0.5)
    ].copy()

    result[f"{prefix}_rows_for_maf_correlation"] = len(maf_data)
    result[f"{prefix}_maf_pearson_correlation"] = safe_corr(maf_data["GWAS_MAF"], maf_data["TARGET_MAF"], "pearson")
    result[f"{prefix}_maf_spearman_correlation"] = safe_corr(maf_data["GWAS_MAF"], maf_data["TARGET_MAF"], "spearman")
    result[f"{prefix}_maf_mean_abs_difference"] = safe_mean_abs_diff(maf_data["GWAS_MAF"], maf_data["TARGET_MAF"])

    result[f"{prefix}_gwas_maf_mean"] = safe_mean(maf_data["GWAS_MAF"])
    result[f"{prefix}_target_maf_mean"] = safe_mean(maf_data["TARGET_MAF"])
    result[f"{prefix}_gwas_maf_median"] = safe_median(maf_data["GWAS_MAF"])
    result[f"{prefix}_target_maf_median"] = safe_median(maf_data["TARGET_MAF"])

    result[f"{prefix}_abs_beta_vs_target_maf_pearson"] = safe_corr(maf_data["ABS_BETA"], maf_data["TARGET_MAF"], "pearson")
    result[f"{prefix}_abs_beta_vs_target_maf_spearman"] = safe_corr(maf_data["ABS_BETA"], maf_data["TARGET_MAF"], "spearman")
    result[f"{prefix}_neglog10p_vs_target_maf_pearson"] = safe_corr(maf_data["NEG_LOG10_P"], maf_data["TARGET_MAF"], "pearson")
    result[f"{prefix}_neglog10p_vs_target_maf_spearman"] = safe_corr(maf_data["NEG_LOG10_P"], maf_data["TARGET_MAF"], "spearman")

    return result


def calculate_feature8(gwas, target, target_freq):
    result = {}

    # -------------------------------------------------------------------------
    # Set overlaps
    # -------------------------------------------------------------------------
    gwas_snp_set = set(gwas["SNP"])
    target_snp_set = set(target["SNP"])
    gwas_chrpos_set = set(gwas["CHRPOS"])
    target_chrpos_set = set(target["CHRPOS"])

    snp_overlap = gwas_snp_set & target_snp_set
    chrpos_overlap = gwas_chrpos_set & target_chrpos_set

    result["gwas_unique_snp"] = len(gwas_snp_set)
    result["target_unique_snp"] = len(target_snp_set)
    result["gwas_unique_chrpos"] = len(gwas_chrpos_set)
    result["target_unique_chrpos"] = len(target_chrpos_set)

    result["snpid_overlap_count"] = len(snp_overlap)
    result["snpid_overlap_pct_of_gwas"] = percent(len(snp_overlap), len(gwas_snp_set))
    result["snpid_overlap_pct_of_target"] = percent(len(snp_overlap), len(target_snp_set))

    result["chrpos_overlap_count"] = len(chrpos_overlap)
    result["chrpos_overlap_pct_of_gwas"] = percent(len(chrpos_overlap), len(gwas_chrpos_set))
    result["chrpos_overlap_pct_of_target"] = percent(len(chrpos_overlap), len(target_chrpos_set))

    # -------------------------------------------------------------------------
    # SNP-based common variants
    # -------------------------------------------------------------------------
    snp_merged = gwas.merge(
        target,
        on="SNP",
        how="inner",
        suffixes=("_GWAS", "_TARGET"),
    )

    result["common_snp_rows"] = len(snp_merged)

    if not snp_merged.empty:
        snp_merged["ALLELE_STATUS"] = snp_merged.apply(classify_allele_status, axis=1)
        snp_merged["PALINDROMIC_OVERLAP"] = [
            is_palindromic(a1, a2)
            for a1, a2 in zip(snp_merged["A1_GWAS"], snp_merged["A2_GWAS"])
        ]

        same = int((snp_merged["ALLELE_STATUS"] == "same_orientation").sum())
        flipped = int((snp_merged["ALLELE_STATUS"] == "flipped_orientation").sum())
        mismatch = int((snp_merged["ALLELE_STATUS"] == "allele_mismatch").sum())
        usable = same + flipped

        result["allele_same_orientation_count"] = same
        result["allele_flipped_orientation_count"] = flipped
        result["allele_mismatch_count"] = mismatch
        result["allele_same_orientation_pct_of_common_snp"] = percent(same, len(snp_merged))
        result["allele_flipped_orientation_pct_of_common_snp"] = percent(flipped, len(snp_merged))
        result["allele_mismatch_pct_of_common_snp"] = percent(mismatch, len(snp_merged))

        result["palindromic_common_snp_count"] = int(snp_merged["PALINDROMIC_OVERLAP"].sum())
        result["non_palindromic_common_snp_count"] = len(snp_merged) - int(snp_merged["PALINDROMIC_OVERLAP"].sum())

        result["usable_for_prs_count"] = usable
        result["usable_for_prs_pct_of_gwas"] = percent(usable, len(gwas))
        result["usable_for_prs_pct_of_target"] = percent(usable, len(target))
        result["usable_for_prs_pct_of_common_snp"] = percent(usable, len(snp_merged))
    else:
        result["allele_same_orientation_count"] = 0
        result["allele_flipped_orientation_count"] = 0
        result["allele_mismatch_count"] = 0
        result["allele_same_orientation_pct_of_common_snp"] = 0.0
        result["allele_flipped_orientation_pct_of_common_snp"] = 0.0
        result["allele_mismatch_pct_of_common_snp"] = 0.0
        result["palindromic_common_snp_count"] = 0
        result["non_palindromic_common_snp_count"] = 0
        result["usable_for_prs_count"] = 0
        result["usable_for_prs_pct_of_gwas"] = 0.0
        result["usable_for_prs_pct_of_target"] = 0.0
        result["usable_for_prs_pct_of_common_snp"] = 0.0

    # Add MAF from target .frq by SNP.
    if not target_freq.empty and not snp_merged.empty:
        snp_merged = snp_merged.merge(target_freq, on="SNP", how="left")
    else:
        snp_merged["TARGET_MAF"] = np.nan

    result = add_correlation_metrics("common_snp", snp_merged, result)

    if "ALLELE_STATUS" in snp_merged.columns:
        usable_snp = snp_merged[snp_merged["ALLELE_STATUS"].isin(["same_orientation", "flipped_orientation"])].copy()
    else:
        usable_snp = pd.DataFrame()

    result = add_correlation_metrics("usable_common_snp", usable_snp, result)

    # -------------------------------------------------------------------------
    # CHR:BP-based common variants
    # This catches files where rsID does not match but coordinates do.
    # -------------------------------------------------------------------------
    chrpos_merged = gwas.merge(
        target,
        on="CHRPOS",
        how="inner",
        suffixes=("_GWAS", "_TARGET"),
    )

    result["common_chrpos_rows"] = len(chrpos_merged)

    if not chrpos_merged.empty:
        chrpos_merged["ALLELE_STATUS"] = chrpos_merged.apply(classify_allele_status, axis=1)
        chrpos_merged["PALINDROMIC_OVERLAP"] = [
            is_palindromic(a1, a2)
            for a1, a2 in zip(chrpos_merged["A1_GWAS"], chrpos_merged["A2_GWAS"])
        ]

        chrpos_same = int((chrpos_merged["ALLELE_STATUS"] == "same_orientation").sum())
        chrpos_flipped = int((chrpos_merged["ALLELE_STATUS"] == "flipped_orientation").sum())
        chrpos_mismatch = int((chrpos_merged["ALLELE_STATUS"] == "allele_mismatch").sum())
        chrpos_usable = chrpos_same + chrpos_flipped

        result["chrpos_allele_same_orientation_count"] = chrpos_same
        result["chrpos_allele_flipped_orientation_count"] = chrpos_flipped
        result["chrpos_allele_mismatch_count"] = chrpos_mismatch
        result["chrpos_usable_for_prs_count"] = chrpos_usable
        result["chrpos_usable_for_prs_pct_of_common_chrpos"] = percent(chrpos_usable, len(chrpos_merged))
    else:
        result["chrpos_allele_same_orientation_count"] = 0
        result["chrpos_allele_flipped_orientation_count"] = 0
        result["chrpos_allele_mismatch_count"] = 0
        result["chrpos_usable_for_prs_count"] = 0
        result["chrpos_usable_for_prs_pct_of_common_chrpos"] = 0.0

    # For CHRPOS MAF, target_freq is keyed by SNP, so merge target MAF using target SNP name.
    if not target_freq.empty and not chrpos_merged.empty:
        chrpos_merged = chrpos_merged.merge(
            target_freq,
            left_on="SNP_TARGET",
            right_on="SNP",
            how="left",
            suffixes=("", "_FRQ"),
        )
    else:
        chrpos_merged["TARGET_MAF"] = np.nan

    result = add_correlation_metrics("common_chrpos", chrpos_merged, result)

    if "ALLELE_STATUS" in chrpos_merged.columns:
        usable_chrpos = chrpos_merged[
            chrpos_merged["ALLELE_STATUS"].isin(["same_orientation", "flipped_orientation"])
        ].copy()
    else:
        usable_chrpos = pd.DataFrame()

    result = add_correlation_metrics("usable_common_chrpos", usable_chrpos, result)

    return result, snp_merged, chrpos_merged


# =============================================================================
# MAIN
# =============================================================================

def process_one_job(index, args):
    job = load_job(index, args.jobs_file)

    file_path = Path(clean_text(job["file_path"]))
    phenotype_dir, phenotype = infer_phenotype_dir(job, file_path)
    accession = clean_text(job.get("accessionId", ""))

    gwas_output_dir = get_gwas_output_dir(job, file_path)

    processed_gwas_file = gwas_output_dir / "ProcessedGWAS.csv"
    if args.processed_gwas:
        processed_gwas_file = Path(args.processed_gwas)

    if not processed_gwas_file.exists():
        raise FileNotFoundError(
            f"Missing ProcessedGWAS.csv: {processed_gwas_file}\n"
            f"Run PRS1-Normalization1.py {index} --force first."
        )

    target_bfile = find_phenotype_bfile(phenotype_dir, phenotype)
    target_bim = Path(str(target_bfile) + ".bim")

    print("=" * 100)
    print("VARIATION 8: GWAS VS PHENOTYPE GENOTYPE COMPATIBILITY + CORRELATION")
    print("=" * 100)
    print(f"[JOB INDEX]       {index}")
    print(f"[PHENOTYPE]       {phenotype}")
    print(f"[ACCESSION]       {accession}")
    print(f"[PHENOTYPE DIR]   {phenotype_dir}")
    print(f"[TARGET BFILE]    {target_bfile}")
    print(f"[TARGET BIM]      {target_bim}")
    print(f"[GWAS OUTPUT DIR] {gwas_output_dir}")
    print(f"[PROCESSED GWAS]  {processed_gwas_file}")
    print("=" * 100)

    output_feature_file = gwas_output_dir / "Features8.csv"
    output_snp_overlap = gwas_output_dir / "Features8_common_snp_overlap.csv"
    output_chrpos_overlap = gwas_output_dir / "Features8_common_chrpos_overlap.csv"
    output_summary_file = gwas_output_dir / "Features8_run_summary.csv"

    gwas, gwas_metric = load_processed_gwas(processed_gwas_file)
    target, target_metric = load_bim(target_bfile)

    frq_file, frq_ok, frq_status = ensure_frq_file(target_bfile, force=args.force_freq)
    target_freq, frq_metric = load_frq(frq_file)

    metrics, snp_overlap_df, chrpos_overlap_df = calculate_feature8(
        gwas=gwas,
        target=target,
        target_freq=target_freq,
    )

    summary = {
        "job_index": index,
        "phenotype": phenotype,
        "accessionId": accession,
        "file_path": str(file_path),
        "processed_gwas_file": str(processed_gwas_file),
        "phenotype_dir": str(phenotype_dir),
        "target_bfile": str(target_bfile),
        "target_bim": str(target_bim),
        "gwas_output_dir": str(gwas_output_dir),
        "features8_file": str(output_feature_file),
        "common_snp_overlap_file": str(output_snp_overlap),
        "common_chrpos_overlap_file": str(output_chrpos_overlap),
        "target_frq_generation_status": frq_status,
        "target_frq_generation_success": frq_ok,
    }

    summary.update(gwas_metric)
    summary.update(target_metric)
    summary.update(frq_metric)
    summary.update(metrics)

    # Save main one-row file directly inside GWAS directory.
    pd.DataFrame([summary]).to_csv(output_feature_file, index=False)
    pd.DataFrame([summary]).to_csv(output_summary_file, index=False)

    # Save overlap details.
    # These contain the actual common SNP / common CHRPOS rows used for correlation.
    if args.save_all_overlap:
        snp_overlap_df.to_csv(output_snp_overlap, index=False)
        chrpos_overlap_df.to_csv(output_chrpos_overlap, index=False)
    else:
        snp_overlap_df.head(args.overlap_limit).to_csv(output_snp_overlap, index=False)
        chrpos_overlap_df.head(args.overlap_limit).to_csv(output_chrpos_overlap, index=False)

    print("=" * 100)
    print("[SAVED]")
    print(f"Features8.csv:                    {output_feature_file}")
    print(f"Features8_run_summary.csv:        {output_summary_file}")
    print(f"Features8_common_snp_overlap.csv: {output_snp_overlap}")
    print(f"Features8_common_chrpos_overlap:  {output_chrpos_overlap}")
    print("=" * 100)

    print()
    print("[KEY METRICS]")
    keys = [
        "processed_gwas_clean_rows",
        "target_bim_clean_rows",
        "target_frq_clean_rows",

        "snpid_overlap_count",
        "snpid_overlap_pct_of_gwas",
        "snpid_overlap_pct_of_target",

        "chrpos_overlap_count",
        "chrpos_overlap_pct_of_gwas",
        "chrpos_overlap_pct_of_target",

        "allele_same_orientation_count",
        "allele_flipped_orientation_count",
        "allele_mismatch_count",
        "allele_mismatch_pct_of_common_snp",

        "usable_for_prs_count",
        "usable_for_prs_pct_of_gwas",
        "usable_for_prs_pct_of_common_snp",

        "common_snp_rows_for_maf_correlation",
        "common_snp_maf_pearson_correlation",
        "common_snp_maf_spearman_correlation",
        "common_snp_maf_mean_abs_difference",

        "usable_common_snp_rows_for_maf_correlation",
        "usable_common_snp_maf_pearson_correlation",
        "usable_common_snp_maf_spearman_correlation",
        "usable_common_snp_maf_mean_abs_difference",

        "common_chrpos_rows_for_maf_correlation",
        "common_chrpos_maf_pearson_correlation",
        "common_chrpos_maf_spearman_correlation",
        "common_chrpos_maf_mean_abs_difference",

        "usable_common_chrpos_rows_for_maf_correlation",
        "usable_common_chrpos_maf_pearson_correlation",
        "usable_common_chrpos_maf_spearman_correlation",
        "usable_common_chrpos_maf_mean_abs_difference",
    ]

    for key in keys:
        print(f"{key}: {summary.get(key, '')}")

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Variation8/PRS2-Correlation: GWAS vs phenotype genotype overlap and MAF correlation."
    )

    parser.add_argument(
        "index",
        type=int,
        help="GWAS job index. Example: python PRS2-Correlation.py 10",
    )

    parser.add_argument(
        "--jobs-file",
        default=DEFAULT_JOBS_FILE,
        help="Default: GWAS_jobs.csv",
    )

    parser.add_argument(
        "--processed-gwas",
        default="",
        help="Optional direct path to ProcessedGWAS.csv.",
    )

    parser.add_argument(
        "--force-freq",
        action="store_true",
        help="Recalculate phenotype-level .frq using PLINK even if it already exists.",
    )

    parser.add_argument(
        "--save-all-overlap",
        action="store_true",
        help="Save all overlap rows. Default saves only first --overlap-limit rows.",
    )

    parser.add_argument(
        "--overlap-limit",
        type=int,
        default=100000,
        help="Number of overlap rows to save unless --save-all-overlap is used. Default: 100000.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Accepted for compatibility. Existing output CSVs are overwritten by default.",
    )

    args = parser.parse_args()

    process_one_job(args.index, args)


if __name__ == "__main__":
    main()