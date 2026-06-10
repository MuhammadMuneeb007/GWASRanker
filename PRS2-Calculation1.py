#!/usr/bin/env python3

"""
PRS2-Calculation1.py

PRS calculation using PLINK for one GWAS job.

Run:
    python PRS2-Calculation1.py 1 --force

This script assumes:
    1. You already ran PRS1-Normalization1.py.
    2. Each GWAS directory contains ProcessedGWAS_hg38.csv.
    3. Shared genotype fold data are in phenotype/Fold_*/.
    4. PLINK is in the current working directory as ./plink.
    5. Phenotypes are binary only.

Main outputs are saved per GWAS:

    <GWAS_DIR>/PRS/Plink_HG38/
        GWAS_hg38_for_plink.tsv
        SNP.pvalue
        range_list
        PRS_run_summary.csv
        PRS_fold_metrics.csv
        PRS_all_folds_results.csv
        PRS_average_performance.csv
        PRS_best_result.csv

Per fold:

    <GWAS_DIR>/PRS/Plink_HG38/Fold_1/
        prune/
        clump/
        genotype_clones/
        scores/
        metrics/
        Results.csv

Important:
    The original genotype files in phenotype/Fold_* are never overwritten.
    All PLINK --out files go inside the GWAS-specific PRS directory.
"""

import argparse
import contextlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# DEFAULT PARAMETERS
# =============================================================================

DEFAULT_JOBS_FILE = "GWAS_jobs.csv"

TRAIN_BFILE_NAME = "train_data.QC"
TEST_BFILE_NAME = "test_data"

DEFAULT_P_WINDOW_SIZE = 200
DEFAULT_P_SLIDE_SIZE = 50
DEFAULT_P_LD_THRESHOLD = 0.25

DEFAULT_CLUMP_P1 = 1
DEFAULT_CLUMP_R2 = 0.1
DEFAULT_CLUMP_KB = 200

DEFAULT_MIN_P_EXPONENT = 10
DEFAULT_N_PVALUE_INTERVALS = 20


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


def safe_float(x):
    try:
        x = float(x)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return np.nan


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_command(command, log_file, cwd=None, print_output=True):
    command = [str(x) for x in command]
    log_file = Path(log_file)
    ensure_dir(log_file.parent)

    command_text = " ".join(command)

    print()
    print("=" * 100)
    print("[COMMAND START]")
    print("[CMD]", command_text)
    print("[CWD]", str(cwd) if cwd else str(Path.cwd()))
    print("[LOG]", log_file)
    print("=" * 100)
    sys.stdout.flush()

    with open(log_file, "a", encoding="utf-8") as log:
        log.write("\n" + "=" * 100 + "\n")
        log.write("[COMMAND START]\n")
        log.write(command_text + "\n")
        log.write(f"[CWD] {str(cwd) if cwd else str(Path.cwd())}\n")
        log.write("=" * 100 + "\n")

        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""

        if print_output:
            print("[STDOUT]")
            print(stdout if stdout else "<empty>")
            print("[STDERR]")
            print(stderr if stderr else "<empty>")
        else:
            print("[STDOUT/STDERR]")
            print(f"Suppressed in terminal; full output saved in {log_file}")
            if completed.returncode != 0:
                print("[STDOUT]")
                print(stdout if stdout else "<empty>")
                print("[STDERR]")
                print(stderr if stderr else "<empty>")
        print("[RETURN CODE]", completed.returncode)
        if completed.returncode != 0:
            print("[COMMAND FAILED]")
        print("[COMMAND END]")
        print("=" * 100)
        sys.stdout.flush()

        log.write("\n[STDOUT]\n")
        log.write(stdout)
        log.write("\n[STDERR]\n")
        log.write(stderr)
        log.write("\n[RETURN CODE]\n")
        log.write(str(completed.returncode) + "\n")
        if completed.returncode != 0:
            log.write("[COMMAND FAILED]\n")
        log.write("[COMMAND END]\n")

    return completed


def find_plink():
    local = Path.cwd() / "plink"
    if local.exists():
        return "./plink"

    local_exe = Path.cwd() / "plink.exe"
    if local_exe.exists():
        return "./plink.exe"

    found = shutil.which("plink")
    if found:
        return found

    raise FileNotFoundError(
        "PLINK not found. Put plink in the current working directory or add it to PATH."
    )


def check_bfile(prefix):
    prefix = Path(prefix)
    missing = []

    for suffix in [".bed", ".bim", ".fam"]:
        path = Path(str(prefix) + suffix)
        if not path.exists():
            missing.append(str(path))

    return missing


def natural_fold_number(path):
    m = re.search(r"Fold_(\d+)", str(path))
    if m:
        return int(m.group(1))
    return 10**9


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

    # Example:
    # migraine/allgwas/GCST90000016/GCST90000016.h.tsv.gz
    parts = Path(file_path).parts
    if len(parts) >= 1:
        candidate = Path(parts[0])
        if candidate.exists() and candidate.is_dir():
            return candidate, candidate.name

    if phenotype:
        return Path(phenotype), phenotype

    raise ValueError(
        "Could not infer phenotype directory. Add phenotype column to GWAS_jobs.csv."
    )


def get_gwas_output_dir(job, file_path):
    if "output_feature_file" in job.index:
        out = clean_text(job.get("output_feature_file", ""))
        if out:
            return Path(out).parent

    return Path(file_path).parent


def find_folds(phenotype_dir, requested_folds):
    phenotype_dir = Path(phenotype_dir)

    if requested_folds != "auto":
        folds = []
        for x in requested_folds.split(","):
            x = clean_text(x)
            if not x:
                continue

            fold_dir = phenotype_dir / f"Fold_{x}"
            if fold_dir.exists():
                folds.append(fold_dir)
            else:
                print("[WARNING] Fold not found:", fold_dir)

        return sorted(folds, key=natural_fold_number)

    folds = sorted(phenotype_dir.glob("Fold_*"), key=natural_fold_number)
    return [f for f in folds if f.is_dir()]


# =============================================================================
# PREPARE GWAS FOR PLINK SCORE
# =============================================================================

def prepare_gwas_for_plink(hg38_gwas_file, prs_dir):
    """
    Prepare PLINK score and p-value files from the already-converted
    ProcessedGWAS_hg38.csv file.

    Important:
    - Uses BETA directly.
    - Does NOT require OR.
    - Does NOT perform liftover.
    - Keeps only rows with SNP, A1, P, and BETA.
    """

    prs_dir = ensure_dir(prs_dir)
    hg38_gwas_file = Path(hg38_gwas_file)

    df = pd.read_csv(hg38_gwas_file, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    print()
    print("=" * 100)
    print("[PLINK SCORE GWAS INPUT CHECK]")
    print("[FILE]", hg38_gwas_file)
    print("[ROWS]", len(df))
    print("[COLUMNS]", ", ".join(map(str, df.columns)))
    print("=" * 100)

    required = ["SNP", "A1", "P"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns in converted GWAS file: {missing}\n"
            f"Expected file: {hg38_gwas_file}\n"
            "This file must be created by PRS1-Normalization2-hg38.py."
        )

    has_beta = "BETA" in df.columns
    has_or = "OR" in df.columns

    if not has_beta and not has_or:
        raise ValueError(
            "Missing effect-size columns in converted GWAS file.\n"
            "Need at least one of: BETA or OR.\n"
            f"Input file: {hg38_gwas_file}"
        )

    out = pd.DataFrame()
    out["SNP"] = df["SNP"].astype(str).map(clean_text)
    out["A1"] = df["A1"].astype(str).map(clean_text).str.upper()
    out["P"] = pd.to_numeric(df["P"], errors="coerce")

    if has_beta:
        out["BETA"] = pd.to_numeric(df["BETA"], errors="coerce")
        effect_source = "BETA"
    else:
        or_values = pd.to_numeric(df["OR"], errors="coerce")
        out["BETA"] = np.where(or_values > 0, np.log(or_values), np.nan)
        effect_source = "OR_to_BETA"

    pre_rows = len(out)

    diagnostic = {
        "rows": pre_rows,
        "nonmissing_snp": int(out["SNP"].ne("").sum()),
        "valid_a1_single_acgt": int(out["A1"].str.match(r"^[ACGT]$", na=False).sum()),
        "valid_p_0_1": int(((out["P"] > 0) & (out["P"] <= 1)).sum()),
        "valid_beta": int(out["BETA"].notna().sum()),
    }

    # Optional: if PRS_READINESS exists, use it.
    if "PRS_READINESS" in df.columns:
        out["PRS_READINESS"] = df["PRS_READINESS"].astype(str).map(clean_text)
        diagnostic["strict_prs_ready"] = int(out["PRS_READINESS"].eq("STRICT_PRS_READY").sum())
        diagnostic["approx_prs_ready"] = int(out["PRS_READINESS"].eq("APPROX_PRS_READY").sum())
    else:
        out["PRS_READINESS"] = "NO_PRS_READINESS_COLUMN"

    print("[PLINK SCORE DIAGNOSTICS]")
    for key, value in diagnostic.items():
        print(f"{key}: {value}")
    print(f"effect_source: {effect_source}")

    valid = (
        out["SNP"].ne("")
        & out["A1"].str.match(r"^[ACGT]$", na=False)
        & out["P"].notna()
        & np.isfinite(out["P"])
        & (out["P"] > 0)
        & (out["P"] <= 1)
        & out["BETA"].notna()
        & np.isfinite(out["BETA"])
    )

    # If PRS_READINESS exists, keep only PRS-ready rows.
    if "PRS_READINESS" in df.columns:
        valid = valid & out["PRS_READINESS"].isin(["STRICT_PRS_READY", "APPROX_PRS_READY"])

    out = out.loc[valid, ["SNP", "A1", "BETA", "P"]].copy()

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["SNP", "A1", "BETA", "P"])

    before_dedup = len(out)
    out = out.sort_values("P").drop_duplicates(subset=["SNP"], keep="first")
    duplicate_removed = before_dedup - len(out)

    score_file = prs_dir / "GWAS_hg38_for_plink.tsv"
    pvalue_file = prs_dir / "SNP.pvalue"

    # PLINK --score columns:
    # 1 SNP, 2 A1, 3 BETA
    out[["SNP", "A1", "BETA", "P"]].to_csv(score_file, sep="\t", index=False)

    # q-score file:
    # SNP P, no header
    out[["SNP", "P"]].to_csv(pvalue_file, sep="\t", index=False, header=False)

    metric = {
        "gwas_for_plink_input_rows": pre_rows,
        "gwas_for_plink_valid_rows": len(out),
        "gwas_for_plink_duplicate_snp_removed": duplicate_removed,
        "gwas_for_plink_retained_pct": percent(len(out), pre_rows),
        "score_file": str(score_file),
        "pvalue_file": str(pvalue_file),
        "used_converted_hg38_file": True,
        "effect_source": effect_source,
        "used_beta_directly": effect_source == "BETA",
        "derived_beta_from_or": effect_source == "OR_to_BETA",
        "required_or_for_scoring": effect_source == "OR_to_BETA",
    }

    print("[PLINK SCORE OUTPUT CHECK]")
    print("[VALID ROWS]", len(out))
    print("[DUPLICATE SNP REMOVED]", duplicate_removed)
    print("[SCORE FILE]", score_file)
    print("[PVALUE FILE]", pvalue_file)
    print("=" * 100)

    if len(out) == 0:
        raise ValueError(
            "No valid SNPs were available for PLINK scoring.\n"
            f"Input file: {hg38_gwas_file}\n"
            f"Input rows={pre_rows}; "
            f"nonmissing SNP={diagnostic['nonmissing_snp']}; "
            f"valid A1={diagnostic['valid_a1_single_acgt']}; "
            f"valid P={diagnostic['valid_p_0_1']}; "
            f"valid BETA={diagnostic['valid_beta']}.\n"
            "Fix ProcessedGWAS_hg38.csv before running PRS2."
        )

    return score_file, pvalue_file, metric


def create_range_list(prs_dir, min_p_exponent, n_intervals):
    prs_dir = ensure_dir(prs_dir)

    values = np.logspace(
        -int(min_p_exponent),
        0,
        int(n_intervals),
        endpoint=True,
    )

    range_file = prs_dir / "range_list"
    labels = []

    with open(range_file, "w", encoding="utf-8") as f:
        for value in values:
            label = f"pv_{value:.12g}"
            labels.append((label, float(value)))
            f.write(f"{label} 0 {value:.12g}\n")

    return range_file, labels


# =============================================================================
# PLINK INTERMEDIATE STEPS PER FOLD
# =============================================================================

def extract_valid_snps_from_clumped(clumped_file, valid_snp_file):
    clumped_file = Path(clumped_file)
    valid_snp_file = Path(valid_snp_file)
    ensure_dir(valid_snp_file.parent)

    snps = []

    if not clumped_file.exists():
        valid_snp_file.write_text("", encoding="utf-8")
        return 0

    with open(clumped_file, "r", encoding="utf-8", errors="replace") as f:
        header = None
        snp_col_idx = None

        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = re.split(r"\s+", line)

            if header is None:
                header = parts
                if "SNP" in header:
                    snp_col_idx = header.index("SNP")
                else:
                    snp_col_idx = 2
                continue

            if line.startswith("Warning") or line.startswith("Error"):
                continue

            if snp_col_idx is not None and len(parts) > snp_col_idx:
                snp = parts[snp_col_idx]
                if snp and snp != "NONE":
                    snps.append(snp)

    with open(valid_snp_file, "w", encoding="utf-8") as out:
        for snp in snps:
            out.write(str(snp) + "\n")

    return len(snps)


def extract_pruned_score_overlap_snps(prune_in_file, score_file, valid_snp_file):
    prune_in_file = Path(prune_in_file)
    score_file = Path(score_file)
    valid_snp_file = Path(valid_snp_file)
    ensure_dir(valid_snp_file.parent)

    if not prune_in_file.exists() or not score_file.exists():
        valid_snp_file.write_text("", encoding="utf-8")
        return 0

    pruned_snps = set()
    with open(prune_in_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            snp = line.strip()
            if snp:
                pruned_snps.add(snp)

    if not pruned_snps:
        valid_snp_file.write_text("", encoding="utf-8")
        return 0

    overlap = set()
    with open(score_file, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip()
        header_parts = re.split(r"\s+", header)
        snp_idx = header_parts.index("SNP") if "SNP" in header_parts else 0

        for line in f:
            parts = re.split(r"\s+", line.strip())
            if len(parts) <= snp_idx:
                continue

            snp = parts[snp_idx]
            if snp in pruned_snps:
                overlap.add(snp)

    with open(valid_snp_file, "w", encoding="utf-8") as out:
        for snp in sorted(overlap):
            out.write(str(snp) + "\n")

    return len(overlap)


def count_lines(path):
    path = Path(path)
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


def run_fold_plink(
    plink,
    fold_dir,
    fold_output_dir,
    score_file,
    pvalue_file,
    range_file,
    train_bfile_name,
    test_bfile_name,
    p_window_size,
    p_slide_size,
    p_ld_threshold,
    clump_p1,
    clump_r2,
    clump_kb,
    number_of_pcs,
    force,
):
    fold_dir = Path(fold_dir)
    fold_output_dir = ensure_dir(fold_output_dir)

    prune_dir = ensure_dir(fold_output_dir / "prune")
    clump_dir = ensure_dir(fold_output_dir / "clump")
    clone_dir = ensure_dir(fold_output_dir / "genotype_clones")
    pca_dir = ensure_dir(fold_output_dir / "pca")
    score_dir = ensure_dir(fold_output_dir / "scores")
    metric_dir = ensure_dir(fold_output_dir / "metrics")

    log_file = fold_output_dir / "plink_commands.log"

    if force and log_file.exists():
        log_file.unlink()

    fold_number = natural_fold_number(fold_dir)

    train_bfile = fold_dir / train_bfile_name
    test_bfile = fold_dir / test_bfile_name

    train_missing = check_bfile(train_bfile)
    test_missing = check_bfile(test_bfile)

    metric = {
        "fold": fold_number,
        "fold_dir": str(fold_dir),
        "fold_output_dir": str(fold_output_dir),
        "train_bfile": str(train_bfile),
        "test_bfile": str(test_bfile),
        "train_bfile_missing": " | ".join(train_missing),
        "test_bfile_missing": " | ".join(test_missing),

        "prune_dir": str(prune_dir),
        "clump_dir": str(clump_dir),
        "genotype_clone_dir": str(clone_dir),
        "pca_dir": str(pca_dir),
        "score_dir": str(score_dir),

        "prune_success": False,
        "clump_success": False,
        "clone_train_success": False,
        "clone_test_success": False,
        "pca_train_success": False,
        "pca_test_success": False,
        "score_train_success": False,
        "score_test_success": False,

        "prune_in_snps": 0,
        "valid_clumped_snps": 0,
        "valid_snps_for_scoring": 0,
        "snp_selection_method": "",
        "clump_fallback_used": False,
        "clump_fallback_reason": "",
        "train_cloned_variants": "",
        "test_cloned_variants": "",
        "train_scored_profiles": 0,
        "test_scored_profiles": 0,
    }

    if train_missing or test_missing:
        return metric

    # -------------------------------------------------------------------------
    # 1. Pruning on shared training genotype, save output in GWAS-specific folder
    # -------------------------------------------------------------------------
    prune_prefix = prune_dir / "train_prune"

    cmd = [
        plink,
        "--bfile", str(train_bfile),
        "--indep-pairwise",
        str(p_window_size),
        str(p_slide_size),
        str(p_ld_threshold),
        "--out", str(prune_prefix),
    ]

    completed = run_command(cmd, log_file, print_output=False)
    metric["prune_success"] = completed.returncode == 0

    prune_in = Path(str(prune_prefix) + ".prune.in")
    metric["prune_in_file"] = str(prune_in)
    metric["prune_in_snps"] = count_lines(prune_in)

    if not prune_in.exists() or metric["prune_in_snps"] == 0:
        return metric

    # -------------------------------------------------------------------------
    # 2. Clumping using GWAS p-values, save clumped file in GWAS-specific folder
    # -------------------------------------------------------------------------
    clump_prefix = clump_dir / "train_clump"

    cmd = [
        plink,
        "--bfile", str(train_bfile),
        "--extract", str(prune_in),
        "--clump", str(score_file),
        "--clump-snp-field", "SNP",
        "--clump-field", "P",
        "--clump-p1", str(clump_p1),
        "--clump-r2", str(clump_r2),
        "--clump-kb", str(clump_kb),
        "--out", str(clump_prefix),
    ]

    completed = run_command(cmd, log_file, print_output=False)
    metric["clump_success"] = completed.returncode == 0

    clumped_file = Path(str(clump_prefix) + ".clumped")
    valid_snp_file = clump_dir / "train.valid.snp"

    metric["clumped_file"] = str(clumped_file)
    metric["valid_snp_file"] = str(valid_snp_file)
    metric["valid_clumped_snps"] = extract_valid_snps_from_clumped(clumped_file, valid_snp_file)
    metric["valid_snps_for_scoring"] = metric["valid_clumped_snps"]
    metric["snp_selection_method"] = "plink_clump"

    if metric["valid_snps_for_scoring"] == 0:
        print()
        print("[CLUMP FALLBACK]")
        print("PLINK produced zero clumped SNPs. Continuing with pruned genotype SNPs that overlap the GWAS score file.")
        print("[PRUNE FILE]", prune_in)
        print("[SCORE FILE]", score_file)
        print("[VALID SNP FILE]", valid_snp_file)

        metric["clump_fallback_used"] = True
        metric["clump_fallback_reason"] = "no_plink_clump_results"
        metric["snp_selection_method"] = "pruned_genotype_x_gwas_score_overlap"
        metric["valid_snps_for_scoring"] = extract_pruned_score_overlap_snps(
            prune_in_file=prune_in,
            score_file=score_file,
            valid_snp_file=valid_snp_file,
        )

        print("[FALLBACK VALID SNPS]", metric["valid_snps_for_scoring"])
        print()

    if metric["valid_snps_for_scoring"] == 0:
        print("[STOP FOLD] No SNPs available for scoring after clump fallback.")
        return metric

    # -------------------------------------------------------------------------
    # 3. Clone/extract clumped/pruned genotype into GWAS-specific folder
    #    Original train/test genotype files are never touched.
    # -------------------------------------------------------------------------
    train_clone_prefix = clone_dir / "train_data.clumped.pruned"
    test_clone_prefix = clone_dir / "test_data.clumped.pruned"

    cmd = [
        plink,
        "--bfile", str(train_bfile),
        "--extract", str(valid_snp_file),
        "--make-bed",
        "--out", str(train_clone_prefix),
    ]

    completed = run_command(cmd, log_file)
    metric["clone_train_success"] = completed.returncode == 0

    cmd = [
        plink,
        "--bfile", str(test_bfile),
        "--extract", str(valid_snp_file),
        "--make-bed",
        "--out", str(test_clone_prefix),
    ]

    completed = run_command(cmd, log_file)
    metric["clone_test_success"] = completed.returncode == 0

    train_bim = Path(str(train_clone_prefix) + ".bim")
    test_bim = Path(str(test_clone_prefix) + ".bim")

    metric["train_cloned_bfile"] = str(train_clone_prefix)
    metric["test_cloned_bfile"] = str(test_clone_prefix)
    metric["train_cloned_variants"] = count_lines(train_bim)
    metric["test_cloned_variants"] = count_lines(test_bim)

    if check_bfile(train_clone_prefix) or check_bfile(test_clone_prefix):
        return metric

    # -------------------------------------------------------------------------
    # 4. PCA on the same clumped/pruned variants for covariate models
    # -------------------------------------------------------------------------
    train_pca_prefix = pca_dir / "train_data"
    test_pca_prefix = pca_dir / "test_data"

    cmd = [
        plink,
        "--bfile", str(train_clone_prefix),
        "--pca", str(number_of_pcs),
        "--out", str(train_pca_prefix),
    ]

    completed = run_command(cmd, log_file, print_output=False)
    metric["pca_train_success"] = completed.returncode == 0

    cmd = [
        plink,
        "--bfile", str(test_clone_prefix),
        "--pca", str(number_of_pcs),
        "--out", str(test_pca_prefix),
    ]

    completed = run_command(cmd, log_file, print_output=False)
    metric["pca_test_success"] = completed.returncode == 0

    metric["number_of_pcs"] = number_of_pcs
    metric["train_pca_file"] = str(train_pca_prefix) + ".eigenvec"
    metric["test_pca_file"] = str(test_pca_prefix) + ".eigenvec"
    metric["train_cov_file"] = str(fold_dir / "train_data.cov")
    metric["test_cov_file"] = str(fold_dir / "test_data.cov")

    # -------------------------------------------------------------------------
    # 5. PLINK score for all p-value thresholds
    # -------------------------------------------------------------------------
    train_score_prefix = score_dir / "train_data"
    test_score_prefix = score_dir / "test_data"

    cmd = [
        plink,
        "--bfile", str(train_clone_prefix),
        "--score", str(score_file), "1", "2", "3", "header",
        "--q-score-range", str(range_file), str(pvalue_file),
        "--out", str(train_score_prefix),
    ]

    completed = run_command(cmd, log_file)
    metric["score_train_success"] = completed.returncode == 0

    cmd = [
        plink,
        "--bfile", str(test_clone_prefix),
        "--score", str(score_file), "1", "2", "3", "header",
        "--q-score-range", str(range_file), str(pvalue_file),
        "--out", str(test_score_prefix),
    ]

    completed = run_command(cmd, log_file)
    metric["score_test_success"] = completed.returncode == 0

    metric["train_scored_profiles"] = len(list(score_dir.glob("train_data.*.profile")))
    metric["test_scored_profiles"] = len(list(score_dir.glob("test_data.*.profile")))

    pd.DataFrame([metric]).to_csv(metric_dir / "fold_plink_metric.csv", index=False)

    return metric


# =============================================================================
# EVALUATION FOR BINARY PHENOTYPES
# =============================================================================

def read_fam_phenotype(bfile_prefix):
    fam_file = Path(str(bfile_prefix) + ".fam")

    if not fam_file.exists():
        raise FileNotFoundError(f"Missing FAM file: {fam_file}")

    fam = pd.read_csv(
        fam_file,
        sep=r"\s+",
        header=None,
        names=["FID", "IID", "PID", "MID", "SEX", "PHENO"],
        dtype={"FID": str, "IID": str},
    )

    fam["PHENO"] = pd.to_numeric(fam["PHENO"], errors="coerce")
    fam = fam.dropna(subset=["PHENO"]).copy()

    # PLINK missing phenotype codes
    fam = fam[~fam["PHENO"].isin([-9, 0])].copy()

    values = sorted(fam["PHENO"].dropna().unique().tolist())

    if values == [1, 2]:
        fam["PHENO_BINARY"] = fam["PHENO"].replace({1: 0, 2: 1})
    elif values == [0, 1]:
        fam["PHENO_BINARY"] = fam["PHENO"]
    elif len(values) == 2:
        fam["PHENO_BINARY"] = fam["PHENO"].replace({values[0]: 0, values[1]: 1})
    else:
        raise ValueError(f"Phenotype is not binary. Values found: {values}")

    return fam[["FID", "IID", "PHENO_BINARY"]]


def read_profile(profile_file):
    profile_file = Path(profile_file)

    if not profile_file.exists():
        return pd.DataFrame()

    df = pd.read_csv(profile_file, sep=r"\s+")

    if "FID" not in df.columns or "IID" not in df.columns:
        return pd.DataFrame()

    score_col = ""

    for candidate in ["SCORE", "SCORESUM", "PRS"]:
        if candidate in df.columns:
            score_col = candidate
            break

    if not score_col:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            score_col = numeric_cols[-1]

    if not score_col:
        return pd.DataFrame()

    df["FID"] = df["FID"].astype(str)
    df["IID"] = df["IID"].astype(str)
    df["SCORE"] = pd.to_numeric(df[score_col], errors="coerce")

    return df[["FID", "IID", "SCORE"]]


def auc_score(y, score):
    try:
        from sklearn.metrics import roc_auc_score
        valid = pd.DataFrame({"y": y, "score": score}).replace([np.inf, -np.inf], np.nan).dropna()

        if valid["y"].nunique() != 2:
            return np.nan

        return float(roc_auc_score(valid["y"], valid["score"]))
    except Exception:
        return np.nan


def score_correlation(y, score):
    valid = pd.DataFrame({"y": y, "score": score}).replace([np.inf, -np.inf], np.nan).dropna()

    if len(valid) < 3:
        return np.nan

    return float(valid["y"].corr(valid["score"]))


def read_pca_file(pca_file, number_of_pcs):
    pca_file = Path(pca_file)
    if not pca_file.exists():
        return pd.DataFrame(columns=["FID", "IID"])

    columns = ["FID", "IID"] + [f"PC{i}" for i in range(1, int(number_of_pcs) + 1)]
    pca = pd.read_csv(pca_file, sep=r"\s+", header=None)
    pca = pca.iloc[:, : len(columns)].copy()
    pca.columns = columns[: pca.shape[1]]
    pca["FID"] = pca["FID"].astype(str)
    pca["IID"] = pca["IID"].astype(str)

    for column in pca.columns:
        if column not in ["FID", "IID"]:
            pca[column] = pd.to_numeric(pca[column], errors="coerce")

    return pca


def read_covariate_file(cov_file):
    cov_file = Path(cov_file)
    if not cov_file.exists():
        return pd.DataFrame(columns=["FID", "IID"])

    cov = pd.read_csv(cov_file, sep=r"\s+")
    if "FID" not in cov.columns or "IID" not in cov.columns:
        return pd.DataFrame(columns=["FID", "IID"])

    cov["FID"] = cov["FID"].astype(str)
    cov["IID"] = cov["IID"].astype(str)

    keep = ["FID", "IID"]
    for column in cov.columns:
        if column in ["FID", "IID"]:
            continue
        numeric = pd.to_numeric(cov[column], errors="coerce")
        if numeric.notna().any():
            cov[column] = numeric
            keep.append(column)

    return cov[keep].copy()


def build_covariate_design(phenotype, cov_file, pca_file, number_of_pcs):
    design = phenotype.copy()
    cov = read_covariate_file(cov_file)
    pca = read_pca_file(pca_file, number_of_pcs)

    if not cov.empty and len(cov.columns) > 2:
        design = design.merge(cov, on=["FID", "IID"], how="left")

    if not pca.empty and len(pca.columns) > 2:
        design = design.merge(pca, on=["FID", "IID"], how="left")

    feature_cols = [c for c in design.columns if c not in ["FID", "IID", "PHENO_BINARY"]]
    for column in feature_cols:
        design[column] = pd.to_numeric(design[column], errors="coerce")

    design[feature_cols] = design[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return design, feature_cols


def logistic_auc_with_features(train_df, test_df, feature_cols):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        if train_df["PHENO_BINARY"].nunique() != 2 or test_df["PHENO_BINARY"].nunique() != 2:
            return np.nan, np.nan

        if not feature_cols:
            prevalence = float(train_df["PHENO_BINARY"].mean())
            train_pred = np.repeat(prevalence, len(train_df))
            test_pred = np.repeat(prevalence, len(test_df))
            return (
                auc_score(train_df["PHENO_BINARY"], train_pred),
                auc_score(test_df["PHENO_BINARY"], test_pred),
            )

        x_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        x_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        model = LogisticRegression(max_iter=2000, solver="liblinear")
        model.fit(x_train_scaled, train_df["PHENO_BINARY"])

        train_pred = model.predict_proba(x_train_scaled)[:, 1]
        test_pred = model.predict_proba(x_test_scaled)[:, 1]

        return (
            auc_score(train_df["PHENO_BINARY"], train_pred),
            auc_score(test_df["PHENO_BINARY"], test_pred),
        )
    except Exception as e:
        print("[MODEL WARNING]", type(e).__name__, e)
        return np.nan, np.nan


def evaluate_fold_profiles(fold_metric, labels):
    if not fold_metric.get("score_train_success", False):
        return pd.DataFrame()

    if not fold_metric.get("score_test_success", False):
        return pd.DataFrame()

    train_clone_bfile = fold_metric.get("train_cloned_bfile", "")
    test_clone_bfile = fold_metric.get("test_cloned_bfile", "")
    score_dir = Path(fold_metric.get("score_dir", ""))

    if not train_clone_bfile or not test_clone_bfile:
        return pd.DataFrame()

    train_pheno = read_fam_phenotype(train_clone_bfile)
    test_pheno = read_fam_phenotype(test_clone_bfile)
    number_of_pcs = int(fold_metric.get("number_of_pcs", 6) or 6)

    train_design, covariate_feature_cols = build_covariate_design(
        train_pheno,
        fold_metric.get("train_cov_file", ""),
        fold_metric.get("train_pca_file", ""),
        number_of_pcs,
    )
    test_design, _ = build_covariate_design(
        test_pheno,
        fold_metric.get("test_cov_file", ""),
        fold_metric.get("test_pca_file", ""),
        number_of_pcs,
    )

    for column in covariate_feature_cols:
        if column not in test_design.columns:
            test_design[column] = 0

    train_null_auc, test_null_auc = logistic_auc_with_features(
        train_design,
        test_design,
        covariate_feature_cols,
    )

    rows = []

    for label, pvalue in labels:
        train_profile = score_dir / f"train_data.{label}.profile"
        test_profile = score_dir / f"test_data.{label}.profile"

        train_score = read_profile(train_profile)
        test_score = read_profile(test_profile)

        if train_score.empty or test_score.empty:
            continue

        train = train_design.merge(train_score, on=["FID", "IID"], how="inner")
        test = test_design.merge(test_score, on=["FID", "IID"], how="inner")

        if train.empty or test.empty:
            continue

        best_feature_cols = covariate_feature_cols + ["SCORE"]
        train_best_auc, test_best_auc = logistic_auc_with_features(
            train,
            test,
            best_feature_cols,
        )

        rows.append({
            "fold": fold_metric.get("fold", ""),
            "pvalue_label": label,
            "pvalue": pvalue,
            "trait_type": "binary",
            "performance_metric": "AUC",

            "Train_pure_prs": auc_score(train["PHENO_BINARY"], train["SCORE"]),
            "Test_pure_prs": auc_score(test["PHENO_BINARY"], test["SCORE"]),
            "Train_null_model": train_null_auc,
            "Test_null_model": test_null_auc,
            "Train_best_model": train_best_auc,
            "Test_best_model": test_best_auc,

            "Train_score_pheno_correlation": score_correlation(train["PHENO_BINARY"], train["SCORE"]),
            "Test_score_pheno_correlation": score_correlation(test["PHENO_BINARY"], test["SCORE"]),

            "train_scored_samples": len(train),
            "test_scored_samples": len(test),

            "train_cases": int(train["PHENO_BINARY"].sum()),
            "train_controls": int((train["PHENO_BINARY"] == 0).sum()),
            "test_cases": int(test["PHENO_BINARY"].sum()),
            "test_controls": int((test["PHENO_BINARY"] == 0).sum()),

            "valid_clumped_snps": fold_metric.get("valid_clumped_snps", ""),
            "valid_snps_for_scoring": fold_metric.get("valid_snps_for_scoring", ""),
            "snp_selection_method": fold_metric.get("snp_selection_method", ""),
            "number_of_pcs": number_of_pcs,
            "covariate_count": len(covariate_feature_cols),
            "train_cloned_variants": fold_metric.get("train_cloned_variants", ""),
            "test_cloned_variants": fold_metric.get("test_cloned_variants", ""),

            "train_profile": str(train_profile),
            "test_profile": str(test_profile),
        })

    return pd.DataFrame(rows)


def summarise_results(all_results, prs_dir):
    prs_dir = Path(prs_dir)

    if not all_results:
        empty = pd.DataFrame()
        empty.to_csv(prs_dir / "PRS_all_folds_results.csv", index=False)
        empty.to_csv(prs_dir / "PRS_average_performance.csv", index=False)
        empty.to_csv(prs_dir / "PRS_best_result.csv", index=False)
        return empty, empty, empty

    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_csv(prs_dir / "PRS_all_folds_results.csv", index=False)

    group_cols = ["pvalue_label", "pvalue", "trait_type", "performance_metric"]

    avg = (
        all_df
        .groupby(group_cols, dropna=False)
        .agg(
            n_folds=("fold", "nunique"),

            Train_pure_prs_mean=("Train_pure_prs", "mean"),
            Train_pure_prs_sd=("Train_pure_prs", "std"),
            Test_pure_prs_mean=("Test_pure_prs", "mean"),
            Test_pure_prs_sd=("Test_pure_prs", "std"),
            Train_null_model_mean=("Train_null_model", "mean"),
            Train_null_model_sd=("Train_null_model", "std"),
            Test_null_model_mean=("Test_null_model", "mean"),
            Test_null_model_sd=("Test_null_model", "std"),
            Train_best_model_mean=("Train_best_model", "mean"),
            Train_best_model_sd=("Train_best_model", "std"),
            Test_best_model_mean=("Test_best_model", "mean"),
            Test_best_model_sd=("Test_best_model", "std"),

            Train_score_pheno_correlation_mean=("Train_score_pheno_correlation", "mean"),
            Test_score_pheno_correlation_mean=("Test_score_pheno_correlation", "mean"),

            train_scored_samples_mean=("train_scored_samples", "mean"),
            test_scored_samples_mean=("test_scored_samples", "mean"),

            train_cases_mean=("train_cases", "mean"),
            train_controls_mean=("train_controls", "mean"),
            test_cases_mean=("test_cases", "mean"),
            test_controls_mean=("test_controls", "mean"),

            valid_clumped_snps_mean=("valid_clumped_snps", "mean"),
            valid_clumped_snps_sd=("valid_clumped_snps", "std"),
            valid_snps_for_scoring_mean=("valid_snps_for_scoring", "mean"),
            covariate_count_mean=("covariate_count", "mean"),
            train_cloned_variants_mean=("train_cloned_variants", "mean"),
            test_cloned_variants_mean=("test_cloned_variants", "mean"),
        )
        .reset_index()
    )

    avg["generalisation_gap"] = (avg["Train_pure_prs_mean"] - avg["Test_pure_prs_mean"]).abs()
    avg["train_test_sum"] = avg["Train_pure_prs_mean"] + avg["Test_pure_prs_mean"]
    avg["best_model_generalisation_gap"] = (avg["Train_best_model_mean"] - avg["Test_best_model_mean"]).abs()
    avg["best_model_train_test_sum"] = avg["Train_best_model_mean"] + avg["Test_best_model_mean"]
    avg["test_incremental_auc_best_minus_null"] = avg["Test_best_model_mean"] - avg["Test_null_model_mean"]
    avg["test_incremental_auc_best_minus_pure_prs"] = avg["Test_best_model_mean"] - avg["Test_pure_prs_mean"]

    avg.to_csv(prs_dir / "PRS_average_performance.csv", index=False)

    if not avg.empty:
        best = avg.sort_values(
            by=["Test_best_model_mean", "best_model_generalisation_gap"],
            ascending=[False, True],
        ).head(1)
    else:
        best = pd.DataFrame()

    best.to_csv(prs_dir / "PRS_best_result.csv", index=False)

    return all_df, avg, best


def cleanup_prs_outputs(prs_dir, keep_names):
    prs_dir = Path(prs_dir)
    keep_names = set(keep_names)

    if not prs_dir.exists():
        return []

    removed = []

    for item in prs_dir.iterdir():
        if item.name in keep_names:
            continue

        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            removed.append(str(item))
        except Exception as e:
            print(f"[WARN] Could not remove {item}: {e}")

    return removed


# =============================================================================
# MAIN
# =============================================================================

def process_one_job(index, args):
    job = load_job(index, args.jobs_file)

    file_path = Path(clean_text(job["file_path"]))
    # Resolve relative job paths under data_root when provided
    if args and getattr(args, "data_root", ""):
        data_root = Path(getattr(args, "data_root"))
        if not file_path.is_absolute():
            file_path = data_root / file_path
    phenotype_dir, phenotype = infer_phenotype_dir(job, file_path)
    # Prefer genotype/phenotype data under the provided data_root when present
    if args and getattr(args, "data_root", ""):
        data_root = Path(getattr(args, "data_root"))
        candidate = data_root / phenotype
        if candidate.exists() and candidate.is_dir():
            print(f"[INFO] Using phenotype directory from data_root: {candidate}")
            phenotype_dir = candidate
    accession = clean_text(job.get("accessionId", ""))

    gwas_output_dir = get_gwas_output_dir(job, file_path)
    processed_gwas = gwas_output_dir / "ProcessedGWAS_hg38.csv"

    # If a relative --processed-gwas was supplied, resolve under data_root
    if args and getattr(args, "data_root", "") and args.processed_gwas:
        proc = Path(args.processed_gwas)
        if not proc.is_absolute():
            args.processed_gwas = str(Path(getattr(args, "data_root")) / proc)

    if args.processed_gwas:
        processed_gwas = Path(args.processed_gwas)

    if not processed_gwas.exists():
        raise FileNotFoundError(
            f"Missing converted GWAS file: {processed_gwas}\n"
            f"Run first:\n"
            f"  python PRS1-Normalization1.py {index} --force\n"
            f"  python PRS1-Normalization2-hg38.py {index} --force\n"
        )

    processed_preview = pd.read_csv(processed_gwas, nrows=5)
    processed_total_rows = sum(1 for _ in open(processed_gwas, "r", encoding="utf-8", errors="replace")) - 1
    print("=" * 100)
    print("[PROCESSED GWAS PREFLIGHT]")
    print(f"[FILE]    {processed_gwas}")
    print(f"[ROWS]    {max(processed_total_rows, 0)}")
    print(f"[COLUMNS] {', '.join(map(str, processed_preview.columns))}")
    print("=" * 100)

    if processed_total_rows <= 0:
        raise ValueError(
            f"ProcessedGWAS_hg38.csv is empty: {processed_gwas}. "
            "This is why PRS2 has zero SNPs and every clump fallback is zero. "
            "Rerun/fix PRS1 normalization for this GWAS before PRS2."
        )

    plink = find_plink()

    prs_dir = ensure_dir(gwas_output_dir / "PRS" / "Plink_HG38")

    print("=" * 100)
    print("PRS2: PLINK PRS CALCULATION")
    print("=" * 100)
    print(f"[JOB INDEX]       {index}")
    print(f"[PHENOTYPE]       {phenotype}")
    print(f"[ACCESSION]       {accession}")
    print(f"[PHENOTYPE DIR]   {phenotype_dir}")
    print(f"[GWAS OUTPUT DIR] {gwas_output_dir}")
    print(f"[PROCESSED GWAS]  {processed_gwas}")
    print(f"[PRS OUTPUT DIR]  {prs_dir}")
    print(f"[PLINK]           {plink}")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # A. Use the already converted hg38 GWAS file
    # -------------------------------------------------------------------------
    hg38_gwas_file = processed_gwas

    liftover_metric = {
        "liftover_requested": False,
        "liftover_success": True,
        "liftover_method": "external_PRS1_Normalization2_hg38",
        "liftover_issue": "",
        "hg38_gwas_file": str(hg38_gwas_file),
    }

    # -------------------------------------------------------------------------
    # B. Prepare PLINK score file and p-value file
    # -------------------------------------------------------------------------
    score_file, pvalue_file, gwas_metric = prepare_gwas_for_plink(
        hg38_gwas_file=hg38_gwas_file,
        prs_dir=prs_dir,
    )

    # -------------------------------------------------------------------------
    # C. Create p-value threshold range file
    # -------------------------------------------------------------------------
    range_file, labels = create_range_list(
        prs_dir=prs_dir,
        min_p_exponent=args.min_p_exponent,
        n_intervals=args.n_pvalue_intervals,
    )

    # -------------------------------------------------------------------------
    # D. Find folds
    # -------------------------------------------------------------------------
    folds = find_folds(phenotype_dir, args.folds)

    if not folds:
        raise FileNotFoundError(f"No Fold_* directories found in: {phenotype_dir}")

    all_results = []
    fold_metrics = []

    for fold_dir in folds:
        fold_number = natural_fold_number(fold_dir)
        fold_output_dir = prs_dir / f"Fold_{fold_number}"

        print("=" * 100)
        print(f"PROCESSING Fold_{fold_number}")
        print("=" * 100)

        fold_metric = run_fold_plink(
            plink=plink,
            fold_dir=fold_dir,
            fold_output_dir=fold_output_dir,
            score_file=score_file,
            pvalue_file=pvalue_file,
            range_file=range_file,
            train_bfile_name=args.train_bfile,
            test_bfile_name=args.test_bfile,
            p_window_size=args.p_window_size,
            p_slide_size=args.p_slide_size,
            p_ld_threshold=args.p_ld_threshold,
            clump_p1=args.clump_p1,
            clump_r2=args.clump_r2,
            clump_kb=args.clump_kb,
            number_of_pcs=args.number_of_pcs,
            force=args.force,
        )

        result = evaluate_fold_profiles(fold_metric, labels)

        if not result.empty:
            result.insert(0, "job_index", index)
            result.insert(1, "phenotype", phenotype)
            result.insert(2, "accessionId", accession)

        result.to_csv(fold_output_dir / "Results.csv", index=False)

        if not result.empty:
            all_results.append(result)

        fold_metrics.append(fold_metric)

    all_df, avg_df, best_df = summarise_results(all_results, prs_dir)

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(prs_dir / "PRS_fold_metrics.csv", index=False)

    summary = {
        "job_index": index,
        "phenotype": phenotype,
        "accessionId": accession,
        "file_path": str(file_path),
        "processed_gwas": str(processed_gwas),
        "hg38_gwas_file": str(hg38_gwas_file),
        "phenotype_dir": str(phenotype_dir),
        "gwas_output_dir": str(gwas_output_dir),
        "prs_dir": str(prs_dir),
        "plink": str(plink),

        "n_folds_found": len(folds),
        "n_folds_with_results": int(all_df["fold"].nunique()) if not all_df.empty else 0,

        "range_file": str(range_file),
        "score_file": str(score_file),
        "pvalue_file": str(pvalue_file),

        "all_folds_results_file": str(prs_dir / "PRS_all_folds_results.csv"),
        "average_performance_file": str(prs_dir / "PRS_average_performance.csv"),
        "best_result_file": str(prs_dir / "PRS_best_result.csv"),
        "fold_metrics_file": str(prs_dir / "PRS_fold_metrics.csv"),
    }

    summary.update(liftover_metric)
    summary.update(gwas_metric)

    pd.DataFrame([summary]).to_csv(prs_dir / "PRS_run_summary.csv", index=False)

    if args.cleanup_intermediate:
        keep_names = {
            "PRS_run_summary.csv",
            "PRS_fold_metrics.csv",
            "PRS_all_folds_results.csv",
            "PRS_average_performance.csv",
            "PRS_best_result.csv",
            Path(score_file).name,
        }
        removed_items = cleanup_prs_outputs(prs_dir, keep_names)
        print("=" * 100)
        print("[CLEANUP]")
        print(f"Kept: {', '.join(sorted(keep_names))}")
        print(f"Removed items: {len(removed_items)}")
        print("=" * 100)

    print("=" * 100)
    print("[FINAL OUTPUTS]")
    print(prs_dir / "PRS_run_summary.csv")
    print(prs_dir / "PRS_fold_metrics.csv")
    print(prs_dir / "PRS_all_folds_results.csv")
    print(prs_dir / "PRS_average_performance.csv")
    print(prs_dir / "PRS_best_result.csv")
    print("=" * 100)

    if not best_df.empty:
        print()
        print("[BEST RESULT]")
        print(best_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="PRS2: Calculate PLINK PRS for one normalized GWAS job without overwriting shared genotype data."
    )

    parser.add_argument(
        "index",
        type=int,
        help="GWAS job index. Example: python PRS2-Calculation1.py 1",
    )

    parser.add_argument(
        "--jobs-file",
        default=DEFAULT_JOBS_FILE,
        help="Default: GWAS_jobs.csv",
    )

    parser.add_argument(
        "--processed-gwas",
        default="",
        help="Optional direct path to converted ProcessedGWAS_hg38.csv.",
    )

    parser.add_argument(
        "--folds",
        default="auto",
        help="auto or comma-separated fold numbers, e.g. 1,2,3,4,5",
    )

    parser.add_argument(
        "--train-bfile",
        default=TRAIN_BFILE_NAME,
        help="Train bfile prefix inside each Fold_* directory. Default: train_data.QC",
    )

    parser.add_argument(
        "--test-bfile",
        default=TEST_BFILE_NAME,
        help="Test bfile prefix inside each Fold_* directory. Default: test_data",
    )

    parser.add_argument("--p-window-size", type=int, default=DEFAULT_P_WINDOW_SIZE)
    parser.add_argument("--p-slide-size", type=int, default=DEFAULT_P_SLIDE_SIZE)
    parser.add_argument("--p-ld-threshold", type=float, default=DEFAULT_P_LD_THRESHOLD)

    parser.add_argument("--clump-p1", type=float, default=DEFAULT_CLUMP_P1)
    parser.add_argument("--clump-r2", type=float, default=DEFAULT_CLUMP_R2)
    parser.add_argument("--clump-kb", type=int, default=DEFAULT_CLUMP_KB)
    parser.add_argument(
        "--number-of-pcs",
        type=int,
        default=6,
        help="Number of principal components to calculate and include with covariates. Default: 6",
    )

    parser.add_argument("--min-p-exponent", type=int, default=DEFAULT_MIN_P_EXPONENT)
    parser.add_argument("--n-pvalue-intervals", type=int, default=DEFAULT_N_PVALUE_INTERVALS)

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite/rerun outputs.",
    )

    parser.add_argument(
        "--cleanup-intermediate",
        action="store_true",
        help="Remove intermediate PLINK files and per-fold working directories after writing the result CSVs.",
    )

    parser.add_argument(
        "--data-root",
        default="/data/ascher01/uqmmune1/finalheritability",
        help="Root folder to resolve relative GWAS job paths. Default: /data/ascher01/uqmmune1/finalheritability",
    )

    args = parser.parse_args()

    process_one_job(args.index, args)


if __name__ == "__main__":
    main()
