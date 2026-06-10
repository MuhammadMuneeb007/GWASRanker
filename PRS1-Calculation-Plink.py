#!/usr/bin/env python3

"""
PRS2-Calculation1-FinalGWAS.py

PRS calculation using PLINK for one GWAS job.

This version uses FinalGWAS.csv directly.

Run:
    python PRS2-Calculation1-FinalGWAS.py 1 --force

Assumptions:
    1. Each GWAS directory contains FinalGWAS.csv.
    2. Shared genotype fold data are in phenotype/Fold_*/.
    3. PLINK is available as ./plink, ./plink.exe, or in PATH.
    4. Phenotypes are binary.
    5. FinalGWAS.csv contains standardised columns such as:
       SNPID, EA, NEA, BETA, OR, P
       or compatible alternatives:
       SNP, A1, A2, BETA, OR, P

Main outputs are saved per GWAS:

    <GWAS_DIR>/PRS/PRS_Plink/
        GWAS_for_plink.tsv
        SNP.pvalue
        range_list
        PRS_run_summary.csv
        PRS_fold_metrics.csv
        PRS_all_folds_results.csv
        PRS_average_performance.csv
        PRS_best_result.csv
        PRS_train_selected_test_results.csv

Per fold:

    <GWAS_DIR>/PRS/PRS_Plink/Fold_1/
        prune/
        clump/
        genotype_clones/
        pca/
        scores/
        metrics/
        Results.csv

Important:
    The original genotype files in phenotype/Fold_* are never overwritten.
    All PLINK --out files go inside the GWAS-specific PRS/PRS_Plink directory.
"""

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
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

DEFAULT_PRS_TOOL_NAME = "PRS_Plink"


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


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def last_meaningful_line(text):
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line:
            return line
    return ""


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def is_finite_numeric(series):
    values = pd.to_numeric(series, errors="coerce")
    return values.notna() & np.isfinite(values)


def normalise_allele(x):
    x = clean_text(x).upper()
    if x in ["", "NAN", "NA", "N/A", "NONE", "NULL", "."]:
        return ""
    return x


def normalise_identifier(x):
    x = clean_text(x)
    if x.upper() in ["", "NAN", "NA", "N/A", "NONE", "NULL", ".", "<NA>"]:
        return ""
    return x


def deduplicate_best_p_rows(df, snp_col="SNP", p_col="P"):
    """
    Keep one row per SNP using the smallest P-value.

    This avoids pandas sort/drop_duplicates edge cases on very large frames by
    using groupby/idxmin and NumPy ordering instead of chaining sort_values().
    """
    if df.empty:
        return df.copy(), 0

    working = df.reset_index(drop=True).copy()
    working[p_col] = pd.to_numeric(working[p_col], errors="coerce")
    working = working.loc[
        working[snp_col].map(normalise_identifier).ne("")
        & working[p_col].notna()
        & np.isfinite(working[p_col])
    ].copy()

    if working.empty:
        return working, 0

    best_idx = working.groupby(snp_col, sort=False)[p_col].idxmin()
    working = working.loc[best_idx].copy()

    order = np.argsort(working[p_col].to_numpy(dtype=np.float64, copy=False), kind="stable")
    working = working.iloc[order].reset_index(drop=True)

    duplicate_removed = len(df) - len(working)
    return working, duplicate_removed


def find_first_column(columns, candidates):
    """
    Case-insensitive column finder.
    """
    exact = {str(c): c for c in columns}
    lower = {str(c).lower(): c for c in columns}

    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    return ""


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


def find_plink(plink_arg=""):
    if plink_arg:
        p = Path(plink_arg)
        if p.exists():
            return str(p)
        return plink_arg

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
        "PLINK not found. Put plink in the current working directory, "
        "pass --plink /path/to/plink, or add it to PATH."
    )


def check_bfile(prefix):
    prefix = Path(prefix)
    missing = []
    for suffix in [".bed", ".bim", ".fam"]:
        path = Path(str(prefix) + suffix)
        if not path.exists():
            missing.append(str(path))
    return missing


def count_lines(path):
    path = Path(path)
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


def natural_fold_number(path):
    m = re.search(r"Fold_(\d+)", str(path))
    if m:
        return int(m.group(1))
    return 10**9


# =============================================================================
# JOB AND FILE RESOLUTION
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

    jobs["job_index"] = pd.to_numeric(jobs["job_index"], errors="coerce").astype("Int64")
    sub = jobs[jobs["job_index"] == int(index)]

    if sub.empty:
        raise ValueError(f"No job found for index: {index}")

    return sub.iloc[0]


def resolve_path(path_text, base_dir=""):
    path = Path(clean_text(path_text))
    if path.is_absolute():
        return path

    if base_dir:
        candidate = Path(base_dir) / path
        if candidate.exists():
            return candidate

    return path


def infer_phenotype_dir(job, file_path, data_root=""):
    phenotype = clean_text(job.get("phenotype", ""))

    candidates = []

    if data_root and phenotype:
        candidates.append(Path(data_root) / phenotype)

    if phenotype:
        candidates.append(Path(phenotype))

    parts = Path(file_path).parts
    if parts:
        if data_root:
            candidates.append(Path(data_root) / parts[0])
        candidates.append(Path(parts[0]))

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate, candidate.name

    if phenotype:
        return Path(data_root) / phenotype if data_root else Path(phenotype), phenotype

    raise ValueError(
        "Could not infer phenotype directory. Add phenotype column to GWAS_jobs.csv."
    )


def get_gwas_output_dir(job, file_path, data_root=""):
    """
    Prefer output_feature_file parent because your feature scripts store:
        <GWAS_DIR>/FinalGWAS.csv
        <GWAS_DIR>/Feature*.csv
    """
    if "output_feature_file" in job.index:
        out = clean_text(job.get("output_feature_file", ""))
        if out:
            out_path = Path(out)
            if not out_path.is_absolute() and data_root:
                candidate = Path(data_root) / out_path
                if candidate.exists():
                    out_path = candidate
            return out_path.parent

    return Path(file_path).parent


def find_final_gwas(gwas_output_dir, explicit_final_gwas=""):
    if explicit_final_gwas:
        p = Path(explicit_final_gwas)
        if p.exists():
            return p
        raise FileNotFoundError(f"Explicit --final-gwas does not exist: {p}")

    candidates = [
        Path(gwas_output_dir) / "FinalGWAS.csv",
        Path(gwas_output_dir) / "finalgwas.csv",
        Path(gwas_output_dir) / "FinalGWAS.tsv",
        Path(gwas_output_dir) / "finalgwas.tsv",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find FinalGWAS.csv.\n"
        f"Expected location: {Path(gwas_output_dir) / 'FinalGWAS.csv'}"
    )


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
# FINALGWAS TO PLINK SCORE INPUT
# =============================================================================

def prepare_finalgwas_for_plink(final_gwas_file, prs_dir):
    """
    Prepare PLINK score and p-value files from FinalGWAS.csv.

    Accepted FinalGWAS column names:
        SNP column:
            SNPID, SNP, rsID, ID, MarkerName, variant_id
        Effect allele:
            EA, A1, effect_allele, effectAllele, ALT
        Non-effect allele:
            NEA, A2, other_allele, non_effect_allele, REF
        Effect:
            BETA, beta, effect, Effect, OR, odds_ratio
        P-value:
            P, p, PVAL, pvalue, p_value

    Output PLINK score file:
        SNP A1 BETA P

    PLINK command uses:
        --score GWAS_for_plink.tsv 1 2 3 header
        --q-score-range range_list SNP.pvalue
    """

    prs_dir = ensure_dir(prs_dir)
    final_gwas_file = Path(final_gwas_file)

    if final_gwas_file.suffix.lower() == ".tsv":
        df = pd.read_csv(final_gwas_file, sep="\t", low_memory=False)
    else:
        df = pd.read_csv(final_gwas_file, low_memory=False)

    df.columns = [str(c).strip() for c in df.columns]

    print()
    print("=" * 100)
    print("[FINALGWAS INPUT CHECK]")
    print("[FILE]", final_gwas_file)
    print("[ROWS]", len(df))
    print("[COLUMNS]", ", ".join(map(str, df.columns)))
    print("=" * 100)

    snp_col = find_first_column(df.columns, [
        "SNPID", "SNP", "rsID", "RSID", "ID", "MarkerName",
        "MARKERNAME", "variant_id", "VARIANT_ID"
    ])
    a1_col = find_first_column(df.columns, [
        "EA", "A1", "effect_allele", "effectAllele", "EFFECT_ALLELE",
        "ALT", "Allele1", "ALLELE1"
    ])
    p_col = find_first_column(df.columns, [
        "P", "p", "PVAL", "P_VALUE", "p_value", "pvalue", "PVALUE"
    ])
    beta_col = find_first_column(df.columns, [
        "BETA", "beta", "Effect", "EFFECT", "effect", "LOG_OR", "logOR"
    ])
    or_col = find_first_column(df.columns, [
        "OR", "odds_ratio", "OddsRatio", "ODDS_RATIO"
    ])

    missing = []
    if not snp_col:
        missing.append("SNPID/SNP/rsID")
    if not a1_col:
        missing.append("EA/A1/effect_allele")
    if not p_col:
        missing.append("P/PVAL/pvalue")
    if not beta_col and not or_col:
        missing.append("BETA or OR")

    if missing:
        raise ValueError(
            "FinalGWAS.csv is missing required PRS columns.\n"
            f"Missing: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["SNP"] = df[snp_col].map(normalise_identifier)
    out["A1"] = df[a1_col].map(normalise_allele)
    out["P"] = safe_numeric(df[p_col])

    if beta_col:
        out["BETA"] = safe_numeric(df[beta_col])
        effect_source = f"BETA:{beta_col}"
    else:
        or_values = safe_numeric(df[or_col])
        out["BETA"] = np.where(or_values > 0, np.log(or_values), np.nan)
        effect_source = f"OR_to_BETA:{or_col}"

    pre_rows = len(out)

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

    diagnostic = {
        "finalgwas_input_rows": pre_rows,
        "mapped_snp_col": snp_col,
        "mapped_a1_col": a1_col,
        "mapped_p_col": p_col,
        "mapped_beta_col": beta_col,
        "mapped_or_col": or_col,
        "effect_source": effect_source,
        "nonmissing_snp": int(out["SNP"].map(normalise_identifier).ne("").sum()),
        "valid_a1_single_acgt": int(out["A1"].str.match(r"^[ACGT]$", na=False).sum()),
        "valid_p_0_1": int(((out["P"] > 0) & (out["P"] <= 1)).sum()),
        "valid_beta": int(out["BETA"].notna().sum()),
    }

    print("[FINALGWAS TO PLINK DIAGNOSTICS]")
    for key, value in diagnostic.items():
        print(f"{key}: {value}")

    out = out.loc[valid, ["SNP", "A1", "BETA", "P"]].copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["SNP", "A1", "BETA", "P"])

    before_dedup = len(out)
    out, duplicate_removed = deduplicate_best_p_rows(out, snp_col="SNP", p_col="P")

    score_file = prs_dir / "GWAS_for_plink.tsv"
    pvalue_file = prs_dir / "SNP.pvalue"

    out[["SNP", "A1", "BETA", "P"]].to_csv(score_file, sep="\t", index=False)
    out[["SNP", "P"]].to_csv(pvalue_file, sep="\t", index=False, header=False)

    metric = {
        "gwas_input_source": "FinalGWAS.csv",
        "final_gwas_file": str(final_gwas_file),
        "finalgwas_input_rows": pre_rows,
        "gwas_for_plink_valid_rows": len(out),
        "gwas_for_plink_duplicate_snp_removed": duplicate_removed,
        "gwas_for_plink_retained_pct": percent(len(out), pre_rows),
        "score_file": str(score_file),
        "pvalue_file": str(pvalue_file),
        "mapped_snp_col": snp_col,
        "mapped_a1_col": a1_col,
        "mapped_p_col": p_col,
        "mapped_beta_col": beta_col,
        "mapped_or_col": or_col,
        "effect_source": effect_source,
        "used_beta_directly": bool(beta_col),
        "derived_beta_from_or": bool((not beta_col) and or_col),
    }

    print("[PLINK SCORE OUTPUT CHECK]")
    print("[VALID ROWS]", len(out))
    print("[DUPLICATE SNP REMOVED]", duplicate_removed)
    print("[SCORE FILE]", score_file)
    print("[PVALUE FILE]", pvalue_file)
    print("=" * 100)

    if len(out) == 0:
        raise ValueError(
            "No valid SNPs were available for PLINK scoring from FinalGWAS.csv.\n"
            f"Input rows={pre_rows}; "
            f"nonmissing SNP={diagnostic['nonmissing_snp']}; "
            f"valid A1={diagnostic['valid_a1_single_acgt']}; "
            f"valid P={diagnostic['valid_p_0_1']}; "
            f"valid BETA={diagnostic['valid_beta']}.\n"
            "Check FinalGWAS.csv columns and allele/effect-size values."
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
        "fold_status": "not_started",
        "fold_failure_reason": "",
    }

    if train_missing or test_missing:
        print("[STOP FOLD] Missing train/test bfile components.")
        metric["fold_status"] = "failed"
        metric["fold_failure_reason"] = "missing train/test genotype bfile components"
        return metric

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
    if completed.returncode != 0:
        metric["fold_status"] = "failed"
        metric["fold_failure_reason"] = last_meaningful_line(completed.stderr) or "pruning failed"
        return metric

    prune_in = Path(str(prune_prefix) + ".prune.in")
    metric["prune_in_file"] = str(prune_in)
    metric["prune_in_snps"] = count_lines(prune_in)

    if not prune_in.exists() or metric["prune_in_snps"] == 0:
        print("[STOP FOLD] No prune.in SNPs.")
        metric["fold_status"] = "failed"
        metric["fold_failure_reason"] = "no variants left after pruning"
        return metric

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
        print("PLINK produced zero clumped SNPs. Using pruned genotype SNPs overlapping the GWAS score file.")
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
        metric["fold_status"] = "failed"
        metric["fold_failure_reason"] = "no variants left after clumping or overlap filtering"
        return metric

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
    if completed.returncode != 0:
        metric["fold_status"] = "failed"
        metric["fold_failure_reason"] = last_meaningful_line(completed.stderr) or "failed to create train clumped/pruned genotype files"
        return metric

    cmd = [
        plink,
        "--bfile", str(test_bfile),
        "--extract", str(valid_snp_file),
        "--make-bed",
        "--out", str(test_clone_prefix),
    ]
    completed = run_command(cmd, log_file)
    metric["clone_test_success"] = completed.returncode == 0
    if completed.returncode != 0:
        metric["fold_status"] = "failed"
        metric["fold_failure_reason"] = last_meaningful_line(completed.stderr) or "failed to create test clumped/pruned genotype files"
        return metric

    train_bim = Path(str(train_clone_prefix) + ".bim")
    test_bim = Path(str(test_clone_prefix) + ".bim")

    metric["train_cloned_bfile"] = str(train_clone_prefix)
    metric["test_cloned_bfile"] = str(test_clone_prefix)
    metric["train_cloned_variants"] = count_lines(train_bim)
    metric["test_cloned_variants"] = count_lines(test_bim)

    if check_bfile(train_clone_prefix) or check_bfile(test_clone_prefix):
        print("[STOP FOLD] Clone bfiles missing.")
        metric["fold_status"] = "failed"
        metric["fold_failure_reason"] = "clumped/pruned genotype files missing after clone step"
        return metric

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
    if completed.returncode != 0:
        metric["fold_status"] = "failed"
        metric["fold_failure_reason"] = last_meaningful_line(completed.stderr) or "PLINK score failed on train fold"
        return metric

    cmd = [
        plink,
        "--bfile", str(test_clone_prefix),
        "--score", str(score_file), "1", "2", "3", "header",
        "--q-score-range", str(range_file), str(pvalue_file),
        "--out", str(test_score_prefix),
    ]
    completed = run_command(cmd, log_file)
    metric["score_test_success"] = completed.returncode == 0
    if completed.returncode != 0:
        metric["fold_status"] = "failed"
        metric["fold_failure_reason"] = last_meaningful_line(completed.stderr) or "PLINK score failed on test fold"
        return metric

    metric["train_scored_profiles"] = len(list(score_dir.glob("train_data.*.profile")))
    metric["test_scored_profiles"] = len(list(score_dir.glob("test_data.*.profile")))
    if metric["train_scored_profiles"] <= 0 or metric["test_scored_profiles"] <= 0:
        metric["fold_status"] = "failed"
        metric["fold_failure_reason"] = "PLINK score profiles were not created"
        return metric

    metric["fold_status"] = "success"
    metric["fold_failure_reason"] = ""

    pd.DataFrame([metric]).to_csv(metric_dir / "fold_plink_metric.csv", index=False)

    return metric


# =============================================================================
# EVALUATION
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

    # Do not remove 0 automatically, because some pipelines use 0/1 coding.
    fam = fam[~fam["PHENO"].isin([-9])].copy()

    values = sorted(fam["PHENO"].dropna().unique().tolist())

    if values == [1, 2]:
        fam["PHENO_TARGET"] = fam["PHENO"].replace({1: 0, 2: 1})
        trait_type = "binary"
    elif values == [0, 1]:
        fam["PHENO_TARGET"] = fam["PHENO"]
        trait_type = "binary"
    elif len(values) == 2:
        fam["PHENO_TARGET"] = fam["PHENO"].replace({values[0]: 0, values[1]: 1})
        trait_type = "binary"
    else:
        fam["PHENO_TARGET"] = fam["PHENO"]
        trait_type = "continuous"

    return fam[["FID", "IID", "PHENO_TARGET"]], trait_type


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
    df["SCORE"] = safe_numeric(df[score_col])

    return df[["FID", "IID", "SCORE"]]


def auc_score(y, score):
    try:
        from sklearn.metrics import roc_auc_score

        valid = pd.DataFrame({"y": y, "score": score})
        valid = valid.replace([np.inf, -np.inf], np.nan).dropna()

        if valid["y"].nunique() != 2:
            return np.nan

        return float(roc_auc_score(valid["y"], valid["score"]))
    except Exception:
        return np.nan


def r2_from_predictions(y, pred):
    try:
        from sklearn.metrics import r2_score

        valid = pd.DataFrame({"y": y, "pred": pred})
        valid = valid.replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < 2:
            return np.nan
        return float(r2_score(valid["y"], valid["pred"]))
    except Exception:
        return np.nan


def score_correlation(y, score):
    valid = pd.DataFrame({"y": y, "score": score})
    valid = valid.replace([np.inf, -np.inf], np.nan).dropna()

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
            pca[column] = safe_numeric(pca[column])

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
        numeric = safe_numeric(cov[column])
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

    feature_cols = [c for c in design.columns if c not in ["FID", "IID", "PHENO_TARGET"]]

    for column in feature_cols:
        design[column] = safe_numeric(design[column])

    if feature_cols:
        design[feature_cols] = design[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    return design, feature_cols


def model_performance_with_features(train_df, test_df, feature_cols, trait_type):
    try:
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.preprocessing import StandardScaler

        if not feature_cols:
            if trait_type == "binary":
                if train_df["PHENO_TARGET"].nunique() != 2 or test_df["PHENO_TARGET"].nunique() != 2:
                    return np.nan, np.nan
                prevalence = float(train_df["PHENO_TARGET"].mean())
                train_pred = np.repeat(prevalence, len(train_df))
                test_pred = np.repeat(prevalence, len(test_df))
                return (
                    auc_score(train_df["PHENO_TARGET"], train_pred),
                    auc_score(test_df["PHENO_TARGET"], test_pred),
                )

            mean_value = float(train_df["PHENO_TARGET"].mean())
            train_pred = np.repeat(mean_value, len(train_df))
            test_pred = np.repeat(mean_value, len(test_df))
            return (
                r2_from_predictions(train_df["PHENO_TARGET"], train_pred),
                r2_from_predictions(test_df["PHENO_TARGET"], test_pred),
            )

        x_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        x_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        if trait_type == "binary":
            if train_df["PHENO_TARGET"].nunique() != 2 or test_df["PHENO_TARGET"].nunique() != 2:
                return np.nan, np.nan

            model = LogisticRegression(max_iter=2000, solver="liblinear")
            model.fit(x_train_scaled, train_df["PHENO_TARGET"])

            train_pred = model.predict_proba(x_train_scaled)[:, 1]
            test_pred = model.predict_proba(x_test_scaled)[:, 1]

            return (
                auc_score(train_df["PHENO_TARGET"], train_pred),
                auc_score(test_df["PHENO_TARGET"], test_pred),
            )

        model = LinearRegression()
        model.fit(x_train_scaled, train_df["PHENO_TARGET"])
        train_pred = model.predict(x_train_scaled)
        test_pred = model.predict(x_test_scaled)

        return (
            r2_from_predictions(train_df["PHENO_TARGET"], train_pred),
            r2_from_predictions(test_df["PHENO_TARGET"], test_pred),
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

    train_pheno, train_trait_type = read_fam_phenotype(train_clone_bfile)
    test_pheno, test_trait_type = read_fam_phenotype(test_clone_bfile)
    trait_type = train_trait_type if train_trait_type == test_trait_type else "mixed"

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

    train_null_metric, test_null_metric = model_performance_with_features(
        train_design,
        test_design,
        covariate_feature_cols,
        trait_type,
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

        train_pure_metric, test_pure_metric = model_performance_with_features(
            train,
            test,
            ["SCORE"],
            trait_type,
        )

        train_best_metric, test_best_metric = model_performance_with_features(
            train,
            test,
            best_feature_cols,
            trait_type,
        )

        rows.append({
            "fold": fold_metric.get("fold", ""),
            "pvalue_label": label,
            "pvalue": pvalue,
            "trait_type": trait_type,
            "performance_metric": "AUC" if trait_type == "binary" else "R2",

            "Train_pure_prs": train_pure_metric,
            "Test_pure_prs": test_pure_metric,
            "Train_null_model": train_null_metric,
            "Test_null_model": test_null_metric,
            "Train_best_model": train_best_metric,
            "Test_best_model": test_best_metric,

            "Train_score_pheno_correlation": score_correlation(train["PHENO_TARGET"], train["SCORE"]),
            "Test_score_pheno_correlation": score_correlation(test["PHENO_TARGET"], test["SCORE"]),

            "train_scored_samples": len(train),
            "test_scored_samples": len(test),

            "train_cases": int(train["PHENO_TARGET"].sum()) if trait_type == "binary" else np.nan,
            "train_controls": int((train["PHENO_TARGET"] == 0).sum()) if trait_type == "binary" else np.nan,
            "test_cases": int(test["PHENO_TARGET"].sum()) if trait_type == "binary" else np.nan,
            "test_controls": int((test["PHENO_TARGET"] == 0).sum()) if trait_type == "binary" else np.nan,

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


# =============================================================================
# SUMMARY
# =============================================================================

def select_train_threshold_test_result(all_df):
    """
    Select best p-value threshold per fold using Train_best_model,
    then report the corresponding test performance.

    This avoids selecting the p-value threshold directly using test performance.
    """
    if all_df.empty:
        return pd.DataFrame()

    rows = []

    for fold, sub in all_df.groupby("fold", dropna=False):
        sub = sub.copy()

        valid = sub.dropna(subset=["Train_best_model"])
        if valid.empty:
            valid = sub.dropna(subset=["Train_pure_prs"])

        if valid.empty:
            continue

        sort_cols = []
        ascending = []

        if "Train_best_model" in valid.columns:
            sort_cols.append("Train_best_model")
            ascending.append(False)

        if "Train_pure_prs" in valid.columns:
            sort_cols.append("Train_pure_prs")
            ascending.append(False)

        sort_cols.append("pvalue")
        ascending.append(True)

        best = valid.sort_values(sort_cols, ascending=ascending).head(1).copy()
        best["threshold_selection_method"] = "selected_by_train_best_model"
        rows.append(best)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def summarise_results(all_results, prs_dir):
    prs_dir = Path(prs_dir)

    if not all_results:
        empty = pd.DataFrame()
        empty.to_csv(prs_dir / "PRS_all_folds_results.csv", index=False)
        empty.to_csv(prs_dir / "PRS_average_performance.csv", index=False)
        empty.to_csv(prs_dir / "PRS_best_result.csv", index=False)
        empty.to_csv(prs_dir / "PRS_train_selected_test_results.csv", index=False)
        return empty, empty, empty, empty

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
    avg["test_incremental_metric_best_minus_null"] = avg["Test_best_model_mean"] - avg["Test_null_model_mean"]
    avg["test_incremental_metric_best_minus_pure_prs"] = avg["Test_best_model_mean"] - avg["Test_pure_prs_mean"]
    avg["test_incremental_auc_best_minus_null"] = avg["test_incremental_metric_best_minus_null"]
    avg["test_incremental_auc_best_minus_pure_prs"] = avg["test_incremental_metric_best_minus_pure_prs"]

    avg.to_csv(prs_dir / "PRS_average_performance.csv", index=False)

    if not avg.empty:
        best = avg.sort_values(
            by=["Test_best_model_mean", "best_model_generalisation_gap"],
            ascending=[False, True],
        ).head(1)
    else:
        best = pd.DataFrame()

    best.to_csv(prs_dir / "PRS_best_result.csv", index=False)

    train_selected = select_train_threshold_test_result(all_df)
    train_selected.to_csv(prs_dir / "PRS_train_selected_test_results.csv", index=False)

    return all_df, avg, best, train_selected


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


def infer_plink_overall_reason(gwas_metric, fold_metrics):
    valid_rows = pd.to_numeric(pd.Series([gwas_metric.get("gwas_for_plink_valid_rows", pd.NA)]), errors="coerce").iloc[0]
    if pd.notna(valid_rows) and float(valid_rows) <= 0:
        return "no valid GWAS variants after FinalGWAS-to-PLINK preparation"

    if not fold_metrics:
        return "no fold metrics available"

    reasons = []
    for metric in fold_metrics:
        reason = clean_text(metric.get("fold_failure_reason", ""))
        if reason:
            reasons.append(reason)

    if reasons:
        counts = pd.Series(reasons).value_counts()
        return str(counts.index[0])

    n_results = sum(1 for metric in fold_metrics if clean_text(metric.get("fold_status", "")) == "success")
    if n_results <= 0:
        return "no successful PRS fold results"
    return ""


# =============================================================================
# MAIN
# =============================================================================

def process_one_job(index, args):
    job = load_job(index, args.jobs_file)

    data_root = clean_text(getattr(args, "data_root", ""))

    file_path = resolve_path(clean_text(job["file_path"]), data_root)

    phenotype_dir, phenotype = infer_phenotype_dir(job, file_path, data_root=data_root)

    accession = clean_text(job.get("accessionId", ""))

    gwas_output_dir = get_gwas_output_dir(job, file_path, data_root=data_root)

    final_gwas = find_final_gwas(
        gwas_output_dir=gwas_output_dir,
        explicit_final_gwas=args.final_gwas,
    )

    final_preview = pd.read_csv(final_gwas, nrows=5)
    try:
        final_total_rows = sum(1 for _ in open(final_gwas, "r", encoding="utf-8", errors="replace")) - 1
    except Exception:
        final_total_rows = len(pd.read_csv(final_gwas, usecols=[0]))

    print("=" * 100)
    print("[FINALGWAS PREFLIGHT]")
    print(f"[FILE]    {final_gwas}")
    print(f"[ROWS]    {max(final_total_rows, 0)}")
    print(f"[COLUMNS] {', '.join(map(str, final_preview.columns))}")
    print("=" * 100)

    if final_total_rows <= 0:
        raise ValueError(
            f"FinalGWAS.csv is empty: {final_gwas}. "
            "PRS cannot be calculated from an empty FinalGWAS file."
        )

    plink = find_plink(args.plink)

    prs_root = ensure_dir(gwas_output_dir / "PRS")
    prs_dir = ensure_dir(prs_root / args.tool_name)

    print("=" * 100)
    print("PRS2: PLINK PRS CALCULATION FROM FINALGWAS")
    print("=" * 100)
    print(f"[JOB INDEX]       {index}")
    print(f"[PHENOTYPE]       {phenotype}")
    print(f"[ACCESSION]       {accession}")
    print(f"[PHENOTYPE DIR]   {phenotype_dir}")
    print(f"[GWAS OUTPUT DIR] {gwas_output_dir}")
    print(f"[FINAL GWAS]      {final_gwas}")
    print(f"[PRS ROOT]        {prs_root}")
    print(f"[PRS OUTPUT DIR]  {prs_dir}")
    print(f"[TOOL NAME]       {args.tool_name}")
    print(f"[PLINK]           {plink}")
    print("=" * 100)

    gwas_metric = {}
    fold_metrics = []
    folds = []
    score_file = ""
    pvalue_file = ""
    range_file = ""
    overall_status = "success"
    overall_failure_reason = ""
    try:
        score_file, pvalue_file, gwas_metric = prepare_finalgwas_for_plink(
            final_gwas_file=final_gwas,
            prs_dir=prs_dir,
        )

        range_file, labels = create_range_list(
            prs_dir=prs_dir,
            min_p_exponent=args.min_p_exponent,
            n_intervals=args.n_pvalue_intervals,
        )

        folds = find_folds(phenotype_dir, args.folds)

        if not folds:
            raise FileNotFoundError(f"No Fold_* directories found in: {phenotype_dir}")

        all_results = []

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
            if result.empty and clean_text(fold_metric.get("fold_status", "")) == "success":
                fold_metric["fold_status"] = "failed"
                fold_metric["fold_failure_reason"] = "PLINK completed but no score or performance rows were produced"

            if not result.empty:
                result.insert(0, "job_index", index)
                result.insert(1, "phenotype", phenotype)
                result.insert(2, "accessionId", accession)
                result.insert(3, "prs_tool", args.tool_name)

            result.to_csv(fold_output_dir / "Results.csv", index=False)

            if not result.empty:
                all_results.append(result)

            fold_metrics.append(fold_metric)

        all_df, avg_df, best_df, train_selected_df = summarise_results(all_results, prs_dir)
        n_folds_with_results = int(all_df["fold"].nunique()) if not all_df.empty else 0
        if n_folds_with_results <= 0:
            overall_status = "failed"
            overall_failure_reason = infer_plink_overall_reason(gwas_metric, fold_metrics)
    except Exception as e:
        overall_status = "failed"
        overall_failure_reason = f"{type(e).__name__}: {e}"
        print("[RUN FAILED]", overall_failure_reason)
        all_df = pd.DataFrame()
        avg_df = pd.DataFrame()
        best_df = pd.DataFrame()
        train_selected_df = pd.DataFrame()

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(prs_dir / "PRS_fold_metrics.csv", index=False)

    summary = {
        "job_index": index,
        "phenotype": phenotype,
        "accessionId": accession,
        "file_path": str(file_path),
        "final_gwas": str(final_gwas),
        "gwas_input_source": "FinalGWAS.csv",
        "phenotype_dir": str(phenotype_dir),
        "gwas_output_dir": str(gwas_output_dir),
        "prs_root": str(prs_root),
        "prs_dir": str(prs_dir),
        "prs_tool": args.tool_name,
        "plink": str(plink),
        "overall_status": overall_status,
        "overall_failure_reason": overall_failure_reason,
        "n_folds_found": len(folds),
        "n_folds_with_results": int(all_df["fold"].nunique()) if not all_df.empty else 0,
        "range_file": str(range_file),
        "score_file": str(score_file),
        "pvalue_file": str(pvalue_file),
        "all_folds_results_file": str(prs_dir / "PRS_all_folds_results.csv"),
        "average_performance_file": str(prs_dir / "PRS_average_performance.csv"),
        "best_result_file": str(prs_dir / "PRS_best_result.csv"),
        "train_selected_test_results_file": str(prs_dir / "PRS_train_selected_test_results.csv"),
        "fold_metrics_file": str(prs_dir / "PRS_fold_metrics.csv"),
    }

    summary.update(gwas_metric)

    if not overall_failure_reason:
        summary["overall_failure_reason"] = infer_plink_overall_reason(gwas_metric, fold_metrics)
        if summary["overall_failure_reason"]:
            summary["overall_status"] = "failed"

    pd.DataFrame([summary]).to_csv(prs_dir / "PRS_run_summary.csv", index=False)

    if args.cleanup_intermediate:
        keep_names = {
            "PRS_run_summary.csv",
            "PRS_fold_metrics.csv",
            "PRS_all_folds_results.csv",
            "PRS_average_performance.csv",
            "PRS_best_result.csv",
            "PRS_train_selected_test_results.csv",
            Path(score_file).name,
            Path(pvalue_file).name,
            Path(range_file).name,
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
    print(prs_dir / "PRS_train_selected_test_results.csv")
    print("=" * 100)

    if not best_df.empty:
        print()
        print("[BEST RESULT: TEST-SORTED SUMMARY]")
        print(best_df.to_string(index=False))

    if not train_selected_df.empty:
        print()
        print("[TRAIN-SELECTED TEST RESULTS]")
        show_cols = [
            "fold", "pvalue_label", "pvalue",
            "Train_best_model", "Test_best_model",
            "Train_pure_prs", "Test_pure_prs",
            "valid_snps_for_scoring",
        ]
        show_cols = [c for c in show_cols if c in train_selected_df.columns]
        print(train_selected_df[show_cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="PRS2: Calculate PLINK PRS for one GWAS job using FinalGWAS.csv."
    )

    parser.add_argument(
        "index",
        type=int,
        help="GWAS job index. Example: python PRS2-Calculation1-FinalGWAS.py 1 --force",
    )

    parser.add_argument(
        "--jobs-file",
        default=DEFAULT_JOBS_FILE,
        help="Default: GWAS_jobs.csv",
    )

    parser.add_argument(
        "--final-gwas",
        default="",
        help="Optional direct path to FinalGWAS.csv. If omitted, script finds it in the GWAS output directory.",
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

    parser.add_argument(
        "--tool-name",
        default=DEFAULT_PRS_TOOL_NAME,
        help="PRS subfolder/tool name under <GWAS_DIR>/PRS/. Default: PRS_Plink",
    )

    parser.add_argument(
        "--plink",
        default="",
        help="Optional PLINK path. Default: ./plink, ./plink.exe, or PATH.",
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
        help="Number of PCs to calculate and include with covariates. Default: 6",
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
        help="Remove intermediate PLINK files and per-fold working directories after writing result CSVs.",
    )

    parser.add_argument(
        "--data-root",
        default="/data/ascher01/uqmmune1/finalheritability",
        help="Root folder to resolve relative paths. Default: /data/ascher01/uqmmune1/finalheritability",
    )

    args = parser.parse_args()

    process_one_job(args.index, args)


if __name__ == "__main__":
    main()
