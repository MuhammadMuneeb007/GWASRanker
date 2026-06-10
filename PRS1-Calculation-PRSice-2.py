#!/usr/bin/env python3

"""
PRS1-Calculation-PRSice-2.py

Simple PRSice-2 execution from FinalGWAS.csv across all folds for one GWAS job.

Run:
    python PRS1-Calculation-PRSice-2.py 1

What it does:
    1. Reads FinalGWAS.csv for the selected job.
    2. Builds one PRSice base file.
    3. Runs PRSice-2 for train and test targets in each Fold_* directory.
    4. Saves fold-level outputs and a simple run summary.

This script intentionally does not do the larger PLINK PRS evaluation workflow.
It just prepares the base file and executes PRSice-2.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_JOBS_FILE = "GWAS_jobs.csv"
DEFAULT_DATA_ROOT = "/data/ascher01/uqmmune1/finalheritability"
DEFAULT_TRAIN_BFILE = "train_data.QC"
DEFAULT_TEST_BFILE = "test_data"
DEFAULT_TOOL_NAME = "PRS_PRSice-2"
DEFAULT_SCORE_MODEL = "avg"
DEFAULT_LOWER = 1e-5
DEFAULT_UPPER = 1.0
DEFAULT_INTERVAL = 0.05
DEFAULT_CLUMP_KB = 200
DEFAULT_CLUMP_R2 = 0.1
DEFAULT_CLUMP_P = 1.0
KEEP_TEMP_FILES = True


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def abs_path(path):
    return str(Path(path).resolve())


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def last_meaningful_line(text):
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line:
            return line
    return ""


def normalise_identifier(x):
    x = clean_text(x)
    if x.upper() in {"", "NAN", "NA", "N/A", "NONE", "NULL", ".", "<NA>"}:
        return ""
    return x


def normalise_allele(x):
    x = clean_text(x).upper()
    if x in {"", "NAN", "NA", "N/A", "NONE", "NULL", "."}:
        return ""
    return x


def find_first_column(columns, candidates):
    exact = {str(c): c for c in columns}
    lower = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return ""


def percent(numerator, denominator):
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 4)


def check_bfile(prefix):
    prefix = Path(prefix)
    missing = []
    for suffix in [".bed", ".bim", ".fam"]:
        p = Path(str(prefix) + suffix)
        if not p.exists():
            missing.append(str(p))
    return missing


def natural_fold_number(path):
    name = Path(path).name
    if name.startswith("Fold_"):
        try:
            return int(name.split("_", 1)[1])
        except Exception:
            return 10**9
    return 10**9


def run_command(command, log_file):
    command = [str(x) for x in command]
    log_file = Path(log_file)
    ensure_dir(log_file.parent)

    print()
    print("=" * 100)
    print("[COMMAND]")
    print(" ".join(command))
    print("[LOG]", log_file)
    print("=" * 100)
    sys.stdout.flush()

    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    with open(log_file, "a", encoding="utf-8") as log:
        log.write("\n" + "=" * 100 + "\n")
        log.write(" ".join(command) + "\n")
        log.write("\n[STDOUT]\n")
        log.write(completed.stdout or "")
        log.write("\n[STDERR]\n")
        log.write(completed.stderr or "")
        log.write("\n[RETURN CODE]\n")
        log.write(str(completed.returncode) + "\n")

    print("[STDOUT]")
    print(completed.stdout or "<empty>")
    print("[STDERR]")
    print(completed.stderr or "<empty>")
    print("[RETURN CODE]", completed.returncode)
    sys.stdout.flush()

    return completed


def find_prsice_command():
    prsice_script = Path("PRSice.R")
    prsice_binary_candidates = [
        Path("PRSice"),
        Path("PRSice.exe"),
        Path("PRSice_linux"),
        Path("PRSice_mac"),
    ]

    prsice_binary = None
    for candidate in prsice_binary_candidates:
        if candidate.exists():
            prsice_binary = candidate
            break

    if prsice_script.exists():
        rscript = shutil.which("Rscript")
        if not rscript:
            raise FileNotFoundError("PRSice.R was found, but Rscript is not available in PATH.")
        if not prsice_binary:
            raise FileNotFoundError("PRSice.R was found, but no PRSice binary was found.")
        return [rscript, str(prsice_script), "--prsice", str(prsice_binary)], str(prsice_binary), str(prsice_script)

    if prsice_binary:
        return [str(prsice_binary)], str(prsice_binary), ""

    found = shutil.which("PRSice")
    if found:
        return [found], found, ""

    found = shutil.which("PRSice_linux")
    if found:
        return [found], found, ""

    raise FileNotFoundError(
        "PRSice-2 not found. Put PRSice/PRSice_linux (and optionally PRSice.R) "
        "in the working directory or add them to PATH."
    )


def load_job(index):
    jobs = pd.read_csv(DEFAULT_JOBS_FILE)
    jobs["job_index"] = pd.to_numeric(jobs["job_index"], errors="coerce").astype("Int64")
    sub = jobs[jobs["job_index"] == int(index)]
    if sub.empty:
        raise ValueError(f"No job found for index {index} in {DEFAULT_JOBS_FILE}")
    return sub.iloc[0]


def infer_phenotype_dir(job, file_path):
    phenotype = clean_text(job.get("phenotype", ""))
    if phenotype:
        candidate = Path(DEFAULT_DATA_ROOT) / phenotype
        if candidate.exists():
            return candidate, phenotype

    parts = Path(file_path).parts
    if parts:
        candidate = Path(DEFAULT_DATA_ROOT) / parts[0]
        if candidate.exists():
            return candidate, candidate.name

    if phenotype:
        return Path(DEFAULT_DATA_ROOT) / phenotype, phenotype

    raise ValueError("Could not infer phenotype directory.")


def get_gwas_output_dir(job, file_path):
    out = clean_text(job.get("output_feature_file", ""))
    if out:
        return Path(out).parent
    return Path(file_path).parent


def find_final_gwas(gwas_output_dir):
    candidates = [
        Path(gwas_output_dir) / "FinalGWAS.csv",
        Path(gwas_output_dir) / "FinalGWAS.tsv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find FinalGWAS.csv in {gwas_output_dir}")


def find_folds(phenotype_dir):
    folds = sorted(Path(phenotype_dir).glob("Fold_*"), key=natural_fold_number)
    return [f for f in folds if f.is_dir()]


def deduplicate_best_p_rows(df, snp_col="SNP", p_col="P"):
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
    return working, len(df) - len(working)


def prepare_finalgwas_for_prsice(final_gwas_file, prs_dir):
    if Path(final_gwas_file).suffix.lower() == ".tsv":
        df = pd.read_csv(final_gwas_file, sep="\t", low_memory=False)
    else:
        df = pd.read_csv(final_gwas_file, low_memory=False)

    df.columns = [str(c).strip() for c in df.columns]

    print("=" * 100)
    print("[FINALGWAS INPUT CHECK]")
    print("[FILE]", final_gwas_file)
    print("[ROWS]", len(df))
    print("[COLUMNS]", ", ".join(df.columns))
    print("=" * 100)

    snp_col = find_first_column(df.columns, ["SNPID", "SNP", "rsID", "ID", "MarkerName", "variant_id"])
    a1_col = find_first_column(df.columns, ["EA", "A1", "effect_allele", "ALT"])
    a2_col = find_first_column(df.columns, ["NEA", "A2", "other_allele", "REF"])
    p_col = find_first_column(df.columns, ["P", "p", "PVAL", "P_VALUE", "pvalue", "p_value"])
    beta_col = find_first_column(df.columns, ["BETA", "beta", "Effect", "effect", "LOG_OR", "logOR"])
    or_col = find_first_column(df.columns, ["OR", "odds_ratio", "OddsRatio"])

    missing = []
    if not snp_col:
        missing.append("SNPID/SNP/rsID")
    if not a1_col:
        missing.append("EA/A1/effect_allele")
    if not a2_col:
        missing.append("NEA/A2/other_allele")
    if not p_col:
        missing.append("P")
    if not beta_col and not or_col:
        missing.append("BETA or OR")

    if missing:
        raise ValueError(
            "FinalGWAS.csv is missing required PRSice columns.\n"
            f"Missing: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["SNP"] = df[snp_col].map(normalise_identifier)
    out["A1"] = df[a1_col].map(normalise_allele)
    out["A2"] = df[a2_col].map(normalise_allele)
    out["P"] = safe_numeric(df[p_col])

    if beta_col:
        out["BETA"] = safe_numeric(df[beta_col])
        stat_mode = "BETA"
        effect_source = f"BETA:{beta_col}"
    else:
        out["OR"] = safe_numeric(df[or_col])
        stat_mode = "OR"
        effect_source = f"OR:{or_col}"

    valid = (
        out["SNP"].ne("")
        & out["A1"].str.match(r"^[ACGT]$", na=False)
        & out["A2"].str.match(r"^[ACGT]$", na=False)
        & out["A1"].ne(out["A2"])
        & out["P"].notna()
        & np.isfinite(out["P"])
        & (out["P"] > 0)
        & (out["P"] <= 1)
    )
    if stat_mode == "BETA":
        valid = valid & out["BETA"].notna() & np.isfinite(out["BETA"])
    else:
        valid = valid & out["OR"].notna() & np.isfinite(out["OR"]) & (out["OR"] > 0)

    keep_cols = ["SNP", "A1", "A2", "P", stat_mode]
    out = out.loc[valid, keep_cols].copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=keep_cols)

    before_dedup = len(out)
    out, duplicate_removed = deduplicate_best_p_rows(out, snp_col="SNP", p_col="P")

    base_file = Path(prs_dir) / "GWAS_for_PRSice.tsv"
    out.to_csv(base_file, sep="\t", index=False)

    metric = {
        "finalgwas_input_rows": int(len(df)),
        "gwas_for_prsice_valid_rows": int(len(out)),
        "gwas_for_prsice_duplicate_snp_removed": int(duplicate_removed),
        "gwas_for_prsice_retained_pct": percent(len(out), len(df)),
        "mapped_snp_col": snp_col,
        "mapped_a1_col": a1_col,
        "mapped_a2_col": a2_col,
        "mapped_p_col": p_col,
        "mapped_beta_col": beta_col,
        "mapped_or_col": or_col,
        "effect_source": effect_source,
        "prsice_stat_mode": stat_mode,
        "prsice_base_file": str(base_file),
    }

    print("[FINALGWAS TO PRSICE DIAGNOSTICS]")
    for key, value in metric.items():
        print(f"{key}: {value}")
    print("=" * 100)

    if len(out) == 0:
        raise ValueError("No valid SNPs remained for PRSice base file generation.")

    return base_file, metric


def read_raw_fam(bfile_prefix):
    fam_file = Path(str(bfile_prefix) + ".fam")
    fam = pd.read_csv(
        fam_file,
        sep=r"\s+",
        header=None,
        names=["FID", "IID", "PID", "MID", "SEX", "PHENO"],
        dtype={"FID": str, "IID": str},
    )
    fam["PHENO"] = safe_numeric(fam["PHENO"])
    fam = fam.dropna(subset=["PHENO"]).copy()
    fam = fam[~fam["PHENO"].isin([-9])].copy()
    return fam


def is_binary_trait(fam):
    values = sorted(fam["PHENO"].dropna().unique().tolist())
    return values in ([1, 2], [0, 1]) or len(values) == 2


def write_pheno_file(bfile_prefix, out_file):
    fam = read_raw_fam(bfile_prefix)
    out = fam[["FID", "IID", "PHENO"]].copy()
    out_file = Path(out_file)
    ensure_dir(out_file.parent)
    out.to_csv(out_file, sep="\t", index=False)
    return out_file, is_binary_trait(fam), len(out)


def build_prsice_cov_file(target_bfile_prefix, cov_source_file, out_file):
    cov_source_file = Path(cov_source_file)
    if not cov_source_file.exists():
        return "", 0

    fam_ids = read_raw_fam(target_bfile_prefix)[["FID", "IID"]].copy()
    fam_ids["FID"] = fam_ids["FID"].astype(str)
    fam_ids["IID"] = fam_ids["IID"].astype(str)

    cov = pd.read_csv(cov_source_file, sep=r"\s+", low_memory=False)
    if "FID" not in cov.columns or "IID" not in cov.columns:
        return "", 0
    cov["FID"] = cov["FID"].astype(str)
    cov["IID"] = cov["IID"].astype(str)

    keep = ["FID", "IID"]
    for column in cov.columns:
        if column in {"FID", "IID"}:
            continue
        numeric = safe_numeric(cov[column])
        if numeric.notna().any():
            cov[column] = numeric
            keep.append(column)

    cov = cov[keep].copy()
    cov = fam_ids.merge(cov, on=["FID", "IID"], how="left")
    feature_cols = [c for c in cov.columns if c not in {"FID", "IID"}]
    if not feature_cols:
        return "", 0

    cov[feature_cols] = cov[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    out_file = Path(out_file)
    ensure_dir(out_file.parent)
    cov.to_csv(out_file, sep="\t", index=False)
    return str(out_file), len(feature_cols)


def safe_delete(path_text):
    path_text = clean_text(path_text)
    if not path_text:
        return
    try:
        path = Path(path_text)
        if path.exists():
            path.unlink()
    except Exception:
        pass


def build_prsice_command(command_prefix, base_file, target_prefix, pheno_file, cov_file, out_prefix, stat_mode, binary_target):
    command = list(command_prefix)
    command.extend([
        "--base", abs_path(base_file),
        "--target", abs_path(target_prefix),
        "--pheno", abs_path(pheno_file),
        "--snp", "SNP",
        "--a1", "A1",
        "--a2", "A2",
        "--stat", stat_mode,
        "--pvalue", "P",
        "--score", DEFAULT_SCORE_MODEL,
        "--lower", str(DEFAULT_LOWER),
        "--upper", str(DEFAULT_UPPER),
        "--interval", str(DEFAULT_INTERVAL),
        "--clump-kb", str(DEFAULT_CLUMP_KB),
        "--clump-r2", str(DEFAULT_CLUMP_R2),
        "--clump-p", str(DEFAULT_CLUMP_P),
        "--binary-target", "T" if binary_target else "F",
        "--all-score",
        "--out", abs_path(out_prefix),
    ])

    if cov_file:
        command.extend(["--cov", abs_path(cov_file)])

    if stat_mode.upper() == "BETA":
        command.append("--beta")
    else:
        command.append("--or")

    return command


def read_fam_phenotype(bfile_prefix):
    fam = read_raw_fam(bfile_prefix).copy()
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


def read_covariate_file(cov_file):
    cov_file = Path(cov_file)
    if not cov_file.exists():
        return pd.DataFrame(columns=["FID", "IID"])

    cov = pd.read_csv(cov_file, sep=r"\s+", low_memory=False)
    if "FID" not in cov.columns or "IID" not in cov.columns:
        return pd.DataFrame(columns=["FID", "IID"])

    cov["FID"] = cov["FID"].astype(str)
    cov["IID"] = cov["IID"].astype(str)

    keep = ["FID", "IID"]
    for column in cov.columns:
        if column in {"FID", "IID"}:
            continue
        numeric = safe_numeric(cov[column])
        if numeric.notna().any():
            cov[column] = numeric
            keep.append(column)

    return cov[keep].copy()


def build_covariate_design(phenotype_df, cov_file):
    design = phenotype_df.copy()
    cov = read_covariate_file(cov_file)
    if not cov.empty and len(cov.columns) > 2:
        design = design.merge(cov, on=["FID", "IID"], how="left")

    feature_cols = [c for c in design.columns if c not in ["FID", "IID", "PHENO_TARGET"]]
    if feature_cols:
        for column in feature_cols:
            design[column] = safe_numeric(design[column])
        design[feature_cols] = design[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    return design, feature_cols


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


def read_prsice_all_score(all_score_file):
    all_score_file = Path(all_score_file)
    if not all_score_file.exists():
        return pd.DataFrame(), []

    df = pd.read_csv(all_score_file, sep=r"\s+")
    if "FID" not in df.columns or "IID" not in df.columns:
        return pd.DataFrame(), []

    df["FID"] = df["FID"].astype(str)
    df["IID"] = df["IID"].astype(str)

    score_cols = [c for c in df.columns if str(c).startswith("Pt_")]
    if not score_cols:
        score_cols = [c for c in df.columns if c not in ["FID", "IID", "IID_FID", "Phenotype"]]

    for column in score_cols:
        df[column] = safe_numeric(df[column])

    return df[["FID", "IID"] + score_cols].copy(), score_cols


def prsice_label_to_pvalue(label):
    label = clean_text(label)
    if label.startswith("Pt_"):
        label = label[3:]
    value = safe_numeric(pd.Series([label])).iloc[0]
    return float(value) if pd.notna(value) else np.nan


def evaluate_fold_prsice_outputs(fold_row):
    train_bfile = fold_row.get("train_bfile", "")
    test_bfile = fold_row.get("test_bfile", "")
    score_dir = Path(fold_row.get("score_dir", ""))
    if not train_bfile or not test_bfile or not score_dir.exists():
        return pd.DataFrame()

    if not fold_row.get("train_success", False) or not fold_row.get("test_success", False):
        return pd.DataFrame()

    train_pheno, train_trait_type = read_fam_phenotype(train_bfile)
    test_pheno, test_trait_type = read_fam_phenotype(test_bfile)
    trait_type = train_trait_type if train_trait_type == test_trait_type else "mixed"

    train_design, covariate_feature_cols = build_covariate_design(
        train_pheno,
        Path(fold_row.get("fold_dir", "")) / "train_data.cov",
    )
    test_design, _ = build_covariate_design(
        test_pheno,
        Path(fold_row.get("fold_dir", "")) / "test_data.cov",
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

    train_scores, train_cols = read_prsice_all_score(score_dir / "train.all_score")
    test_scores, test_cols = read_prsice_all_score(score_dir / "test.all_score")
    if train_scores.empty or test_scores.empty:
        return pd.DataFrame()

    shared_cols = [c for c in train_cols if c in set(test_cols)]
    rows = []
    for label in shared_cols:
        train_score = train_scores[["FID", "IID", label]].rename(columns={label: "SCORE"})
        test_score = test_scores[["FID", "IID", label]].rename(columns={label: "SCORE"})

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
        train_best_metric, test_best_metric = model_performance_with_features(train, test, best_feature_cols, trait_type)

        rows.append({
            "fold": fold_row.get("fold", ""),
            "pvalue_label": label,
            "pvalue": prsice_label_to_pvalue(label),
            "prsice_model": fold_row.get("prsice_model", DEFAULT_SCORE_MODEL),
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
            "covariate_count": len(covariate_feature_cols),
            "train_profile": str(score_dir / "train.all_score"),
            "test_profile": str(score_dir / "test.all_score"),
        })

    return pd.DataFrame(rows)


def select_train_threshold_test_result(all_df):
    if all_df.empty:
        return pd.DataFrame()

    rows = []
    for fold, sub in all_df.groupby("fold", dropna=False):
        valid = sub.dropna(subset=["Train_best_model"]).copy()
        if valid.empty:
            valid = sub.dropna(subset=["Train_pure_prs"]).copy()
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

    group_cols = ["prsice_model", "pvalue_label", "pvalue", "trait_type", "performance_metric"]
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
            covariate_count_mean=("covariate_count", "mean"),
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


def infer_prsice_overall_reason(gwas_metric, fold_rows):
    valid_rows = pd.to_numeric(pd.Series([gwas_metric.get("gwas_for_prsice_valid_rows", pd.NA)]), errors="coerce").iloc[0]
    if pd.notna(valid_rows) and float(valid_rows) <= 0:
        return "no valid GWAS variants after FinalGWAS-to-PRSice preparation"

    if not fold_rows:
        return "no fold metrics available"

    reasons = []
    for row in fold_rows:
        reason = clean_text(row.get("fold_failure_reason", ""))
        if reason:
            reasons.append(reason)

    if reasons:
        counts = pd.Series(reasons).value_counts()
        return str(counts.index[0])

    n_results = sum(1 for row in fold_rows if clean_text(row.get("fold_status", "")) == "success")
    if n_results <= 0:
        return "no successful PRSice fold results"
    return ""


def process_one_job(index):
    job = load_job(index)
    file_path = Path(clean_text(job["file_path"]))
    phenotype_dir, phenotype = infer_phenotype_dir(job, file_path)
    accession = clean_text(job.get("accessionId", ""))
    gwas_output_dir = get_gwas_output_dir(job, file_path)
    final_gwas = find_final_gwas(gwas_output_dir)
    prs_root = ensure_dir(Path(gwas_output_dir) / "PRS")
    prs_dir = ensure_dir(prs_root / DEFAULT_TOOL_NAME)

    final_preview = pd.read_csv(final_gwas, nrows=5)
    try:
        final_total_rows = sum(1 for _ in open(final_gwas, "r", encoding="utf-8", errors="replace")) - 1
    except Exception:
        final_total_rows = len(pd.read_csv(final_gwas, usecols=[0]))

    print("=" * 100)
    print("[FINALGWAS PREFLIGHT]")
    print("[FILE]   ", final_gwas)
    print("[ROWS]   ", max(final_total_rows, 0))
    print("[COLUMNS]", ", ".join(map(str, final_preview.columns)))
    print("=" * 100)

    prsice_command_prefix, prsice_binary, prsice_script = find_prsice_command()

    print("=" * 100)
    print("PRS1: PRSICE-2 CALCULATION FROM FINALGWAS")
    print("=" * 100)
    print("[JOB INDEX]      ", index)
    print("[PHENOTYPE]      ", phenotype)
    print("[ACCESSION]      ", accession)
    print("[PHENOTYPE DIR]  ", phenotype_dir)
    print("[GWAS OUTPUT DIR]", gwas_output_dir)
    print("[FINAL GWAS]     ", final_gwas)
    print("[PRS ROOT]       ", prs_root)
    print("[PRS OUTPUT DIR] ", prs_dir)
    print("[PRSICE]         ", prsice_binary)
    if prsice_script:
        print("[PRSICE SCRIPT]  ", prsice_script)
    print("=" * 100)

    gwas_metric = {}
    folds = []
    fold_rows = []
    all_results = []
    base_file = ""
    overall_status = "success"
    overall_failure_reason = ""
    try:
        base_file, gwas_metric = prepare_finalgwas_for_prsice(final_gwas, prs_dir)
        folds = find_folds(phenotype_dir)
        if not folds:
            raise FileNotFoundError(f"No Fold_* directories found in {phenotype_dir}")

        for fold_dir in folds:
            fold_number = natural_fold_number(fold_dir)
            fold_output_dir = ensure_dir(prs_dir / f"Fold_{fold_number}")
            score_dir = ensure_dir(fold_output_dir / "scores")
            log_file = fold_output_dir / "prsice_commands.log"

            train_bfile = Path(fold_dir) / DEFAULT_TRAIN_BFILE
            test_bfile = Path(fold_dir) / DEFAULT_TEST_BFILE

            print("=" * 100)
            print(f"PROCESSING Fold_{fold_number}")
            print("=" * 100)

            train_missing = check_bfile(train_bfile)
            test_missing = check_bfile(test_bfile)

            row = {
                "fold": fold_number,
                "fold_dir": str(fold_dir),
                "score_dir": str(score_dir),
                "train_bfile": str(train_bfile),
                "test_bfile": str(test_bfile),
                "train_bfile_missing": " | ".join(train_missing),
                "test_bfile_missing": " | ".join(test_missing),
                "prsice_base_file": str(base_file),
                "prsice_model": DEFAULT_SCORE_MODEL,
                "train_success": False,
                "test_success": False,
                "binary_target": "",
                "train_pheno_rows": 0,
                "test_pheno_rows": 0,
                "train_covariate_count": 0,
                "test_covariate_count": 0,
                "train_all_score_file": str(score_dir / "train.all_score"),
                "test_all_score_file": str(score_dir / "test.all_score"),
                "train_best_file": str(score_dir / "train.best"),
                "test_best_file": str(score_dir / "test.best"),
                "command_log": str(log_file),
                "fold_status": "not_started",
                "fold_failure_reason": "",
                "train_cov_file": "",
                "test_cov_file": "",
                "train_cov_exists_before_run": False,
                "test_cov_exists_before_run": False,
            }

            if train_missing or test_missing:
                row["fold_status"] = "failed"
                row["fold_failure_reason"] = "missing train/test genotype bfile components"
                fold_rows.append(row)
                pd.DataFrame().to_csv(fold_output_dir / "Results.csv", index=False)
                continue

            train_cov_file = ""
            test_cov_file = ""
            train_pheno_file, binary_target, train_pheno_rows = write_pheno_file(
                train_bfile,
                score_dir / "train.PHENO",
            )
            test_pheno_file, _, test_pheno_rows = write_pheno_file(
                test_bfile,
                score_dir / "test.PHENO",
            )

            train_cov_file, train_cov_count = build_prsice_cov_file(
                train_bfile,
                Path(fold_dir) / "train_data.cov",
                score_dir / "train.PRSice.COV",
            )
            test_cov_file, test_cov_count = build_prsice_cov_file(
                test_bfile,
                Path(fold_dir) / "test_data.cov",
                score_dir / "test.PRSice.COV",
            )

            row["binary_target"] = binary_target
            row["train_pheno_rows"] = int(train_pheno_rows)
            row["test_pheno_rows"] = int(test_pheno_rows)
            row["train_covariate_count"] = int(train_cov_count)
            row["test_covariate_count"] = int(test_cov_count)
            row["train_cov_file"] = abs_path(train_cov_file) if train_cov_file else ""
            row["test_cov_file"] = abs_path(test_cov_file) if test_cov_file else ""
            row["train_cov_exists_before_run"] = bool(train_cov_file and Path(train_cov_file).exists())
            row["test_cov_exists_before_run"] = bool(test_cov_file and Path(test_cov_file).exists())

            try:
                train_command = build_prsice_command(
                    command_prefix=prsice_command_prefix,
                    base_file=base_file,
                    target_prefix=train_bfile,
                    pheno_file=train_pheno_file,
                    cov_file=train_cov_file,
                    out_prefix=score_dir / "train",
                    stat_mode=gwas_metric["prsice_stat_mode"],
                    binary_target=binary_target,
                )
                if train_cov_file and not Path(train_cov_file).exists():
                    raise FileNotFoundError(f"PRSice train cov file missing before run: {train_cov_file}")
                completed_train = run_command(train_command, log_file)
                row["train_success"] = completed_train.returncode == 0
                if completed_train.returncode != 0:
                    row["fold_status"] = "failed"
                    row["fold_failure_reason"] = last_meaningful_line(completed_train.stderr) or "PRSice execution failed on train fold"

                test_command = build_prsice_command(
                    command_prefix=prsice_command_prefix,
                    base_file=base_file,
                    target_prefix=test_bfile,
                    pheno_file=test_pheno_file,
                    cov_file=test_cov_file,
                    out_prefix=score_dir / "test",
                    stat_mode=gwas_metric["prsice_stat_mode"],
                    binary_target=binary_target,
                )
                if test_cov_file and not Path(test_cov_file).exists():
                    raise FileNotFoundError(f"PRSice test cov file missing before run: {test_cov_file}")
                completed_test = run_command(test_command, log_file)
                row["test_success"] = completed_test.returncode == 0
                if completed_test.returncode != 0 and not row["fold_failure_reason"]:
                    row["fold_status"] = "failed"
                    row["fold_failure_reason"] = last_meaningful_line(completed_test.stderr) or "PRSice execution failed on test fold"
            except Exception as fold_error:
                row["fold_status"] = "failed"
                if not row["fold_failure_reason"]:
                    row["fold_failure_reason"] = f"{type(fold_error).__name__}: {fold_error}"
            finally:
                if KEEP_TEMP_FILES:
                    print("[INFO] KEEP_TEMP_FILES=True, keeping PRSice temporary covariate files.")
                    if train_cov_file:
                        print("[KEEP TRAIN COV]", train_cov_file)
                    if test_cov_file:
                        print("[KEEP TEST COV]", test_cov_file)
                else:
                    safe_delete(train_cov_file)
                    safe_delete(test_cov_file)

            if row["fold_status"] == "not_started":
                if row["train_success"] and row["test_success"]:
                    row["fold_status"] = "success"
                    row["fold_failure_reason"] = ""
                elif not row["train_cov_exists_before_run"]:
                    row["fold_status"] = "failed"
                    row["fold_failure_reason"] = "train PRSice covariate file was not created"
                elif not row["test_cov_exists_before_run"]:
                    row["fold_status"] = "failed"
                    row["fold_failure_reason"] = "test PRSice covariate file was not created"
                else:
                    row["fold_status"] = "failed"
                    row["fold_failure_reason"] = "PRSice fold did not complete successfully"

            fold_rows.append(row)

            result = evaluate_fold_prsice_outputs(row)
            if result.empty and row["fold_status"] == "success":
                row["fold_status"] = "failed"
                row["fold_failure_reason"] = "PRSice completed but no score or performance rows were produced"
            if not result.empty:
                result.insert(0, "job_index", index)
                result.insert(1, "phenotype", phenotype)
                result.insert(2, "accessionId", accession)
                result.insert(3, "prs_tool", DEFAULT_TOOL_NAME)
                all_results.append(result)
            fold_rows[-1] = row
            result.to_csv(fold_output_dir / "Results.csv", index=False)
    except Exception as e:
        overall_status = "failed"
        overall_failure_reason = f"{type(e).__name__}: {e}"
        print("[RUN FAILED]", overall_failure_reason)

    fold_df = pd.DataFrame(fold_rows)
    fold_file = prs_dir / "PRS_fold_metrics.csv"
    fold_df.to_csv(fold_file, index=False)

    all_df, avg_df, best_df, train_selected_df = summarise_results(all_results, prs_dir)

    summary = {
        "job_index": index,
        "phenotype": phenotype,
        "accessionId": accession,
        "file_path": str(file_path),
        "final_gwas": str(final_gwas),
        "prs_dir": str(prs_dir),
        "prs_tool": DEFAULT_TOOL_NAME,
        "prsice_binary": prsice_binary,
        "prsice_script": prsice_script,
        "overall_status": overall_status,
        "overall_failure_reason": overall_failure_reason,
        "n_folds_found": len(folds),
        "n_folds_train_success": int(fold_df["train_success"].sum()) if not fold_df.empty else 0,
        "n_folds_test_success": int(fold_df["test_success"].sum()) if not fold_df.empty else 0,
        "n_folds_with_results": int(all_df["fold"].nunique()) if not all_df.empty else 0,
        "base_file": str(base_file),
        "score_model": DEFAULT_SCORE_MODEL,
        "lower": DEFAULT_LOWER,
        "upper": DEFAULT_UPPER,
        "interval": DEFAULT_INTERVAL,
        "clump_kb": DEFAULT_CLUMP_KB,
        "clump_r2": DEFAULT_CLUMP_R2,
        "clump_p": DEFAULT_CLUMP_P,
        "fold_metrics_file": str(fold_file),
        "all_folds_results_file": str(prs_dir / "PRS_all_folds_results.csv"),
        "average_performance_file": str(prs_dir / "PRS_average_performance.csv"),
        "best_result_file": str(prs_dir / "PRS_best_result.csv"),
        "train_selected_test_results_file": str(prs_dir / "PRS_train_selected_test_results.csv"),
    }
    summary.update(gwas_metric)

    if not summary["overall_failure_reason"]:
        summary["overall_failure_reason"] = infer_prsice_overall_reason(gwas_metric, fold_rows)
        if summary["overall_failure_reason"]:
            summary["overall_status"] = "failed"

    summary_file = prs_dir / "PRS_run_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_file, index=False)

    print("=" * 100)
    print("[FINAL OUTPUTS]")
    print(summary_file)
    print(fold_file)
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
        ]
        show_cols = [c for c in show_cols if c in train_selected_df.columns]
        print(train_selected_df[show_cols].to_string(index=False))


def main():
    if len(sys.argv) != 2:
        print("Usage: python PRS1-Calculation-PRSice-2.py <job_index>")
        sys.exit(1)

    process_one_job(int(sys.argv[1]))


if __name__ == "__main__":
    main()
