#!/usr/bin/env python3

import argparse
import contextlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# BASIC HELPERS
# =============================================================================

def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalise_chr(x):
    x = clean_text(x).upper().replace("CHR", "").strip()

    if x == "23":
        return "X"
    if x == "24":
        return "Y"
    if x in ["25", "M", "MT"]:
        return "MT"

    return x


class TeeTextIO:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


# =============================================================================
# JOB HANDLING
# =============================================================================

def load_job(index, jobs_file):
    jobs_file = Path(jobs_file)

    if not jobs_file.exists():
        raise FileNotFoundError(f"Missing jobs file: {jobs_file}")

    jobs = pd.read_csv(jobs_file)

    if "job_index" not in jobs.columns or "file_path" not in jobs.columns:
        raise ValueError("GWAS_jobs.csv must contain job_index and file_path columns.")

    sub = jobs[jobs["job_index"].astype(int) == int(index)]

    if sub.empty:
        raise ValueError(f"No job found for index: {index}")

    return sub.iloc[0]


def get_output_dir(job, gwas_file):
    if "output_feature_file" in job.index:
        output_feature_file = clean_text(job.get("output_feature_file", ""))
        if output_feature_file:
            return Path(output_feature_file).parent

    return Path(gwas_file).parent


# =============================================================================
# LOAD ProcessedGWAS.csv
# =============================================================================

def load_processed_gwas(processed_file):
    processed_file = Path(processed_file)

    if not processed_file.exists():
        raise FileNotFoundError(f"Missing ProcessedGWAS.csv: {processed_file}")

    df = pd.read_csv(processed_file, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    required = ["CHR", "BP", "SNP", "A1", "A2"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"ProcessedGWAS.csv is missing required columns: {missing}")

    df["CHR"] = df["CHR"].map(normalise_chr)
    df["BP"] = pd.to_numeric(df["BP"], errors="coerce")

    print("=" * 100)
    print("[LOADED ProcessedGWAS.csv]")
    print(f"Input:   {processed_file}")
    print(f"Rows:    {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("Header:")
    print(list(df.columns))
    print("=" * 100)

    return df


def processed_to_gwaslab_df(df):
    out = pd.DataFrame()

    out["SNPID"] = df["SNP"].astype(str)
    out["CHR"] = df["CHR"].map(normalise_chr)
    out["POS"] = pd.to_numeric(df["BP"], errors="coerce")
    out["EA"] = df["A1"].astype(str).str.upper()
    out["NEA"] = df["A2"].astype(str).str.upper()

    if "BETA" in df.columns:
        out["BETA"] = pd.to_numeric(df["BETA"], errors="coerce")

    if "SE" in df.columns:
        out["SE"] = pd.to_numeric(df["SE"], errors="coerce")

    if "P" in df.columns:
        out["P"] = pd.to_numeric(df["P"], errors="coerce")

    if "N" in df.columns:
        out["N"] = pd.to_numeric(df["N"], errors="coerce")

    if "OR" in df.columns:
        out["OR"] = pd.to_numeric(df["OR"], errors="coerce")

    return out


# =============================================================================
# GWASLAB LOADING / BUILD DETECTION / LIFTOVER
# =============================================================================

def load_into_gwaslab(processed_df, output_dir):
    import gwaslab as gl

    output_dir = Path(output_dir)
    log_file = output_dir / "PRS1_Normalization2_hg38_gwaslab.log"

    tmpdir = tempfile.TemporaryDirectory(prefix="prs2_hg38_")
    tmp_file = Path(tmpdir.name) / "ProcessedGWAS_for_gwaslab.tsv"

    gwaslab_df = processed_to_gwaslab_df(processed_df)
    gwaslab_df.to_csv(tmp_file, sep="\t", index=False)

    log_buffer = io.StringIO()
    tee_stdout = TeeTextIO(sys.stdout, log_buffer)
    tee_stderr = TeeTextIO(sys.stderr, log_buffer)

    print("=" * 100)
    print("[LOAD ProcessedGWAS INTO GWASLab]")
    print(f"Temporary file: {tmp_file}")
    print("=" * 100)

    with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
        sumstats = gl.Sumstats(
            str(tmp_file),
            fmt="gwaslab",
            verbose=True,
        )

    log_file.write_text(log_buffer.getvalue(), encoding="utf-8")

    print("[GWASLab loaded]")
    print(f"GWASLab log: {log_file}")
    print()

    return sumstats, tmpdir, log_file


def parse_build_value(x):
    text = str(x).lower()

    if any(v in text for v in ["38", "hg38", "grch38", "build38"]):
        return "hg38"

    if any(v in text for v in ["19", "37", "hg19", "grch37", "build37"]):
        return "hg19"

    return "unknown"


def detect_build_with_gwaslab(sumstats):
    """
    Try common GWASLab build-detection methods.
    Different GWASLab versions may expose different method names.
    """
    print("=" * 100)
    print("[DETECT BUILD USING GWASLab]")
    print("=" * 100)

    method_names = [
        "infer_build",
        "inferbuild",
        "check_build",
        "checkbuild",
    ]

    for method_name in method_names:
        if hasattr(sumstats, method_name):
            method = getattr(sumstats, method_name)

            attempts = [
                {},
                {"verbose": True},
            ]

            for kwargs in attempts:
                try:
                    print(f"Trying GWASLab method: {method_name}({kwargs})")
                    result = method(**kwargs)

                    build = parse_build_value(result)

                    if build != "unknown":
                        print(f"[BUILD DETECTED] {build} from return value: {result}")
                        return build

                    # Some GWASLab methods update attributes instead of returning value.
                    for attr in ["build", "genome_build", "build_inferred", "build_detected"]:
                        if hasattr(sumstats, attr):
                            attr_value = getattr(sumstats, attr)
                            build = parse_build_value(attr_value)
                            if build != "unknown":
                                print(f"[BUILD DETECTED] {build} from attribute {attr}: {attr_value}")
                                return build

                except Exception as e:
                    print(f"  Failed: {type(e).__name__}: {e}")

    # Last fallback: inspect known attributes.
    for attr in ["build", "genome_build", "build_inferred", "build_detected"]:
        if hasattr(sumstats, attr):
            attr_value = getattr(sumstats, attr)
            build = parse_build_value(attr_value)
            if build != "unknown":
                print(f"[BUILD DETECTED] {build} from attribute {attr}: {attr_value}")
                return build

    print("[BUILD DETECTED] unknown")
    print("=" * 100)

    return "unknown"


def liftover_with_gwaslab(sumstats):
    """
    Try GWASLab liftover from hg19/build37 to hg38.
    GWASLab versions differ, so try several signatures.
    """
    if not hasattr(sumstats, "liftover"):
        raise AttributeError("This GWASLab version does not expose sumstats.liftover().")

    print("=" * 100)
    print("[LIFTOVER hg19/build37 TO hg38 USING GWASLab]")
    print("=" * 100)

    attempts = [
        {"from_build": "19", "to_build": "38"},
        {"from_build": 19, "to_build": 38},
        {"to_build": "38"},
        {"to_build": 38},
        {"build": "38"},
        {"build": 38},
    ]

    last_error = None

    for kwargs in attempts:
        try:
            print(f"Trying liftover kwargs: {kwargs}")
            result = sumstats.liftover(**kwargs)

            if result is not None:
                return result

            return sumstats

        except Exception as e:
            last_error = e
            print(f"  Failed: {type(e).__name__}: {e}")

    raise RuntimeError(f"GWASLab liftover failed. Last error: {last_error}")


def extract_gwaslab_data(sumstats):
    df = getattr(sumstats, "data", None)

    if df is None:
        raise AttributeError("GWASLab Sumstats object does not expose .data dataframe.")

    return df.copy()


def gwaslab_to_processed_schema(gwaslab_df, original_df):
    df = gwaslab_df.copy()

    lower = {str(c).lower(): c for c in df.columns}

    def get_col(*names):
        for n in names:
            if n in df.columns:
                return n
            if n.lower() in lower:
                return lower[n.lower()]
        return ""

    chr_col = get_col("CHR")
    bp_col = get_col("POS", "BP")
    snp_col = get_col("SNPID", "SNP", "rsID")
    a1_col = get_col("EA", "A1")
    a2_col = get_col("NEA", "A2")
    n_col = get_col("N")
    se_col = get_col("SE")
    p_col = get_col("P")
    beta_col = get_col("BETA")
    or_col = get_col("OR")

    out = pd.DataFrame({
        "CHR": df[chr_col].map(normalise_chr) if chr_col else "",
        "BP": pd.to_numeric(df[bp_col], errors="coerce") if bp_col else np.nan,
        "SNP": df[snp_col].astype(str) if snp_col else "",
        "A1": df[a1_col].astype(str).str.upper() if a1_col else "",
        "A2": df[a2_col].astype(str).str.upper() if a2_col else "",
        "N": pd.to_numeric(df[n_col], errors="coerce") if n_col else np.nan,
        "SE": pd.to_numeric(df[se_col], errors="coerce") if se_col else np.nan,
        "P": pd.to_numeric(df[p_col], errors="coerce") if p_col else np.nan,
        "BETA": pd.to_numeric(df[beta_col], errors="coerce") if beta_col else np.nan,
        "OR": pd.to_numeric(df[or_col], errors="coerce") if or_col else np.nan,
    })

    # Keep extra original columns if row counts match.
    if len(original_df) == len(out):
        for col in original_df.columns:
            if col not in out.columns:
                out[col] = original_df[col].values

        ordered = [c for c in original_df.columns if c in out.columns]
        extras = [c for c in out.columns if c not in ordered]
        out = out[ordered + extras]

    return out


# =============================================================================
# MAIN PROCESS
# =============================================================================

def process_index(index, jobs_file, force):
    job = load_job(index, jobs_file)

    phenotype = clean_text(job.get("phenotype", ""))
    accession = clean_text(job.get("accessionId", ""))
    gwas_file = Path(clean_text(job["file_path"]))

    output_dir = get_output_dir(job, gwas_file)
    processed_file = output_dir / "ProcessedGWAS.csv"
    output_file = output_dir / "ProcessedGWAS_hg38.csv"

    print("=" * 100)
    print("ProcessedGWAS TO hg38 BY INDEX USING GWASLab")
    print("=" * 100)
    print(f"[JOB INDEX]    {index}")
    print(f"[PHENOTYPE]    {phenotype}")
    print(f"[ACCESSION]    {accession}")
    print(f"[GWAS FILE]    {gwas_file}")
    print(f"[OUTPUT DIR]   {output_dir}")
    print(f"[INPUT FILE]   {processed_file}")
    print(f"[OUTPUT FILE]  {output_file}")
    print("=" * 100)

    if output_file.exists() and not force:
        print("[SKIP] ProcessedGWAS_hg38.csv already exists.")
        print("Use --force to overwrite.")
        return

    processed_df = load_processed_gwas(processed_file)

    sumstats, tmpdir, log_file = load_into_gwaslab(processed_df, output_dir)

    try:
        detected_build = detect_build_with_gwaslab(sumstats)

        print("=" * 100)
        print("[BUILD DECISION]")
        print(f"Detected build: {detected_build}")
        print("=" * 100)

        if detected_build == "hg38":
            print("[NO LIFTOVER] Already hg38.")
            out_df = processed_df.copy()
            out_df["LIFTOVER_SUCCESS"] = True
            out_df["GWASLAB_DETECTED_BUILD"] = detected_build

        elif detected_build == "hg19":
            lifted_sumstats = liftover_with_gwaslab(sumstats)
            lifted_data = extract_gwaslab_data(lifted_sumstats)
            out_df = gwaslab_to_processed_schema(lifted_data, processed_df)
            out_df["LIFTOVER_SUCCESS"] = True
            out_df["GWASLAB_DETECTED_BUILD"] = detected_build

        else:
            print("=" * 100)
            print("[STOPPED]")
            print("GWASLab could not confidently detect the build.")
            print("I will not guess, because wrong liftover will corrupt coordinates.")
            print("=" * 100)
            raise ValueError("GWASLab could not detect build.")

        # Remove the two liftover-related columns from the final hg38 file
        for col in ["LIFTOVER_SUCCESS", "GWASLAB_DETECTED_BUILD"]:
            if col in out_df.columns:
                out_df = out_df.drop(columns=[col])

        out_df.to_csv(output_file, index=False)

        # Delete the original ProcessedGWAS.csv to keep only the hg38 file
        try:
            if processed_file.exists():
                processed_file.unlink()
                print(f"[REMOVED] Original processed file: {processed_file}")
        except Exception as e:
            print(f"[WARN] Could not remove original processed file: {e}")

        print("=" * 100)
        print("[SAVED]")
        print(f"Output: {output_file}")
        print(f"Rows:   {len(out_df)}")
        print("Header:")
        print(list(out_df.columns))
        print("=" * 100)

    finally:
        tmpdir.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Detect build with GWASLab, then convert ProcessedGWAS.csv to ProcessedGWAS_hg38.csv."
    )

    parser.add_argument(
        "index",
        type=int,
        help="GWAS job index from GWAS_jobs.csv."
    )

    parser.add_argument(
        "--jobs-file",
        default="GWAS_jobs.csv",
        help="Default: GWAS_jobs.csv"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing ProcessedGWAS_hg38.csv."
    )

    args = parser.parse_args()

    process_index(
        index=args.index,
        jobs_file=args.jobs_file,
        force=args.force,
    )


if __name__ == "__main__":
    main()