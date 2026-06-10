#!/usr/bin/env python3

import argparse
import contextlib
import importlib.util
import io
import math
import re
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def load_module2_file_rules():
    module2_path = Path(__file__).with_name("Module2-Search_Poke_Normalize_Scan.py")
    spec = importlib.util.spec_from_file_location("module2_search_poke_normalize_scan", module2_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Module 2 file rules from {module2_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LIKELY_GWAS_EXTENSIONS, module.SKIP_FILE_PATTERNS


LIKELY_GWAS_EXTENSIONS, SKIP_FILE_PATTERNS = load_module2_file_rules()

STANDARD_COLUMNS = [
    "SNPID", "CHR", "POS", "EA", "NEA", "BETA", "OR", "Z", "SE", "P",
    "N", "N_CASES", "N_CONTROLS", "EAF", "MAF", "INFO", "DIRECTION",
]


# =============================================================================
# Basic helpers
# =============================================================================

def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def clean_bool(x):
    if isinstance(x, bool):
        return x
    return clean_text(x).lower() in ["true", "1", "yes", "y"]


def percent(numerator, denominator):
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 4)


def detect_compression(path):
    name = str(path).lower()

    if name.endswith((".tar.gz", ".tgz")):
        return "tar.gz"
    if name.endswith(".tar"):
        return "tar"
    if name.endswith(".zip"):
        return "zip"
    if name.endswith((".gz", ".bgz")):
        return "gzip"

    return "plain"


def get_file_extension(path):
    name = str(path).lower()

    for suffix in [
        ".tar.gz", ".tgz", ".tsv.gz", ".txt.gz", ".csv.gz", ".bgz", ".gz",
        ".sumstats", ".summary", ".assoc", ".linear", ".logistic",
        ".tbl", ".ma", ".meta",
    ]:
        if name.endswith(suffix):
            return suffix

    return Path(path).suffix.lower()


def is_likely_gwas_file(filename):
    name = str(filename).lower()

    if any(pattern in name for pattern in SKIP_FILE_PATTERNS):
        return False

    return name.endswith(LIKELY_GWAS_EXTENSIONS)


def get_series(df, column):
    if column in df.columns:
        return df[column]
    return pd.Series(dtype=object)


def get_text_series(df, column):
    series = get_series(df, column)
    if series.empty:
        return pd.Series(dtype=str)
    return series.astype(str).map(clean_text)


def get_numeric_series(df, column):
    series = get_series(df, column)
    if series.empty:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def is_missing(series):
    if series.empty:
        return pd.Series(dtype=bool)

    text = series.astype(str).str.strip()
    return (
        series.isna()
        | (text == "")
        | text.str.lower().isin(["nan", "na", "n/a", "none", "null", "."])
    )


def missing_percentage(df, column):
    series = get_series(df, column)
    if series.empty:
        return 100.0
    return percent(is_missing(series).sum(), len(series))


def join_counts(counts):
    if not counts:
        return ""
    return " | ".join(f"{key}:{value}" for key, value in counts.items())


def correlation_or_blank(x, y):
    valid = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()

    if len(valid) < 3:
        return ""

    return round(float(valid["x"].corr(valid["y"])), 6)


def build_gwaslab_kwargs():
    return {"fmt": "auto"}


def load_final_gwas_dataframe(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [clean_text(c) for c in df.columns]
    return df.copy(), "FinalGWAS.csv"


# =============================================================================
# Archive handling for GWASLab
# =============================================================================

def select_zip_member(path):
    with zipfile.ZipFile(path) as z:
        members = [m for m in z.infolist() if not m.is_dir()]
        likely = [m for m in members if is_likely_gwas_file(m.filename)]
        choices = likely or members

        if not choices:
            return "", 0

        largest = max(choices, key=lambda x: x.file_size)
        return largest.filename, largest.file_size


def select_tar_member(path):
    mode = "r:gz" if str(path).lower().endswith((".tar.gz", ".tgz")) else "r:"

    with tarfile.open(path, mode) as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        likely = [m for m in members if is_likely_gwas_file(m.name)]
        choices = likely or members

        if not choices:
            return "", 0

        largest = max(choices, key=lambda x: x.size)
        return largest.name, largest.size


def extract_zip_member(path, tmpdir):
    member, _ = select_zip_member(path)
    if not member:
        raise ValueError(f"No usable GWAS member found in ZIP archive: {path}")

    out_path = Path(tmpdir) / Path(member).name
    with zipfile.ZipFile(path) as z:
        with z.open(member) as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

    return out_path, f"zip_member:{member}"


def extract_tar_member(path, tmpdir):
    member, _ = select_tar_member(path)
    if not member:
        raise ValueError(f"No usable GWAS member found in TAR archive: {path}")

    mode = "r:gz" if str(path).lower().endswith((".tar.gz", ".tgz")) else "r:"
    out_path = Path(tmpdir) / Path(member).name

    with tarfile.open(path, mode) as tar:
        src = tar.extractfile(member)
        if src is None:
            raise ValueError(f"Could not extract TAR member for GWASLab: {member}")
        with src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

    return out_path, f"tar_member:{member}"


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


def get_gwaslab_data(sumstats):
    data = getattr(sumstats, "data", None)

    if data is None:
        raise AttributeError("GWASLab Sumstats object does not expose a .data dataframe.")

    return data


def parse_gwaslab_inferred_formats(log_text):
    formats = []
    pattern = re.compile(
        r"(?P<rank>\d+)\.\s+(?P<name>[^\s\[]+)"
        r"(?:\s+\[[^\]]+\])?\s+\(score:\s*(?P<score>[\d\.]+)\)"
    )

    for match in pattern.finditer(log_text):
        formats.append({
            "rank": int(match.group("rank")),
            "name": match.group("name"),
            "score": float(match.group("score")),
        })

    return sorted(formats, key=lambda x: x["rank"])


def parse_gwaslab_build_matches(log_text):
    m19 = re.search(r"Matching variants for hg19:\s*num_hg19\s*=\s*([\d,]+)", log_text)
    m38 = re.search(r"Matching variants for hg38:\s*num_hg38\s*=\s*([\d,]+)", log_text)

    hg19 = int(m19.group(1).replace(",", "")) if m19 else ""
    hg38 = int(m38.group(1).replace(",", "")) if m38 else ""
    return hg19, hg38


def calculate_build_match_ratio(hg19_count, hg38_count):
    if hg19_count == "" or hg38_count == "":
        return ""

    denominator = max(int(hg19_count), int(hg38_count), 1)
    numerator = min(int(hg19_count), int(hg38_count))
    return round(numerator / denominator, 6)


def normalise_gwaslab_build(build):
    text = clean_text(build)

    if text in ["19", "hg19", "HG19", "GRCh37", "grch37"]:
        return "GRCh37"
    if text in ["38", "hg38", "HG38", "GRCh38", "grch38"]:
        return "GRCh38"
    if text in ["", "99", "unknown", "Unknown", "UNKNOWN"]:
        return "unknown"

    return text


def load_gwaslab_sumstats(logical_path, log_file):
    import gwaslab as gl

    kwargs = build_gwaslab_kwargs()
    log_buffer = io.StringIO()
    tee_stdout = TeeTextIO(sys.stdout, log_buffer)
    tee_stderr = TeeTextIO(sys.stderr, log_buffer)

    with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
        sumstats = gl.Sumstats(str(logical_path), **kwargs)

    log_file.write_text(log_buffer.getvalue(), encoding="utf-8")
    return sumstats, log_buffer.getvalue(), kwargs


def load_gwaslab_for_file(file_path, output_dir):
    compression = detect_compression(file_path)
    log_file = output_dir / "Feature5_gwaslab.log"

    if compression == "zip":
        tmpdir = tempfile.TemporaryDirectory(prefix="variation5_gwaslab_")
        extracted_path, source = extract_zip_member(file_path, tmpdir.name)
        sumstats, log_text, kwargs = load_gwaslab_sumstats(extracted_path, log_file)
        return sumstats, log_text, kwargs, source, tmpdir

    if compression in ["tar", "tar.gz"]:
        tmpdir = tempfile.TemporaryDirectory(prefix="variation5_gwaslab_")
        extracted_path, source = extract_tar_member(file_path, tmpdir.name)
        sumstats, log_text, kwargs = load_gwaslab_sumstats(extracted_path, log_file)
        return sumstats, log_text, kwargs, source, tmpdir

    sumstats, log_text, kwargs = load_gwaslab_sumstats(file_path, log_file)
    return sumstats, log_text, kwargs, "direct_file", None


def infer_build_with_log(sumstats, log_file):
    log_buffer = io.StringIO()
    tee_stdout = TeeTextIO(sys.stdout, log_buffer)
    tee_stderr = TeeTextIO(sys.stderr, log_buffer)

    with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
        sumstats.infer_build()

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_buffer.getvalue())

    return log_buffer.getvalue()


def basic_check_with_log(sumstats, log_file):
    log_buffer = io.StringIO()
    tee_stdout = TeeTextIO(sys.stdout, log_buffer)
    tee_stderr = TeeTextIO(sys.stderr, log_buffer)

    with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
        sumstats.basic_check(remove=True, remove_dup=True)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_buffer.getvalue())

    return log_buffer.getvalue()


# =============================================================================
# Feature groups
# =============================================================================

def analyse_standard_columns(df):
    loaded = [col for col in STANDARD_COLUMNS if col in df.columns]

    result = {
        "final_gwas_loaded_variant_count": len(df),
        "final_gwas_dataframe_memory_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 3),
        "final_gwas_standard_columns_loaded": " | ".join(loaded),
        "final_gwas_standard_column_count": len(loaded),
    }

    for col in STANDARD_COLUMNS:
        result[f"has_{col}"] = col in df.columns

    return result


def analyse_chromosome_coverage(df):
    result = {
        "missing_chr_percentage": missing_percentage(df, "CHR"),
        "missing_pos_percentage": missing_percentage(df, "POS"),
        "chr_numeric_valid_percentage": 0.0,
        "pos_numeric_valid_percentage": 0.0,
        "invalid_chromosome_percentage": 0.0,
        "non_autosomal_percentage": 0.0,
        "autosomal_variant_count": 0,
        "chrX_variant_count": 0,
        "chrY_variant_count": 0,
        "chrMT_variant_count": 0,
        "missing_autosome_count": "",
        "chromosome_count_signature": "",
    }

    if "CHR" not in df.columns:
        return result

    chr_text = get_text_series(df, "CHR").str.replace("^chr", "", case=False, regex=True).str.upper()
    chr_nonmissing = chr_text[~is_missing(chr_text)]

    if chr_nonmissing.empty:
        return result

    canonical = set([str(i) for i in range(1, 23)] + ["X", "Y", "XY", "M", "MT"])
    autosomes = set(str(i) for i in range(1, 23))

    counts = chr_nonmissing.value_counts(dropna=False).sort_index().to_dict()
    result["chromosome_count_signature"] = join_counts(counts)
    result["invalid_chromosome_percentage"] = percent((~chr_nonmissing.isin(canonical)).sum(), len(chr_nonmissing))
    result["non_autosomal_percentage"] = percent((~chr_nonmissing.isin(autosomes)).sum(), len(chr_nonmissing))
    result["autosomal_variant_count"] = int(chr_nonmissing.isin(autosomes).sum())
    result["chrX_variant_count"] = int(chr_nonmissing.isin(["X", "XY"]).sum())
    result["chrY_variant_count"] = int((chr_nonmissing == "Y").sum())
    result["chrMT_variant_count"] = int(chr_nonmissing.isin(["M", "MT"]).sum())

    present_autosomes = set(chr_nonmissing[chr_nonmissing.isin(autosomes)].unique())
    result["missing_autosome_count"] = 22 - len(present_autosomes)

    chr_numeric = pd.to_numeric(chr_nonmissing, errors="coerce")
    result["chr_numeric_valid_percentage"] = percent(chr_numeric.notna().sum(), len(chr_nonmissing))

    if "POS" in df.columns:
        pos = get_numeric_series(df, "POS")
        non_missing_pos = pos.dropna()
        result["pos_numeric_valid_percentage"] = percent(pos.notna().sum(), len(pos))
        result["nonpositive_pos_percentage"] = percent((non_missing_pos <= 0).sum(), len(non_missing_pos))
    else:
        result["nonpositive_pos_percentage"] = 0.0

    return result


def analyse_variant_identity(df):
    result = {
        "missing_snpid_percentage": missing_percentage(df, "SNPID"),
        "duplicated_snpid_count": "",
        "duplicated_snpid_percentage": 0.0,
        "duplicated_chr_pos_count": "",
        "duplicated_chr_pos_percentage": 0.0,
        "duplicated_chr_pos_ea_nea_count": "",
        "duplicated_chr_pos_ea_nea_percentage": 0.0,
    }

    if "SNPID" in df.columns:
        snp = get_text_series(df, "SNPID")
        valid = snp[~is_missing(snp)]
        duplicated = valid.duplicated(keep=False).sum()
        result["duplicated_snpid_count"] = int(duplicated)
        result["duplicated_snpid_percentage"] = percent(duplicated, len(valid))

    if {"CHR", "POS"}.issubset(df.columns):
        valid = df[["CHR", "POS"]].dropna()
        duplicated = valid.astype(str).duplicated(keep=False).sum()
        result["duplicated_chr_pos_count"] = int(duplicated)
        result["duplicated_chr_pos_percentage"] = percent(duplicated, len(valid))

    if {"CHR", "POS", "EA", "NEA"}.issubset(df.columns):
        valid = df[["CHR", "POS", "EA", "NEA"]].dropna()
        duplicated = valid.astype(str).duplicated(keep=False).sum()
        result["duplicated_chr_pos_ea_nea_count"] = int(duplicated)
        result["duplicated_chr_pos_ea_nea_percentage"] = percent(duplicated, len(valid))

    return result


def analyse_alleles_and_variant_types(df):
    result = {
        "missing_ea_percentage": missing_percentage(df, "EA"),
        "missing_nea_percentage": missing_percentage(df, "NEA"),
        "single_base_snp_percentage": 0.0,
        "indel_percentage": 0.0,
        "multi_character_allele_percentage": 0.0,
        "non_acgt_allele_percentage": 0.0,
        "symbolic_allele_percentage": 0.0,
        "palindromic_snp_percentage": 0.0,
    }

    if not {"EA", "NEA"}.issubset(df.columns):
        return result

    alleles = pd.DataFrame({
        "ea": get_text_series(df, "EA").str.upper(),
        "nea": get_text_series(df, "NEA").str.upper(),
    })
    alleles = alleles[
        (~is_missing(alleles["ea"]))
        & (~is_missing(alleles["nea"]))
    ]

    if alleles.empty:
        return result

    all_alleles = pd.concat([alleles["ea"], alleles["nea"]])
    result["multi_character_allele_percentage"] = percent((all_alleles.str.len() > 1).sum(), len(all_alleles))
    result["non_acgt_allele_percentage"] = percent(
        (~all_alleles.str.match(r"^[ACGT]+$", na=False)).sum(), len(all_alleles)
    )
    result["symbolic_allele_percentage"] = percent(
        all_alleles.str.contains(r"INS|DEL|INSERTION|DELETION|<|>|-", regex=True, na=False).sum(),
        len(all_alleles),
    )

    single_base = (
        alleles["ea"].str.match(r"^[ACGT]$", na=False)
        & alleles["nea"].str.match(r"^[ACGT]$", na=False)
    )
    result["single_base_snp_percentage"] = percent(single_base.sum(), len(alleles))

    symbolic_indel = (
        alleles["ea"].str.contains(r"INS|DEL|INSERTION|DELETION|<|>|-", regex=True, na=False)
        | alleles["nea"].str.contains(r"INS|DEL|INSERTION|DELETION|<|>|-", regex=True, na=False)
    )
    length_indel = alleles["ea"].str.len() != alleles["nea"].str.len()
    result["indel_percentage"] = percent((symbolic_indel | length_indel).sum(), len(alleles))

    single_base_alleles = alleles[single_base]
    if not single_base_alleles.empty:
        pairs = single_base_alleles["ea"] + "/" + single_base_alleles["nea"]
        result["palindromic_snp_percentage"] = percent(
            pairs.isin(["A/T", "T/A", "C/G", "G/C"]).sum(),
            len(single_base_alleles),
        )

    return result


def analyse_pvalues(df):
    result = {
        "missing_p_percentage": missing_percentage(df, "P"),
        "p_numeric_percentage": 0.0,
        "p_out_of_range_percentage": 0.0,
        "p_zero_count": 0,
        "p_equal_one_count": 0,
        "min_p": "",
        "median_p": "",
        "p_scientific_notation_percentage": 0.0,
        "genome_wide_significant_count": 0,
        "genome_wide_significant_percentage": 0.0,
        "suggestive_significant_count": 0,
        "suggestive_significant_percentage": 0.0,
    }

    if "P" not in df.columns:
        return result

    p_raw = get_text_series(df, "P")
    p_numeric = pd.to_numeric(p_raw, errors="coerce")
    result["p_numeric_percentage"] = percent(p_numeric.notna().sum(), len(p_raw))
    result["p_scientific_notation_percentage"] = percent(
        p_raw.str.contains(r"e[-+]?\d+", case=False, regex=True, na=False).sum(),
        len(p_raw),
    )

    p_valid = p_numeric.dropna()
    if p_valid.empty:
        return result

    result["p_out_of_range_percentage"] = percent(((p_valid < 0) | (p_valid > 1)).sum(), len(p_valid))
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


def analyse_frequency_and_info(df):
    result = {
        "missing_eaf_percentage": missing_percentage(df, "EAF"),
        "missing_maf_percentage": missing_percentage(df, "MAF"),
        "missing_info_percentage": missing_percentage(df, "INFO"),
        "eaf_out_of_range_percentage": 0.0,
        "maf_out_of_range_percentage": 0.0,
        "info_out_of_range_percentage": 0.0,
        "median_eaf": "",
        "median_maf": "",
        "median_info": "",
        "rare_variant_percentage": 0.0,
        "low_frequency_variant_percentage": 0.0,
        "common_variant_percentage": 0.0,
        "info_below_0_3_percentage": 0.0,
        "info_below_0_8_percentage": 0.0,
    }

    derived_maf = pd.Series(dtype=float)

    if "EAF" in df.columns:
        eaf = get_numeric_series(df, "EAF").dropna()
        if not eaf.empty:
            result["eaf_out_of_range_percentage"] = percent(((eaf < 0) | (eaf > 1)).sum(), len(eaf))
            eaf_in_range = eaf[(eaf >= 0) & (eaf <= 1)]
            if not eaf_in_range.empty:
                result["median_eaf"] = round(float(eaf_in_range.median()), 4)
                derived_maf = eaf_in_range.map(lambda x: min(x, 1 - x))

    if "MAF" in df.columns:
        maf = get_numeric_series(df, "MAF").dropna()
        if not maf.empty:
            result["maf_out_of_range_percentage"] = percent(((maf < 0) | (maf > 0.5)).sum(), len(maf))
            maf_in_range = maf[(maf >= 0) & (maf <= 0.5)]
            if not maf_in_range.empty:
                result["median_maf"] = round(float(maf_in_range.median()), 4)
                derived_maf = maf_in_range

    if not derived_maf.empty:
        result["rare_variant_percentage"] = percent((derived_maf < 0.01).sum(), len(derived_maf))
        result["low_frequency_variant_percentage"] = percent(
            ((derived_maf >= 0.01) & (derived_maf < 0.05)).sum(), len(derived_maf)
        )
        result["common_variant_percentage"] = percent((derived_maf >= 0.05).sum(), len(derived_maf))

    if "INFO" in df.columns:
        info = get_numeric_series(df, "INFO").dropna()
        if not info.empty:
            result["info_out_of_range_percentage"] = percent(((info < 0) | (info > 1)).sum(), len(info))
            info_in_range = info[(info >= 0) & (info <= 1)]
            if not info_in_range.empty:
                result["median_info"] = round(float(info_in_range.median()), 4)
                result["info_below_0_3_percentage"] = percent((info_in_range < 0.3).sum(), len(info_in_range))
                result["info_below_0_8_percentage"] = percent((info_in_range < 0.8).sum(), len(info_in_range))

    return result


def analyse_sample_size(df):
    result = {
        "missing_N_percentage": missing_percentage(df, "N"),
        "has_total_N": "N" in df.columns,
        "has_N_cases": "N_CASES" in df.columns,
        "has_N_controls": "N_CONTROLS" in df.columns,
        "N_constant_or_variable": "not_available",
        "median_N": "",
        "min_N": "",
        "max_N": "",
        "N_variability_ratio": "",
        "median_N_cases": "",
        "median_N_controls": "",
    }

    if "N" in df.columns:
        n = get_numeric_series(df, "N").dropna()
        if not n.empty:
            result["median_N"] = round(float(n.median()), 0)
            result["min_N"] = int(n.min())
            result["max_N"] = int(n.max())
            if n.max() > 0:
                ratio = (n.max() - n.min()) / n.max()
                result["N_variability_ratio"] = round(float(ratio), 6)
                result["N_constant_or_variable"] = "constant" if ratio < 0.001 else "variable"

    if "N_CASES" in df.columns:
        n_cases = get_numeric_series(df, "N_CASES").dropna()
        if not n_cases.empty:
            result["median_N_cases"] = round(float(n_cases.median()), 0)

    if "N_CONTROLS" in df.columns:
        n_controls = get_numeric_series(df, "N_CONTROLS").dropna()
        if not n_controls.empty:
            result["median_N_controls"] = round(float(n_controls.median()), 0)

    return result


def analyse_effects_and_consistency(df):
    result = {
        "effect_column_type": "missing_effect_column",
        "has_beta_and_or": bool("BETA" in df.columns and "OR" in df.columns),
        "or_nonpositive_percentage": 0.0,
        "or_extreme_percentage": 0.0,
        "beta_extreme_abs_gt_10_percentage": 0.0,
        "beta_se_to_z_correlation": "",
        "beta_se_z_concordance_percentage": "",
        "p_z_correlation": "",
        "p_z_concordance_percentage": "",
    }

    if "BETA" in df.columns:
        result["effect_column_type"] = "BETA"
    elif "OR" in df.columns:
        result["effect_column_type"] = "OR_requires_log_conversion"
    elif "Z" in df.columns:
        result["effect_column_type"] = "Z"

    if "OR" in df.columns:
        or_values = get_numeric_series(df, "OR").dropna()
        if not or_values.empty:
            result["or_nonpositive_percentage"] = percent((or_values <= 0).sum(), len(or_values))
            result["or_extreme_percentage"] = percent(((or_values > 100) | (or_values < 0.01)).sum(), len(or_values))

    if "BETA" in df.columns:
        beta = get_numeric_series(df, "BETA").dropna()
        if not beta.empty:
            result["beta_extreme_abs_gt_10_percentage"] = percent((beta.abs() > 10).sum(), len(beta))

    if {"BETA", "SE", "Z"}.issubset(df.columns):
        beta = get_numeric_series(df, "BETA")
        se = get_numeric_series(df, "SE")
        z = get_numeric_series(df, "Z")
        z_from_beta = beta / se.replace(0, np.nan)
        valid = pd.DataFrame({"observed_z": z, "calculated_z": z_from_beta}).replace([np.inf, -np.inf], np.nan).dropna()
        result["beta_se_to_z_correlation"] = correlation_or_blank(valid["observed_z"], valid["calculated_z"])
        if not valid.empty:
            result["beta_se_z_concordance_percentage"] = percent(
                (valid["observed_z"] - valid["calculated_z"]).abs().le(0.01).sum(),
                len(valid),
            )

    z_for_p = pd.Series(dtype=float)
    if "Z" in df.columns:
        z_for_p = get_numeric_series(df, "Z")
    elif {"BETA", "SE"}.issubset(df.columns):
        z_for_p = get_numeric_series(df, "BETA") / get_numeric_series(df, "SE").replace(0, np.nan)

    if "P" in df.columns and not z_for_p.empty:
        observed_p = get_numeric_series(df, "P")
        valid = pd.DataFrame({"p": observed_p, "z": z_for_p}).replace([np.inf, -np.inf], np.nan).dropna()
        valid = valid[(valid["p"] >= 0) & (valid["p"] <= 1)]

        if not valid.empty:
            calculated_p = valid["z"].map(lambda z: math.erfc(abs(z) / math.sqrt(2)))
            observed_logp = -np.log10(valid["p"].replace(0, np.nan))
            calculated_logp = -np.log10(calculated_p.replace(0, np.nan))
            result["p_z_correlation"] = correlation_or_blank(observed_logp, calculated_logp)
            concordant = (observed_logp - calculated_logp).abs().le(0.05)
            result["p_z_concordance_percentage"] = percent(concordant.sum(), concordant.notna().sum())

    return result


def analyse_readiness(df):
    columns = set(df.columns)
    has_effect = bool({"BETA", "OR", "Z"} & columns)

    gwas_ssf_mandatory = bool(
        {"CHR", "POS", "EA", "NEA", "SE", "P", "EAF"}.issubset(columns)
        and bool({"BETA", "OR"} & columns)
    )
    prs_minimum = bool(
        ("SNPID" in columns or {"CHR", "POS"}.issubset(columns))
        and "EA" in columns
        and "P" in columns
        and has_effect
    )
    ldsc_ready = bool(
        "SNPID" in columns
        and {"EA", "NEA", "N"}.issubset(columns)
        and ("Z" in columns or "P" in columns or {"BETA", "SE"}.issubset(columns))
    )
    plink_ready = bool(
        ("SNPID" in columns or {"CHR", "POS"}.issubset(columns))
        and "EA" in columns
        and "P" in columns
        and has_effect
    )

    return {
        "gwas_ssf_mandatory_columns_present": gwas_ssf_mandatory,
        "prs_minimum_columns_present": prs_minimum,
        "ldsc_minimum_columns_present": ldsc_ready,
        "plink_minimum_columns_present": plink_ready,
    }


def calculate_all_features(df):
    row = {}
    row.update(analyse_standard_columns(df))
    row.update(analyse_chromosome_coverage(df))
    row.update(analyse_variant_identity(df))
    row.update(analyse_alleles_and_variant_types(df))
    row.update(analyse_pvalues(df))
    row.update(analyse_frequency_and_info(df))
    row.update(analyse_sample_size(df))
    row.update(analyse_effects_and_consistency(df))
    row.update(analyse_readiness(df))
    return row


# =============================================================================
# Main profiling
# =============================================================================

def profile_variation5(job_index, phenotype, accession, file_path, final_gwas_file, output_feature_file):
    file_path = Path(file_path)
    final_gwas_file = Path(final_gwas_file)
    output_feature_file = Path(output_feature_file)
    output_dir = output_feature_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_size_bytes = file_path.stat().st_size if file_path.exists() else 0
    compression = detect_compression(file_path)
    final_gwas_size_bytes = final_gwas_file.stat().st_size if final_gwas_file.exists() else 0
    loaded_df, load_source = load_final_gwas_dataframe(final_gwas_file)
    features = calculate_all_features(loaded_df)

    row = {
        "job_index": job_index,
        "phenotype": phenotype,
        "accessionId": accession,
        "file_path": str(file_path),
        "file_name": file_path.name,
        "final_gwas_file": str(final_gwas_file),
        "final_gwas_name": final_gwas_file.name,
        "file_extension": get_file_extension(file_path),
        "compression_type": compression,
        "raw_file_size_bytes": raw_size_bytes,
        "raw_file_size_mb": round(raw_size_bytes / (1024 ** 2), 3),
        "final_gwas_size_bytes": final_gwas_size_bytes,
        "final_gwas_size_mb": round(final_gwas_size_bytes / (1024 ** 2), 3),
        "final_gwas_load_source": load_source,
        "genome_build": "GRCh38",
    }

    row.update(features)
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
    output_feature_file = base_output.parent / "Feature5.csv"

    print("=" * 100)
    print("VARIATION 5: FINALGWAS VARIATION AND REPORTING READINESS")
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
        print(f"[SKIP] Feature5.csv already exists: {output_feature_file}")
        return

    row = profile_variation5(
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
    print(f"loaded_variants:              {row['final_gwas_loaded_variant_count']}")
    print(f"genome_build:                 {row['genome_build']}")
    print(f"standard_columns_loaded:      {row['final_gwas_standard_columns_loaded']}")
    print(f"GWAS-SSF mandatory present:   {row['gwas_ssf_mandatory_columns_present']}")
    print("=" * 100)


def combine_feature5(jobs_file, output_file):
    jobs = pd.read_csv(jobs_file)
    rows = []

    for _, job in jobs.iterrows():
        base_output = Path(clean_text(job["output_feature_file"]))
        feature5 = base_output.parent / "Feature5.csv"
        if feature5.exists():
            rows.append(pd.read_csv(feature5))

    out_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out_df.to_csv(output_file, index=False)

    print("=" * 100)
    print("[COMBINED Feature5.csv FILES]")
    print(f"[FILES COMBINED] {len(rows)}")
    print(f"[OUTPUT] {output_file}")
    print("=" * 100)

    if not out_df.empty:
        for col in [
            "genome_build",
            "gwas_ssf_mandatory_columns_present",
            "prs_minimum_columns_present",
            "effect_column_type",
        ]:
            if col in out_df.columns:
                print()
                print(col)
                print(out_df[col].value_counts(dropna=False).to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Variation5: features computed from FinalGWAS.csv for GWAS variation and reporting readiness."
    )
    parser.add_argument("index", nargs="?", type=int, help="GWAS job index from GWAS_jobs.csv")
    parser.add_argument(
        "--jobs-file",
        default="GWAS_jobs.csv",
        help="GWAS job manifest. Default: GWAS_jobs.csv",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing Feature5.csv")
    parser.add_argument("--combine", action="store_true", help="Combine all per-GWAS Feature5.csv files.")
    parser.add_argument(
        "--combine-output",
        default="Variation5_allGWAS.csv",
        help="Output CSV for --combine. Default: Variation5_allGWAS.csv",
    )

    args = parser.parse_args()

    if args.combine:
        combine_feature5(args.jobs_file, args.combine_output)
        return

    if args.index is None:
        parser.error("index is required unless --combine is used")

    run_one_job(args.index, args.jobs_file, args.force)


if __name__ == "__main__":
    main()
