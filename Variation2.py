#!/usr/bin/env python3

import argparse
import gzip
import hashlib
import importlib.util
import math
import os
import re
import tarfile
import zipfile
from pathlib import Path

import pandas as pd


def load_module2_column_synonyms():
    module2_path = Path(__file__).with_name("Module2-Search_Poke_Normalize_Scan.py")
    spec = importlib.util.spec_from_file_location("module2_search_poke_normalize_scan", module2_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Module 2 mapping from {module2_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.COLUMN_SYNONYMS


# Use the exact same GWAS header mapping dictionary as Module 2.
COLUMN_SYNONYMS = load_module2_column_synonyms()


STANDARD_COLUMNS_ORDER = [
    "SNP", "CHR", "BP", "A1", "A2", "BETA", "OR", "Z",
    "SE", "P", "N", "N_CASES", "N_CONTROLS", "EAF", "MAF",
    "INFO", "DIRECTION"
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
    text = clean_text(x).lower()
    return text in ["true", "1", "yes", "y"]


def normalise_header_name(x):
    x = clean_text(x).lower()
    x = x.replace("\ufeff", "")
    x = x.strip()
    x = x.replace("/", "_")
    x = x.replace(":", "_")
    x = x.replace("-", "_")
    x = x.replace(".", "_")
    x = x.replace("[", "")
    x = x.replace("]", "")
    x = re.sub(r"\s+", "_", x)
    x = re.sub(r"__+", "_", x)
    return x.strip("_")


def build_synonym_lookup():
    lookup = {}

    for standard_col, synonyms in COLUMN_SYNONYMS.items():
        for synonym in synonyms:
            norm = normalise_header_name(synonym)
            if norm and norm not in lookup:
                lookup[norm] = standard_col

    return lookup


SYNONYM_LOOKUP = build_synonym_lookup()


def percent(numerator, denominator):
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 4)


def sha1_text(text):
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def split_line(line, delimiter):
    if delimiter == "tab":
        return line.split("\t")
    if delimiter == "comma":
        return line.split(",")
    if delimiter == "semicolon":
        return line.split(";")
    if delimiter == "pipe":
        return line.split("|")
    if delimiter == "space":
        return re.split(r"\s+", line.strip())

    return [line]


def delimiter_to_sep(delimiter):
    if delimiter == "tab":
        return "\t"
    if delimiter == "comma":
        return ","
    if delimiter == "semicolon":
        return ";"
    if delimiter == "pipe":
        return "|"
    if delimiter == "space":
        return r"\s+"

    return None


def map_headers(raw_headers):
    normalised_headers = [normalise_header_name(h) for h in raw_headers]

    mapped = {}
    unmapped = []
    mapped_standard_headers = []

    for raw, norm in zip(raw_headers, normalised_headers):
        standard = SYNONYM_LOOKUP.get(norm, "")

        if standard:
            if standard not in mapped:
                mapped[standard] = []
            mapped[standard].append(raw)
            mapped_standard_headers.append(standard)
        else:
            unmapped.append(raw)
            mapped_standard_headers.append("UNMAPPED")

    return mapped, unmapped, normalised_headers, mapped_standard_headers


def first_mapped_column(mapped, standard_col):
    values = mapped.get(standard_col, [])
    if values:
        return values[0]
    return ""


def split_joined_columns(value):
    text = clean_text(value)
    if not text:
        return []
    return [part.strip() for part in text.split("|") if part.strip()]


def load_feature1_metadata(output_feature_file):
    feature1_path = Path(output_feature_file).parent / "Feature1.csv"

    if not feature1_path.exists():
        raise FileNotFoundError(
            f"Missing Feature1 metadata: {feature1_path}. Run Variation1 first."
        )

    feature1 = pd.read_csv(feature1_path)

    if feature1.empty:
        raise ValueError(f"Feature1 metadata is empty: {feature1_path}")

    row = feature1.iloc[0]
    raw_headers = split_joined_columns(row.get("raw_headers_joined", ""))

    return {
        "feature1_file": str(feature1_path),
        "file_extension": clean_text(row.get("file_extension", "")),
        "compression_type": clean_text(row.get("compression_type", "")),
        "compression_from_extension": clean_text(row.get("compression_from_extension", "")),
        "compression_from_magic_bytes": clean_text(row.get("compression_from_magic_bytes", "")),
        "compression_detection_source": clean_text(row.get("compression_detection_source", "")),
        "compression_suffix_magic_mismatch": clean_bool(row.get("compression_suffix_magic_mismatch", False)),
        "reader_used": clean_text(row.get("reader_used", "")),
        "raw_file_size_bytes": row.get("raw_file_size_bytes", 0),
        "raw_file_size_mb": row.get("raw_file_size_mb", 0),
        "extracted_member": clean_text(row.get("extracted_member", "")),
        "extracted_file_size_bytes": row.get("extracted_file_size_bytes", 0),
        "extracted_file_size_mb": row.get("extracted_file_size_mb", 0),
        "delimiter": clean_text(row.get("delimiter", "")),
        "estimated_n_columns": row.get("n_columns", len(raw_headers)),
        "header_detected": clean_bool(row.get("header_detected", False)),
        "header_row_index": clean_text(row.get("header_row_index", "")),
        "comment_or_metadata_lines_before_header": clean_text(row.get("comment_or_metadata_lines_before_header", "")),
        "raw_headers": raw_headers,
        "raw_headers_joined": clean_text(row.get("raw_headers_joined", "")),
        "n_columns": row.get("n_columns", len(raw_headers)),
        "read_status": clean_text(row.get("read_status", "")),
        "encoding_used": clean_text(row.get("encoding_used", "")),
        "encoding_issues": clean_text(row.get("encoding_issues", "")),
        "quote_usage": clean_bool(row.get("quote_usage", False)),
        "n_rows_physical_lines": row.get("n_rows_physical_lines", ""),
        "n_sample_lines_read": row.get("n_sample_lines_read", 0),
    }


# =============================================================================
# Pandas complete-file loading
# =============================================================================

def open_for_pandas(path, compression_type, extracted_member=""):
    compression = clean_text(compression_type)
    extracted_member = clean_text(extracted_member)

    if compression == "plain":
        return open(path, "rt", encoding="utf-8", errors="replace"), ""

    if compression == "gzip":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace"), ""

    if compression == "zip":
        if not extracted_member:
            return None, "zip_empty"

        z = zipfile.ZipFile(path)
        raw = z.open(extracted_member)
        text = pd.io.common.TextIOWrapper(raw, encoding="utf-8", errors="replace")
        return _ClosableWrapper(text, [text, raw, z]), ""

    if compression in ["tar", "tar.gz"]:
        if not extracted_member:
            return None, "tar_empty"

        mode = "r:*"
        tar = tarfile.open(path, mode)
        raw = tar.extractfile(extracted_member)

        if raw is None:
            tar.close()
            return None, "tar_extract_failed"

        text = pd.io.common.TextIOWrapper(raw, encoding="utf-8", errors="replace")
        return _ClosableWrapper(text, [text, raw, tar]), ""

    return None, "unsupported_compression"


class _ClosableWrapper:
    def __init__(self, handle, closables):
        self.handle = handle
        self.closables = closables

    def __getattr__(self, item):
        return getattr(self.handle, item)

    def close(self):
        for obj in self.closables:
            try:
                obj.close()
            except Exception:
                pass


def read_complete_dataframe(path, feature1):
    delimiter = feature1["delimiter"]
    header_row_index = feature1["header_row_index"]
    sep = delimiter_to_sep(delimiter)

    if sep is None or header_row_index == "":
        return pd.DataFrame(), "cannot_read_dataframe_no_sep_or_header"

    handle, issue = open_for_pandas(
        path=path,
        compression_type=feature1["compression_type"],
        extracted_member=feature1["extracted_member"],
    )

    if handle is None:
        return pd.DataFrame(), issue

    try:
        df = pd.read_csv(
            handle,
            sep=sep,
            header=0,
            skiprows=int(float(header_row_index)),
            dtype=str,
            engine="python",
            on_bad_lines="skip",
        )

        df.columns = [clean_text(c) for c in df.columns]
        return df, ""

    except Exception as e:
        return pd.DataFrame(), f"{type(e).__name__}: {e}"

    finally:
        try:
            handle.close()
        except Exception:
            pass


# =============================================================================
# Format classification helpers
# =============================================================================

def get_series(df, col):
    if col and col in df.columns:
        return df[col].astype(str).map(clean_text)
    return pd.Series(dtype=str)


def missing_percentage(series):
    if series.empty:
        return 100.0

    missing = (
        series.isna()
        | (series.astype(str).str.strip() == "")
        | (series.astype(str).str.lower().isin(["nan", "na", "n/a", "none", "null", "."]))
    )

    return percent(missing.sum(), len(series))


def classify_chr_format(series):
    if series.empty:
        return "missing_column"

    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]

    if values.empty:
        return "all_missing"

    chr_prefix = values.str.match(r"^chr[0-9xyXYmMtT]+$", na=False).mean()
    numeric = values.str.match(r"^[0-9]+$", na=False).mean()
    special = values.str.match(r"^(X|Y|XY|MT|M)$", case=False, na=False).mean()

    if chr_prefix >= 0.8:
        return "chr_prefix_chr1_chr2"
    if numeric >= 0.8:
        return "numeric_1_2_3"
    if special >= 0.5:
        return "special_X_Y_XY_MT"
    return "mixed_or_other"


def classify_snp_format(series):
    if series.empty:
        return "missing_column"

    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]

    if values.empty:
        return "all_missing"

    rsid = values.str.match(r"^rs\d+$", case=False, na=False).mean()
    chr_pos = values.str.match(r"^(chr)?[0-9XYMTxy]+[:_][0-9]+$", na=False).mean()
    chr_pos_alleles = values.str.match(
        r"^(chr)?[0-9XYMTxy]+[:_][0-9]+[:_][ACGT]+[:_][ACGT]+$",
        case=False,
        na=False,
    ).mean()

    if rsid >= 0.8:
        return "rsID"
    if chr_pos_alleles >= 0.8:
        return "chr_pos_alleles"
    if chr_pos >= 0.8:
        return "chr_pos"
    return "mixed_or_other"


def classify_bp_format(series):
    if series.empty:
        return "missing_column"

    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]

    if values.empty:
        return "all_missing"

    numeric = values.str.match(r"^\d+$", na=False).mean()

    if numeric >= 0.95:
        return "integer"
    if numeric >= 0.5:
        return "mostly_integer_with_non_numeric"
    return "missing_or_non_numeric"


def classify_allele_format(series):
    if series.empty:
        return "missing_column"

    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]

    if values.empty:
        return "all_missing"

    single_upper = values.str.match(r"^[ACGT]$", na=False).mean()
    single_lower = values.str.match(r"^[acgt]$", na=False).mean()
    multi = values.str.match(r"^[ACGTacgt]{2,}$", na=False).mean()
    indel = values.str.contains(r"ins|del|insertion|deletion|-", case=False, regex=True, na=False).mean()

    if single_upper >= 0.8:
        return "single_base_uppercase_A_C_G_T"
    if single_lower >= 0.8:
        return "single_base_lowercase_a_c_g_t"
    if indel >= 0.2:
        return "contains_indels"
    if multi >= 0.2:
        return "multi_character_alleles"
    return "mixed_or_invalid"


def classify_numeric_range(series):
    if series.empty:
        return "missing_column"

    x = pd.to_numeric(series, errors="coerce")
    x = x.dropna()

    if x.empty:
        return "non_numeric_or_missing"

    min_v = x.min()
    max_v = x.max()

    if min_v >= 0 and max_v <= 1:
        return "range_0_1"

    if min_v >= 0 and max_v <= 100:
        return "range_0_100"

    return "outside_expected_range"


def classify_pvalue_format(series):
    if series.empty:
        return "missing_column"

    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]

    if values.empty:
        return "all_missing"

    scientific = values.str.contains(r"e[-+]?\d+", case=False, regex=True, na=False).mean()
    numeric = pd.to_numeric(values, errors="coerce")
    numeric_valid = numeric.notna().mean()

    if numeric_valid == 0:
        return "non_numeric"

    numeric_nonmissing = numeric.dropna()

    if not numeric_nonmissing.empty:
        if numeric_nonmissing.min() >= 0 and numeric_nonmissing.max() <= 1:
            if scientific >= 0.5:
                return "scientific_notation_0_1"
            return "decimal_0_1"

        if numeric_nonmissing.min() >= 0 and numeric_nonmissing.max() > 1:
            return "possible_neg_log10_p"

    return "mixed_or_other"


def classify_effect_type(mapped):
    has_beta = bool(mapped.get("BETA"))
    has_or = bool(mapped.get("OR"))
    has_z = bool(mapped.get("Z"))

    if has_beta:
        return "BETA"
    if has_or:
        return "OR_requires_log_conversion"
    if has_z:
        return "Z"
    return "missing_effect_column"


# =============================================================================
# Missingness and usability
# =============================================================================

def count_invalid_alleles(series):
    if series.empty:
        return 0, 0

    values = series.astype(str).map(clean_text)
    non_missing = values[
        ~values.str.lower().isin(["", "nan", "na", "n/a", "none", "null", "."])
    ]

    if non_missing.empty:
        return 0, 0

    valid = non_missing.str.upper().str.match(r"^[ACGT]+$", na=False)
    invalid_count = int((~valid).sum())

    return invalid_count, len(non_missing)


def ambiguous_allele_percentage(a1, a2):
    if a1.empty or a2.empty:
        return 100.0

    df = pd.DataFrame({
        "a1": a1.astype(str).str.upper().map(clean_text),
        "a2": a2.astype(str).str.upper().map(clean_text),
    })

    df = df[
        (df["a1"] != "")
        & (df["a2"] != "")
        & (~df["a1"].isin(["NAN", "NA", "N/A", "NONE", "NULL", "."]))
        & (~df["a2"].isin(["NAN", "NA", "N/A", "NONE", "NULL", "."]))
    ]

    if df.empty:
        return 0.0

    pairs = df["a1"] + "/" + df["a2"]

    ambiguous = pairs.isin(["A/T", "T/A", "C/G", "G/C"])

    return percent(ambiguous.sum(), len(df))


def duplicated_snp_percentage(snp):
    if snp.empty:
        return 100.0

    values = snp.astype(str).map(clean_text)
    values = values[
        ~values.str.lower().isin(["", "nan", "na", "n/a", "none", "null", "."])
    ]

    if values.empty:
        return 100.0

    duplicated = values.duplicated(keep=False).sum()

    return percent(duplicated, len(values))


# =============================================================================
# Main profiling logic
# =============================================================================

def profile_variation2(job_index, phenotype, accession, file_path, output_feature_file):
    file_path = Path(file_path)
    output_feature_file = Path(output_feature_file)

    feature1 = load_feature1_metadata(output_feature_file)
    compression = feature1["compression_type"]
    delimiter = feature1["delimiter"]
    raw_headers = feature1["raw_headers"]
    header_row_index = feature1["header_row_index"]
    header_detected = feature1["header_detected"]

    mapped, unmapped, normalised_headers, mapped_standard_headers = map_headers(raw_headers)

    raw_header_signature = sha1_text("|".join(raw_headers))
    normalised_header_signature = sha1_text("|".join(normalised_headers))
    mapped_standard_header_signature = sha1_text("|".join(mapped_standard_headers))

    df, dataframe_issue = read_complete_dataframe(
        path=file_path,
        feature1=feature1,
    )

    n_rows_loaded = len(df)

    col_snp = first_mapped_column(mapped, "SNP")
    col_chr = first_mapped_column(mapped, "CHR")
    col_bp = first_mapped_column(mapped, "BP")
    col_a1 = first_mapped_column(mapped, "A1")
    col_a2 = first_mapped_column(mapped, "A2")
    col_beta = first_mapped_column(mapped, "BETA")
    col_or = first_mapped_column(mapped, "OR")
    col_z = first_mapped_column(mapped, "Z")
    col_p = first_mapped_column(mapped, "P")
    col_eaf = first_mapped_column(mapped, "EAF")
    col_maf = first_mapped_column(mapped, "MAF")
    col_info = first_mapped_column(mapped, "INFO")

    snp = get_series(df, col_snp)
    chr_col = get_series(df, col_chr)
    bp = get_series(df, col_bp)
    a1 = get_series(df, col_a1)
    a2 = get_series(df, col_a2)
    beta = get_series(df, col_beta)
    or_col = get_series(df, col_or)
    z = get_series(df, col_z)
    p = get_series(df, col_p)
    eaf = get_series(df, col_eaf)
    maf = get_series(df, col_maf)
    info = get_series(df, col_info)

    effect_series = beta
    if effect_series.empty:
        effect_series = or_col
    if effect_series.empty:
        effect_series = z

    invalid_a1_count, valid_a1_denominator = count_invalid_alleles(a1)
    invalid_a2_count, valid_a2_denominator = count_invalid_alleles(a2)

    total_invalid_alleles = invalid_a1_count + invalid_a2_count
    total_allele_denominator = valid_a1_denominator + valid_a2_denominator

    prs_minimum_columns_present = bool(
        (col_snp or (col_chr and col_bp))
        and col_a1
        and col_p
        and (col_beta or col_or or col_z)
    )

    has_standard_required_core = bool(
        col_snp and col_chr and col_bp and col_a1 and col_a2 and col_p and (col_beta or col_or or col_z)
    )

    requires_column_renaming = len(unmapped) > 0 or any(
        first_mapped_column(mapped, c) == "" for c in ["SNP", "A1", "P"]
    )

    requires_delimiter_fix = delimiter not in ["tab", "comma", "space"]
    requires_decompression_or_extraction = compression in ["gzip", "zip", "tar", "tar.gz"]
    requires_effect_conversion = bool(col_or and not col_beta)

    cannot_normalise_safely = bool(
        not prs_minimum_columns_present
        or header_detected is False
        or delimiter == "unknown"
        or n_rows_loaded == 0
    )

    close_to_gwas_ssf = bool(
        has_standard_required_core
        and delimiter in ["tab", "comma"]
        and not requires_effect_conversion
        and not cannot_normalise_safely
    )

    row = {
        "job_index": job_index,
        "phenotype": phenotype,
        "accessionId": accession,
        "file_path": str(file_path),
        "file_name": file_path.name,
        "feature1_file": feature1["feature1_file"],
        "feature1_header_source": True,
        "file_extension": feature1["file_extension"],
        "compression_type": feature1["compression_type"],
        "compression_from_extension": feature1["compression_from_extension"],
        "compression_from_magic_bytes": feature1["compression_from_magic_bytes"],
        "compression_detection_source": feature1["compression_detection_source"],
        "compression_suffix_magic_mismatch": feature1["compression_suffix_magic_mismatch"],
        "reader_used": feature1["reader_used"],
        "raw_file_size_bytes": feature1["raw_file_size_bytes"],
        "raw_file_size_mb": feature1["raw_file_size_mb"],

        "extracted_member": feature1["extracted_member"],
        "extracted_file_size_bytes": feature1["extracted_file_size_bytes"],
        "extracted_file_size_mb": feature1["extracted_file_size_mb"],

        "read_issue": feature1["read_status"],
        "encoding_issues": feature1["encoding_issues"],
        "encoding_used": feature1["encoding_used"],
        "dataframe_issue": dataframe_issue,
        "n_rows_loaded": n_rows_loaded,
        "n_rows_physical_lines": feature1["n_rows_physical_lines"],

        "delimiter": delimiter,
        "estimated_n_columns": feature1["estimated_n_columns"],
        "header_detected": header_detected,
        "header_row_index": header_row_index,
        "comment_or_metadata_lines_before_header": feature1["comment_or_metadata_lines_before_header"],
        "quote_usage": feature1["quote_usage"],
        "n_sample_lines_read": feature1["n_sample_lines_read"],

        "n_columns": feature1["n_columns"],
        "raw_headers_joined": feature1["raw_headers_joined"],
        "normalised_headers_joined": " | ".join(normalised_headers),
        "mapped_standard_headers_joined": " | ".join(mapped_standard_headers),
        "raw_header_signature": raw_header_signature,
        "normalised_header_signature": normalised_header_signature,
        "mapped_standard_header_signature": mapped_standard_header_signature,

        "n_unique_raw_column_names_in_file": len(set(raw_headers)),
        "n_unique_normalised_column_names_in_file": len(set(normalised_headers)),
        "n_unique_mapped_standard_columns_in_file": len(set([
            x for x in mapped_standard_headers if x != "UNMAPPED"
        ])),
        "n_mapped_standard_headers": sum(1 for x in mapped_standard_headers if x != "UNMAPPED"),
        "n_unmapped_standard_headers": sum(1 for x in mapped_standard_headers if x == "UNMAPPED"),
        "n_unmapped_columns": len(unmapped),
        "unmapped_columns_joined": " | ".join(unmapped),
    }

    for standard_col in STANDARD_COLUMNS_ORDER:
        mapped_values = mapped.get(standard_col, [])
        row[f"has_{standard_col}"] = bool(mapped_values)
        row[f"mapped_{standard_col}"] = " | ".join(mapped_values)

    row.update({
        "effect_column_type": classify_effect_type(mapped),

        "chromosome_format": classify_chr_format(chr_col),
        "snp_format": classify_snp_format(snp),
        "bp_format": classify_bp_format(bp),
        "a1_allele_format": classify_allele_format(a1),
        "a2_allele_format": classify_allele_format(a2),
        "pvalue_format": classify_pvalue_format(p),
        "eaf_range": classify_numeric_range(eaf),
        "maf_range": classify_numeric_range(maf),
        "info_range": classify_numeric_range(info),

        "missing_snp_percentage": missing_percentage(snp),
        "missing_chr_percentage": missing_percentage(chr_col),
        "missing_bp_percentage": missing_percentage(bp),
        "missing_chr_bp_percentage": max(
            missing_percentage(chr_col),
            missing_percentage(bp),
        ),
        "missing_a1_percentage": missing_percentage(a1),
        "missing_a2_percentage": missing_percentage(a2),
        "missing_a1_a2_percentage": max(
            missing_percentage(a1),
            missing_percentage(a2),
        ),
        "missing_effect_beta_or_z_percentage": missing_percentage(effect_series),
        "missing_p_percentage": missing_percentage(p),

        "duplicated_snp_percentage": duplicated_snp_percentage(snp),
        "ambiguous_allele_percentage": ambiguous_allele_percentage(a1, a2),
        "invalid_allele_percentage": percent(total_invalid_alleles, total_allele_denominator),

        "prs_minimum_columns_present": prs_minimum_columns_present,

        "close_to_gwas_ssf": close_to_gwas_ssf,
        "requires_column_renaming": requires_column_renaming,
        "requires_delimiter_fix": requires_delimiter_fix,
        "requires_decompression_or_extraction": requires_decompression_or_extraction,
        "requires_effect_conversion_or_to_log_or": requires_effect_conversion,
        "cannot_normalise_safely": cannot_normalise_safely,
    })

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

    required = [
        "job_index",
        "phenotype",
        "accessionId",
        "file_path",
        "output_feature_file",
    ]

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
    output_feature_file = base_output.parent / "Feature2.csv"

    print("=" * 100)
    print("VARIATION 2: HEADER-LEVEL AND USABILITY PROFILING")
    print("=" * 100)
    print(f"[JOB INDEX] {index}")
    print(f"[PHENOTYPE] {phenotype}")
    print(f"[ACCESSION] {accession}")
    print(f"[GWAS FILE] {file_path}")
    print(f"[OUTPUT] {output_feature_file}")
    print("=" * 100)

    if not file_path.exists():
        raise FileNotFoundError(f"Missing GWAS file: {file_path}")

    if output_feature_file.exists() and not force:
        print(f"[SKIP] Feature2.csv already exists: {output_feature_file}")
        return

    row = profile_variation2(
        job_index=index,
        phenotype=phenotype,
        accession=accession,
        file_path=file_path,
        output_feature_file=output_feature_file,
    )

    print()
    print("[SAVED]")
    print(output_feature_file)
    print()
    print("[KEY RESULTS]")
    print(f"n_columns: {row['n_columns']}")
    print(f"delimiter: {row['delimiter']}")
    print(f"effect_column_type: {row['effect_column_type']}")
    print(f"prs_minimum_columns_present: {row['prs_minimum_columns_present']}")
    print(f"cannot_normalise_safely: {row['cannot_normalise_safely']}")
    print("=" * 100)


def combine_feature2(jobs_file, output_file):
    jobs = pd.read_csv(jobs_file)

    rows = []

    for _, job in jobs.iterrows():
        base_output = Path(clean_text(job["output_feature_file"]))
        feature2 = base_output.parent / "Feature2.csv"

        if feature2.exists():
            try:
                rows.append(pd.read_csv(feature2))
            except Exception:
                pass

    if rows:
        out_df = pd.concat(rows, ignore_index=True)
    else:
        out_df = pd.DataFrame()

    out_df.to_csv(output_file, index=False)

    print("=" * 100)
    print("[COMBINED Feature2.csv FILES]")
    print(f"[INPUT JOBS] {jobs_file}")
    print(f"[FILES COMBINED] {len(rows)}")
    print(f"[OUTPUT] {output_file}")
    print("=" * 100)

    if not out_df.empty:
        print()
        print("[SUMMARY]")
        summary_cols = [
            "close_to_gwas_ssf",
            "requires_column_renaming",
            "requires_delimiter_fix",
            "requires_decompression_or_extraction",
            "requires_effect_conversion_or_to_log_or",
            "cannot_normalise_safely",
        ]

        for col in summary_cols:
            if col in out_df.columns:
                print()
                print(col)
                print(out_df[col].value_counts(dropna=False).to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Variation2: process one GWAS file by index and save Feature2.csv."
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
        help="GWAS job manifest file. Default: GWAS_jobs.csv"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing Feature2.csv"
    )

    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine all per-GWAS Feature2.csv files into one CSV."
    )

    parser.add_argument(
        "--combine-output",
        default="Variation2_allGWAS.csv",
        help="Output CSV for --combine. Default: Variation2_allGWAS.csv"
    )

    args = parser.parse_args()

    if args.combine:
        combine_feature2(
            jobs_file=args.jobs_file,
            output_file=args.combine_output,
        )
        return

    if args.index is None:
        raise ValueError("Please provide a GWAS job index, e.g. python Variation2.py 1")

    run_one_job(
        index=args.index,
        jobs_file=args.jobs_file,
        force=args.force,
    )


if __name__ == "__main__":
    main()
