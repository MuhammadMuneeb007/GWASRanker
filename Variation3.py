#!/usr/bin/env python3

import argparse
import contextlib
import gzip
import importlib.util
import io
import re
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import pandas as pd


def load_module2_file_rules():
    module2_path = Path(__file__).with_name("Module2-Search_Poke_Normalize_Scan.py")
    spec = importlib.util.spec_from_file_location("module2_search_poke_normalize_scan", module2_path)

    if spec is None or spec.loader is None:
        return (
            (".gz", ".bgz", ".txt", ".tsv", ".csv", ".tbl", ".ma", ".assoc",
             ".meta", ".linear", ".logistic", ".sumstats", ".summary",
             ".zip", ".tar", ".tar.gz", ".tgz"),
            ("readme", "md5", "manifest", "license", "licence", ".pdf",
             ".html", ".htm", ".png", ".jpg", ".jpeg", ".json", ".yaml",
             ".yml", ".xlsx", ".xls"),
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LIKELY_GWAS_EXTENSIONS, module.SKIP_FILE_PATTERNS


LIKELY_GWAS_EXTENSIONS, SKIP_FILE_PATTERNS = load_module2_file_rules()


def load_module2_column_synonyms():
    module2_path = Path(__file__).with_name("Module2-Search_Poke_Normalize_Scan.py")
    spec = importlib.util.spec_from_file_location("module2_search_poke_normalize_scan", module2_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Module 2 mapping from {module2_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.COLUMN_SYNONYMS


COLUMN_SYNONYMS = load_module2_column_synonyms()
STANDARD_COLUMNS_ORDER = [
    "SNP", "CHR", "BP", "A1", "A2", "BETA", "OR", "Z",
    "SE", "P", "N", "N_CASES", "N_CONTROLS", "EAF", "MAF",
    "INFO", "DIRECTION",
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


def infer_gzip_member_name(path):
    name = Path(path).name
    lower = name.lower()

    if lower.endswith(".tar.gz"):
        return name[:-7]
    if lower.endswith(".tgz"):
        return name[:-4] + ".tar"
    if lower.endswith(".bgz"):
        return name[:-4]
    if lower.endswith(".gz"):
        return name[:-3]

    return name


def get_gzip_uncompressed_size(path):
    try:
        with open(path, "rb") as f:
            f.seek(-4, 2)
            return int.from_bytes(f.read(4), "little")
    except Exception:
        return 0


def get_file_extension(path):
    name = str(path).lower()
    if name.endswith(".tar.gz"):    return ".tar.gz"
    if name.endswith(".tgz"):       return ".tgz"
    if name.endswith(".tsv.gz"):    return ".tsv.gz"
    if name.endswith(".txt.gz"):    return ".txt.gz"
    if name.endswith(".csv.gz"):    return ".csv.gz"
    if name.endswith(".bgz"):       return ".bgz"
    if name.endswith(".gz"):        return ".gz"
    if name.endswith(".sumstats"):  return ".sumstats"
    if name.endswith(".summary"):   return ".summary"
    if name.endswith(".assoc"):     return ".assoc"
    if name.endswith(".linear"):    return ".linear"
    if name.endswith(".logistic"):  return ".logistic"
    if name.endswith(".tbl"):       return ".tbl"
    if name.endswith(".ma"):        return ".ma"
    if name.endswith(".meta"):      return ".meta"
    return Path(path).suffix.lower()


# =============================================================================
# SNP count
# =============================================================================

def is_likely_gwas_file(filename):
    name = str(filename).lower()

    if any(pattern in name for pattern in SKIP_FILE_PATTERNS):
        return False

    return name.endswith(LIKELY_GWAS_EXTENSIONS)


def select_zip_member(path):
    try:
        with zipfile.ZipFile(path) as z:
            members = [m for m in z.infolist() if not m.is_dir()]
            likely = [m for m in members if is_likely_gwas_file(m.filename)]
            choices = likely or members

            if not choices:
                return "", 0

            largest = max(choices, key=lambda m: m.file_size)
            return largest.filename, largest.file_size
    except Exception:
        return "", 0


def select_tar_member(path):
    try:
        mode = "r:gz" if str(path).lower().endswith((".tar.gz", ".tgz")) else "r:"

        with tarfile.open(path, mode) as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            likely = [m for m in members if is_likely_gwas_file(m.name)]
            choices = likely or members

            if not choices:
                return "", 0

            largest = max(choices, key=lambda m: m.size)
            return largest.name, largest.size
    except Exception:
        return "", 0


def get_data_member_info(path):
    compression = detect_compression(path)

    if compression == "gzip":
        return infer_gzip_member_name(path), get_gzip_uncompressed_size(path)

    if compression == "zip":
        return select_zip_member(path)

    if compression in ["tar", "tar.gz"]:
        return select_tar_member(path)

    return "", 0


def normalise_header_index(value):
    try:
        if clean_text(value) == "":
            return None
        return int(float(value))
    except Exception:
        return None


def count_text_lines(handle):
    return sum(1 for _ in handle)


def read_plain_lines(path, max_lines):
    lines = []

    try:
        with open(path, "rt", encoding="utf-8", errors="replace") as f:
            for _ in range(max_lines):
                line = f.readline()
                if not line:
                    break
                lines.append(line.rstrip("\n"))
        return lines, "utf-8", ""
    except Exception as e:
        return [], "utf-8", f"{type(e).__name__}: {e}"


def read_gzip_lines(path, max_lines):
    lines = []

    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            for _ in range(max_lines):
                line = f.readline()
                if not line:
                    break
                lines.append(line.rstrip("\n"))
        return lines, "utf-8", ""
    except Exception as e:
        return [], "utf-8", f"{type(e).__name__}: {e}"


def read_zip_lines(path, max_lines):
    member, _ = select_zip_member(path)
    if not member:
        return [], "utf-8", "zip_empty"

    try:
        with zipfile.ZipFile(path) as z:
            lines = []
            with z.open(member) as f:
                for i, raw_line in enumerate(f):
                    if i >= max_lines:
                        break
                    line = raw_line.decode("utf-8", errors="replace")
                    lines.append(line.rstrip("\n"))
        return lines, "utf-8", ""
    except Exception as e:
        return [], "utf-8", f"{type(e).__name__}: {e}"


def read_tar_lines(path, max_lines):
    member, _ = select_tar_member(path)
    if not member:
        return [], "utf-8", "tar_empty"

    try:
        mode = "r:gz" if str(path).lower().endswith((".tar.gz", ".tgz")) else "r:"
        with tarfile.open(path, mode) as tar:
            f = tar.extractfile(member)
            if f is None:
                return [], "utf-8", "tar_extract_failed"

            lines = []
            for i, raw_line in enumerate(f):
                if i >= max_lines:
                    break
                line = raw_line.decode("utf-8", errors="replace")
                lines.append(line.rstrip("\n"))
        return lines, "utf-8", ""
    except Exception as e:
        return [], "utf-8", f"{type(e).__name__}: {e}"


def read_sample_lines(path, max_lines=200):
    compression = detect_compression(path)

    if compression == "plain":
        lines, encoding, issue = read_plain_lines(path, max_lines)
    elif compression == "gzip":
        lines, encoding, issue = read_gzip_lines(path, max_lines)
    elif compression == "zip":
        lines, encoding, issue = read_zip_lines(path, max_lines)
    elif compression in ["tar", "tar.gz"]:
        lines, encoding, issue = read_tar_lines(path, max_lines)
    else:
        lines, encoding, issue = [], "utf-8", "unsupported_compression"

    return {
        "lines": lines,
        "encoding_used": encoding,
        "encoding_issues": issue,
        "read_status": "read_ok" if lines else "read_failed_or_empty",
    }


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


def detect_delimiter(lines):
    data_lines = [
        line for line in lines
        if clean_text(line) and not clean_text(line).startswith("#")
    ]

    if not data_lines:
        return "unknown", 0

    delimiters = {
        "tab": "\t",
        "comma": ",",
        "semicolon": ";",
        "pipe": "|",
        "space": "space",
    }

    best_delimiter = "unknown"
    best_cols = 0

    for delimiter_name, delimiter_symbol in delimiters.items():
        counts = []
        for line in data_lines[:50]:
            if delimiter_name == "space":
                parts = re.split(r"\s+", line.strip())
            else:
                parts = line.split(delimiter_symbol)
            counts.append(len(parts))

        if not counts:
            continue

        median_cols = sorted(counts)[len(counts) // 2]
        if median_cols > best_cols:
            best_cols = median_cols
            best_delimiter = delimiter_name

    return best_delimiter, best_cols


def detect_header(lines, delimiter):
    comment_lines = 0

    for i, line in enumerate(lines):
        raw = clean_text(line)
        if raw == "":
            continue

        if raw.startswith("#"):
            comment_lines += 1
            continue

        parts = split_line(raw, delimiter)
        if len(parts) < 2:
            continue

        non_numeric = 0
        for value in parts:
            try:
                float(clean_text(value))
            except Exception:
                non_numeric += 1

        non_numeric_ratio = non_numeric / max(len(parts), 1)
        if non_numeric_ratio >= 0.5:
            return {
                "header_detected": True,
                "header_row_index": i,
                "comment_or_metadata_lines_before_header": comment_lines,
                "headers": parts,
            }

    return {
        "header_detected": False,
        "header_row_index": "",
        "comment_or_metadata_lines_before_header": comment_lines,
        "headers": [],
    }


def map_headers(raw_headers):
    mapped = {}
    normalised_headers = [normalise_header_name(h) for h in raw_headers]

    for raw, norm in zip(raw_headers, normalised_headers):
        standard = SYNONYM_LOOKUP.get(norm, "")
        if standard:
            mapped.setdefault(standard, []).append(raw)

    return mapped, normalised_headers


def first_mapped_column(mapped, standard_col):
    values = mapped.get(standard_col, [])
    return values[0] if values else ""


def detect_header_profile(file_path):
    sample = read_sample_lines(file_path, max_lines=200)
    lines = sample["lines"]
    delimiter, estimated_n_columns = detect_delimiter(lines)
    header = detect_header(lines, delimiter)
    raw_headers = [clean_text(x) for x in header["headers"]]
    mapped, normalised_headers = map_headers(raw_headers)

    profile = {
        "header_detection_source": "Variation3_self_detected",
        "read_status": sample["read_status"],
        "encoding_used": sample["encoding_used"],
        "encoding_issues": sample["encoding_issues"],
        "delimiter": delimiter,
        "header_detected": header["header_detected"],
        "header_row_index": header["header_row_index"],
        "comment_or_metadata_lines_before_header": header["comment_or_metadata_lines_before_header"],
        "n_columns": len(raw_headers) if raw_headers else estimated_n_columns,
        "raw_headers": raw_headers,
        "raw_headers_joined": " | ".join(raw_headers),
        "normalised_headers_joined": " | ".join(normalised_headers),
        "mapped_header_count": sum(len(values) for values in mapped.values()),
    }

    for standard_col in STANDARD_COLUMNS_ORDER:
        profile[f"mapped_{standard_col}"] = first_mapped_column(mapped, standard_col)

    return profile


def count_snps_fast(path, compression, header_row_index):
    """Count data rows in the logical GWAS table across plain/compressed/archive files."""
    try:
        header_index = normalise_header_index(header_row_index)
        skip = header_index + 1 if header_index is not None else 0

        if compression == "plain":
            with open(path, "rt", encoding="utf-8", errors="replace") as f:
                total = count_text_lines(f)
            return max(0, total - skip)

        if compression == "gzip":
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                total = count_text_lines(f)
            return max(0, total - skip)

        if compression == "zip":
            member, _ = select_zip_member(path)
            if not member:
                return ""

            with zipfile.ZipFile(path) as z:
                with z.open(member) as raw:
                    text = pd.io.common.TextIOWrapper(raw, encoding="utf-8", errors="replace")
                    total = count_text_lines(text)
                    text.close()
            return max(0, total - skip)

        if compression in ["tar", "tar.gz"]:
            member, _ = select_tar_member(path)
            if not member:
                return ""

            mode = "r:gz" if str(path).lower().endswith((".tar.gz", ".tgz")) else "r:"
            with tarfile.open(path, mode) as tar:
                raw = tar.extractfile(member)
                if raw is None:
                    return ""

                text = pd.io.common.TextIOWrapper(raw, encoding="utf-8", errors="replace")
                total = count_text_lines(text)
                text.close()
            return max(0, total - skip)

        return ""

    except Exception:
        return ""


# =============================================================================
# Genome build detection
# =============================================================================

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


def normalise_gwaslab_build(build):
    build = clean_text(build)

    if build in ["19", "hg19", "HG19", "GRCh37", "grch37"]:
        return "GRCh37"

    if build in ["38", "hg38", "HG38", "GRCh38", "grch38"]:
        return "GRCh38"

    if build in ["", "99", "unknown", "Unknown", "UNKNOWN"]:
        return "unknown"

    return build


def get_gwaslab_meta_build(sumstats):
    meta = getattr(sumstats, "meta", {})

    if isinstance(meta, dict):
        gwaslab_meta = meta.get("gwaslab", {})
        if isinstance(gwaslab_meta, dict):
            return gwaslab_meta.get("genome_build", "")

    return ""


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

    formats = sorted(formats, key=lambda x: x["rank"])
    return formats


def parse_gwaslab_build_matches(log_text):
    hg19 = ""
    hg38 = ""

    m19 = re.search(r"Matching variants for hg19:\s*num_hg19\s*=\s*([\d,]+)", log_text)
    m38 = re.search(r"Matching variants for hg38:\s*num_hg38\s*=\s*([\d,]+)", log_text)

    if m19:
        hg19 = int(m19.group(1).replace(",", ""))
    if m38:
        hg38 = int(m38.group(1).replace(",", ""))

    return hg19, hg38


def calculate_build_match_ratio(hg19_count, hg38_count):
    if hg19_count == "" or hg38_count == "":
        return ""

    denominator = max(int(hg19_count), int(hg38_count), 1)
    numerator = min(int(hg19_count), int(hg38_count))
    return round(numerator / denominator, 6)


def get_gwaslab_data(sumstats):
    data = getattr(sumstats, "data", None)

    if data is None:
        raise AttributeError("GWASLab Sumstats object does not expose a .data dataframe.")

    return data


def column_numeric_success(data, column):
    if column not in data.columns:
        return ""

    converted = pd.to_numeric(data[column], errors="coerce")
    non_missing = data[column].notna().sum()

    if non_missing == 0:
        return False

    return bool(converted.notna().sum() == non_missing)


def count_duplicated_column(data, column):
    if column not in data.columns:
        return ""

    return int(data[column].astype(str).duplicated(keep=False).sum())


def count_duplicated_compound(data, columns):
    columns = [col for col in columns if col in data.columns]

    if len(columns) == 0:
        return ""

    return int(data[columns].astype(str).duplicated(keep=False).sum())


def build_gwaslab_kwargs(header_profile):
    kwargs = {"fmt": "auto"}

    optional_columns = {
        "snpid": header_profile["mapped_SNP"],
        "chrom": header_profile["mapped_CHR"],
        "pos": header_profile["mapped_BP"],
        "ea": header_profile["mapped_A1"],
        "nea": header_profile["mapped_A2"],
        "beta": header_profile["mapped_BETA"],
        "se": header_profile["mapped_SE"],
        "p": header_profile["mapped_P"],
        "n": header_profile["mapped_N"],
        "eaf": header_profile["mapped_EAF"],
        "info": header_profile["mapped_INFO"],
        "OR": header_profile["mapped_OR"],
        "z": header_profile["mapped_Z"],
        "direction": header_profile["mapped_DIRECTION"],
    }

    for key, value in optional_columns.items():
        if value:
            kwargs[key] = value

    return kwargs


def materialize_gwaslab_input(path, compression):
    path = Path(path)
    if compression in ["plain", "gzip"]:
        return str(path), ""

    if compression == "zip":
        member, _ = select_zip_member(path)
        if not member:
            raise ValueError("zip archive did not contain a readable GWAS member")
        with zipfile.ZipFile(path) as z:
            with z.open(member) as raw:
                suffix = "".join(Path(member).suffixes) or ".txt"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(raw.read())
                    return tmp.name, member

    if compression in ["tar", "tar.gz"]:
        member, _ = select_tar_member(path)
        if not member:
            raise ValueError("tar archive did not contain a readable GWAS member")
        mode = "r:gz" if str(path).lower().endswith((".tar.gz", ".tgz")) else "r:"
        with tarfile.open(path, mode) as tar:
            raw = tar.extractfile(member)
            if raw is None:
                raise ValueError(f"could not extract tar member: {member}")
            suffix = "".join(Path(member).suffixes) or ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(raw.read())
                return tmp.name, member

    return str(path), ""


def profile_with_gwaslab(path, compression, header_profile):
    import gwaslab as gl

    gwaslab_log_file = Path(path).parent / "Feature3_gwaslab.log"
    log_buffer = io.StringIO()
    tee_stdout = TeeTextIO(sys.stdout, log_buffer)
    tee_stderr = TeeTextIO(sys.stderr, log_buffer)

    standard_columns = [
        "SNPID", "CHR", "POS", "EA", "NEA", "BETA", "SE", "P",
        "N", "OR", "EAF", "INFO", "Z", "DIRECTION"
    ]
    base_failure = {
        "genome_build": "unknown",
        "genome_build_evidence": "",
        "gwaslab_load_success": False,
        "gwaslab_load_error": "",
        "gwaslab_log_file": str(gwaslab_log_file),
        "gwaslab_column_detection_mode": "",
        "gwaslab_input_member": "",
        "gwaslab_loaded_variant_count": "",
        "gwaslab_dataframe_memory_mb": "",
        "gwaslab_standard_columns_loaded": "",
        "gwaslab_standard_column_count": "",
        "gwaslab_chr_numeric_success": "",
        "gwaslab_pos_numeric_success": "",
        "gwaslab_p_numeric_success": "",
        "gwaslab_beta_numeric_success": "",
        "gwaslab_se_numeric_success": "",
        "gwaslab_duplicated_snpid_count": "",
        "gwaslab_duplicated_chr_pos_count": "",
        "gwaslab_duplicated_chr_pos_ea_nea_count": "",
        "gwaslab_build_hg19_match_count": "",
        "gwaslab_build_hg38_match_count": "",
        "gwaslab_build_match_ratio": "",
        "gwaslab_qc_pre_variant_count": "",
        "gwaslab_qc_pass_variant_count": "",
        "gwaslab_qc_removed_variant_count": "",
        "gwaslab_qc_removed_variant_percentage": "",
        "gwaslab_post_qc_standard_columns_loaded": "",
    }

    for i in range(3):
        base_failure[f"gwaslab_inferred_format_{i + 1}"] = ""
        base_failure[f"gwaslab_inferred_format_{i + 1}_score"] = ""

    materialized_path = ""
    input_member = ""

    try:
        materialized_path, input_member = materialize_gwaslab_input(path, compression)
        attempts = [
            ("gwaslab_auto", {"fmt": "auto"}),
            ("gwaslab_auto_plus_local_header_mapping", build_gwaslab_kwargs(header_profile)),
        ]

        last_error = ""
        chosen_mode = ""
        sumstats = None

        for mode_name, kwargs in attempts:
            try:
                with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
                    sumstats = gl.Sumstats(str(materialized_path), **kwargs)
                chosen_mode = mode_name
                break
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"

        if sumstats is None:
            raise RuntimeError(last_error or "GWASLab failed to load the file")

        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
            loaded_data = get_gwaslab_data(sumstats)
            loaded_variant_count = len(loaded_data)
            loaded_memory_mb = round(loaded_data.memory_usage(deep=True).sum() / (1024 ** 2), 3)
            loaded_standard_columns = [col for col in standard_columns if col in loaded_data.columns]
            loaded_chr_numeric_success = column_numeric_success(loaded_data, "CHR")
            loaded_pos_numeric_success = column_numeric_success(loaded_data, "POS")
            loaded_p_numeric_success = column_numeric_success(loaded_data, "P")
            loaded_beta_numeric_success = column_numeric_success(loaded_data, "BETA")
            loaded_se_numeric_success = column_numeric_success(loaded_data, "SE")
            duplicated_snpid_count = count_duplicated_column(loaded_data, "SNPID")
            duplicated_chr_pos_count = count_duplicated_compound(loaded_data, ["CHR", "POS"])
            duplicated_chr_pos_ea_nea_count = count_duplicated_compound(
                loaded_data,
                ["CHR", "POS", "EA", "NEA"],
            )

            sumstats.infer_build()
            pre_qc_count = len(get_gwaslab_data(sumstats))
            sumstats.basic_check(remove=True, remove_dup=True)
            post_qc_data = get_gwaslab_data(sumstats)
            post_qc_count = len(post_qc_data)

        log_text = log_buffer.getvalue()
        gwaslab_log_file.write_text(log_text, encoding="utf-8")

        raw_build = clean_text(getattr(sumstats, "build", "")) or clean_text(get_gwaslab_meta_build(sumstats))
        genome_build = normalise_gwaslab_build(raw_build)
        inferred_formats = parse_gwaslab_inferred_formats(log_text)
        hg19_count, hg38_count = parse_gwaslab_build_matches(log_text)

        gwaslab_features = dict(base_failure)
        gwaslab_features.update({
            "genome_build": genome_build,
            "genome_build_evidence": f"gwaslab_infer_build:{raw_build}",
            "gwaslab_load_success": True,
            "gwaslab_load_error": "",
            "gwaslab_column_detection_mode": chosen_mode,
            "gwaslab_input_member": input_member,
            "gwaslab_loaded_variant_count": loaded_variant_count,
            "gwaslab_dataframe_memory_mb": loaded_memory_mb,
            "gwaslab_standard_columns_loaded": " | ".join(loaded_standard_columns),
            "gwaslab_standard_column_count": len(loaded_standard_columns),
            "gwaslab_chr_numeric_success": loaded_chr_numeric_success,
            "gwaslab_pos_numeric_success": loaded_pos_numeric_success,
            "gwaslab_p_numeric_success": loaded_p_numeric_success,
            "gwaslab_beta_numeric_success": loaded_beta_numeric_success,
            "gwaslab_se_numeric_success": loaded_se_numeric_success,
            "gwaslab_duplicated_snpid_count": duplicated_snpid_count,
            "gwaslab_duplicated_chr_pos_count": duplicated_chr_pos_count,
            "gwaslab_duplicated_chr_pos_ea_nea_count": duplicated_chr_pos_ea_nea_count,
            "gwaslab_build_hg19_match_count": hg19_count,
            "gwaslab_build_hg38_match_count": hg38_count,
            "gwaslab_build_match_ratio": calculate_build_match_ratio(hg19_count, hg38_count),
            "gwaslab_qc_pre_variant_count": pre_qc_count,
            "gwaslab_qc_pass_variant_count": post_qc_count,
            "gwaslab_qc_removed_variant_count": pre_qc_count - post_qc_count,
            "gwaslab_qc_removed_variant_percentage": round(
                ((pre_qc_count - post_qc_count) / pre_qc_count) * 100,
                6,
            ) if pre_qc_count else 0,
            "gwaslab_post_qc_standard_columns_loaded": " | ".join(
                [col for col in standard_columns if col in post_qc_data.columns]
            ),
        })

        for i in range(3):
            item = inferred_formats[i] if i < len(inferred_formats) else {}
            rank = i + 1
            gwaslab_features[f"gwaslab_inferred_format_{rank}"] = item.get("name", "")
            gwaslab_features[f"gwaslab_inferred_format_{rank}_score"] = item.get("score", "")

        return gwaslab_features

    except Exception as e:
        log_text = log_buffer.getvalue()
        gwaslab_log_file.write_text(log_text, encoding="utf-8")
        gwaslab_features = dict(base_failure)
        gwaslab_features["gwaslab_load_error"] = f"{type(e).__name__}: {e}"
        gwaslab_features["gwaslab_input_member"] = input_member
        return gwaslab_features

    finally:
        if materialized_path and materialized_path != str(path):
            try:
                Path(materialized_path).unlink(missing_ok=True)
            except Exception:
                pass


# =============================================================================
# Harmonised file detection
# =============================================================================

HARMONISED_HINTS = [
    "harmonised",
    "harmonized",
    "harm",
    ".h.tsv",
    ".h.tsv.gz",
    "Build38",
    "build38",
    "grch38",
    "hg38",
]


def check_harmonised_file(file_path, accession):
    """
    Check whether a harmonised version of this GWAS file exists.

    Looks in:
    - Same directory as the downloaded file
    - GWAS Catalog harmonised naming convention: h.tsv.gz suffix
    - Sibling directory named 'harmonised' or 'harmonized'
    """
    file_path = Path(file_path)
    parent_dir = file_path.parent

    result = {
        "harmonised_file_exists": False,
        "harmonised_file_path": "",
        "harmonised_detection_method": "",
        "current_file_is_harmonised": False,
    }

    # Check if the current file itself is harmonised
    fname = file_path.name.lower()

    if any(h.lower() in fname for h in HARMONISED_HINTS):
        result["current_file_is_harmonised"] = True
        result["harmonised_file_exists"] = True
        result["harmonised_file_path"] = str(file_path)
        result["harmonised_detection_method"] = "current_file_name_hint"
        return result

    # Look for harmonised sibling files in the same directory
    for sibling in parent_dir.iterdir():
        if not sibling.is_file():
            continue

        if sibling == file_path:
            continue

        sname = sibling.name.lower()

        if any(h.lower() in sname for h in HARMONISED_HINTS):
            result["harmonised_file_exists"] = True
            result["harmonised_file_path"] = str(sibling)
            result["harmonised_detection_method"] = "sibling_file_in_same_dir"
            return result

    # Look in harmonised/harmonized subdirectory
    for subdir_name in ["harmonised", "harmonized"]:
        subdir = parent_dir / subdir_name

        if subdir.exists() and subdir.is_dir():
            files = list(subdir.iterdir())

            if files:
                result["harmonised_file_exists"] = True
                result["harmonised_file_path"] = str(files[0])
                result["harmonised_detection_method"] = f"subdir_{subdir_name}"
                return result

    # Check parent GWAS Catalog download folder for .h.tsv.gz pattern
    # GWAS Catalog convention: GCST12345.h.tsv.gz is harmonised, GCST12345.tsv.gz is raw
    stem = file_path.name

    for suffix in [".h.tsv.gz", ".h.tsv", ".harmonised.tsv.gz", ".harm.tsv.gz"]:
        candidate = parent_dir / (accession + suffix)

        if candidate.exists() and candidate != file_path:
            result["harmonised_file_exists"] = True
            result["harmonised_file_path"] = str(candidate)
            result["harmonised_detection_method"] = f"gwas_catalog_h_suffix:{suffix}"
            return result

    return result


# =============================================================================
# Main profiling function
# =============================================================================

def profile_variation3(
    job_index,
    phenotype,
    accession,
    file_path,
    output_feature_file,
):
    file_path = Path(file_path)
    output_feature_file = Path(output_feature_file)

    compression = detect_compression(file_path)
    raw_size_bytes = file_path.stat().st_size if file_path.exists() else 0

    header_profile = detect_header_profile(file_path)

    delimiter = header_profile["delimiter"]
    header_row_index = header_profile["header_row_index"]
    comment_lines = header_profile["comment_or_metadata_lines_before_header"]
    extracted_member, extracted_size = get_data_member_info(file_path)

    # SNP count
    snp_count = count_snps_fast(
        path=file_path,
        compression=compression,
        header_row_index=header_row_index,
    )

    # GWASLab profile, including genome build and QC retention.
    gwaslab_features = profile_with_gwaslab(
        path=file_path,
        compression=compression,
        header_profile=header_profile,
    )

    # Harmonised file
    harm = check_harmonised_file(file_path, accession)

    row = {
        "job_index": job_index,
        "phenotype": phenotype,
        "accessionId": accession,
        "file_path": str(file_path),
        "file_name": file_path.name,
        "file_extension": get_file_extension(file_path),
        "compression_type": compression,
        "raw_file_size_bytes": raw_size_bytes,
        "raw_file_size_mb": round(raw_size_bytes / (1024 ** 2), 3),
        "extracted_member": extracted_member,
        "extracted_file_size_bytes": extracted_size,
        "extracted_file_size_mb": round(extracted_size / (1024 ** 2), 3) if extracted_size else 0,
        "header_detection_source": header_profile["header_detection_source"],
        "read_status": header_profile["read_status"],
        "encoding_used": header_profile["encoding_used"],
        "encoding_issues": header_profile["encoding_issues"],
        "delimiter": delimiter,
        "header_detected": header_profile["header_detected"],
        "header_row_index": header_row_index,
        "comment_or_metadata_lines_before_header": comment_lines,
        "n_columns": header_profile["n_columns"],
        "raw_headers_joined": header_profile["raw_headers_joined"],
        "normalised_headers_joined": header_profile["normalised_headers_joined"],
        "mapped_header_count": header_profile["mapped_header_count"],
        "mapped_SNP": header_profile["mapped_SNP"],
        "mapped_CHR": header_profile["mapped_CHR"],
        "mapped_BP": header_profile["mapped_BP"],
        "mapped_A1": header_profile["mapped_A1"],
        "mapped_A2": header_profile["mapped_A2"],
        "mapped_BETA": header_profile["mapped_BETA"],
        "mapped_OR": header_profile["mapped_OR"],
        "mapped_Z": header_profile["mapped_Z"],
        "mapped_SE": header_profile["mapped_SE"],
        "mapped_P": header_profile["mapped_P"],
        "mapped_N": header_profile["mapped_N"],
        "mapped_N_CASES": header_profile["mapped_N_CASES"],
        "mapped_N_CONTROLS": header_profile["mapped_N_CONTROLS"],
        "mapped_EAF": header_profile["mapped_EAF"],
        "mapped_MAF": header_profile["mapped_MAF"],
        "mapped_INFO": header_profile["mapped_INFO"],
        "mapped_DIRECTION": header_profile["mapped_DIRECTION"],
        # SNP count
        "snp_count": snp_count,
        "snp_count_source": "logical_file_line_count_minus_variation3_detected_header",
        # Genome build
        "genome_build": gwaslab_features["genome_build"],
        "genome_build_evidence": gwaslab_features["genome_build_evidence"],
        "genome_build_is_known": gwaslab_features["genome_build"] not in ["unknown", ""],
        "genome_build_is_grch37": "37" in gwaslab_features["genome_build"],
        "genome_build_is_grch38": "38" in gwaslab_features["genome_build"],
        # Harmonised
        "harmonised_file_exists": harm["harmonised_file_exists"],
        "harmonised_file_path": harm["harmonised_file_path"],
        "harmonised_detection_method": harm["harmonised_detection_method"],
        "current_file_is_harmonised": harm["current_file_is_harmonised"],
    }

    row.update(gwaslab_features)

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
    output_feature_file = base_output.parent / "Feature3.csv"

    print("=" * 100)
    print("VARIATION 3: GENOME BUILD, SNP COUNT, HARMONISED FILE CHECK")
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
        print(f"[SKIP] Feature3.csv already exists: {output_feature_file}")
        return

    row = profile_variation3(
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
    print(f"snp_count:              {row['snp_count']}")
    print(f"genome_build:           {row['genome_build']}")
    print(f"genome_build_evidence:  {row['genome_build_evidence']}")
    print(f"gwaslab_format_1:       {row['gwaslab_inferred_format_1']}")
    print(f"gwaslab_hg19_matches:   {row['gwaslab_build_hg19_match_count']}")
    print(f"gwaslab_hg38_matches:   {row['gwaslab_build_hg38_match_count']}")
    print(f"gwaslab_qc_removed_%:   {row['gwaslab_qc_removed_variant_percentage']}")
    print(f"harmonised_exists:      {row['harmonised_file_exists']}")
    print(f"current_is_harmonised:  {row['current_file_is_harmonised']}")
    print("=" * 100)


def combine_feature3(jobs_file, output_file):
    jobs = pd.read_csv(jobs_file)

    rows = []

    for _, job in jobs.iterrows():
        base_output = Path(clean_text(job["output_feature_file"]))
        feature3 = base_output.parent / "Feature3.csv"

        if feature3.exists():
            try:
                rows.append(pd.read_csv(feature3))
            except Exception:
                pass

    out_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out_df.to_csv(output_file, index=False)

    print("=" * 100)
    print("[COMBINED Feature3.csv FILES]")
    print(f"[INPUT JOBS] {jobs_file}")
    print(f"[FILES COMBINED] {len(rows)}")
    print(f"[OUTPUT] {output_file}")
    print("=" * 100)

    if not out_df.empty:
        print()
        print("[SUMMARY]")

        for col in ["genome_build", "harmonised_file_exists", "current_file_is_harmonised", "genome_build_is_known"]:
            if col in out_df.columns:
                print()
                print(col)
                print(out_df[col].value_counts(dropna=False).to_string())

        if "snp_count" in out_df.columns:
            snp_numeric = pd.to_numeric(out_df["snp_count"], errors="coerce").dropna()
            print()
            print("snp_count (numeric files only)")
            print(f"  min:    {snp_numeric.min():,.0f}")
            print(f"  median: {snp_numeric.median():,.0f}")
            print(f"  max:    {snp_numeric.max():,.0f}")
            print(f"  mean:   {snp_numeric.mean():,.0f}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Variation3: genome build detection, SNP count, harmonised file check per GWAS."
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
        help="Overwrite existing Feature3.csv"
    )

    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine all per-GWAS Feature3.csv files into one CSV."
    )

    parser.add_argument(
        "--combine-output",
        default="Variation3_allGWAS.csv",
        help="Output CSV for --combine. Default: Variation3_allGWAS.csv"
    )

    args = parser.parse_args()

    if args.combine:
        combine_feature3(
            jobs_file=args.jobs_file,
            output_file=args.combine_output,
        )
        return

    if args.index is None:
        raise ValueError("Please provide a GWAS job index, e.g. python Variation3.py 1")

    run_one_job(
        index=args.index,
        jobs_file=args.jobs_file,
        force=args.force,
    )


if __name__ == "__main__":
    main()
