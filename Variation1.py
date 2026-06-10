#!/usr/bin/env python3

import argparse
import gzip
import os
import re
import tarfile
import zipfile
from pathlib import Path

import pandas as pd


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


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


def sniff_compression_from_magic_bytes(path):
    try:
        with open(path, "rb") as f:
            first_bytes = f.read(512)

        if len(first_bytes) >= 2 and first_bytes[:2] == b"\x1f\x8b":
            return "gzip"

        if len(first_bytes) >= 4 and first_bytes[:4] == b"PK\x03\x04":
            return "zip"

        if len(first_bytes) >= 262 and first_bytes[257:262] == b"ustar":
            return "tar"

        return "plain"

    except Exception:
        return "unknown"


def resolve_compression(path):
    extension_guess = detect_compression(path)
    magic_guess = sniff_compression_from_magic_bytes(path)

    if extension_guess in ["tar.gz", "tar"] and magic_guess == "gzip":
        final = extension_guess
        source = "file_extension_tar_over_gzip_magic"
    elif magic_guess == "plain" and extension_guess != "plain":
        final = "plain"
        source = "magic_bytes_plain_override"
    elif magic_guess != "unknown" and magic_guess != "plain":
        final = magic_guess
        source = "magic_bytes"
    else:
        final = extension_guess
        source = "file_extension"

    mismatch = (
        magic_guess not in ["unknown", ""]
        and extension_guess != magic_guess
    )

    return {
        "compression_type": final,
        "compression_from_extension": extension_guess,
        "compression_from_magic_bytes": magic_guess,
        "compression_detection_source": source,
        "compression_suffix_magic_mismatch": mismatch,
    }


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
            f.seek(-4, os.SEEK_END)
            return int.from_bytes(f.read(4), "little")
    except Exception:
        return 0


def read_gzip_lines(path, max_lines):
    lines = []
    extracted_member = infer_gzip_member_name(path)
    extracted_size = get_gzip_uncompressed_size(path)

    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            for _ in range(max_lines):
                line = f.readline()
                if not line:
                    break
                lines.append(line.rstrip("\n"))

        return lines, "utf-8", "", extracted_member, extracted_size

    except Exception as e:
        return [], "utf-8", f"{type(e).__name__}: {e}", extracted_member, extracted_size


def read_zip_lines(path, max_lines):
    try:
        with zipfile.ZipFile(path) as z:
            members = [m for m in z.infolist() if not m.is_dir()]

            if not members:
                return [], "utf-8", "zip_empty", "", 0

            largest = max(members, key=lambda x: x.file_size)

            lines = []

            with z.open(largest.filename) as f:
                for i, raw_line in enumerate(f):
                    if i >= max_lines:
                        break

                    line = raw_line.decode("utf-8", errors="replace")
                    lines.append(line.rstrip("\n"))

            return lines, "utf-8", "", largest.filename, largest.file_size

    except Exception as e:
        return [], "utf-8", f"{type(e).__name__}: {e}", "", 0


def read_tar_lines(path, max_lines):
    try:
        mode = "r:*"

        with tarfile.open(path, mode) as tar:
            members = [m for m in tar.getmembers() if m.isfile()]

            if not members:
                return [], "utf-8", "tar_empty", "", 0

            largest = max(members, key=lambda x: x.size)

            f = tar.extractfile(largest)

            if f is None:
                return [], "utf-8", "tar_extract_failed", largest.name, largest.size

            lines = []

            for i, raw_line in enumerate(f):
                if i >= max_lines:
                    break

                line = raw_line.decode("utf-8", errors="replace")
                lines.append(line.rstrip("\n"))

            return lines, "utf-8", "", largest.name, largest.size

    except Exception as e:
        return [], "utf-8", f"{type(e).__name__}: {e}", "", 0


def read_sample_lines_with_mode(path, max_lines, compression):
    if compression == "gzip":
        return read_gzip_lines(path, max_lines)

    if compression == "zip":
        return read_zip_lines(path, max_lines)

    if compression in ["tar", "tar.gz"]:
        return read_tar_lines(path, max_lines)

    lines, encoding, issue = read_plain_lines(path, max_lines)
    return lines, encoding, issue, "", 0


def build_read_attempt_order(resolved_compression):
    primary = resolved_compression["compression_type"]
    extension_guess = resolved_compression["compression_from_extension"]
    magic_guess = resolved_compression["compression_from_magic_bytes"]

    attempts = []

    for candidate in [primary, extension_guess, magic_guess, "plain", "gzip", "zip", "tar"]:
        if candidate in ["unknown", ""] or candidate is None:
            continue
        if candidate not in attempts:
            attempts.append(candidate)

    return attempts


def read_sample_lines(path, max_lines):
    resolved = resolve_compression(path)
    attempts = build_read_attempt_order(resolved)
    issues = []

    for compression in attempts:
        lines, encoding, issue, extracted_member, extracted_size = read_sample_lines_with_mode(
            path,
            max_lines=max_lines,
            compression=compression,
        )

        if len(lines) > 0:
            return {
                "lines": lines,
                "encoding_used": encoding,
                "encoding_issues": " | ".join(issues),
                "extracted_member": extracted_member,
                "extracted_file_size_bytes": extracted_size,
                "resolved_compression": resolved,
                "reader_used": compression,
                "read_status": "read_ok",
            }

        issues.append(f"{compression}:{issue}")

    return {
        "lines": [],
        "encoding_used": "utf-8",
        "encoding_issues": " | ".join(issues),
        "extracted_member": "",
        "extracted_file_size_bytes": 0,
        "resolved_compression": resolved,
        "reader_used": "",
        "read_status": "read_failed_or_empty",
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


def has_quotes(lines):
    text = "\n".join(lines[:50])
    return '"' in text or "'" in text


def count_rows_fast(path, resolved_compression):
    attempts = build_read_attempt_order(resolved_compression)

    for compression in attempts:
        try:
            if compression == "gzip":
                with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                    return sum(1 for _ in f)

            if compression == "plain":
                with open(path, "rt", encoding="utf-8", errors="replace") as f:
                    return sum(1 for _ in f)

        except Exception:
            continue

    return ""


def profile_file(job_index, phenotype, accession, file_path, max_lines):
    file_path = Path(file_path)
    resolved_compression = resolve_compression(file_path)
    compression = resolved_compression["compression_type"]

    raw_size_bytes = file_path.stat().st_size if file_path.exists() else 0

    read_result = read_sample_lines(
        file_path,
        max_lines=max_lines,
    )
    lines = read_result["lines"]
    encoding = read_result["encoding_used"]
    encoding_issue = read_result["encoding_issues"]
    extracted_member = read_result["extracted_member"]
    extracted_size = read_result["extracted_file_size_bytes"]
    reader_used = read_result["reader_used"]

    delimiter, estimated_n_columns = detect_delimiter(lines)
    header_info = detect_header(lines, delimiter)

    headers = header_info["headers"]
    n_columns = len(headers) if headers else estimated_n_columns
    n_rows = count_rows_fast(file_path, resolved_compression)

    return {
        "job_index": job_index,
        "phenotype": phenotype,
        "accessionId": accession,
        "file_path": str(file_path),
        "file_name": file_path.name,
        "file_extension": get_file_extension(file_path),
        "compression_type": compression,
        "compression_from_extension": resolved_compression["compression_from_extension"],
        "compression_from_magic_bytes": resolved_compression["compression_from_magic_bytes"],
        "compression_detection_source": resolved_compression["compression_detection_source"],
        "compression_suffix_magic_mismatch": resolved_compression["compression_suffix_magic_mismatch"],
        "reader_used": reader_used,
        "raw_file_size_bytes": raw_size_bytes,
        "raw_file_size_mb": round(raw_size_bytes / (1024 ** 2), 3),
        "extracted_member": extracted_member,
        "extracted_file_size_bytes": extracted_size,
        "extracted_file_size_mb": round(extracted_size / (1024 ** 2), 3) if extracted_size else 0,
        "n_rows_physical_lines": n_rows,
        "n_columns": n_columns,
        "delimiter": delimiter,
        "quote_usage": has_quotes(lines),
        "comment_or_metadata_lines_before_header": header_info["comment_or_metadata_lines_before_header"],
        "header_detected": header_info["header_detected"],
        "header_row_index": header_info["header_row_index"],
        "encoding_used": encoding,
        "encoding_issues": encoding_issue,
        "raw_headers_joined": " | ".join(headers),
        "n_sample_lines_read": len(lines),
        "read_status": read_result["read_status"],
    }


def run_one_job(index, jobs_file, max_lines, force):
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

    row = sub.iloc[0]

    phenotype = clean_text(row["phenotype"])
    accession = clean_text(row["accessionId"])
    file_path = Path(clean_text(row["file_path"]))
    output_feature_file = Path(clean_text(row["output_feature_file"]))

    print("=" * 100)
    print("VARIATION 1: ONE GWAS FILE FEATURE EXTRACTION")
    print("=" * 100)
    print(f"[JOB INDEX] {index}")
    print(f"[PHENOTYPE] {phenotype}")
    print(f"[ACCESSION] {accession}")
    print(f"[FILE] {file_path}")
    print(f"[OUTPUT] {output_feature_file}")
    print("=" * 100)

    if not file_path.exists():
        raise FileNotFoundError(f"Missing GWAS file: {file_path}")

    if output_feature_file.exists() and not force:
        print(f"[SKIP] Feature file already exists: {output_feature_file}")
        return

    output_feature_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        feature_row = profile_file(
            job_index=index,
            phenotype=phenotype,
            accession=accession,
            file_path=file_path,
            max_lines=max_lines,
        )

    except Exception as e:
        raw_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

        feature_row = {
            "job_index": index,
            "phenotype": phenotype,
            "accessionId": accession,
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_extension": get_file_extension(file_path),
            "compression_type": detect_compression(file_path),
            "compression_from_extension": detect_compression(file_path),
            "compression_from_magic_bytes": sniff_compression_from_magic_bytes(file_path),
            "compression_detection_source": "profile_failed",
            "compression_suffix_magic_mismatch": False,
            "reader_used": "",
            "raw_file_size_bytes": raw_size,
            "raw_file_size_mb": round(raw_size / (1024 ** 2), 3),
            "extracted_member": "",
            "extracted_file_size_bytes": 0,
            "extracted_file_size_mb": 0,
            "n_rows_physical_lines": "",
            "n_columns": 0,
            "delimiter": "unknown",
            "quote_usage": False,
            "comment_or_metadata_lines_before_header": "",
            "header_detected": False,
            "header_row_index": "",
            "encoding_used": "",
            "encoding_issues": f"{type(e).__name__}: {e}",
            "raw_headers_joined": "",
            "n_sample_lines_read": 0,
            "read_status": "profile_failed",
        }

    pd.DataFrame([feature_row]).to_csv(output_feature_file, index=False)

    print()
    print("[SAVED]")
    print(output_feature_file)
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Variation1: process one GWAS file by job index and save Feature1.csv."
    )

    parser.add_argument(
        "index",
        type=int,
        help="GWAS job index from GWAS_jobs.csv"
    )

    parser.add_argument(
        "--jobs-file",
        default="GWAS_jobs.csv",
        help="GWAS job manifest file. Default: GWAS_jobs.csv"
    )

    parser.add_argument(
        "--max-lines",
        type=int,
        default=500,
        help="Number of sample lines to read. Default: 500"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing Feature1.csv"
    )

    args = parser.parse_args()

    run_one_job(
        index=args.index,
        jobs_file=args.jobs_file,
        max_lines=args.max_lines,
        force=args.force,
    )


if __name__ == "__main__":
    main()
