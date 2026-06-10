#!/usr/bin/env python3

import argparse
import contextlib
import gzip
import io
import re
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


STANDARD_COLUMNS = ["CHR", "SNPID", "rsID", "POS", "EA", "NEA", "BETA", "OR", "Z", "SE", "P", "N", "EAF", "INFO"]
FINAL_EXPORT_COLUMNS = [
    "SNPID",
    "CHR",
    "POS",
    "EA",
    "NEA",
    "EAF",
    "MAF",
    "BETA",
    "SE",
    "OR",
    "OR_95U",
    "OR_95L",
    "Z",
    "P",
    "N",
    "N_CASES",
    "N_CONTROLS",
    "INFO",
    "DIRECTION",
]


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


def select_zip_member(path):
    with zipfile.ZipFile(path) as z:
        members = [m for m in z.infolist() if not m.is_dir()]

        if not members:
            return "", 0

        largest = max(members, key=lambda x: x.file_size)
        return largest.filename, largest.file_size


def select_tar_member(path):
    mode = "r:gz" if str(path).lower().endswith((".tar.gz", ".tgz")) else "r:"

    with tarfile.open(path, mode) as tar:
        members = [m for m in tar.getmembers() if m.isfile()]

        if not members:
            return "", 0

        largest = max(members, key=lambda x: x.size)
        return largest.name, largest.size


def materialize_input_file(path):
    path = Path(path)
    compression = detect_compression(path)

    if compression in ["plain", "gzip"]:
        return str(path), compression, "", None

    if compression == "zip":
        member, _ = select_zip_member(path)
        if not member:
            raise ValueError(f"No readable GWAS member found in ZIP: {path}")

        tmpdir = tempfile.TemporaryDirectory(prefix="variation31_zip_")
        out_path = Path(tmpdir.name) / Path(member).name

        with zipfile.ZipFile(path) as z:
            with z.open(member) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

        return str(out_path), "plain", member, tmpdir

    if compression in ["tar", "tar.gz"]:
        member, _ = select_tar_member(path)
        if not member:
            raise ValueError(f"No readable GWAS member found in TAR: {path}")

        tmpdir = tempfile.TemporaryDirectory(prefix="variation31_tar_")
        out_path = Path(tmpdir.name) / Path(member).name
        mode = "r:gz" if str(path).lower().endswith((".tar.gz", ".tgz")) else "r:"

        with tarfile.open(path, mode) as tar:
            src = tar.extractfile(member)
            if src is None:
                raise ValueError(f"Could not extract TAR member: {member}")
            with src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

        return str(out_path), "plain", member, tmpdir

    return str(path), compression, "", None


def load_preview_dataframe(path, nrows):
    attempts = [
        {"sep": None, "engine": "python"},
        {"sep": "\t", "engine": "python"},
        {"sep": ",", "engine": "python"},
        {"sep": r"\s+", "engine": "python"},
        {"sep": ";", "engine": "python"},
        {"sep": "|", "engine": "python"},
    ]

    last_error = ""

    for attempt in attempts:
        try:
            df = pd.read_csv(
                path,
                nrows=nrows,
                dtype=str,
                comment="#",
                on_bad_lines="skip",
                compression="infer",
                **attempt,
            )
            df.columns = [clean_text(c) for c in df.columns]
            return df, attempt, ""
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"

    return pd.DataFrame(), {}, last_error


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


def normalise_gwaslab_build(build):
    build = clean_text(build)

    if build in ["19", "hg19", "HG19", "GRCh37", "grch37"]:
        return "19"
    if build in ["38", "hg38", "HG38", "GRCh38", "grch38"]:
        return "38"
    return ""


def detect_build_hint_from_path(path):
    name = str(path).lower()

    if any(token in name for token in ["build38", "grch38", "hg38"]):
        return "38"
    if any(token in name for token in ["build37", "grch37", "hg19"]):
        return "19"
    return ""


def normalise_label(text):
    return re.sub(r"[^a-z0-9]+", "", clean_text(text).lower())


def choose_preferred_column(columns, aliases):
    normalised = {normalise_label(col): col for col in columns}

    for alias in aliases:
        chosen = normalised.get(normalise_label(alias), "")
        if chosen:
            return chosen

    return ""


def parse_compound_variant_series(series):
    text = series.astype(str).map(clean_text)
    coordinate_pattern = re.compile(
        r"^(?:chr)?(?P<chr>[A-Za-z0-9]+)[:_](?P<pos>\d+)(?:[:_](?P<nea>[A-Za-z]+)[:_](?P<ea>[A-Za-z]+))?$",
        flags=re.IGNORECASE,
    )
    rs_allele_pattern = re.compile(
        r"^(?P<rs>rs\d+)[:_](?P<nea>[A-Za-z]+)[:_](?P<ea>[A-Za-z]+)$",
        flags=re.IGNORECASE,
    )

    parsed = text.str.extract(coordinate_pattern)
    parsed.columns = ["CHR", "POS", "PARSED_NEA", "PARSED_EA"]
    parsed = parsed.astype("object")
    parsed["PARSED_RS"] = pd.Series(pd.NA, index=parsed.index, dtype="object")

    rs_mask = text.str.match(r"^rs\d+$", case=False, na=False)
    parsed.loc[rs_mask, "PARSED_RS"] = text.loc[rs_mask]

    rs_allele = text.str.extract(rs_allele_pattern)
    rs_allele_mask = rs_allele["rs"].notna()
    parsed.loc[rs_allele_mask, "PARSED_RS"] = rs_allele.loc[rs_allele_mask, "rs"]
    parsed.loc[rs_allele_mask, "PARSED_NEA"] = rs_allele.loc[rs_allele_mask, "nea"]
    parsed.loc[rs_allele_mask, "PARSED_EA"] = rs_allele.loc[rs_allele_mask, "ea"]
    return parsed


def choose_best_compound_id_column(df):
    candidates = []
    for column in df.columns:
        label = normalise_label(column)
        if label in {
            "variantid", "markername", "marker", "snp", "snpid", "snp1", "snp2",
            "rsid", "rs", "id",
        }:
            candidates.append(column)

    best_column = ""
    best_parsed = None
    best_score = -1

    for column in candidates:
        parsed = parse_compound_variant_series(df[column])
        coordinate_count = int((parsed["CHR"].notna() & parsed["POS"].notna()).sum())
        allele_count = int((parsed["PARSED_NEA"].notna() & parsed["PARSED_EA"].notna()).sum())
        rsid_count = int(parsed["PARSED_RS"].notna().sum())
        score = (coordinate_count * 4) + (allele_count * 2) + rsid_count
        if score > best_score:
            best_column = column
            best_parsed = parsed
            best_score = score

    return best_column, best_parsed, best_score


def build_standardized_preview_dataframe(df):
    if df is None or df.empty:
        return df, {"parser_strategy": "empty_dataframe", "selected_columns": {}, "compound_id_column": ""}

    canonical_aliases = {
        "CHR": ["CHR", "chromosome", "chrom", "chr"],
        "POS": ["POS", "base_pair_location", "bp", "position", "basepairlocation"],
        "EA": ["effect_allele", "ea", "allele1", "a1", "tested_allele", "alt", "alt_allele"],
        "NEA": ["other_allele", "nea", "allele2", "a2", "ref", "ref_allele", "non_effect_allele"],
        "SNPID": ["variant_id", "snpid", "markername", "marker", "snp", "snp.1", "id"],
        "rsID": ["rsid", "rs_id", "rs"],
        "EAF": [
            "effect_allele_frequency", "eaf", "freq", "freq1", "freq.allele1.hapmapceu",
            "freq1.hapmap", "af", "alt_af",
        ],
        "BETA": ["beta", "b", "effect", "logor", "log_or"],
        "SE": ["standard_error", "se", "stderr", "stderrlogor", "std_error", "stderror", "stderr_logor"],
        "P": ["p_value", "p-value", "pval", "p"],
        "N": ["n", "samples", "sample_size"],
        "OR": ["odds_ratio", "or"],
        "OR_95L": ["ci_lower", "or_95l", "or95l", "lower_ci"],
        "OR_95U": ["ci_upper", "or_95u", "or95u", "upper_ci"],
        "Z": ["z_stat", "z", "zscore", "z-score"],
        "INFO": ["info"],
        "DIRECTION": ["direction"],
        "TEST": ["test"],
    }

    selected_columns = {}
    out = pd.DataFrame(index=df.index)

    for canonical, aliases in canonical_aliases.items():
        chosen = choose_preferred_column(df.columns, aliases)
        if chosen:
            selected_columns[canonical] = chosen
            out[canonical] = df[chosen]

    compound_column, compound_parsed, compound_score = choose_best_compound_id_column(df)
    parser_strategy = "direct_columns"

    if compound_column and compound_parsed is not None and compound_score > 0:
        parser_strategy = f"compound_id:{compound_column}"

        if "SNPID" not in out.columns:
            out["SNPID"] = df[compound_column]

        if "rsID" not in out.columns and compound_parsed["PARSED_RS"].notna().any():
            out["rsID"] = compound_parsed["PARSED_RS"]

        if "CHR" not in out.columns and compound_parsed["CHR"].notna().any():
            out["CHR"] = compound_parsed["CHR"]

        if "POS" not in out.columns and compound_parsed["POS"].notna().any():
            out["POS"] = compound_parsed["POS"]

        if "NEA" not in out.columns and compound_parsed["PARSED_NEA"].notna().any():
            out["NEA"] = compound_parsed["PARSED_NEA"]

        if "EA" not in out.columns and compound_parsed["PARSED_EA"].notna().any():
            out["EA"] = compound_parsed["PARSED_EA"]

    if "rsID" not in out.columns and "SNPID" in out.columns:
        snpid_text = out["SNPID"].astype(str).map(clean_text)
        rs_mask = snpid_text.str.match(r"^rs\d+$", case=False, na=False)
        if rs_mask.any():
            out["rsID"] = pd.Series(pd.NA, index=out.index, dtype="object")
            out.loc[rs_mask, "rsID"] = snpid_text.loc[rs_mask]
            parser_strategy = f"{parser_strategy}|rsid_from_snpid"

    # GWASLab maps both rsID and SNPID to SNPID. Consolidate them before
    # writing the standardized handoff file to prevent duplicate columns.
    if "rsID" in out.columns:
        rsid_text = out["rsID"].astype(str).map(clean_text)
        valid_rsid = rsid_text.str.match(r"^rs\d+$", case=False, na=False)

        if "SNPID" not in out.columns:
            out["SNPID"] = pd.Series(pd.NA, index=out.index, dtype="object")

        out.loc[valid_rsid, "SNPID"] = rsid_text.loc[valid_rsid]
        out = out.drop(columns=["rsID"])
        parser_strategy = f"{parser_strategy}|single_snpid_handoff"

    if "CHR" in out.columns:
        out["CHR"] = out["CHR"].astype(str).map(clean_text).str.replace("^chr", "", regex=True, case=False)

    for allele_column in ["EA", "NEA"]:
        if allele_column in out.columns:
            out[allele_column] = out[allele_column].astype(str).map(clean_text).str.upper()

    for id_column in ["SNPID", "rsID"]:
        if id_column in out.columns:
            out[id_column] = out[id_column].astype(str).map(clean_text)
            out.loc[out[id_column].isin(["", "nan", "NaN", "<NA>", "NA"]), id_column] = pd.NA

    preferred_order = [
        "SNPID", "CHR", "POS", "EA", "NEA", "EAF", "BETA", "SE", "OR",
        "OR_95U", "OR_95L", "Z", "P", "N", "INFO", "DIRECTION", "TEST",
    ]
    out = out[[col for col in preferred_order if col in out.columns]].copy()

    return out, {
        "parser_strategy": parser_strategy,
        "selected_columns": selected_columns,
        "compound_id_column": compound_column,
    }


def extract_dataframe(obj):
    if isinstance(obj, pd.DataFrame):
        return obj

    data = getattr(obj, "data", None)
    if isinstance(data, pd.DataFrame):
        return data

    return None


def run_gwaslab_preview(df, source_path):
    if df.empty:
        return None, None, "", "", "", "", "preview dataframe is empty"

    try:
        import gwaslab as gl
    except Exception as e:
        return None, None, "", "", "", "", f"GWASLab import failed: {type(e).__name__}: {e}"

    standardized_df, parser_meta = build_standardized_preview_dataframe(df)
    if standardized_df.empty:
        return None, None, "", "", "", "", "local standardization produced an empty dataframe"

    tmpdir = tempfile.TemporaryDirectory(prefix="variation31_gwaslab_")
    preview_path = Path(tmpdir.name) / "preview.tsv"
    standardized_df.to_csv(preview_path, sep="\t", index=False)

    log_buffer = io.StringIO()
    tee_stdout = TeeTextIO(sys.stdout, log_buffer)
    tee_stderr = TeeTextIO(sys.stderr, log_buffer)

    try:
        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
            sumstats = gl.Sumstats(str(preview_path), fmt="gwaslab")
            parsed = extract_dataframe(sumstats)
            build_hint = detect_build_hint_from_path(source_path)
            can_infer_build = {"CHR", "POS"}.issubset(standardized_df.columns)

            if build_hint:
                inferred_build = ""
                build_source = "filename_hint"
            elif can_infer_build:
                sumstats.infer_build()
                inferred_build_raw = clean_text(getattr(sumstats, "build", ""))
                inferred_build = normalise_gwaslab_build(inferred_build_raw)
                build_source = "gwaslab_infer_build"
            else:
                inferred_build = ""
                build_source = "no_chr_pos_for_build_inference"

            build_for_processing = build_hint or inferred_build

            if build_for_processing == "19":
                sumstats.liftover(from_build="19", to_build="38", remove=False)
                liftover_status = "liftover_19_to_38_applied"
                current_df = extract_dataframe(sumstats)
            elif build_for_processing == "38":
                liftover_status = "already_hg38_no_liftover_needed"
                current_df = extract_dataframe(sumstats)
            else:
                liftover_status = "build_unknown_liftover_skipped"
                current_df = extract_dataframe(sumstats)

            rsid_status = "rsid_assignment_skipped"
            current_has_rsid = False
            if current_df is not None and "SNPID" in current_df.columns:
                current_has_rsid = current_df["SNPID"].astype(str).str.match(
                    r"^rs\d+$", case=False, na=False
                ).any()

            if build_for_processing in ["19", "38"] or current_has_rsid:
                try:
                    snpid_built_count = 0
                    if current_df is not None:
                        current_df, snpid_built_count = fill_snpid_from_columns(current_df)
                    rsid_status = f"snpid_built_from_columns:{snpid_built_count}"

                    ref_key = "1kg_dbsnp151_hg38_auto"
                    ref_path, ref_status = ensure_local_reference(gl, ref_key)
                    rsid_status = f"{rsid_status}|{ref_status}"

                    current_df, rsid_meta = fill_rsid_from_reference(current_df, ref_path)
                    filled_fields = ",".join(
                        f"{key}:{value}"
                        for key, value in rsid_meta.get("reference_fill_counts", {}).items()
                        if value
                    ) or "none"
                    rsid_status = (
                        f"{rsid_status}|manual_rsid_join:{rsid_meta['match_mode']}"
                        f"|rsid_filled:{rsid_meta['filled_count']}"
                        f"|rsid_still_missing:{rsid_meta['still_missing_count']}"
                        f"|reference_fields_filled:{filled_fields}"
                    )

                    if not build_for_processing and {
                        "CHR", "POS"
                    }.issubset(current_df.columns):
                        build_for_processing = "38"
                        liftover_status = "hg38_coordinates_filled_from_rsid_reference"
                except Exception as rsid_error:
                    rsid_status = f"rsid_assignment_failed:{type(rsid_error).__name__}:{rsid_error}"

            parsed_hg38 = current_df

        if parsed is None:
            return None, None, log_buffer.getvalue(), "", "", "", "GWASLab did not expose a dataframe"

        if parsed_hg38 is None:
            return (
                parsed.copy(),
                None,
                log_buffer.getvalue(),
                liftover_status,
                rsid_status,
                f"build_source:{build_source}|build_hint:{build_hint or 'none'}|gwaslab_inferred:{inferred_build or 'skipped'}|processing_build:{build_for_processing or 'unknown'}|parser_strategy:{parser_meta['parser_strategy']}",
                "GWASLab liftover result dataframe is missing",
            )

        return (
            parsed.copy(),
            parsed_hg38.copy(),
            log_buffer.getvalue(),
            liftover_status,
            rsid_status,
            f"build_source:{build_source}|build_hint:{build_hint or 'none'}|gwaslab_inferred:{inferred_build or 'skipped'}|processing_build:{build_for_processing or 'unknown'}|parser_strategy:{parser_meta['parser_strategy']}",
            "",
        )
    except Exception as e:
        return None, None, log_buffer.getvalue(), "", "", f"parser_strategy:{parser_meta['parser_strategy']}", f"{type(e).__name__}: {e}"
    finally:
        tmpdir.cleanup()


def print_value_preview(series, label, max_unique=8):
    if series is None or series.empty:
        print(f"{label:<8} <missing>")
        return

    values = series.dropna().astype(str).map(clean_text)
    values = values[values != ""]

    if values.empty:
        print(f"{label:<8} <all missing>")
        return

    examples = values.head(max_unique).tolist()
    print(f"{label:<8} {examples}")


def derive_se_if_missing(df):
    if df is None or df.empty:
        return df, {"derived_se_count": 0, "sources": {}}

    out = df.copy()

    if "SE" in out.columns:
        se_numeric = pd.to_numeric(out["SE"], errors="coerce")
        se_missing = se_numeric.isna()
    else:
        se_numeric = pd.Series([pd.NA] * len(out), index=out.index, dtype="object")
        se_missing = pd.Series([True] * len(out), index=out.index)

    derived_source = pd.Series([""] * len(out), index=out.index, dtype="object")
    derived_values = pd.Series([pd.NA] * len(out), index=out.index, dtype="object")

    z_numeric = pd.to_numeric(out["Z"], errors="coerce") if "Z" in out.columns else pd.Series([pd.NA] * len(out), index=out.index)
    beta_numeric = pd.to_numeric(out["BETA"], errors="coerce") if "BETA" in out.columns else pd.Series([pd.NA] * len(out), index=out.index)
    or_numeric = pd.to_numeric(out["OR"], errors="coerce") if "OR" in out.columns else pd.Series([pd.NA] * len(out), index=out.index)
    or95u_numeric = pd.to_numeric(out["OR_95U"], errors="coerce") if "OR_95U" in out.columns else pd.Series([pd.NA] * len(out), index=out.index)
    or95l_numeric = pd.to_numeric(out["OR_95L"], errors="coerce") if "OR_95L" in out.columns else pd.Series([pd.NA] * len(out), index=out.index)

    valid_z = z_numeric.notna() & (z_numeric != 0)

    # 1. BETA and Z
    mask_beta_z = se_missing & derived_source.eq("") & beta_numeric.notna() & valid_z
    derived_values.loc[mask_beta_z] = (beta_numeric.loc[mask_beta_z] / z_numeric.loc[mask_beta_z]).abs()
    derived_source.loc[mask_beta_z] = "BETA_over_Z"

    # 2. OR and Z
    valid_or = or_numeric.notna() & (or_numeric > 0)
    mask_or_z = se_missing & derived_source.eq("") & valid_or & valid_z
    derived_values.loc[mask_or_z] = (pd.Series(np.log(or_numeric.loc[mask_or_z]), index=or_numeric.loc[mask_or_z].index) / z_numeric.loc[mask_or_z]).abs()
    derived_source.loc[mask_or_z] = "logOR_over_Z"

    # 3. OR and confidence interval
    valid_ci = valid_or & or95u_numeric.notna() & or95l_numeric.notna() & (or95u_numeric > 0) & (or95l_numeric > 0)
    mask_or_ci = se_missing & derived_source.eq("") & valid_ci
    if mask_or_ci.any():
        upper = pd.Series(np.log(or95u_numeric.loc[mask_or_ci]), index=or95u_numeric.loc[mask_or_ci].index)
        lower = pd.Series(np.log(or95l_numeric.loc[mask_or_ci]), index=or95l_numeric.loc[mask_or_ci].index)
        derived_values.loc[mask_or_ci] = (upper - lower) / 3.92
        derived_source.loc[mask_or_ci] = "OR_CI"

    derived_numeric = pd.to_numeric(derived_values, errors="coerce")
    fill_mask = se_missing & derived_numeric.notna()

    if "SE" not in out.columns:
        out["SE"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="object")

    out.loc[fill_mask, "SE"] = derived_numeric.loc[fill_mask]
    out["SE_DERIVED_SOURCE"] = derived_source

    source_counts = derived_source[derived_source != ""].value_counts().to_dict()
    return out, {
        "derived_se_count": int(fill_mask.sum()),
        "sources": source_counts,
    }


def derive_beta_or_if_missing(df):
    if df is None or df.empty:
        return df, {
            "derived_beta_count": 0,
            "derived_or_count": 0,
            "beta_sources": {},
            "or_sources": {},
        }

    out = df.copy()

    if "BETA" not in out.columns:
        out["BETA"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="object")
    if "OR" not in out.columns:
        out["OR"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="object")

    beta_numeric = pd.to_numeric(out["BETA"], errors="coerce")
    or_numeric = pd.to_numeric(out["OR"], errors="coerce")

    beta_missing = beta_numeric.isna()
    or_missing = or_numeric.isna()
    valid_or = or_numeric.notna() & (or_numeric > 0)
    valid_beta = beta_numeric.notna()

    beta_derived_source = pd.Series([""] * len(out), index=out.index, dtype="object")
    or_derived_source = pd.Series([""] * len(out), index=out.index, dtype="object")

    beta_from_or = beta_missing & valid_or
    if beta_from_or.any():
        out.loc[beta_from_or, "BETA"] = np.log(or_numeric.loc[beta_from_or])
        beta_derived_source.loc[beta_from_or] = "logOR"

    or_from_beta = or_missing & valid_beta
    if or_from_beta.any():
        out.loc[or_from_beta, "OR"] = np.exp(beta_numeric.loc[or_from_beta])
        or_derived_source.loc[or_from_beta] = "expBETA"

    out["BETA_DERIVED_SOURCE"] = beta_derived_source
    out["OR_DERIVED_SOURCE"] = or_derived_source

    return out, {
        "derived_beta_count": int(beta_from_or.sum()),
        "derived_or_count": int(or_from_beta.sum()),
        "beta_sources": beta_derived_source[beta_derived_source != ""].value_counts().to_dict(),
        "or_sources": or_derived_source[or_derived_source != ""].value_counts().to_dict(),
    }


def fill_snpid_from_columns(df):
    if df is None or df.empty:
        return df, 0

    out = df.copy()
    required = ["CHR", "POS", "NEA", "EA"]
    if not all(col in out.columns for col in required):
        return out, 0

    if "SNPID" not in out.columns:
        out["SNPID"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="object")

    snpid_missing = out["SNPID"].isna() | out["SNPID"].astype(str).str.strip().isin(["", "nan", "NA", "<NA>"])

    chr_values = out["CHR"].astype(str).map(clean_text)
    pos_values = out["POS"].astype(str).map(clean_text)
    nea_values = out["NEA"].astype(str).map(clean_text).str.upper()
    ea_values = out["EA"].astype(str).map(clean_text).str.upper()

    buildable = (
        snpid_missing
        & chr_values.ne("")
        & pos_values.ne("")
        & nea_values.ne("")
        & ea_values.ne("")
        & ~chr_values.str.lower().isin(["nan", "na", "<na>"])
        & ~pos_values.str.lower().isin(["nan", "na", "<na>"])
        & ~nea_values.str.lower().isin(["nan", "na", "<na>"])
        & ~ea_values.str.lower().isin(["nan", "na", "<na>"])
    )

    out.loc[buildable, "SNPID"] = (
        chr_values.loc[buildable]
        + ":"
        + pos_values.loc[buildable]
        + ":"
        + nea_values.loc[buildable]
        + ":"
        + ea_values.loc[buildable]
    )

    return out, int(buildable.sum())


def clean_chr_for_join(series):
    return series.astype(str).map(clean_text).str.replace("^chr", "", regex=True, case=False)


def clean_allele_for_join(series):
    return series.astype(str).map(clean_text).str.upper()


def ensure_local_reference(gl, ref_key):
    local_ref_path = Path.cwd() / f"{ref_key}.txt.gz"
    if local_ref_path.exists():
        return str(local_ref_path), f"rsid_reference_local:{local_ref_path}"

    ref_path = gl.get_path(ref_key)
    if not ref_path:
        gl.download_ref(ref_key)
        ref_path = gl.get_path(ref_key)

    if not ref_path:
        raise FileNotFoundError(f"GWASLab reference path unavailable after download: {ref_key}")

    shutil.copy2(ref_path, local_ref_path)
    return str(local_ref_path), f"rsid_reference_downloaded_to_cwd:{local_ref_path}"


def fill_rsid_from_reference(gwas_df, ref_path):
    if gwas_df is None or gwas_df.empty:
        return gwas_df, {"filled_count": 0, "still_missing_count": 0, "match_mode": "empty_dataframe"}

    gwas = gwas_df.copy()
    gwas, _ = fill_snpid_from_columns(gwas)

    ref = pd.read_csv(ref_path, sep="\t", dtype=str, compression="infer")
    ref.columns = [clean_text(c) for c in ref.columns]

    rename = {}
    for c in ref.columns:
        cl = c.lower()
        if cl in ["chr", "chrom", "chromosome"]:
            rename[c] = "CHR"
        elif cl in ["pos", "bp", "base_pair_location", "position"]:
            rename[c] = "POS"
        elif cl in ["rsid", "rs_id", "rs"]:
            rename[c] = "rsID"
        elif cl in ["ea", "effect_allele", "a1", "alt"]:
            rename[c] = "EA"
        elif cl in ["nea", "other_allele", "a2", "ref"]:
            rename[c] = "NEA"
        elif cl == "snpid":
            rename[c] = "SNPID"

    ref = ref.rename(columns=rename)

    required = ["CHR", "POS", "rsID"]
    missing_required = [c for c in required if c not in ref.columns]
    if missing_required:
        raise ValueError(f"Reference is missing required columns: {missing_required}")

    if "rsID" not in gwas.columns:
        gwas["rsID"] = pd.NA

    if "SNPID" in gwas.columns:
        snpid_text = gwas["SNPID"].astype(str).map(clean_text)
        valid_snpid_rsid = snpid_text.str.match(r"^rs\d+$", case=False, na=False)
        missing_rsid = gwas["rsID"].isna() | gwas["rsID"].astype(str).str.strip().isin(
            ["", "nan", "NaN", "<NA>"]
        )
        gwas.loc[missing_rsid & valid_snpid_rsid, "rsID"] = snpid_text.loc[
            missing_rsid & valid_snpid_rsid
        ]

    # rsID-only files can be enriched directly from the hg38 reference before
    # coordinate/allele matching. Existing non-missing fields are preserved.
    reference_fill_counts = {}
    valid_ref_rsid = ref["rsID"].astype(str).str.match(r"^rs\d+$", case=False, na=False)
    ref_by_rsid = ref.loc[valid_ref_rsid].drop_duplicates("rsID", keep="first").set_index("rsID")
    gwas_rsid = gwas["rsID"].astype(str).map(clean_text)

    for column in ["CHR", "POS", "EA", "NEA"]:
        if column not in ref_by_rsid.columns:
            continue

        if column not in gwas.columns:
            gwas[column] = pd.NA

        missing = gwas[column].isna() | gwas[column].astype(str).str.strip().isin(
            ["", "nan", "NaN", "<NA>", "NA"]
        )
        mapped = gwas_rsid.map(ref_by_rsid[column])
        fill_mask = missing & mapped.notna()
        gwas.loc[fill_mask, column] = mapped.loc[fill_mask]
        reference_fill_counts[column] = int(fill_mask.sum())

    if not {"CHR", "POS"}.issubset(gwas.columns):
        missing_after = gwas["rsID"].isna() | gwas["rsID"].astype(str).str.strip().isin(
            ["", "nan", "NaN", "<NA>"]
        )
        return gwas, {
            "filled_count": int((~missing_after).sum()),
            "still_missing_count": int(missing_after.sum()),
            "match_mode": "rsid_reference_enrichment_no_coordinates",
            "reference_fill_counts": reference_fill_counts,
        }

    gwas["CHR"] = clean_chr_for_join(gwas["CHR"])
    ref["CHR"] = clean_chr_for_join(ref["CHR"])
    gwas["POS"] = pd.to_numeric(gwas["POS"], errors="coerce").astype("Int64").astype(str)
    ref["POS"] = pd.to_numeric(ref["POS"], errors="coerce").astype("Int64").astype(str)

    missing_before = gwas["rsID"].isna() | gwas["rsID"].astype(str).str.strip().isin(["", "nan", "NaN", "<NA>"])

    match_mode = "position_only"

    if all(col in ref.columns for col in ["EA", "NEA"]):
        match_mode = "chr_pos_alleles_direct_and_reverse"
        gwas["EA"] = clean_allele_for_join(gwas["EA"])
        gwas["NEA"] = clean_allele_for_join(gwas["NEA"])
        ref["EA"] = clean_allele_for_join(ref["EA"])
        ref["NEA"] = clean_allele_for_join(ref["NEA"])

        ref_direct = ref[["CHR", "POS", "EA", "NEA", "rsID"]].drop_duplicates()
        merged = gwas.merge(ref_direct, on=["CHR", "POS", "EA", "NEA"], how="left", suffixes=("", "_REF"))
        fill_mask = missing_before & merged["rsID_REF"].notna()
        gwas.loc[fill_mask, "rsID"] = merged.loc[fill_mask, "rsID_REF"].values

        still_missing = gwas["rsID"].isna() | gwas["rsID"].astype(str).str.strip().isin(["", "nan", "NaN", "<NA>"])
        ref_rev = ref[["CHR", "POS", "EA", "NEA", "rsID"]].copy()
        ref_rev = ref_rev.rename(columns={"EA": "NEA", "NEA": "EA", "rsID": "rsID_REF"})
        merged_rev = gwas.merge(ref_rev, on=["CHR", "POS", "EA", "NEA"], how="left")
        fill_mask = still_missing & merged_rev["rsID_REF"].notna()
        gwas.loc[fill_mask, "rsID"] = merged_rev.loc[fill_mask, "rsID_REF"].values

    elif "SNPID" in ref.columns:
        match_mode = "snpid_direct"
        gwas["SNPID"] = gwas["SNPID"].astype(str).map(clean_text)
        ref["SNPID"] = ref["SNPID"].astype(str).map(clean_text)
        ref_snpid = ref[["SNPID", "rsID"]].drop_duplicates()
        merged = gwas.merge(ref_snpid, on="SNPID", how="left", suffixes=("", "_REF"))
        fill_mask = missing_before & merged["rsID_REF"].notna()
        gwas.loc[fill_mask, "rsID"] = merged.loc[fill_mask, "rsID_REF"].values

    else:
        ref_pos = ref[["CHR", "POS", "rsID"]].drop_duplicates(["CHR", "POS"], keep="first")
        merged = gwas.merge(ref_pos, on=["CHR", "POS"], how="left", suffixes=("", "_REF"))
        fill_mask = missing_before & merged["rsID_REF"].notna()
        gwas.loc[fill_mask, "rsID"] = merged.loc[fill_mask, "rsID_REF"].values

    missing_after = gwas["rsID"].isna() | gwas["rsID"].astype(str).str.strip().isin(["", "nan", "NaN", "<NA>"])
    return gwas, {
        "filled_count": int((missing_before & ~missing_after).sum()),
        "still_missing_count": int(missing_after.sum()),
        "match_mode": match_mode,
        "reference_fill_counts": reference_fill_counts,
    }


def replace_snpid_with_rsid(df):
    if df is None or df.empty or "rsID" not in df.columns:
        return df

    out = df.copy()
    if "SNPID" not in out.columns:
        out["SNPID"] = pd.NA

    rsid_present = ~(
        out["rsID"].isna()
        | out["rsID"].astype(str).str.strip().isin(["", "nan", "NaN", "<NA>"])
    )
    out.loc[rsid_present, "SNPID"] = out.loc[rsid_present, "rsID"]
    return out


def sanitize_final_snpid(df):
    if df is None or df.empty:
        return df, 0

    out = df.copy()
    if "SNPID" not in out.columns:
        return out, 0

    snpid_values = out["SNPID"].astype(str).map(clean_text)
    keep_mask = snpid_values.str.match(r"^rs\d+$", case=False, na=False)
    clear_mask = snpid_values.ne("") & ~snpid_values.isin(["nan", "NaN", "<NA>", "NA"]) & ~keep_mask
    out.loc[clear_mask, "SNPID"] = pd.NA
    return out, int(clear_mask.sum())


def prepare_final_gwas(df):
    if df is None or df.empty:
        return df, {
            "synthetic_snpid_cleared": 0,
            "rsid_present_count": 0,
            "beta_or_meta": {
                "derived_beta_count": 0,
                "derived_or_count": 0,
                "beta_sources": {},
                "or_sources": {},
            },
        }

    out, beta_or_meta = derive_beta_or_if_missing(df)
    out, se_meta = derive_se_if_missing(out)
    out = replace_snpid_with_rsid(out)
    out, synthetic_cleared = sanitize_final_snpid(out)

    present_columns = [col for col in FINAL_EXPORT_COLUMNS if col in out.columns]
    out = out[present_columns].copy()

    rsid_present_count = 0
    if "SNPID" in out.columns:
        snpid_present = ~(
            out["SNPID"].isna()
            | out["SNPID"].astype(str).str.strip().isin(["", "nan", "NaN", "<NA>"])
        )
        rsid_present_count = int(snpid_present.sum())

    return out, {
        "synthetic_snpid_cleared": synthetic_cleared,
        "rsid_present_count": rsid_present_count,
        "beta_or_meta": beta_or_meta,
        "se_meta": se_meta,
    }


def save_final_gwas(df, source_file):
    output_path = Path(source_file).parent / "FinalGWAS.csv"
    df.to_csv(output_path, index=False)
    return output_path


def parse_numeric_metadata_value(value):
    text = clean_text(value).replace(",", "")
    if text == "":
        return pd.NA

    try:
        numeric = float(text)
    except Exception:
        return pd.NA

    if not np.isfinite(numeric):
        return pd.NA

    return int(round(numeric))


def load_phenotype_sample_metadata(phenotype, accession, base_dir=None):
    phenotype = clean_text(phenotype)
    accession = clean_text(accession)
    base_dir = Path(base_dir or Path.cwd())

    result = {
        "phenotype_metadata_file": "",
        "phenotype_metadata_found": False,
        "sample_n": pd.NA,
        "sample_cases": pd.NA,
        "sample_controls": pd.NA,
    }

    if not phenotype or not accession:
        return result

    metadata_path = base_dir / f"{phenotype}.csv"
    result["phenotype_metadata_file"] = str(metadata_path)

    if not metadata_path.exists():
        return result

    metadata = pd.read_csv(metadata_path, low_memory=False)
    if metadata.empty or "accessionId" not in metadata.columns:
        return result

    accession_series = metadata["accessionId"].astype(str).str.strip().str.lower()
    matched = metadata[accession_series == accession.lower()]
    if matched.empty:
        return result

    row = matched.iloc[0]
    result["phenotype_metadata_found"] = True
    result["sample_n"] = parse_numeric_metadata_value(row.get("SAMPLES", pd.NA))
    result["sample_cases"] = parse_numeric_metadata_value(row.get("CASES", pd.NA))
    result["sample_controls"] = parse_numeric_metadata_value(row.get("CONTROLS", pd.NA))
    return result


def fill_sample_size_from_metadata(df, phenotype, accession, base_dir=None):
    if df is None or df.empty:
        return df, {
            "phenotype_metadata_found": False,
            "filled_N_rows": 0,
            "filled_N_CASES_rows": 0,
            "filled_N_CONTROLS_rows": 0,
            "sample_n": pd.NA,
            "sample_cases": pd.NA,
            "sample_controls": pd.NA,
            "phenotype_metadata_file": "",
        }

    metadata = load_phenotype_sample_metadata(phenotype, accession, base_dir=base_dir)
    out = df.copy()

    def fill_column(column_name, fill_value):
        if pd.isna(fill_value):
            return 0

        if column_name not in out.columns:
            out[column_name] = pd.Series([pd.NA] * len(out), index=out.index, dtype="object")

        missing_mask = (
            out[column_name].isna()
            | out[column_name].astype(str).str.strip().isin(["", "nan", "NaN", "<NA>", "NA"])
        )
        out.loc[missing_mask, column_name] = fill_value
        return int(missing_mask.sum())

    filled_n = fill_column("N", metadata["sample_n"])
    filled_cases = fill_column("N_CASES", metadata["sample_cases"])
    filled_controls = fill_column("N_CONTROLS", metadata["sample_controls"])

    return out, {
        "phenotype_metadata_found": metadata["phenotype_metadata_found"],
        "filled_N_rows": filled_n,
        "filled_N_CASES_rows": filled_cases,
        "filled_N_CONTROLS_rows": filled_controls,
        "sample_n": metadata["sample_n"],
        "sample_cases": metadata["sample_cases"],
        "sample_controls": metadata["sample_controls"],
        "phenotype_metadata_file": metadata["phenotype_metadata_file"],
    }


def resolve_job_from_manifest(index, jobs_file):
    jobs_path = Path(jobs_file)
    if not jobs_path.exists():
        raise FileNotFoundError(f"Missing jobs file: {jobs_path}")

    jobs = pd.read_csv(jobs_path)
    required = ["job_index", "phenotype", "accessionId", "file_path"]
    missing = [c for c in required if c not in jobs.columns]
    if missing:
        raise ValueError(f"Missing columns in {jobs_path}: {missing}")

    sub = jobs[jobs["job_index"].astype(int) == int(index)]
    if sub.empty:
        raise ValueError(f"No job found for index: {index}")

    job = sub.iloc[0]
    return {
        "job_index": int(job["job_index"]),
        "phenotype": clean_text(job["phenotype"]),
        "accessionId": clean_text(job["accessionId"]),
        "file_path": clean_text(job["file_path"]),
    }


def infer_metadata_from_file_path(file_path):
    path = Path(file_path)
    parts = [part for part in path.as_posix().split("/") if part]

    phenotype = ""
    accession = ""

    if "allgwas" in parts:
        idx = parts.index("allgwas")
        if idx >= 1:
            phenotype = parts[idx - 1]
        if idx + 1 < len(parts):
            accession = parts[idx + 1]

    return phenotype, accession


def manual_check(file_path, label, nrows, phenotype="", accession=""):
    file_path = Path(file_path)
    materialized_path, effective_compression, member, tmp_resource = materialize_input_file(file_path)
    if not phenotype or not accession:
        inferred_phenotype, inferred_accession = infer_metadata_from_file_path(file_path)
        phenotype = phenotype or inferred_phenotype
        accession = accession or inferred_accession

    try:
        df, read_attempt, read_error = load_preview_dataframe(materialized_path, nrows=nrows)

        print("=" * 100)
        print("VARIATION 3.1 MANUAL CHECK")
        print("=" * 100)
        print(f"[LABEL]                {label}")
        print(f"[GWAS FILE]            {file_path}")
        print(f"[COMPRESSION]          {detect_compression(file_path)}")
        print(f"[ARCHIVE MEMBER]       {member if member else '<none>'}")
        print(f"[PREVIEW ROW LIMIT]    {nrows if nrows is not None else 'FULL FILE'}")
        print(f"[PANDAS READ SUCCESS]  {not df.empty}")
        print(f"[PANDAS READ MODE]     {read_attempt if read_attempt else '<failed>'}")
        if read_error:
            print(f"[PANDAS READ ERROR]    {read_error}")
        print("=" * 100)

        print()
        print("[RAW HEADERS]")
        if df.empty:
            print("<no rows loaded>")
        else:
            for i, col in enumerate(df.columns, start=1):
                print(f"{i:>3}. {col}")

        print()
        print("[RAW PREVIEW SHAPE]")
        print(f"rows={len(df)} cols={len(df.columns)}")

        if not df.empty:
            print()
            print("[RAW FIRST 5 ROWS]")
            with pd.option_context("display.max_columns", None, "display.width", 160):
                print(df.head(5).to_string(index=False))

        parsed_df, parsed_hg38_df, gwaslab_log, liftover_status, rsid_status, build_status, gwaslab_error = run_gwaslab_preview(df, file_path)

        print()
        print("=" * 100)
        print("[GWASLAB PREVIEW]")
        print("=" * 100)
        print(f"[GWASLAB PARSE SUCCESS] {parsed_df is not None}")
        if gwaslab_error:
            print(f"[GWASLAB ERROR]         {gwaslab_error}")

        inferred_formats = parse_gwaslab_inferred_formats(gwaslab_log)
        if inferred_formats:
            print("[GWASLAB TOP FORMATS]")
            for item in inferred_formats[:3]:
                print(f"  {item['rank']}. {item['name']} (score={item['score']})")

        if parsed_df is not None:
            parsed_df, se_meta = derive_se_if_missing(parsed_df)
            print()
            print("[GWASLAB COLUMNS]")
            print(" | ".join(parsed_df.columns.astype(str)))

            print()
            print("[GWASLAB FIRST 5 ROWS]")
            with pd.option_context("display.max_columns", None, "display.width", 160):
                print(parsed_df.head(5).to_string(index=False))

            print()
            print("[SE DERIVATION]")
            print(f"derived_se_count = {se_meta['derived_se_count']}")
            if se_meta["sources"]:
                for source_name, source_count in se_meta["sources"].items():
                    print(f"  {source_name}: {source_count}")
            else:
                print("  no SE values were derived")

            print()
            print("[KEY COLUMN VALUE PREVIEW]")
            for col in STANDARD_COLUMNS:
                if col in parsed_df.columns:
                    print_value_preview(parsed_df[col], col)
                else:
                    print(f"{col:<8} <missing>")

        print()
        print("=" * 100)
        print("[GWASLAB HG38 PREVIEW]")
        print("=" * 100)
        print(f"[BUILD STATUS]          {build_status if build_status else '<not available>'}")
        print(f"[LIFTOVER STATUS]       {liftover_status if liftover_status else '<not available>'}")
        print(f"[RSID STATUS]           {rsid_status if rsid_status else '<not available>'}")

        if parsed_hg38_df is None:
            print("[GWASLAB HG38 SUCCESS]  False")
        else:
            parsed_hg38_df, final_meta = prepare_final_gwas(parsed_hg38_df)
            parsed_hg38_df, sample_meta = fill_sample_size_from_metadata(
                parsed_hg38_df,
                phenotype=phenotype,
                accession=accession,
                base_dir=Path.cwd(),
            )
            se_hg38_meta = final_meta["se_meta"]
            print("[GWASLAB HG38 SUCCESS]  True")
            print()
            print("[GWASLAB HG38 COLUMNS]")
            print(" | ".join(parsed_hg38_df.columns.astype(str)))

            print()
            print("[GWASLAB HG38 FIRST 5 ROWS]")
            with pd.option_context("display.max_columns", None, "display.width", 160):
                print(parsed_hg38_df.head(5).to_string(index=False))

            print()
            print("[HG38 SE DERIVATION]")
            print(f"derived_se_count = {se_hg38_meta['derived_se_count']}")
            if se_hg38_meta["sources"]:
                for source_name, source_count in se_hg38_meta["sources"].items():
                    print(f"  {source_name}: {source_count}")
            else:
                print("  no SE values were derived")

            print()
            print("[HG38 KEY COLUMN VALUE PREVIEW]")
            for col in STANDARD_COLUMNS:
                if col in parsed_hg38_df.columns:
                    print_value_preview(parsed_hg38_df[col], col)
                else:
                    print(f"{col:<8} <missing>")

            print()
            print("[FINAL GWAS STATUS]")
            print(f"derived_beta_count      = {final_meta['beta_or_meta']['derived_beta_count']}")
            print(f"derived_or_count        = {final_meta['beta_or_meta']['derived_or_count']}")
            print(f"synthetic_snpid_cleared = {final_meta['synthetic_snpid_cleared']}")
            print(f"rsid_present_count      = {final_meta['rsid_present_count']}")
            print(f"phenotype_metadata_used = {sample_meta['phenotype_metadata_found']}")
            print(f"filled_N_rows           = {sample_meta['filled_N_rows']}")
            print(f"filled_N_CASES_rows     = {sample_meta['filled_N_CASES_rows']}")
            print(f"filled_N_CONTROLS_rows  = {sample_meta['filled_N_CONTROLS_rows']}")

            if nrows is None:
                final_output_path = save_final_gwas(parsed_hg38_df, file_path)
                print(f"saved_final_gwas        = {final_output_path}")
            else:
                print("saved_final_gwas        = skipped_preview_mode_use_--full-file")

        print("=" * 100)

    finally:
        if tmp_resource is not None:
            tmp_resource.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Manual GWAS check and final standardized GWAS export using pandas and GWASLab."
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
        "--file-path",
        default="",
        help="Direct GWAS file path. Use this instead of a job index if needed."
    )

    parser.add_argument(
        "--label",
        default="manual_check",
        help="Label to print in the output."
    )

    parser.add_argument(
        "--nrows",
        type=int,
        default=500,
        help="Number of rows to preview with pandas. Default: 500"
    )

    parser.add_argument(
        "--full-file",
        action="store_true",
        help="Load the complete file instead of only the first preview rows."
    )

    args = parser.parse_args()

    if args.file_path:
        file_path = args.file_path
        label = args.label
        phenotype, accession = infer_metadata_from_file_path(file_path)
    elif args.index is not None:
        job = resolve_job_from_manifest(args.index, args.jobs_file)
        file_path = job["file_path"]
        label = f"{job['phenotype']} | {job['accessionId']} | job_index={job['job_index']}"
        phenotype = job["phenotype"]
        accession = job["accessionId"]
    else:
        raise ValueError("Provide either a job index or --file-path.")

    manual_check(
        file_path=file_path,
        label=label,
        nrows=None if args.full_file else args.nrows,
        phenotype=phenotype,
        accession=accession,
    )


if __name__ == "__main__":
    main()
