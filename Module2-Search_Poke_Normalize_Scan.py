#!/usr/bin/env python3

import argparse
import gzip
import hashlib
import html
import io
import os
import re
import tarfile
import time
import zipfile
import zlib
from pathlib import Path
from urllib.parse import urljoin, unquote

import pandas as pd
import requests
from bs4 import BeautifulSoup


# =============================================================================
# Detailed GWAS header mapping
# =============================================================================

COLUMN_SYNONYMS = {
    "CHR": [
        "chr", "chrom", "chromosome", "chromo", "chro", "ch", "#chrom",
        "chr_build36", "chr_pos_b36", "chr:pos", "chr:position",
        "chromosome_name", "chromosome_id",
        "chromosome_position_reference_allele_other_allele_b37"
    ],
    "SNP": [
        "snp", "markername", "marker", "rs", "rsid", "snpid", "snp_id",
        "snp_ids", "snpname", "rs_number", "rs_numbers", "rs_id",
        "rs_id_all", "rsnumber", "rsq", "snprsid", "dbsnprsid",
        "dbsnprsid_marker", "dbsnprsid/marker", "variant_id",
        "variantid", "variant_id_when_present", "var_id", "varid",
        "rs_marker_id", "marker_id", "markerid", "locus_id", "uniqid",
        "oldid", "illuminasnp", "vcf_ids", "pmarkername", "id",
        "name", "var_name"
    ],
    "BP": [
        "bp", "b", "pos", "position", "phys_pos", "bpos", "genpos",
        "base_pair_location", "base_pair_position", "base_pair",
        "base_pair_locations", "base_pair_location_grch37",
        "base_pair_location_grch38", "base_pair_lo", "base_pa", "base_",
        "snp_pos", "positionhg19", "position_hg19", "position_build36",
        "position_b37", "position_grch37", "pos_b37", "posb37",
        "pos_build36", "chr_pos_b36", "chr:position", "chr:pos",
        "markername", "marker_id", "markerid", "locus_id", "id",
        "name", "var_name", "chromosome_position_reference_allele_other_allele_b37"
    ],
    "A1": [
        "a1", "a_1", "allele1", "allele_1", "allelea", "allele.1",
        "effect_allele", "effectallele", "effect_allel", "effect_all",
        "effect_allele_all", "effect-allele", "eff_allele", "ea", "e",
        "coded_allele", "tested_allele", "inc_allele", "t_allele",
        "minor_allele", "minorallele", "minor_alele", "ref",
        "reference_allele", "ref_allele", "referenceallelesa/b",
        "effect_allele_plink", "effect_allele_saige", "allele_0",
        "allele0", "a0"
    ],
    "A2": [
        "a2", "a_2", "allele2", "allele_2", "alleleb",
        "other_allele", "otherallele", "other-allele", "other",
        "other_all", "other.all", "a2_other", "non_effect_allele",
        "non_effect_allele_all", "noneffect_allele", "noneffectallele",
        "noneff_allele", "none_effect_alllele", "noncoded_allele",
        "non_coded_allele", "dec_allele", "nea", "oa", "alt",
        "alt_allele", "alternate_allele", "alternative_allele",
        "alternate_ids", "majorallele"
    ],
    "N": [
        "n", "n_total", "ntotal", "total_n", "total_sample_size",
        "total_samplesize", "totalsamplesize", "sample_size",
        "samplesize", "sample_n", "sample_num", "n_samples", "nsample",
        "n_obs", "weight", "all_total", "mantra_n_samples",
        "n_effective", "effective_n", "effective_sample_size",
        "n_effective_samplesize", "n_effective_samples",
        "n_effective_samplesize_all", "n_effective_samplesize_females",
        "n_effective_samplesize_males", "n_effective_samples_study_level",
        "n_effective_samples_variant_level", "n_samples_study_level",
        "n_samples_variant_level", "chr_x_all_total",
        "chr_x_all_total_females", "chr_x_all_total_males"
    ],
    "N_CASES": [
        "ncase", "n_case", "n_cases", "cases_n", "cases_total",
        "n_cas", "num_cases", "sample-size-cases"
    ],
    "N_CONTROLS": [
        "ncontrol", "n_control", "n_controls", "controls_n",
        "controls_total", "n_con", "num_controls"
    ],
    "P": [
        "p", "pval", "p_value", "pvalue", "p-value", "p.value",
        "p_val", "p-val", "p.metal", "gc_pvalue", "gc.pvalue",
        "hetpval", "het_pvalue", "het_p_value", "p2_value",
        "meta.pval", "meta_pval", "score.pval", "p_score",
        "p_bolt_lmm", "pvalue_plink", "frequentist_add_pvalue",
        "frequentist_add_wald_pvalue_1", "p-value_association",
        "p.value_association", "p-value_ancestry_het",
        "p.value_ancestry_het", "p-value_residual_het",
        "p.value_residual_het", "robust_pval", "robust_pva",
        "neg_log_10_p_value", "_value", "pvalue_all",
        "all.fixed.pval", "all.random.pval", "ff.fixed.pval",
        "ff.random.pval", "pval_fe", "pval_re", "pval_re2",
        "pval_q", "pval_dds", "pval_uganda", "pval_aadm",
        "pval_dcc", "pval_eas", "pval_ind", "pval_ira",
        "jass_pval", "univariate_min_pval", "univariate_min_qval",
        "mtag_pval", "european_ancestry_pval_fix",
        "european_ancestry_pval_hetqtest",
        "european_ancestry_pval_rand",
        "multiancestry_pval_fix", "multiancestry_pval_hetqtest",
        "multiancestry_pval_rand"
    ],
    "INFO": [
        "info", "info-score", "info_all", "info_score", "info_ukbb",
        "info.ukb", "info_ukb", "all_info", "variant_info_score",
        "additional_info", "info.new", "median_info", "n_informative",
        "imputation", "imputation_quality", "imputationaccuracy",
        "imputationinfo", "impute_info", "imp_quality", "imp_rsqr",
        "impu_rsq", "oncoarray_imputation_r2"
    ],
    "EAF": [
        "eaf", "frq", "frq1", "freq", "freq1", "frequency",
        "effect_allele_frequency", "effectallelefreq",
        "effect_allele_freq", "effect_allele_fre",
        "effect_allele_frequenc", "effect_allele_freque",
        "effectallelefrequencyeaf", "effect_af",
        "effect_allle_frequency", "effect-allele-frequency",
        "freq_effect_allele", "freq_effect_allele_all",
        "eff_allele_freq", "eff_all_freq", "coded_allele_freq",
        "a1freq", "a1_freq", "freq_a1", "freqa1", "a1.af",
        "af", "af1", "af_coded_all", "allelefreq",
        "allele_frequencies", "eafreq", "ref_allele_frequency",
        "_allele_frequency", "t_allele_frequen",
        "freq.allele1.hapmapceu", "freq.a1.esp.eur",
        "freq.a1.1000g.eur", "freq1.hapmap", "freq_hapmap",
        "1000g_allele_freq", "avreaf", "all_meta_af",
        "all.freq.var", "ff.freq.var", "eaf_ukb", "eaf_a1",
        "eaf_cases", "eaf_controls",
        "effect_allele_frequency_cases",
        "effect_allele_frequency_controls", "allelefreq.cases",
        "allelefreq.controls", "eaf_eur", "eaf_eas", "eaf_ind",
        "eaf_ira"
    ],
    "MAF": [
        "maf", "minorallelefrequency", "minorallelefreq",
        "minorallelefreq.cases", "minorallelefreq.controls",
        "expected_minor_allele_freq", "global_maf", "cases_maf",
        "controls_maf", "maf_ukb", "maf_nw", "maf_hapmap_rel27",
        "maxeafreq", "mineafreq"
    ],
    "BETA": [
        "beta", "b", "effect", "effects", "effect1", "effect.1",
        "effect_size", "effectsize", "estimate", "meta.effect",
        "effect_a1", "effect_a2", "effect_all", "effect_nw",
        "a1_effect", "maineffects", "beta_plink", "beta_gc",
        "frequentist_add_beta_1",
        "frequentist_add_beta_1:add/plink_pheno=1",
        "p_bolt_lmm_inf", "betazscale", "effec", "effec.1"
    ],
    "SE": [
        "se", "stderr", "standard_error", "standarderror",
        "standard_error_of_beta", "standard", "sebeta", "se_beta",
        "beta_se", "se_plink", "se_saige", "se_lm", "se_error",
        "serror", "stderr_all", "stderrlogor", "stderr_nw",
        "stderr_females", "stderr_males", "log_or_ste", "logor_se",
        "or_se", "log_odds_se", "logor.se", "se1", "se2",
        "mtag_se", "freqse", "sebeta_snp_add", "robust.se",
        "se.coef.", "se_gc", "meta.se", "odds_ratio_se",
        "se_of_beta", "se_0", "se_1", "se_2", "se_3",
        "frequentist_add_se_1", "standard_error_asian",
        "standard_error_black", "standard_error_chinese",
        "standard_error_white", "standard_error_site1",
        "standard_error_site2", "tandard_error", "dard_error",
        "andard_error", "_error"
    ],
    "OR": [
        "or", "odds_ratio", "oddsratio", "oddsratiominorallele",
        "hm_odds_ratio", "odds_rat", "odds_ration", "odds",
        "odds_ra", "odd", "__odds_ratio__", "or_random",
        "heterozygous_or", "homozygous_or", "log_odds",
        "log_or", "logor"
    ],
    "Z": [
        "z", "z_value", "zscore", "z-score", "z_score", "z_stat",
        "zval", "z.meta", "z.weightedsumz", "mtag_z",
        "gc.zscore", "gz_zscore", "gz-score", "__z__",
        "weighted_z", "zscore_metal", "betazscale"
    ],
    "DIRECTION": [
        "direction", "direct", "dir", "effect_direction",
        "direction_ukbb_tagc", "direction_effects_cohorts",
        "direction_effects_cohorts_all",
        "direction_effects_cohorts_females",
        "direction_effects_cohorts_males", "direction_by_study",
        "mantra_dir", "cohort_dir"
    ],
}


LIKELY_GWAS_EXTENSIONS = (
    ".gz", ".bgz", ".txt", ".tsv", ".csv", ".tbl", ".ma", ".assoc",
    ".meta", ".linear", ".logistic", ".sumstats", ".summary",
    ".zip", ".tar", ".tar.gz", ".tgz"
)

SKIP_FILE_PATTERNS = (
    "readme", "md5", "manifest", "license", "licence", ".pdf",
    ".html", ".htm", ".png", ".jpg", ".jpeg", ".json", ".yaml",
    ".yml", ".xlsx", ".xls"
)


# =============================================================================
# Basic helpers
# =============================================================================

def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def safe_name(x):
    x = clean_text(x)
    x = re.sub(r"[^\w\-\.]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_") or "missing"


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


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
    rows = []

    for standard_col, synonyms in COLUMN_SYNONYMS.items():
        for synonym in synonyms:
            norm = normalise_header_name(synonym)
            if not norm:
                continue

            if norm not in lookup:
                lookup[norm] = standard_col

            rows.append({
                "standard_column": standard_col,
                "synonym": norm,
            })

    table = pd.DataFrame(rows).drop_duplicates()
    return lookup, table


SYNONYM_LOOKUP, SYNONYM_TABLE = build_synonym_lookup()


def sha256_file(path, max_bytes=None):
    h = hashlib.sha256()
    total = 0

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break

            if max_bytes is not None and total + len(chunk) > max_bytes:
                chunk = chunk[: max_bytes - total]

            h.update(chunk)
            total += len(chunk)

            if max_bytes is not None and total >= max_bytes:
                break

    return h.hexdigest()


def normalise_url(url):
    url = clean_text(url)

    if url.startswith("http://ftp.ebi.ac.uk"):
        url = "https://ftp.ebi.ac.uk" + url[len("http://ftp.ebi.ac.uk"):]

    url = re.sub(r"gcst\d+", lambda m: m.group(0).upper(), url, flags=re.I)

    return url


def detect_compression_from_name(filename):
    name = filename.lower()

    if name.endswith((".tar.gz", ".tgz")):
        return "tar.gz"
    if name.endswith((".gz", ".bgz")):
        return "gzip"
    if name.endswith(".zip"):
        return "zip"
    if name.endswith(".tar"):
        return "tar"

    return "plain"


def is_likely_gwas_file(filename):
    name = filename.lower()

    if any(p in name for p in SKIP_FILE_PATTERNS):
        return False

    return name.endswith(LIKELY_GWAS_EXTENSIONS)


def parse_size_to_bytes(size_text):
    size_text = clean_text(size_text)

    if size_text in ["", "-"]:
        return 0

    m = re.match(r"^([\d\.]+)\s*([KMGTPE]?)(?:B)?$", size_text, flags=re.I)
    if not m:
        return 0

    value = float(m.group(1))
    unit = m.group(2).upper()

    multipliers = {
        "": 1,
        "K": 1024,
        "M": 1024 ** 2,
        "G": 1024 ** 3,
        "T": 1024 ** 4,
        "P": 1024 ** 5,
        "E": 1024 ** 6,
    }

    return int(value * multipliers.get(unit, 1))


def make_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "GWASPoker/1.0 partial GWAS scanner"
    })
    return session


SESSION = make_session()


def url_exists(url, timeout=8):
    url = normalise_url(url)

    try:
        r = SESSION.get(
            url,
            headers={"Range": "bytes=0-1023"},
            stream=True,
            timeout=(4, timeout),
            allow_redirects=True,
        )

        ok = r.status_code in [200, 206]
        r.close()
        return ok

    except Exception:
        return False


# =============================================================================
# Candidate discovery
# =============================================================================

def list_directory_files(url, timeout=8):
    rows = []
    url = normalise_url(url)

    urls_to_try = [url]
    if not url.endswith("/"):
        urls_to_try.append(url + "/")

    response = None
    final_url = None

    for try_url in urls_to_try:
        try:
            r = SESSION.get(
                try_url,
                timeout=(4, timeout),
                allow_redirects=True,
            )

            if r.status_code == 200 and len(r.text) > 0:
                response = r
                final_url = r.url
                break

        except Exception:
            continue

    if response is None:
        return pd.DataFrame(rows)

    soup = BeautifulSoup(response.text, "html.parser")

    for a in soup.find_all("a"):
        href = a.get("href", "")
        name = a.text.strip()

        if not href or href in ["../", "./", "/"]:
            continue

        if name.lower() in ["parent directory", "name", "last modified", "size"]:
            continue

        file_url = urljoin(final_url.rstrip("/") + "/", href)
        decoded_name = unquote(name or href)

        parent_text = a.parent.get_text(" ", strip=True) if a.parent else ""
        size_text = "-"

        for part in parent_text.split():
            if re.match(r"^[\d\.]+[KMGTPE]?$", part, flags=re.I):
                size_text = part
                break

        rows.append({
            "name": decoded_name,
            "url": file_url,
            "size_text": size_text,
            "size_bytes": parse_size_to_bytes(size_text),
            "is_directory": decoded_name.endswith("/") or href.endswith("/"),
        })

    return pd.DataFrame(rows)


def get_accession_from_url_or_row(url, accession_from_row=""):
    if accession_from_row:
        return safe_name(accession_from_row).upper()

    url = clean_text(url).rstrip("/")
    return safe_name(url.split("/")[-1]).upper()


def guessed_candidate_urls(summary_url, accession_id):
    summary_url = normalise_url(summary_url).rstrip("/")
    accession_id = get_accession_from_url_or_row(summary_url, accession_id)

    guesses = []

    names = [
        f"{accession_id}.tsv.gz",
        f"{accession_id}.txt.gz",
        f"{accession_id}.csv.gz",
        f"{accession_id}.tsv",
        f"{accession_id}.txt",
        f"{accession_id}.csv",
        f"{accession_id}.h.tsv.gz",
        f"{accession_id}.h.txt.gz",
        f"{accession_id}.sumstats.gz",
        f"{accession_id}.sumstats.tsv.gz",
    ]

    for name in names:
        guesses.append((name, f"{summary_url}/{name}"))
        guesses.append((name, f"{summary_url}/harmonised/{name}"))
        guesses.append((name, f"{summary_url}/harmonized/{name}"))

    return guesses


def discover_candidate_files(summary_url, accession_id="", timeout=8, max_depth=3):
    summary_url = normalise_url(summary_url)
    candidates = []

    if not summary_url:
        return candidates

    basename = unquote(summary_url.rstrip("/").split("/")[-1])

    # Direct file URL
    if is_likely_gwas_file(basename):
        candidates.append({
            "candidate_name": basename,
            "candidate_url": summary_url,
            "candidate_size_bytes": 0,
            "candidate_source": "direct_url",
        })
        return candidates

    # Directory scrape
    start_urls = [summary_url]
    if not summary_url.endswith("/"):
        start_urls.append(summary_url + "/")

    to_visit = [(u, 0) for u in start_urls]
    visited = set()

    while to_visit:
        current_url, depth = to_visit.pop(0)

        if current_url in visited or depth > max_depth:
            continue

        visited.add(current_url)

        listing = list_directory_files(current_url, timeout=timeout)

        if listing.empty:
            continue

        for _, row in listing.iterrows():
            name = clean_text(row.get("name", ""))
            file_url = clean_text(row.get("url", ""))

            if row.get("is_directory", False):
                if depth + 1 <= max_depth:
                    if "harmonised" in name.lower() or "harmonized" in name.lower():
                        to_visit.insert(0, (file_url, depth + 1))
                    else:
                        to_visit.append((file_url, depth + 1))
                continue

            if is_likely_gwas_file(name):
                candidates.append({
                    "candidate_name": name,
                    "candidate_url": normalise_url(file_url),
                    "candidate_size_bytes": int(row.get("size_bytes", 0)),
                    "candidate_source": "directory_listing",
                })

    # If directory listing failed, try common GWAS Catalog filenames
    if not candidates:
        for name, guess_url in guessed_candidate_urls(summary_url, accession_id):
            if url_exists(guess_url, timeout=timeout):
                candidates.append({
                    "candidate_name": name,
                    "candidate_url": normalise_url(guess_url),
                    "candidate_size_bytes": 0,
                    "candidate_source": "guessed_common_filename",
                })

    # Deduplicate
    seen = set()
    unique = []

    for c in candidates:
        u = c["candidate_url"]
        if u not in seen:
            seen.add(u)
            unique.append(c)

    return unique


def choose_best_candidate(candidates):
    if not candidates:
        return None

    def score(candidate):
        url = candidate["candidate_url"].lower()
        name = candidate["candidate_name"].lower()
        size = candidate.get("candidate_size_bytes", 0)

        s = 0

        if "harmonised" in url or "harmonized" in url:
            s += 1000

        if name.endswith((".tsv.gz", ".txt.gz", ".csv.gz", ".gz", ".tsv", ".txt", ".csv")):
            s += 400

        if name.endswith((".zip", ".tar", ".tar.gz", ".tgz")):
            s += 100

        if "sumstats" in name or "summary" in name:
            s += 100

        if any(p in name for p in SKIP_FILE_PATTERNS):
            s -= 1000

        s += min(size / (1024 ** 2), 500)

        return s

    return sorted(candidates, key=score, reverse=True)[0]


# =============================================================================
# Partial download + parse
# =============================================================================

def partial_download(url, output_path, max_mb=10, timeout=30):
    url = normalise_url(url)
    max_bytes = int(max_mb * 1024 * 1024)
    ensure_dir(Path(output_path).parent)

    try:
        with SESSION.get(
            url,
            headers={"Range": f"bytes=0-{max_bytes - 1}"},
            stream=True,
            timeout=(5, timeout),
            allow_redirects=True,
        ) as response:
            if response.status_code not in [200, 206]:
                return False, f"HTTP {response.status_code}", 0

            total = 0

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if not chunk:
                        continue

                    if total + len(chunk) > max_bytes:
                        chunk = chunk[: max_bytes - total]

                    f.write(chunk)
                    total += len(chunk)

                    if total >= max_bytes:
                        break

        return True, "downloaded_partial", total

    except Exception as e:
        return False, str(e), 0


def decode_partial_gzip_bytes(raw):
    try:
        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
        data = decompressor.decompress(raw)
        return data.decode("utf-8", errors="replace")
    except Exception:
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
                return gz.read().decode("utf-8", errors="replace")
        except Exception:
            return ""


def extract_sample_lines(path, original_name, max_lines=250):
    compression = detect_compression_from_name(original_name)

    try:
        raw = Path(path).read_bytes()

        if compression == "gzip":
            text = decode_partial_gzip_bytes(raw)
            if text:
                return text.splitlines()[:max_lines], "gzip_partial_zlib"
            return [], "gzip_failed"

        if compression == "zip":
            lines = []
            with zipfile.ZipFile(path) as z:
                members = [m for m in z.namelist() if is_likely_gwas_file(m)]
                if not members:
                    members = z.namelist()

                if not members:
                    return [], "zip_empty"

                member = members[0]
                with z.open(member) as f:
                    text = io.TextIOWrapper(f, errors="replace")
                    for _ in range(max_lines):
                        line = text.readline()
                        if not line:
                            break
                        lines.append(line.rstrip("\n"))
            return lines, "zip"

        if compression in ["tar", "tar.gz"]:
            mode = "r:gz" if compression == "tar.gz" else "r:"
            lines = []

            with tarfile.open(path, mode) as tar:
                members = [m for m in tar.getmembers() if m.isfile()]
                members = sorted(members, key=lambda m: m.size, reverse=True)

                if not members:
                    return [], f"{compression}_empty"

                f = tar.extractfile(members[0])
                text = io.TextIOWrapper(f, errors="replace")

                for _ in range(max_lines):
                    line = text.readline()
                    if not line:
                        break
                    lines.append(line.rstrip("\n"))

            return lines, compression

        text = raw.decode("utf-8", errors="replace")
        return text.splitlines()[:max_lines], "plain"

    except Exception as e:
        return [], f"read_failed_{type(e).__name__}"


def detect_delimiter_from_lines(lines):
    candidate_delimiters = {
        "tab": "\t",
        "comma": ",",
        "semicolon": ";",
        "pipe": "|",
        "space": r"\s+",
    }

    data_lines = [
        line for line in lines
        if clean_text(line) and not clean_text(line).startswith("#")
    ]

    if not data_lines:
        return "unknown", None, 0

    best_name = "unknown"
    best_sep = None
    best_cols = 0

    for name, sep in candidate_delimiters.items():
        counts = []

        for line in data_lines[:30]:
            if sep == r"\s+":
                parts = re.split(r"\s+", line.strip())
            else:
                parts = line.split(sep)

            counts.append(len(parts))

        median_cols = sorted(counts)[len(counts) // 2]

        if median_cols > best_cols:
            best_cols = median_cols
            best_name = name
            best_sep = sep

    return best_name, best_sep, best_cols


def split_line(line, sep):
    if sep is None:
        return [line]

    if sep == r"\s+":
        return re.split(r"\s+", line.strip())

    return line.split(sep)


def detect_header_row(lines, sep):
    for i, line in enumerate(lines):
        raw = clean_text(line)

        if not raw or raw.startswith("#"):
            continue

        parts = split_line(raw, sep)

        if len(parts) < 2:
            continue

        non_numeric = 0

        for part in parts:
            try:
                float(clean_text(part))
            except Exception:
                non_numeric += 1

        if non_numeric / max(len(parts), 1) >= 0.5:
            return i, parts

    return None, []


def match_columns(headers):
    mapping_rows = []
    found = {standard_col: [] for standard_col in COLUMN_SYNONYMS.keys()}

    for raw_header in headers:
        norm_header = normalise_header_name(raw_header)
        standard_col = SYNONYM_LOOKUP.get(norm_header, "")

        if standard_col:
            found[standard_col].append(raw_header)

        mapping_rows.append({
            "raw_header": raw_header,
            "normalised_header": norm_header,
            "standard_column": standard_col,
            "is_identified": bool(standard_col),
        })

    has_position = bool(found["CHR"] and found["BP"])
    has_snp_or_position = bool(found["SNP"] or has_position)
    has_effect_size = bool(found["BETA"] or found["OR"] or found["Z"])

    coverage = {
        "has_chr": bool(found["CHR"]),
        "has_bp": bool(found["BP"]),
        "has_snp": bool(found["SNP"]),
        "has_a1": bool(found["A1"]),
        "has_a2": bool(found["A2"]),
        "has_beta": bool(found["BETA"]),
        "has_or": bool(found["OR"]),
        "has_se": bool(found["SE"]),
        "has_p": bool(found["P"]),
        "has_n": bool(found["N"]),
        "has_n_cases": bool(found["N_CASES"]),
        "has_n_controls": bool(found["N_CONTROLS"]),
        "has_eaf": bool(found["EAF"]),
        "has_maf": bool(found["MAF"]),
        "has_info": bool(found["INFO"]),
        "has_z": bool(found["Z"]),
        "has_direction": bool(found["DIRECTION"]),
        "has_effect_size": has_effect_size,
        "has_snp_or_position": has_snp_or_position,
        "prs_minimum_columns": bool(
            has_snp_or_position and found["A1"] and found["P"] and has_effect_size
        ),
    }

    for standard_col, values in found.items():
        coverage[f"mapped_{standard_col}"] = ";".join(values)

    return mapping_rows, coverage


def scan_partial_file(partial_path, candidate_name):
    lines, read_mode = extract_sample_lines(partial_path, candidate_name)
    delimiter_name, sep, estimated_n_columns = detect_delimiter_from_lines(lines)
    header_row_index, headers = detect_header_row(lines, sep)

    if not headers:
        return {
            "read_mode": read_mode,
            "delimiter_name": delimiter_name,
            "estimated_n_columns": estimated_n_columns,
            "header_row_index": "",
            "n_headers": 0,
            "raw_headers_joined": "",
            "parse_status": "header_not_detected",
        }, []

    mapping_rows, coverage = match_columns(headers)

    scan_summary = {
        "read_mode": read_mode,
        "delimiter_name": delimiter_name,
        "estimated_n_columns": estimated_n_columns,
        "header_row_index": header_row_index,
        "n_headers": len(headers),
        "raw_headers_joined": " | ".join(headers),
        "parse_status": "parsed_header",
    }

    scan_summary.update(coverage)
    return scan_summary, mapping_rows


# =============================================================================
# Input/output/report
# =============================================================================

def load_input_table(path):
    df = pd.read_csv(path)

    required = ["summaryStatistics", "accessionId", "reportedTrait"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    if "pubmedId" not in df.columns:
        df["pubmedId"] = ""

    if "Rank" not in df.columns:
        df["Rank"] = range(1, len(df) + 1)

    df["summaryStatistics"] = df["summaryStatistics"].astype(str).str.strip()
    df["summaryStatistics"] = df["summaryStatistics"].apply(normalise_url)
    df["accessionId"] = df["accessionId"].astype(str).str.strip()
    df["reportedTrait"] = df["reportedTrait"].astype(str).str.strip()

    bad_values = ["", "nan", "none", "null", "na", "n/a", "-"]
    df = df[~df["summaryStatistics"].str.lower().isin(bad_values)].copy()

    df = df.drop_duplicates(
        subset=["accessionId", "summaryStatistics"],
        keep="first"
    ).copy()

    df = df.reset_index(drop=True)
    return df


def make_html_report(scan_report, mapping_report, output_html):
    if mapping_report is None or mapping_report.empty:
        mapping_report = pd.DataFrame(
            columns=[
                "accessionId", "raw_header", "normalised_header",
                "standard_column", "is_identified"
            ]
        )

    html_parts = []

    html_parts.append("""
<html>
<head>
<meta charset="utf-8">
<title>GWAS Partial Scan Report</title>
<style>
body { font-family: Arial, sans-serif; margin: 30px; background: #f7f7f7; }
.card { background: white; border-radius: 10px; padding: 18px; margin-bottom: 24px; box-shadow: 0 1px 5px rgba(0,0,0,0.12); }
table { border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 13px; }
th, td { border: 1px solid #ddd; padding: 6px; vertical-align: top; }
th { background: #eef2ff; }
.ok { color: #0a7f32; font-weight: bold; }
.bad { color: #b00020; font-weight: bold; }
.small { font-size: 12px; color: #555; word-break: break-all; }
code { background: #eee; padding: 2px 4px; border-radius: 4px; }
</style>
</head>
<body>
<h1>GWAS Partial Download + Detailed Header Mapping Report</h1>
""")

    for _, row in scan_report.iterrows():
        accession = html.escape(str(row.get("accessionId", "")))
        trait = html.escape(str(row.get("reportedTrait", "")))
        status = html.escape(str(row.get("parse_status", "")))
        prs_ok = bool(row.get("prs_minimum_columns", False))

        html_parts.append("<div class='card'>")
        html_parts.append(f"<h2>{accession} — {trait}</h2>")
        html_parts.append(f"<p><strong>Status:</strong> {status}</p>")
        html_parts.append(f"<p><strong>Candidate file:</strong> <code>{html.escape(str(row.get('candidate_name', '')))}</code></p>")
        html_parts.append(f"<p><strong>URL:</strong> <span class='small'>{html.escape(str(row.get('candidate_url', '')))}</span></p>")
        html_parts.append(f"<p><strong>Download:</strong> {html.escape(str(row.get('download_status', '')))}</p>")
        html_parts.append(f"<p><strong>Delimiter:</strong> {html.escape(str(row.get('delimiter_name', '')))} | <strong>Headers:</strong> {row.get('n_headers', '')}</p>")
        html_parts.append(f"<p><strong>PRS minimum columns:</strong> <span class='{'ok' if prs_ok else 'bad'}'>{prs_ok}</span></p>")

        bool_cols = [
            "has_chr", "has_bp", "has_snp", "has_a1", "has_a2",
            "has_beta", "has_or", "has_se", "has_p", "has_n",
            "has_n_cases", "has_n_controls", "has_eaf", "has_maf",
            "has_info", "has_z", "has_direction"
        ]

        bool_cols = [col for col in bool_cols if col in scan_report.columns]
        html_parts.append(
            pd.DataFrame([{col: row.get(col, False) for col in bool_cols}])
            .to_html(index=False, escape=True)
        )

        accession_mapping = mapping_report[
            mapping_report["accessionId"].astype(str) == str(row.get("accessionId", ""))
        ]

        if not accession_mapping.empty:
            html_parts.append("<h3>Detected Column Mapping</h3>")
            html_parts.append(
                accession_mapping[["raw_header", "normalised_header", "standard_column", "is_identified"]]
                .to_html(index=False, escape=True)
            )

        html_parts.append("<h3>Raw Headers</h3>")
        html_parts.append(f"<p class='small'>{html.escape(str(row.get('raw_headers_joined', '')))}</p>")
        html_parts.append("</div>")

    html_parts.append("</body></html>")

    with open(output_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))


def run_scan(
    processedfile,
    output_dir=None,
    max_rows=None,
    partial_mb=10,
    timeout=10,
    sleep_seconds=0.2,
):
    input_path = Path(processedfile)
    phenotype_name = input_path.stem

    if output_dir is None:
        output_dir = phenotype_name

    output_dir = Path(output_dir)
    allgwas_dir = output_dir / "allgwas"
    ensure_dir(allgwas_dir)

    df = load_input_table(processedfile)

    if max_rows is not None:
        df = df.head(max_rows).copy()

    scan_rows = []
    mapping_rows_all = []
    candidates_rows_all = []

    print("=" * 100)
    print("MODULE 2: GWAS PARTIAL DOWNLOAD + DETAILED HEADER SCAN")
    print("=" * 100)
    print(f"[INPUT] {processedfile}")
    print(f"[OUTPUT DIR] {output_dir}")
    print(f"[ROWS TO SCAN] {len(df):,}")
    print(f"[PARTIAL DOWNLOAD MB] {partial_mb}")
    print(f"[TIMEOUT] {timeout}")
    print(f"[MAPPING TERMS] {len(SYNONYM_TABLE):,}")
    print("=" * 100)

    synonym_out = output_dir / "header_synonym_dictionary.csv"
    SYNONYM_TABLE.to_csv(synonym_out, index=False)

    for idx, row in df.iterrows():
        accession = safe_name(row.get("accessionId", f"row_{idx + 1}")).upper()
        trait = clean_text(row.get("reportedTrait", ""))
        url = normalise_url(row.get("summaryStatistics", ""))
        rank = row.get("Rank", idx + 1)

        study_dir = allgwas_dir / accession
        ensure_dir(study_dir)

        print(f"\n[{idx + 1}/{len(df)}] {accession} | {trait}")
        print(f"URL: {url}")

        base_info = {
            "Rank": rank,
            "accessionId": accession,
            "reportedTrait": trait,
            "pubmedId": row.get("pubmedId", ""),
            "summaryStatistics": url,
        }

        try:
            candidates = discover_candidate_files(
                url,
                accession_id=accession,
                timeout=timeout,
                max_depth=3,
            )
        except Exception as e:
            candidates = []
            print(f"  -> Candidate discovery failed: {type(e).__name__}: {e}")

        for candidate in candidates:
            candidates_rows_all.append({
                **base_info,
                **candidate,
            })

        if not candidates:
            scan_rows.append({
                **base_info,
                "candidate_name": "",
                "candidate_url": "",
                "candidate_size_bytes": 0,
                "candidate_source": "",
                "download_status": "no_candidate_file_found",
                "downloaded_bytes": 0,
                "partial_file": "",
                "partial_sha256": "",
                "parse_status": "not_scanned",
            })
            print("  -> No candidate file found")
            continue

        best = choose_best_candidate(candidates)

        candidate_url = best["candidate_url"]
        candidate_name = safe_name(unquote(best["candidate_name"]))
        partial_path = study_dir / f"partial_{candidate_name}"

        ok, msg, downloaded_bytes = partial_download(
            candidate_url,
            partial_path,
            max_mb=partial_mb,
            timeout=timeout,
        )

        if not ok:
            scan_rows.append({
                **base_info,
                **best,
                "download_status": msg,
                "downloaded_bytes": downloaded_bytes,
                "partial_file": str(partial_path),
                "partial_sha256": "",
                "parse_status": "download_failed",
            })
            print(f"  -> Download failed: {msg}")
            continue

        try:
            partial_hash = sha256_file(partial_path)
        except Exception:
            partial_hash = ""

        scan_summary, mapping_rows = scan_partial_file(partial_path, candidate_name)

        scan_row = {
            **base_info,
            **best,
            "download_status": msg,
            "downloaded_bytes": downloaded_bytes,
            "partial_file": str(partial_path),
            "partial_sha256": partial_hash,
            **scan_summary,
        }

        scan_rows.append(scan_row)

        for mapping in mapping_rows:
            mapping_rows_all.append({
                **base_info,
                "candidate_name": best.get("candidate_name", ""),
                "candidate_url": best.get("candidate_url", ""),
                **mapping,
            })

        print(
            f"  -> {scan_summary.get('parse_status')} | "
            f"headers={scan_summary.get('n_headers')} | "
            f"delimiter={scan_summary.get('delimiter_name')} | "
            f"PRS_minimum={scan_summary.get('prs_minimum_columns', False)}"
        )

        time.sleep(sleep_seconds)

    scan_report = pd.DataFrame(scan_rows)
    mapping_report = pd.DataFrame(mapping_rows_all)
    candidates_report = pd.DataFrame(candidates_rows_all)

    if mapping_report.empty:
        mapping_report = pd.DataFrame(
            columns=[
                "Rank", "accessionId", "reportedTrait", "pubmedId",
                "summaryStatistics", "candidate_name", "candidate_url",
                "raw_header", "normalised_header", "standard_column",
                "is_identified"
            ]
        )

    if candidates_report.empty:
        candidates_report = pd.DataFrame(
            columns=[
                "Rank", "accessionId", "reportedTrait", "pubmedId",
                "summaryStatistics", "candidate_name", "candidate_url",
                "candidate_size_bytes", "candidate_source"
            ]
        )

    scan_csv = output_dir / "scan_report.csv"
    mapping_csv = output_dir / "normalised_column_mapping.csv"
    candidates_csv = output_dir / "candidate_files_report.csv"
    html_report = output_dir / "scan_report.html"

    scan_report.to_csv(scan_csv, index=False)
    mapping_report.to_csv(mapping_csv, index=False)
    candidates_report.to_csv(candidates_csv, index=False)

    make_html_report(scan_report, mapping_report, html_report)

    print("\n" + "=" * 100)
    print("[DONE]")
    print(f"[SCAN REPORT] {scan_csv}")
    print(f"[COLUMN MAPPING] {mapping_csv}")
    print(f"[CANDIDATE FILES] {candidates_csv}")
    print(f"[SYNONYM DICTIONARY] {synonym_out}")
    print(f"[HTML REPORT] {html_report}")
    print("=" * 100)

    if not scan_report.empty:
        summary_cols = [
            "accessionId", "reportedTrait", "candidate_name",
            "download_status", "parse_status", "delimiter_name",
            "n_headers", "prs_minimum_columns", "has_snp", "has_chr",
            "has_bp", "has_a1", "has_a2", "has_beta", "has_or",
            "has_se", "has_p", "has_n", "has_eaf", "has_maf", "has_info"
        ]

        summary_cols = [col for col in summary_cols if col in scan_report.columns]

        print("\n[PREVIEW]")
        print(scan_report[summary_cols].head(30).to_string(index=False))

    return scan_report, mapping_report, candidates_report


def main():
    parser = argparse.ArgumentParser(
        description="Partially download/poke GWAS summary statistics and scan headers using detailed GWAS mapping."
    )

    parser.add_argument(
        "--processedfile",
        required=True,
        help="Input CSV from Module1, e.g. migraine.csv",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: input file stem, e.g. migraine",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Only scan first N rows for testing.",
    )

    parser.add_argument(
        "--partial-mb",
        type=float,
        default=10,
        help="Maximum MB to download per GWAS file.",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="HTTP timeout in seconds.",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Sleep seconds between downloads.",
    )

    args = parser.parse_args()

    run_scan(
        processedfile=args.processedfile,
        output_dir=args.output_dir,
        max_rows=args.max_rows,
        partial_mb=args.partial_mb,
        timeout=args.timeout,
        sleep_seconds=args.sleep,
    )


if __name__ == "__main__":
    main()
