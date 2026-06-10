#!/usr/bin/env python3

import argparse
import gzip
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple, Set, Any

import numpy as np
import pandas as pd


# =============================================================================
# DEFAULT INPUTS
# =============================================================================

REF_DIR = Path("references")

HAPMAP_HG19 = REF_DIR / "PAN.hapmap3.hg19.EAF.tsv.gz"
HAPMAP_HG38 = REF_DIR / "PAN.hapmap3.hg38.EAF.tsv.gz"

DBSNP_HG19 = REF_DIR / "1kg_dbsnp151_hg19_auto.txt.gz"
DBSNP_HG38 = REF_DIR / "1kg_dbsnp151_hg38_auto.txt.gz"

KG_EUR_HG19_VCF = REF_DIR / "EUR.ALL.split_norm_af.1kgp3v5.hg19.vcf.gz"
KG_EUR_HG38_VCF = REF_DIR / "EUR.ALL.split_norm_af.1kg_30x.hg38.vcf.gz"

# This bfile is usually hg19/GRCh37 unless you made a lifted hg38 version.
# IMPORTANT:
# If you have an hg38 PLINK bfile, put it in BIGSNPR_BFILE_HG38.
BIGSNPR_BFILE_HG19 = REF_DIR / "bigsnpr_1000G/1000G_phase3_common_norel"
BIGSNPR_BFILE_HG38 = None

PLINK = "/data/ascher02/uqmmune1/VariationInGWAS/allfiles/plink"
PLINK2 = "/data/ascher02/uqmmune1/VariationInGWAS/allfiles/plink2"

OUTDIR = Path("Feature7_files")


# =============================================================================
# REFERENCE CONFIGURATION
# =============================================================================

REFERENCE_CONFIG = {
    "hg19": {
        "gwaslab_build": "19",
        "label": "hg19 / GRCh37",
        "hapmap": HAPMAP_HG19,
        "dbsnp": DBSNP_HG19,
        "kg_vcf": KG_EUR_HG19_VCF,
        "plink_bfile": BIGSNPR_BFILE_HG19,
    },
    "hg38": {
        "gwaslab_build": "38",
        "label": "hg38 / GRCh38",
        "hapmap": HAPMAP_HG38,
        "dbsnp": DBSNP_HG38,
        "kg_vcf": KG_EUR_HG38_VCF,
        "plink_bfile": BIGSNPR_BFILE_HG38,
    },
}


def configure_paths(ref_dir, plink="", plink2="", bigsnpr_hg19="", bigsnpr_hg38=""):
    global REF_DIR
    global HAPMAP_HG19, HAPMAP_HG38, DBSNP_HG19, DBSNP_HG38
    global KG_EUR_HG19_VCF, KG_EUR_HG38_VCF
    global BIGSNPR_BFILE_HG19, BIGSNPR_BFILE_HG38
    global PLINK, PLINK2, REFERENCE_CONFIG

    REF_DIR = Path(ref_dir)

    HAPMAP_HG19 = REF_DIR / "PAN.hapmap3.hg19.EAF.tsv.gz"
    HAPMAP_HG38 = REF_DIR / "PAN.hapmap3.hg38.EAF.tsv.gz"
    DBSNP_HG19 = REF_DIR / "1kg_dbsnp151_hg19_auto.txt.gz"
    DBSNP_HG38 = REF_DIR / "1kg_dbsnp151_hg38_auto.txt.gz"
    KG_EUR_HG19_VCF = REF_DIR / "EUR.ALL.split_norm_af.1kgp3v5.hg19.vcf.gz"
    KG_EUR_HG38_VCF = REF_DIR / "EUR.ALL.split_norm_af.1kg_30x.hg38.vcf.gz"

    BIGSNPR_BFILE_HG19 = Path(bigsnpr_hg19) if clean_text(bigsnpr_hg19) else REF_DIR / "bigsnpr_1000G/1000G_phase3_common_norel"
    BIGSNPR_BFILE_HG38 = Path(bigsnpr_hg38) if clean_text(bigsnpr_hg38) else None

    PLINK = clean_text(plink) or "plink"
    PLINK2 = clean_text(plink2) or "plink2"

    REFERENCE_CONFIG = {
        "hg19": {
            "gwaslab_build": "19",
            "label": "hg19 / GRCh37",
            "hapmap": HAPMAP_HG19,
            "dbsnp": DBSNP_HG19,
            "kg_vcf": KG_EUR_HG19_VCF,
            "plink_bfile": BIGSNPR_BFILE_HG19,
        },
        "hg38": {
            "gwaslab_build": "38",
            "label": "hg38 / GRCh38",
            "hapmap": HAPMAP_HG38,
            "dbsnp": DBSNP_HG38,
            "kg_vcf": KG_EUR_HG38_VCF,
            "plink_bfile": BIGSNPR_BFILE_HG38,
        },
    }


# =============================================================================
# HELPERS
# =============================================================================

def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def clean_chr(x):
    x = clean_text(x)
    x = x.replace("chr", "").replace("CHR", "")
    if x == "23":
        return "X"
    if x == "24":
        return "Y"
    if x in ["25", "M", "MT", "Mt", "mt"]:
        return "MT"
    return x


def clean_rsid(x):
    x = clean_text(x)
    if x.lower().startswith("rs"):
        return x.lower()
    return ""


def make_chrpos(chrom, pos):
    try:
        chrom = clean_chr(chrom)
        pos = int(float(pos))
        if chrom and pos > 0:
            return f"{chrom}:{pos}"
    except Exception:
        return ""
    return ""


def percent(n, d):
    if d == 0:
        return 0.0
    return round((n / d) * 100, 4)


def choose_col(columns, candidates):
    lower = {str(c).lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def open_text(path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def detect_sep(path):
    with open_text(path) as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            if "\t" in line:
                return "\t"
            if "," in line:
                return ","
            return r"\s+"
    return "\t"


def normalise_build_string(x: Any) -> Optional[str]:
    """
    Converts possible build strings to hg19 or hg38.
    """
    if x is None:
        return None

    s = str(x).lower()

    if "hg19" in s or "grch37" in s or s in {"19", "37", "b37"}:
        return "hg19"

    if "hg38" in s or "grch38" in s or s in {"38", "b38"}:
        return "hg38"

    return None


def recursively_find_build(obj: Any, depth: int = 0, max_depth: int = 4) -> Optional[str]:
    """
    Searches gwaslab Sumstats object metadata for build-like values.
    This is deliberately defensive because gwaslab versions expose metadata differently.
    """
    if depth > max_depth:
        return None

    b = normalise_build_string(obj)
    if b:
        return b

    if isinstance(obj, dict):
        for k, v in obj.items():
            key_build = normalise_build_string(k)
            val_build = normalise_build_string(v)

            if key_build:
                return key_build
            if val_build:
                return val_build

            nested = recursively_find_build(v, depth + 1, max_depth)
            if nested:
                return nested

    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            nested = recursively_find_build(v, depth + 1, max_depth)
            if nested:
                return nested

    return None


# =============================================================================
# 1. LOAD GWAS WITH GWASLAB
# =============================================================================

def load_gwas_with_gwaslab(gwas_file: str, build: Optional[str] = None):
    print("=" * 100)
    print("1. LOAD GWAS USING GWASLab")
    print("=" * 100)

    import gwaslab as gl

    kwargs = {
        "fmt": "auto",
        "verbose": True,
    }

    if build is not None:
        kwargs["build"] = build

    ss = gl.Sumstats(gwas_file, **kwargs)
    df = ss.data.copy()

    print("[GWASLab columns]")
    print(list(df.columns))
    print("[GWAS rows]", len(df))

    return ss, df


def extract_gwas_variant_sets(df: pd.DataFrame) -> Tuple[Set[str], Set[str]]:
    """
    Extract GWAS rsID and CHR:POS sets after gwaslab standardisation.
    """
    gwas_rsids = set()
    gwas_chrpos = set()

    if "SNPID" in df.columns:
        gwas_rsids.update(x for x in df["SNPID"].map(clean_rsid) if x)

    if "rsID" in df.columns:
        gwas_rsids.update(x for x in df["rsID"].map(clean_rsid) if x)

    if {"CHR", "POS"}.issubset(df.columns):
        gwas_chrpos.update(
            x for x in (
                make_chrpos(c, p)
                for c, p in zip(df["CHR"], df["POS"])
            )
            if x
        )

    print("[GWAS unique rsIDs]", len(gwas_rsids))
    print("[GWAS unique CHR:POS]", len(gwas_chrpos))
    print()

    return gwas_rsids, gwas_chrpos


def create_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds BETA/Z/SNPID if possible.
    """
    df = df.copy()

    if "BETA" not in df.columns and "OR" in df.columns:
        df["BETA"] = np.log(pd.to_numeric(df["OR"], errors="coerce"))
        print("[INFO] Created BETA = log(OR)")

    if "Z" not in df.columns and {"BETA", "SE"}.issubset(df.columns):
        beta = pd.to_numeric(df["BETA"], errors="coerce")
        se = pd.to_numeric(df["SE"], errors="coerce").replace(0, np.nan)
        df["Z"] = beta / se
        print("[INFO] Created Z = BETA / SE")

    if "rsID" in df.columns:
        rsid = df["rsID"].astype(str).map(clean_text)
        valid = rsid.str.match(r"^rs\d+$", case=False, na=False)
        df.loc[valid, "SNPID"] = rsid[valid]
        print("[INFO] Copied valid rsID values into SNPID")

    return df


# =============================================================================
# 2. BUILD DETECTION
# =============================================================================

def try_detect_build_from_gwaslab(ss) -> Optional[str]:
    """
    First tries to detect build from gwaslab object/metadata.
    Different gwaslab versions expose this differently, so this function is defensive.
    """
    print("=" * 100)
    print("2A. TRY BUILD DETECTION FROM GWASLab")
    print("=" * 100)

    possible_attrs = [
        "build",
        "genome_build",
        "ref_build",
        "meta",
        "metadata",
        "log",
    ]

    for attr in possible_attrs:
        if hasattr(ss, attr):
            value = getattr(ss, attr)
            build = recursively_find_build(value)
            if build:
                print(f"[GWASLab detected build from attribute: {attr}] {build}")
                print()
                return build

    possible_methods = [
        "infer_build",
        "inferbuild",
        "check_build",
        "detect_build",
    ]

    for method_name in possible_methods:
        method = getattr(ss, method_name, None)
        if callable(method):
            try:
                result = method()
                build = recursively_find_build(result)
                if build:
                    print(f"[GWASLab detected build from method: {method_name}] {build}")
                    print()
                    return build
            except Exception as e:
                print(f"[GWASLab method skipped] {method_name}: {e}")

    print("[GWASLab build detection] No clear build found in metadata.")
    print()
    return None


def load_reference_chrpos_only(path: Path) -> Set[str]:
    """
    Loads only CHR:POS from a table reference.
    Used for build detection only.
    """
    if path is None or not Path(path).exists():
        print("[MISSING]", path)
        return set()

    sep = detect_sep(path)

    header = pd.read_csv(
        path,
        sep=sep,
        compression="gzip" if str(path).endswith(".gz") else None,
        nrows=0,
    )

    columns = list(header.columns)

    chr_col = choose_col(columns, ["CHR", "CHROM", "chromosome", "#CHROM"])
    pos_col = choose_col(columns, ["POS", "BP", "position", "base_pair_location"])

    if chr_col is None or pos_col is None:
        print("[WARNING] Cannot find CHR/POS columns in:", path)
        print("[COLUMNS]", columns)
        return set()

    ref_chrpos = set()

    for chunk in pd.read_csv(
        path,
        sep=sep,
        compression="gzip" if str(path).endswith(".gz") else None,
        usecols=[chr_col, pos_col],
        dtype=str,
        chunksize=500000,
        low_memory=False,
    ):
        ref_chrpos.update(
            x for x in (
                make_chrpos(c, p)
                for c, p in zip(chunk[chr_col], chunk[pos_col])
            )
            if x
        )

    return ref_chrpos


def detect_build_by_hapmap_overlap(gwas_chrpos: Set[str]) -> str:
    """
    Fallback build detection.
    Only compares HapMap hg19 vs HapMap hg38, not all reference files.
    """
    print("=" * 100)
    print("2B. FALLBACK BUILD DETECTION USING HAPMAP CHR:POS OVERLAP")
    print("=" * 100)

    if not gwas_chrpos:
        raise ValueError(
            "Cannot detect build by CHR:POS overlap because GWAS CHR:POS set is empty. "
            "Check whether gwaslab parsed CHR and POS correctly."
        )

    hg19_chrpos = load_reference_chrpos_only(HAPMAP_HG19)
    hg38_chrpos = load_reference_chrpos_only(HAPMAP_HG38)

    hg19_overlap = len(gwas_chrpos & hg19_chrpos)
    hg38_overlap = len(gwas_chrpos & hg38_chrpos)

    hg19_pct = percent(hg19_overlap, len(gwas_chrpos))
    hg38_pct = percent(hg38_overlap, len(gwas_chrpos))

    print("[GWAS CHR:POS]", len(gwas_chrpos))
    print("[HapMap hg19 CHR:POS]", len(hg19_chrpos))
    print("[HapMap hg38 CHR:POS]", len(hg38_chrpos))
    print("[hg19 overlap]", hg19_overlap, f"({hg19_pct}%)")
    print("[hg38 overlap]", hg38_overlap, f"({hg38_pct}%)")

    build_detection_file = OUTDIR / "build_detection_hapmap_overlap.csv"
    pd.DataFrame([
        {
            "reference": "HapMap3 hg19 EAF",
            "build": "hg19",
            "reference_chrpos": len(hg19_chrpos),
            "gwas_chrpos": len(gwas_chrpos),
            "chrpos_overlap": hg19_overlap,
            "chrpos_overlap_pct_of_gwas": hg19_pct,
        },
        {
            "reference": "HapMap3 hg38 EAF",
            "build": "hg38",
            "reference_chrpos": len(hg38_chrpos),
            "gwas_chrpos": len(gwas_chrpos),
            "chrpos_overlap": hg38_overlap,
            "chrpos_overlap_pct_of_gwas": hg38_pct,
        },
    ]).to_csv(build_detection_file, index=False)

    print("[SAVED]", build_detection_file)

    if hg19_overlap > hg38_overlap:
        detected = "hg19"
    elif hg38_overlap > hg19_overlap:
        detected = "hg38"
    else:
        raise ValueError(
            "Build detection is unclear: hg19 and hg38 overlaps are equal. "
            "Please inspect build_detection_hapmap_overlap.csv manually."
        )

    print("[Detected build by HapMap overlap]", detected)
    print()

    return detected


def detect_gwas_build(ss, gwas_chrpos: Set[str]) -> str:
    """
    Main build detection controller.
    Uses gwaslab first, then HapMap overlap fallback.
    """
    detected = try_detect_build_from_gwaslab(ss)

    if detected in REFERENCE_CONFIG:
        print("=" * 100)
        print("BUILD DETECTION FINAL")
        print("=" * 100)
        print("[SOURCE] GWASLab")
        print("[BUILD]", detected)
        print()
        return detected

    detected = detect_build_by_hapmap_overlap(gwas_chrpos)

    print("=" * 100)
    print("BUILD DETECTION FINAL")
    print("=" * 100)
    print("[SOURCE] HapMap overlap fallback")
    print("[BUILD]", detected)
    print()

    return detected


# =============================================================================
# 3. LOAD ONLY SELECTED BUILD REFERENCES
# =============================================================================

def load_table_reference(path: Path) -> Tuple[Set[str], Set[str]]:
    if path is None or not Path(path).exists():
        print("[MISSING]", path)
        return set(), set()

    sep = detect_sep(path)

    header = pd.read_csv(
        path,
        sep=sep,
        compression="gzip" if str(path).endswith(".gz") else None,
        nrows=0,
    )

    columns = list(header.columns)

    rsid_col = choose_col(columns, ["rsid", "rsID", "SNP", "SNPID", "ID", "variant_id"])
    chr_col = choose_col(columns, ["CHR", "CHROM", "chromosome", "#CHROM"])
    pos_col = choose_col(columns, ["POS", "BP", "position", "base_pair_location"])

    usecols = [c for c in [rsid_col, chr_col, pos_col] if c is not None]

    ref_rsids = set()
    ref_chrpos = set()

    if not usecols:
        print("[WARNING] No usable rsID/CHR/POS columns in:", path)
        print("[COLUMNS]", columns)
        return ref_rsids, ref_chrpos

    for chunk in pd.read_csv(
        path,
        sep=sep,
        compression="gzip" if str(path).endswith(".gz") else None,
        usecols=usecols,
        dtype=str,
        chunksize=500000,
        low_memory=False,
    ):
        if rsid_col:
            ref_rsids.update(x for x in chunk[rsid_col].map(clean_rsid) if x)

        if chr_col and pos_col:
            ref_chrpos.update(
                x for x in (
                    make_chrpos(c, p)
                    for c, p in zip(chunk[chr_col], chunk[pos_col])
                )
                if x
            )

    return ref_rsids, ref_chrpos


def load_vcf_reference(path: Path) -> Tuple[Set[str], Set[str]]:
    if path is None or not Path(path).exists():
        print("[MISSING]", path)
        return set(), set()

    ref_rsids = set()
    ref_chrpos = set()

    with open_text(path) as f:
        for line in f:
            if line.startswith("#"):
                continue

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue

            chrom = parts[0]
            pos = parts[1]
            rsid = parts[2]

            r = clean_rsid(rsid)
            if r:
                ref_rsids.add(r)

            cp = make_chrpos(chrom, pos)
            if cp:
                ref_chrpos.add(cp)

    return ref_rsids, ref_chrpos


def load_bim_reference(bfile_prefix: Optional[Path]) -> Tuple[Set[str], Set[str]]:
    if bfile_prefix is None:
        print("[SKIP] No PLINK bfile configured for this build.")
        return set(), set()

    bim_path = Path(str(bfile_prefix) + ".bim")

    if not bim_path.exists():
        print("[MISSING]", bim_path)
        return set(), set()

    df = pd.read_csv(
        bim_path,
        sep=r"\s+",
        header=None,
        usecols=[0, 1, 3],
        names=["CHR", "SNP", "POS"],
        dtype=str,
    )

    ref_rsids = set(x for x in df["SNP"].map(clean_rsid) if x)

    ref_chrpos = set(
        x for x in (
            make_chrpos(c, p)
            for c, p in zip(df["CHR"], df["POS"])
        )
        if x
    )

    return ref_rsids, ref_chrpos


def report_overlap(label, build, gwas_rsids, gwas_chrpos, ref_rsids, ref_chrpos):
    rs_overlap = gwas_rsids & ref_rsids
    pos_overlap = gwas_chrpos & ref_chrpos

    row = {
        "reference": label,
        "build": build,
        "reference_rsids": len(ref_rsids),
        "reference_chrpos": len(ref_chrpos),
        "rsid_overlap": len(rs_overlap),
        "chrpos_overlap": len(pos_overlap),
        "rsid_overlap_pct_of_gwas": percent(len(rs_overlap), len(gwas_rsids)),
        "chrpos_overlap_pct_of_gwas": percent(len(pos_overlap), len(gwas_chrpos)),
    }

    print("=" * 100)
    print(label)
    print("=" * 100)
    for k, v in row.items():
        print(f"{k}: {v}")
    print()

    return row


def calculate_selected_reference_overlaps(
    selected_build: str,
    gwas_rsids: Set[str],
    gwas_chrpos: Set[str],
):
    """
    This function now uses ONLY the selected build references.
    It no longer loops over hg19 and hg38 together.
    """
    print("=" * 100)
    print("3. SELECTED BUILD REFERENCE OVERLAP")
    print("=" * 100)

    config = REFERENCE_CONFIG[selected_build]

    print("[Selected build]", selected_build)
    print("[Selected label]", config["label"])
    print("[HapMap]", config["hapmap"])
    print("[dbSNP]", config["dbsnp"])
    print("[1000G VCF]", config["kg_vcf"])
    print("[PLINK bfile]", config["plink_bfile"])
    print()

    rows = []

    ref_rsids, ref_chrpos = load_table_reference(config["hapmap"])
    rows.append(
        report_overlap(
            label=f"HapMap3 {selected_build} EAF",
            build=selected_build,
            gwas_rsids=gwas_rsids,
            gwas_chrpos=gwas_chrpos,
            ref_rsids=ref_rsids,
            ref_chrpos=ref_chrpos,
        )
    )

    ref_rsids, ref_chrpos = load_table_reference(config["dbsnp"])
    rows.append(
        report_overlap(
            label=f"dbSNP151 {selected_build} auto",
            build=selected_build,
            gwas_rsids=gwas_rsids,
            gwas_chrpos=gwas_chrpos,
            ref_rsids=ref_rsids,
            ref_chrpos=ref_chrpos,
        )
    )

    ref_rsids, ref_chrpos = load_vcf_reference(config["kg_vcf"])
    rows.append(
        report_overlap(
            label=f"1000G EUR {selected_build} VCF",
            build=selected_build,
            gwas_rsids=gwas_rsids,
            gwas_chrpos=gwas_chrpos,
            ref_rsids=ref_rsids,
            ref_chrpos=ref_chrpos,
        )
    )

    ref_rsids, ref_chrpos = load_bim_reference(config["plink_bfile"])
    if ref_rsids or ref_chrpos:
        rows.append(
            report_overlap(
                label=f"PLINK BIM {selected_build}",
                build=selected_build,
                gwas_rsids=gwas_rsids,
                gwas_chrpos=gwas_chrpos,
                ref_rsids=ref_rsids,
                ref_chrpos=ref_chrpos,
            )
        )

    out = pd.DataFrame(rows)
    out_file = OUTDIR / f"reference_overlap_selected_{selected_build}.csv"
    out.to_csv(out_file, index=False)

    print("[SAVED]", out_file)
    print()

    return out


# =============================================================================
# 4. PREPARE GWAS FOR PLINK CLUMPING
# =============================================================================

def prepare_plink_clump_file(df: pd.DataFrame):
    print("=" * 100)
    print("4. PREPARE GWAS FOR PLINK CLUMPING")
    print("=" * 100)

    required = ["SNPID", "P"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        print(f"[SKIP] Cannot prepare clump file. Missing columns: {missing}")
        print()
        return None, {
            "plink_clump_input_variant_count": 0,
            "plink_clump_input_file": "",
            "plink_clump_input_reason": f"missing_columns:{'|'.join(missing)}",
        }

    clump = pd.DataFrame({
        "SNP": df["SNPID"].astype(str).map(clean_text),
        "P": pd.to_numeric(df["P"], errors="coerce"),
    })

    clump = clump[
        clump["SNP"].str.match(r"^rs\d+$", case=False, na=False)
        & clump["P"].notna()
        & (clump["P"] > 0)
        & (clump["P"] <= 1)
    ].copy()

    clump = clump.drop_duplicates(subset=["SNP"], keep="first")

    out_file = OUTDIR / "gwas_for_plink_clumping.tsv"
    clump.to_csv(out_file, sep="\t", index=False)

    print("[Clump SNPs]", len(clump))
    print("[SAVED]", out_file)
    print()

    reason = "" if len(clump) > 0 else "no_valid_rsid_p_rows"
    return out_file, {
        "plink_clump_input_variant_count": len(clump),
        "plink_clump_input_file": str(out_file),
        "plink_clump_input_reason": reason,
    }


# =============================================================================
# 5. RUN PLINK CLUMPING USING SELECTED BUILD ONLY
# =============================================================================

def count_clumps(clumped_file):
    if not Path(clumped_file).exists():
        return 0

    count = 0
    with open(clumped_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("CHR"):
                continue
            if line.startswith("Warning") or line.startswith("Error"):
                continue
            count += 1
    return count


def run_plink_clumping(clump_file: Path, selected_build: str):
    print("=" * 100)
    print("5. PLINK CLUMPING")
    print("=" * 100)

    if clump_file is None:
        print("[SKIP] No clump input file was created.")
        print()
        return {
            "plink_clumping_attempted": False,
            "plink_return_code": "",
            "plink_clump_count": "",
            "plink_clump_file": "",
            "plink_log": "",
            "plink_reason": "no_clump_input_file",
        }

    config = REFERENCE_CONFIG[selected_build]
    bfile = config["plink_bfile"]

    if bfile is None:
        print(f"[SKIP] No PLINK bfile configured for selected build: {selected_build}")
        print("[IMPORTANT] Do not use hg19 bfile for hg38 GWAS.")
        print()

        return {
            "plink_clumping_attempted": False,
            "plink_clump_count": "",
            "plink_clump_file": "",
            "plink_log": "",
            "plink_reason": f"no_plink_bfile_for_{selected_build}",
        }

    bed_path = Path(str(bfile) + ".bed")
    bim_path = Path(str(bfile) + ".bim")
    fam_path = Path(str(bfile) + ".fam")

    missing = [p for p in [bed_path, bim_path, fam_path] if not p.exists()]
    if missing:
        print("[SKIP] Missing PLINK bfile components:")
        for p in missing:
            print(" -", p)
        print()

        return {
            "plink_clumping_attempted": False,
            "plink_clump_count": "",
            "plink_clump_file": "",
            "plink_log": "",
            "plink_reason": "missing_bfile_components",
        }

    plink_exe = PLINK if Path(PLINK).exists() else shutil.which("plink")

    if not plink_exe:
        print("[SKIP] PLINK not found")
        print()

        return {
            "plink_clumping_attempted": False,
            "plink_clump_count": "",
            "plink_clump_file": "",
            "plink_log": "",
            "plink_reason": "plink_not_found",
        }

    out_prefix = OUTDIR / f"plink_clumps_{selected_build}"

    cmd = [
        str(plink_exe),
        "--bfile", str(bfile),
        "--clump", str(clump_file),
        "--clump-snp-field", "SNP",
        "--clump-field", "P",
        "--clump-p1", "5e-8",
        "--clump-p2", "1",
        "--clump-r2", "0.1",
        "--clump-kb", "250",
        "--out", str(out_prefix),
    ]

    print("[Selected build]", selected_build)
    print("[Using bfile]", bfile)
    print("[COMMAND]")
    print(" ".join(cmd))

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    print("[PLINK stdout]")
    print(completed.stdout)

    if completed.stderr:
        print("[PLINK stderr]")
        print(completed.stderr)

    clumped_file = str(out_prefix) + ".clumped"
    log_file = str(out_prefix) + ".log"
    n_clumps = count_clumps(clumped_file)

    print("[PLINK return code]", completed.returncode)
    print("[CLUMPED FILE]", clumped_file)
    print("[NUMBER OF CLUMPS]", n_clumps)
    print()

    return {
        "plink_clumping_attempted": True,
        "plink_return_code": completed.returncode,
        "plink_clump_count": n_clumps,
        "plink_clump_file": clumped_file,
        "plink_log": log_file,
        "plink_reason": "",
    }


# =============================================================================
# 6. OPTIONAL PLINK2 CHECK USING SELECTED BUILD ONLY
# =============================================================================

def run_plink2_variant_count(selected_build: str):
    print("=" * 100)
    print("6. PLINK2 SELECTED REFERENCE VARIANT COUNT CHECK")
    print("=" * 100)

    config = REFERENCE_CONFIG[selected_build]
    bfile = config["plink_bfile"]

    if bfile is None:
        print(f"[SKIP] No PLINK bfile configured for selected build: {selected_build}")
        print()

        return {
            "plink2_available": False,
            "plink2_reference_variant_count": "",
            "plink2_reason": f"no_plink_bfile_for_{selected_build}",
        }

    plink2_exe = PLINK2 if Path(PLINK2).exists() else shutil.which("plink2")

    if not plink2_exe:
        print("[SKIP] PLINK2 not found")
        print()

        return {
            "plink2_available": False,
            "plink2_reference_variant_count": "",
            "plink2_reason": "plink2_not_found",
        }

    out_prefix = OUTDIR / f"plink2_ref_check_{selected_build}"

    cmd = [
        str(plink2_exe),
        "--bfile", str(bfile),
        "--write-snplist",
        "--out", str(out_prefix),
    ]

    print("[Selected build]", selected_build)
    print("[Using bfile]", bfile)
    print("[COMMAND]")
    print(" ".join(cmd))

    completed = subprocess.run(cmd, capture_output=True, text=True)

    snplist = str(out_prefix) + ".snplist"

    if Path(snplist).exists():
        with open(snplist, "r", encoding="utf-8", errors="replace") as f:
            n = sum(1 for _ in f)
    else:
        n = ""

    print("[PLINK2 return code]", completed.returncode)
    print("[REFERENCE VARIANTS]", n)
    print()

    return {
        "plink2_available": True,
        "plink2_return_code": completed.returncode,
        "plink2_reference_variant_count": n,
        "plink2_reason": "",
    }


# =============================================================================
# JOB EXECUTION
# =============================================================================

def flatten_overlap_rows(overlap_df):
    result = {}

    if overlap_df is None or overlap_df.empty:
        return result

    for _, row in overlap_df.iterrows():
        reference = clean_text(row.get("reference", "")).lower()
        if reference.startswith("hapmap3"):
            prefix = "hapmap3"
        elif reference.startswith("dbsnp"):
            prefix = "dbsnp151"
        elif reference.startswith("1000g"):
            prefix = "kg_eur"
        elif reference.startswith("plink"):
            prefix = "plink_bim"
        else:
            prefix = reference.replace(" ", "_").replace("/", "_")

        for column in [
            "reference_rsids",
            "reference_chrpos",
            "rsid_overlap",
            "chrpos_overlap",
            "rsid_overlap_pct_of_gwas",
            "chrpos_overlap_pct_of_gwas",
        ]:
            result[f"{prefix}_{column}"] = row.get(column, "")

    return result


def load_prior_build_from_feature_files(output_feature_file):
    output_dir = Path(output_feature_file).parent

    for feature_name in ["Feature3.csv", "Feature5.csv", "Feature6.csv"]:
        feature_file = output_dir / feature_name
        if not feature_file.exists():
            continue

        data = pd.read_csv(feature_file)
        if data.empty:
            continue

        for column in ["genome_build", "gwaslab_build_used", "selected_build"]:
            if column in data.columns:
                build = normalise_build_string(data.iloc[0].get(column, ""))
                if build:
                    return build, str(feature_file)

    return None, ""


def profile_variation7(job_index, phenotype, accession, file_path, final_gwas_file, output_feature_file, args):
    global OUTDIR

    file_path = Path(file_path)
    final_gwas_file = Path(final_gwas_file)
    output_feature_file = Path(output_feature_file)
    output_dir = output_feature_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    OUTDIR = output_dir / "Feature7_files"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    configure_paths(
        ref_dir=args.ref_dir,
        plink=args.plink,
        plink2=args.plink2,
        bigsnpr_hg19=args.bigsnpr_bfile_hg19,
        bigsnpr_hg38=args.bigsnpr_bfile_hg38,
    )

    prior_build, prior_build_source = load_prior_build_from_feature_files(output_feature_file)

    if prior_build:
        selected_build = prior_build
        build_source = "prior_feature"
        ss_initial = None
        df_initial = pd.DataFrame()
        gwas_rsids_initial = set()
        gwas_chrpos_initial = set()
    else:
        ss_initial, df_initial = load_gwas_with_gwaslab(str(final_gwas_file), build=None)
        df_initial = create_derived_columns(df_initial)
        gwas_rsids_initial, gwas_chrpos_initial = extract_gwas_variant_sets(df_initial)
        selected_build = detect_gwas_build(ss_initial, gwas_chrpos_initial)
        build_source = "gwaslab_or_hapmap_overlap"

    selected_config = REFERENCE_CONFIG[selected_build]
    gwaslab_build = selected_config["gwaslab_build"]

    print("=" * 100)
    print("RELOAD GWAS WITH SELECTED BUILD")
    print("=" * 100)
    print("[Selected build]", selected_build)
    print("[Build source]", build_source)
    print("[Prior build source]", prior_build_source)
    print("[GWASLab build argument]", gwaslab_build)
    print()

    ss, df = load_gwas_with_gwaslab(str(final_gwas_file), build=gwaslab_build)
    df = create_derived_columns(df)
    gwas_rsids, gwas_chrpos = extract_gwas_variant_sets(df)

    overlap_df = calculate_selected_reference_overlaps(
        selected_build=selected_build,
        gwas_rsids=gwas_rsids,
        gwas_chrpos=gwas_chrpos,
    )

    clump_file, clump_input_result = prepare_plink_clump_file(df)
    clump_result = run_plink_clumping(
        clump_file=clump_file,
        selected_build=selected_build,
    )
    plink2_result = run_plink2_variant_count(selected_build=selected_build)

    row = {
        "job_index": job_index,
        "phenotype": phenotype,
        "accessionId": accession,
        "file_path": str(file_path),
        "file_name": file_path.name,
        "final_gwas_file": str(final_gwas_file),
        "final_gwas_name": final_gwas_file.name,
        "feature7_work_dir": str(OUTDIR),
        "selected_build": selected_build,
        "selected_build_source": build_source,
        "prior_build_source": prior_build_source,
        "selected_build_label": selected_config["label"],
        "gwaslab_build_used": gwaslab_build,
        "gwaslab_rows": len(df),
        "initial_gwas_unique_rsids": len(gwas_rsids_initial),
        "initial_gwas_unique_chrpos": len(gwas_chrpos_initial),
        "gwas_unique_rsids": len(gwas_rsids),
        "gwas_unique_chrpos": len(gwas_chrpos),
        "selected_hapmap": str(selected_config["hapmap"]),
        "selected_dbsnp": str(selected_config["dbsnp"]),
        "selected_kg_vcf": str(selected_config["kg_vcf"]),
        "selected_plink_bfile": str(selected_config["plink_bfile"]),
        "reference_overlap_file": str(OUTDIR / f"reference_overlap_selected_{selected_build}.csv"),
    }

    row.update(flatten_overlap_rows(overlap_df))
    row.update(clump_input_result)
    row.update(clump_result)
    row.update(plink2_result)

    pd.DataFrame([row]).to_csv(output_feature_file, index=False)
    return row


def run_one_job(index, jobs_file, force, args):
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
    output_feature_file = base_output.parent / "Feature7.csv"

    print("=" * 100)
    print("VARIATION 7: BUILD-MATCHED REFERENCE OVERLAP AND PLINK CLUMPING READINESS")
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
        print(f"[SKIP] Feature7.csv already exists: {output_feature_file}")
        return

    row = profile_variation7(
        job_index=index,
        phenotype=phenotype,
        accession=accession,
        file_path=file_path,
        final_gwas_file=final_gwas_file,
        output_feature_file=output_feature_file,
        args=args,
    )

    print()
    print("[SAVED]")
    print(output_feature_file)
    print()
    print("[KEY RESULTS]")
    print(f"selected_build:               {row['selected_build']}")
    print(f"gwas_unique_rsids:            {row['gwas_unique_rsids']}")
    print(f"gwas_unique_chrpos:           {row['gwas_unique_chrpos']}")
    print(f"hapmap3_rsid_overlap:         {row.get('hapmap3_rsid_overlap', '')}")
    print(f"hapmap3_chrpos_overlap:       {row.get('hapmap3_chrpos_overlap', '')}")
    print(f"kg_eur_rsid_overlap:          {row.get('kg_eur_rsid_overlap', '')}")
    print(f"plink_clumping_attempted:     {row['plink_clumping_attempted']}")
    print(f"plink_clump_count:            {row['plink_clump_count']}")
    print("=" * 100)


def combine_feature7(jobs_file, output_file):
    jobs = pd.read_csv(jobs_file)
    rows = []

    for _, job in jobs.iterrows():
        base_output = Path(clean_text(job["output_feature_file"]))
        feature7 = base_output.parent / "Feature7.csv"
        if feature7.exists():
            rows.append(pd.read_csv(feature7))

    out_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out_df.to_csv(output_file, index=False)

    print("=" * 100)
    print("[COMBINED Feature7.csv FILES]")
    print(f"[FILES COMBINED] {len(rows)}")
    print(f"[OUTPUT] {output_file}")
    print("=" * 100)

    if not out_df.empty and "selected_build" in out_df.columns:
        print()
        print("selected_build")
        print(out_df["selected_build"].value_counts(dropna=False).to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Variation7: build-matched reference overlap and PLINK clumping readiness."
    )
    parser.add_argument("index", nargs="?", type=int, help="GWAS job index from GWAS_jobs.csv")
    parser.add_argument(
        "--jobs-file",
        default="GWAS_jobs.csv",
        help="GWAS job manifest. Default: GWAS_jobs.csv",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing Feature7.csv")
    parser.add_argument("--combine", action="store_true", help="Combine all per-GWAS Feature7.csv files.")
    parser.add_argument(
        "--combine-output",
        default="Variation7_allGWAS.csv",
        help="Output CSV for --combine. Default: Variation7_allGWAS.csv",
    )
    parser.add_argument(
        "--ref-dir",
        default="references",
        help="Reference directory. Default: references",
    )
    parser.add_argument(
        "--plink",
        default="/data/ascher02/uqmmune1/VariationInGWAS/allfiles/plink",
        help="PLINK executable path or command.",
    )
    parser.add_argument(
        "--plink2",
        default="/data/ascher02/uqmmune1/VariationInGWAS/allfiles/plink2",
        help="PLINK2 executable path or command.",
    )
    parser.add_argument(
        "--bigsnpr-bfile-hg19",
        default="",
        help="PLINK bfile prefix for hg19. Default: references/bigsnpr_1000G/1000G_phase3_common_norel",
    )
    parser.add_argument(
        "--bigsnpr-bfile-hg38",
        default="",
        help="Optional PLINK bfile prefix for hg38. Leave blank to skip hg38 clumping.",
    )

    args = parser.parse_args()

    if args.combine:
        combine_feature7(args.jobs_file, args.combine_output)
        return

    if args.index is None:
        parser.error("index is required unless --combine is used")

    run_one_job(args.index, args.jobs_file, args.force, args)


if __name__ == "__main__":
    main()
