#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd


DEFAULT_JOBS_FILE = "GWAS_jobs.csv"
DEFAULT_SCORE_RELATIVE = "PRS/Plink_HG38/GWAS_hg38_for_plink.tsv"


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def load_job(index, jobs_file):
    jobs_file = Path(jobs_file)

    if not jobs_file.exists():
        raise FileNotFoundError(f"Missing jobs file: {jobs_file}")

    jobs = pd.read_csv(jobs_file)

    required = ["job_index", "file_path"]
    missing = [c for c in required if c not in jobs.columns]
    if missing:
        raise ValueError(f"Missing required columns in {jobs_file}: {missing}")

    sub = jobs[jobs["job_index"].astype(int) == int(index)]

    if sub.empty:
        raise ValueError(f"No job found for index: {index}")

    return sub.iloc[0]


def get_gwas_output_dir(job, file_path):
    if "output_feature_file" in job.index:
        out = clean_text(job.get("output_feature_file", ""))
        if out:
            return Path(out).parent

    return Path(file_path).parent


def infer_phenotype_dir(job, file_path):
    phenotype = clean_text(job.get("phenotype", ""))

    if phenotype:
        return Path(phenotype), phenotype

    parts = Path(file_path).parts
    if len(parts) >= 1:
        return Path(parts[0]), parts[0]

    raise ValueError("Could not infer phenotype directory.")


def read_bim(bim_file):
    bim_file = Path(bim_file)

    if not bim_file.exists():
        raise FileNotFoundError(f"Missing BIM file: {bim_file}")

    bim = pd.read_csv(
        bim_file,
        sep=r"\s+",
        header=None,
        names=["CHR", "SNP", "CM", "BP", "A1", "A2"],
        dtype=str,
    )

    bim["SNP"] = bim["SNP"].astype(str).str.strip()
    bim["CHR"] = bim["CHR"].astype(str).str.strip()
    bim["BP"] = bim["BP"].astype(str).str.strip()
    bim["A1"] = bim["A1"].astype(str).str.upper().str.strip()
    bim["A2"] = bim["A2"].astype(str).str.upper().str.strip()

    return bim


def read_gwas_snp_column(gwas_file):
    gwas_file = Path(gwas_file)

    if not gwas_file.exists():
        raise FileNotFoundError(f"Missing GWAS file: {gwas_file}")

    preview = pd.read_csv(gwas_file, sep=None, engine="python", nrows=5)

    if "SNP" not in preview.columns:
        raise ValueError(f"GWAS file must contain SNP column. Found: {list(preview.columns)}")

    gwas = pd.read_csv(
        gwas_file,
        sep=None,
        engine="python",
        usecols=["SNP"],
        dtype=str,
    )

    gwas["SNP"] = gwas["SNP"].astype(str).str.strip()
    gwas = gwas[gwas["SNP"].ne("")].copy()

    return preview, gwas


def main():
    parser = argparse.ArgumentParser(
        description="Check SNP overlap between one GWAS job and PLINK BIM file."
    )

    parser.add_argument(
        "index",
        type=int,
        help="GWAS job index. Example: python CheckGWASBIMOverlap.py 1",
    )

    parser.add_argument(
        "--jobs-file",
        default=DEFAULT_JOBS_FILE,
        help="Default: GWAS_jobs.csv",
    )

    parser.add_argument(
        "--fold",
        default="0",
        help="Fold number to check. Default: 0",
    )

    parser.add_argument(
        "--bfile",
        default="train_data.QC",
        help="Bfile prefix inside Fold_* directory. Default: train_data.QC",
    )

    parser.add_argument(
        "--gwas",
        default="",
        help="Optional direct GWAS score file. Default: <GWAS_DIR>/PRS/Plink_HG38/GWAS_hg38_for_plink.tsv",
    )

    parser.add_argument(
        "--bim",
        default="",
        help="Optional direct BIM file. Default: <PHENOTYPE>/Fold_<fold>/<bfile>.bim",
    )

    parser.add_argument(
        "--out",
        default="",
        help="Optional output prefix.",
    )

    args = parser.parse_args()

    job = load_job(args.index, args.jobs_file)

    file_path = Path(clean_text(job["file_path"]))
    phenotype_dir, phenotype = infer_phenotype_dir(job, file_path)
    accession = clean_text(job.get("accessionId", ""))

    gwas_output_dir = get_gwas_output_dir(job, file_path)

    processed_hg38 = gwas_output_dir / "ProcessedGWAS_hg38.csv"
    score_file = gwas_output_dir / DEFAULT_SCORE_RELATIVE

    if args.gwas:
        score_file = Path(args.gwas)

    bim_file = phenotype_dir / f"Fold_{args.fold}" / f"{args.bfile}.bim"

    if args.bim:
        bim_file = Path(args.bim)

    if args.out:
        out_prefix = Path(args.out)
    else:
        out_prefix = gwas_output_dir / "PRS" / "Plink_HG38" / f"Fold_{args.fold}" / "gwas_bim_overlap_check"

    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("GWAS vs BIM OVERLAP CHECK BY JOB INDEX")
    print("=" * 100)
    print(f"[JOB INDEX]       {args.index}")
    print(f"[PHENOTYPE]       {phenotype}")
    print(f"[ACCESSION]       {accession}")
    print(f"[GWAS OUTPUT DIR] {gwas_output_dir}")
    print(f"[PROCESSED HG38]  {processed_hg38}")
    print(f"[GWAS SCORE FILE] {score_file}")
    print(f"[BIM FILE]        {bim_file}")
    print(f"[OUT PREFIX]      {out_prefix}")
    print("=" * 100)

    if processed_hg38.exists():
        processed_preview = pd.read_csv(processed_hg38, nrows=5)
        print("\n[ProcessedGWAS_hg38.csv HEADER]")
        print(list(processed_preview.columns))
        print("\n[ProcessedGWAS_hg38.csv FIRST 5 ROWS]")
        print(processed_preview.to_string(index=False))
    else:
        print("\n[WARNING] ProcessedGWAS_hg38.csv not found:", processed_hg38)

    gwas_preview, gwas = read_gwas_snp_column(score_file)
    bim = read_bim(bim_file)

    print("\n[GWAS SCORE HEADER]")
    print(list(gwas_preview.columns))

    print("\n[GWAS SCORE FIRST 5 ROWS]")
    print(gwas_preview.to_string(index=False))

    print("\n[BIM HEADER]")
    print(["CHR", "SNP", "CM", "BP", "A1", "A2"])

    print("\n[BIM FIRST 5 ROWS]")
    print(bim.head().to_string(index=False))

    gwas_snps = set(gwas["SNP"])
    bim_snps = set(bim["SNP"])

    common = gwas_snps.intersection(bim_snps)
    gwas_only = gwas_snps.difference(bim_snps)
    bim_only = bim_snps.difference(gwas_snps)

    print("\n" + "=" * 100)
    print("[SNP ID OVERLAP SUMMARY]")
    print("=" * 100)
    print(f"GWAS rows:             {len(gwas):,}")
    print(f"GWAS unique SNPs:      {len(gwas_snps):,}")
    print(f"BIM variants:          {len(bim):,}")
    print(f"BIM unique SNPs:       {len(bim_snps):,}")
    print(f"Common SNPs:           {len(common):,}")
    print(f"GWAS-only SNPs:        {len(gwas_only):,}")
    print(f"BIM-only SNPs:         {len(bim_only):,}")
    print(f"Overlap % of GWAS:     {len(common) / max(len(gwas_snps), 1) * 100:.4f}%")
    print(f"Overlap % of BIM:      {len(common) / max(len(bim_snps), 1) * 100:.4f}%")

    print("\n[EXAMPLE GWAS SNP IDs]")
    print(list(gwas["SNP"].head(10)))

    print("\n[EXAMPLE BIM SNP IDs]")
    print(list(bim["SNP"].head(10)))

    print("\n[EXAMPLE COMMON SNPs]")
    print(list(sorted(common))[:10])

    print("\n[EXAMPLE GWAS-ONLY SNPs]")
    print(list(sorted(gwas_only))[:10])

    print("\n[EXAMPLE BIM-ONLY SNPs]")
    print(list(sorted(bim_only))[:10])

    common_file = Path(str(out_prefix) + ".common_snps.txt")
    gwas_only_file = Path(str(out_prefix) + ".gwas_only_snps.first100k.txt")
    bim_only_file = Path(str(out_prefix) + ".bim_only_snps.first100k.txt")
    summary_file = Path(str(out_prefix) + ".summary.csv")

    pd.Series(sorted(common)).to_csv(common_file, index=False, header=False)
    pd.Series(sorted(gwas_only)).head(100000).to_csv(gwas_only_file, index=False, header=False)
    pd.Series(sorted(bim_only)).head(100000).to_csv(bim_only_file, index=False, header=False)

    summary = pd.DataFrame([{
        "job_index": args.index,
        "phenotype": phenotype,
        "accessionId": accession,
        "gwas_output_dir": str(gwas_output_dir),
        "processed_hg38": str(processed_hg38),
        "gwas_score_file": str(score_file),
        "bim_file": str(bim_file),
        "gwas_rows": len(gwas),
        "gwas_unique_snps": len(gwas_snps),
        "bim_variants": len(bim),
        "bim_unique_snps": len(bim_snps),
        "common_snps": len(common),
        "gwas_only_snps": len(gwas_only),
        "bim_only_snps": len(bim_only),
        "overlap_pct_of_gwas": len(common) / max(len(gwas_snps), 1) * 100,
        "overlap_pct_of_bim": len(common) / max(len(bim_snps), 1) * 100,
    }])

    summary.to_csv(summary_file, index=False)

    print("\n" + "=" * 100)
    print("[SAVED]")
    print(common_file)
    print(gwas_only_file)
    print(bim_only_file)
    print(summary_file)
    print("=" * 100)


if __name__ == "__main__":
    main()