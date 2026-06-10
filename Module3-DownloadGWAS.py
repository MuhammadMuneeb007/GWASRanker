#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path

import pandas as pd


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def make_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_filename(x):
    x = clean_text(x)

    if not x:
        return "downloaded_gwas_file"

    bad_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
    for bad in bad_chars:
        x = x.replace(bad, "_")

    x = x.replace(" ", "_")

    while "__" in x:
        x = x.replace("__", "_")

    return x.strip("_")


def download_file(url, outfile):
    make_dir(outfile.parent)

    cmd = [
        "wget",
        "-c",
        "--tries", "3",
        "--timeout", "60",
        "-O", str(outfile),
        url,
    ]

    print("[COMMAND]", " ".join(cmd))

    result = subprocess.run(cmd, check=False)

    if result.returncode == 0:
        print(f"[DOWNLOADED] {outfile}")
    else:
        print(f"[FAILED] return code = {result.returncode}")

    return result.returncode


def download_phenotype(phenotype):
    phenotype_dir = Path(phenotype)
    scan_file = phenotype_dir / "scan_report.csv"

    if not scan_file.exists():
        raise FileNotFoundError(f"Missing file: {scan_file}")

    df = pd.read_csv(scan_file)

    required_cols = [
        "accessionId",
        "candidate_url",
        "candidate_name",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {scan_file}: {missing}")

    df = df[
        df["candidate_url"].notna()
        & (df["candidate_url"].astype(str).str.strip() != "")
    ].copy()

    df = df.drop_duplicates(subset=["accessionId", "candidate_url"])

    print("=" * 100)
    print("MODULE 3: DOWNLOAD COMPLETE GWAS FILES ONLY")
    print("=" * 100)
    print(f"[PHENOTYPE] {phenotype}")
    print(f"[SCAN FILE] {scan_file}")
    print(f"[TOTAL FILES TO DOWNLOAD] {len(df)}")
    print("=" * 100)

    manifest_rows = []

    for count, (_, row) in enumerate(df.iterrows(), start=1):
        accession = clean_text(row["accessionId"]).upper()
        url = clean_text(row["candidate_url"])
        candidate_name = clean_text(row["candidate_name"])

        if not candidate_name:
            candidate_name = url.rstrip("/").split("/")[-1]

        filename = safe_filename(candidate_name)

        outdir = phenotype_dir / "allgwas" / accession
        outfile = outdir / filename

        print()
        print("-" * 100)
        print(f"[{count}/{len(df)}] {accession}")
        print(f"[URL] {url}")
        print(f"[OUTDIR] {outdir}")
        print(f"[OUTFILE] {outfile}")

        return_code = download_file(url, outfile)

        file_exists = outfile.exists()
        file_size = outfile.stat().st_size if file_exists else 0

        manifest_rows.append({
            "phenotype": phenotype,
            "accessionId": accession,
            "candidate_name": candidate_name,
            "candidate_url": url,
            "output_directory": str(outdir),
            "downloaded_file": str(outfile),
            "wget_return_code": return_code,
            "file_exists": file_exists,
            "file_size_bytes": file_size,
        })

    manifest = pd.DataFrame(manifest_rows)

    out_manifest = phenotype_dir / "download_manifest.csv"
    manifest.to_csv(out_manifest, index=False)

    print()
    print("=" * 100)
    print("[DONE]")
    print(f"[MANIFEST] {out_manifest}")
    print("=" * 100)

    if not manifest.empty:
        print()
        print("[DOWNLOAD SUMMARY]")
        print(manifest[[
            "accessionId",
            "file_exists",
            "file_size_bytes",
            "downloaded_file"
        ]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Download complete GWAS files from Module2 scan_report.csv."
    )

    parser.add_argument(
        "phenotype",
        type=str,
        help="Phenotype folder, e.g. migraine or allfiles/high_cholesterol"
    )

    args = parser.parse_args()

    download_phenotype(args.phenotype)


if __name__ == "__main__":
    main()