#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


PHENOTYPES = [
    "migraine",
    "asthma",
    "depression",
    "body_mass_index_bmi",
    "hypertension",
    "high_cholesterol",
    "hayfever_allergic_rhinitis",
    "hypothyroidism_myxoedema",
    "irritable_bowel_syndrome",
    "gastro_oesophageal_reflux_gord_gastric_reflux",
    "osteoarthritis",
    "blood_pressure_medication",
    "cholesterol_lowering_medication",
]


def is_complete_gwas_file(path):
    name = path.name.lower()

    if name.startswith("partial_"):
        return False

    if name == "feature1.csv":
        return False

    if name.endswith(".part"):
        return False

    skip_terms = [
        "readme",
        "license",
        "licence",
        "manifest",
        "md5",
        ".pdf",
        ".html",
        ".htm",
        ".png",
        ".jpg",
        ".jpeg",
        ".json",
        ".yaml",
        ".yml",
        ".xlsx",
        ".xls",
        "scan_report",
        "download_manifest",
        "normalised_column_mapping",
        "candidate_files_report",
        "header_synonym_dictionary",
    ]

    if any(term in name for term in skip_terms):
        return False

    allowed_extensions = [
        ".gz",
        ".bgz",
        ".txt",
        ".tsv",
        ".csv",
        ".tbl",
        ".ma",
        ".assoc",
        ".meta",
        ".linear",
        ".logistic",
        ".sumstats",
        ".summary",
        ".zip",
        ".tar",
        ".tgz",
        ".tar.gz",
    ]

    return any(name.endswith(ext) for ext in allowed_extensions)


def main():
    rows = []

    for phenotype in PHENOTYPES:
        root = Path(phenotype) / "allgwas"

        if not root.exists():
            print(f"[WARNING] Missing folder: {root}")
            continue

        for accession_dir in sorted(root.iterdir()):
            if not accession_dir.is_dir():
                continue

            accession = accession_dir.name

            for file_path in sorted(accession_dir.iterdir()):
                if not file_path.is_file():
                    continue

                if not is_complete_gwas_file(file_path):
                    continue

                rows.append({
                    "job_index": len(rows) + 1,
                    "phenotype": phenotype,
                    "accessionId": accession,
                    "file_path": str(file_path),
                    "output_feature_file": str(accession_dir / "Feature1.csv"),
                })

    df = pd.DataFrame(rows)

    out = Path("GWAS_jobs.csv")
    df.to_csv(out, index=False)

    print("=" * 100)
    print("[DONE] GWAS job list created")
    print(f"[FILE] {out}")
    print(f"[TOTAL GWAS FILES] {len(df)}")
    print("=" * 100)

    if not df.empty:
        print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()