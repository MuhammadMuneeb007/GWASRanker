from pathlib import Path

phenotypes = [p for p in Path(".").iterdir() if p.is_dir() and (p / "allgwas").exists()]
print("=" * 100)
print("DOWNLOADED GWAS FILES BY PHENOTYPE")
print("=" * 100)
for phenotype in sorted(phenotypes):
    files = [f for f in (phenotype / "allgwas").rglob("*") if f.is_file()]
    print(f"{phenotype.name}: {len(files)} files")
    for f in files:
        print(f"  {f}")
print("=" * 100)