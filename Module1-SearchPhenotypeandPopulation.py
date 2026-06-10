#!/usr/bin/env python3

import argparse
import re
import pandas as pd

try:
    from rapidfuzz import fuzz
except ImportError:
    from fuzzywuzzy import fuzz


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_text(x):
    return clean_text(x).lower().strip()


def normalize_number(x):
    if x is None:
        return None

    x = str(x).replace(",", "").replace(" ", "").replace(".", "")
    nums = re.findall(r"\d+", x)

    if not nums:
        return None

    return int(nums[0])


def has_summary_statistics(x):
    text = clean_text(x)

    if text == "":
        return False

    bad_values = {
        "-",
        "nan",
        "none",
        "null",
        "na",
        "n/a",
        "no",
        "not available",
        "not reported",
    }

    if text.lower().strip() in bad_values:
        return False

    return True


def is_european(sample_text):
    text = normalize_text(sample_text)

    european_terms = [
        "european",
        "european ancestry",
        "eur",
    ]

    return any(term in text for term in european_terms)


def extract_cases_controls_samples(sample_text):
    text = normalize_text(sample_text)

    cases = None
    controls = None
    samples = None

    case_patterns = [
        r"([\d,.\s]+)\s*(?:[a-zA-Z\s\-]*?)\bcase(?:s)?\b",
        r"\bcase(?:s)?\b\s*[:=]?\s*([\d,.\s]+)",
    ]

    control_patterns = [
        r"([\d,.\s]+)\s*(?:[a-zA-Z\s\-]*?)\bcontrol(?:s)?\b",
        r"\bcontrol(?:s)?\b\s*[:=]?\s*([\d,.\s]+)",
    ]

    sample_patterns = [
        r"([\d,.\s]+)\s*(?:[a-zA-Z\s\-]*?)\bindividual(?:s)?\b",
        r"([\d,.\s]+)\s*(?:[a-zA-Z\s\-]*?)\bsample(?:s)?\b",
        r"([\d,.\s]+)\s*(?:[a-zA-Z\s\-]*?)\bparticipant(?:s)?\b",
        r"([\d,.\s]+)\s*(?:[a-zA-Z\s\-]*?)\bsubject(?:s)?\b",
    ]

    for pattern in case_patterns:
        match = re.search(pattern, text)
        if match:
            cases = normalize_number(match.group(1))
            break

    for pattern in control_patterns:
        match = re.search(pattern, text)
        if match:
            controls = normalize_number(match.group(1))
            break

    for pattern in sample_patterns:
        match = re.search(pattern, text)
        if match:
            samples = normalize_number(match.group(1))
            break

    if cases is not None and controls is not None:
        samples = cases + controls

    return cases, controls, samples


def map_gwas_catalog_columns(df):
    column_mapping = {
        "DATE ADDED TO CATALOG": "dateAdded",
        "PUBMEDID": "pubmedId",
        "PUBMED ID": "pubmedId",
        "FIRST AUTHOR": "firstAuthor",
        "DATE": "date",
        "JOURNAL": "journal",
        "LINK": "link",
        "STUDY": "study",
        "DISEASE/TRAIT": "reportedTrait",
        "REPORTED TRAIT": "reportedTrait",
        "MAPPED_TRAIT": "efoTraits",
        "MAPPED TRAIT": "efoTraits",
        "MAPPED_TRAIT_URI": "mappedTraitUri",
        "INITIAL SAMPLE SIZE": "initialSampleDescription",
        "REPLICATION SAMPLE SIZE": "replicationSampleDescription",
        "PLATFORM [SNPS PASSING QC]": "platformSnps",
        "ASSOCIATION COUNT": "associationCount",
        "STUDY ACCESSION": "accessionId",
        "GENOTYPING TECHNOLOGY": "genotypingTechnology",
        "FULL SUMMARY STATISTICS": "fullSummaryStatistics",
        "SUMMARY STATS LOCATION": "summaryStatistics",
        "SUMMARY STATISTICS LOCATION": "summaryStatistics",
    }

    rename_dict = {}

    for col in df.columns:
        clean_col = col.strip()
        if clean_col in column_mapping:
            rename_dict[col] = column_mapping[clean_col]

    df = df.rename(columns=rename_dict)

    required_cols = [
        "reportedTrait",
        "efoTraits",
        "initialSampleDescription",
        "replicationSampleDescription",
        "summaryStatistics",
        "fullSummaryStatistics",
        "accessionId",
        "pubmedId",
        "firstAuthor",
        "journal",
        "study",
        "date",
        "platformSnps",
        "genotypingTechnology",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    return df


def remove_duplicate_gwas_rows(df):
    before = len(df)

    dedup_cols = [
        "pubmedId",
        "firstAuthor",
        "reportedTrait",
        "efoTraits",
        "accessionId",
        "initialSampleDescription",
        "summaryStatistics",
        "study",
    ]

    for col in dedup_cols:
        if col not in df.columns:
            df[col] = ""

    for col in dedup_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()

    df = df.drop_duplicates(subset=dedup_cols, keep="first").copy()

    after = len(df)

    print(f"[DEDUPLICATION] Removed {before - after:,} duplicate rows")
    print(f"[DEDUPLICATION] Remaining rows: {after:,}")

    return df


def search_phenotype_and_population(
    phenotype,
    population,
    input_file,
    output_file,
):
    phenotype_clean = phenotype.replace("_", " ").lower().strip()

    print("=" * 100)
    print("GWAS CATALOG SEARCH")
    print("=" * 100)
    print(f"[INPUT] {input_file}")
    print(f"[PHENOTYPE FOR RANKING] {phenotype_clean}")
    print(f"[POPULATION FILTER] {population}")
    print(f"[OUTPUT] {output_file}")
    print("[FILTERING RULES]")
    print("  1. Remove missing summaryStatistics")
    print("  2. Remove non-European rows")
    print("  3. Remove rows where FuzzySimilarity_reportedTrait <= 0")
    print("=" * 100)

    df = pd.read_csv(input_file, sep="\t", low_memory=False)
    df = map_gwas_catalog_columns(df)

    df["reportedTrait_clean"] = df["reportedTrait"].astype(str).str.lower().str.strip()
    df["efoTraits_clean"] = df["efoTraits"].astype(str).str.lower().str.strip()
    df["sample_clean"] = df["initialSampleDescription"].astype(str).str.lower().str.strip()

    # 1. Remove rows with missing summaryStatistics
    before_sumstats = len(df)
    df = df[df["summaryStatistics"].apply(has_summary_statistics)].copy()
    after_sumstats = len(df)

    print(f"[SUMMARY STATISTICS FILTER] Removed {before_sumstats - after_sumstats:,} rows")
    print(f"[SUMMARY STATISTICS FILTER] Remaining rows: {after_sumstats:,}")

    # 2. Remove rows that are not European
    if population is not None:
        population = population.lower().strip()

        if population in ["european", "eur", "europe"]:
            before_pop = len(df)
            df = df[df["sample_clean"].apply(is_european)].copy()
            after_pop = len(df)

            print(f"[EUROPEAN FILTER] Removed {before_pop - after_pop:,} non-European rows")
            print(f"[EUROPEAN FILTER] Remaining rows: {after_pop:,}")
        else:
            before_pop = len(df)
            df = df[df["sample_clean"].str.contains(population, case=False, na=False)].copy()
            after_pop = len(df)

            print(f"[POPULATION FILTER] Removed {before_pop - after_pop:,} non-matching rows")
            print(f"[POPULATION FILTER] Remaining rows: {after_pop:,}")

    # 3. Calculate fuzzy scores
    df["FuzzySimilarity_reportedTrait"] = df["reportedTrait_clean"].apply(
        lambda x: fuzz.token_sort_ratio(phenotype_clean, x)
    )

    df["FuzzySimilarity_mappedTrait"] = df["efoTraits_clean"].apply(
        lambda x: fuzz.token_sort_ratio(phenotype_clean, x)
    )

    # 4. Remove only rows where FuzzySimilarity_reportedTrait <= 0
    before_fuzzy = len(df)
    df = df[df["FuzzySimilarity_reportedTrait"] >=50].copy()
    after_fuzzy = len(df)

    print(f"[FUZZY > 0 FILTER] Removed {before_fuzzy - after_fuzzy:,} rows")
    print(f"[FUZZY > 0 FILTER] Remaining rows: {after_fuzzy:,}")

    # 5. Ranking flags
    df["ExactReportedTraitMatch"] = (
        df["reportedTrait_clean"] == phenotype_clean
    ).astype(int)

    df["ReportedTraitContainsPhenotype"] = df["reportedTrait_clean"].apply(
        lambda x: 1 if re.search(rf"\b{re.escape(phenotype_clean)}\b", x) else 0
    )

    df["PhenotypeRankScore"] = (
        df["FuzzySimilarity_reportedTrait"]
        + (df["ExactReportedTraitMatch"] * 100)
        + (df["ReportedTraitContainsPhenotype"] * 25)
    )

    # 6. Extract cases, controls, samples
    extracted = df["initialSampleDescription"].apply(extract_cases_controls_samples)

    df["CASES"] = extracted.apply(lambda x: x[0] if x[0] is not None else "-")
    df["CONTROLS"] = extracted.apply(lambda x: x[1] if x[1] is not None else "-")
    df["SAMPLES"] = extracted.apply(lambda x: x[2] if x[2] is not None else "-")

    df["SearchPhenotype"] = phenotype_clean
    df["SearchPopulation"] = population if population else "-"

    df["_samples_sort"] = pd.to_numeric(df["SAMPLES"], errors="coerce").fillna(0)

    # 7. Sort before deduplication so best row is retained
    df = df.sort_values(
        by=[
            "ExactReportedTraitMatch",
            "PhenotypeRankScore",
            "FuzzySimilarity_reportedTrait",
            "_samples_sort",
        ],
        ascending=[False, False, False, False],
    ).copy()

    # 8. Remove duplicate rows
    df = remove_duplicate_gwas_rows(df)

    # 9. Sort again and add rank
    df = df.sort_values(
        by=[
            "ExactReportedTraitMatch",
            "PhenotypeRankScore",
            "FuzzySimilarity_reportedTrait",
            "_samples_sort",
        ],
        ascending=[False, False, False, False],
    ).copy()

    df.insert(0, "Rank", range(1, len(df) + 1))

    preferred_columns = [
        "Rank",
        "SearchPhenotype",
        "SearchPopulation",
        "pubmedId",
        "firstAuthor",
        "date",
        "journal",
        "reportedTrait",
        "efoTraits",
        "accessionId",
        "study",
        "initialSampleDescription",
        "replicationSampleDescription",
        "CASES",
        "CONTROLS",
        "SAMPLES",
        "fullSummaryStatistics",
        "summaryStatistics",
        "genotypingTechnology",
        "platformSnps",
        "ExactReportedTraitMatch",
        "ReportedTraitContainsPhenotype",
        "FuzzySimilarity_reportedTrait",
        "FuzzySimilarity_mappedTrait",
        "PhenotypeRankScore",
    ]

    final_columns = [c for c in preferred_columns if c in df.columns]

    remaining_columns = [
        c
        for c in df.columns
        if c not in final_columns
        and not c.startswith("_")
        and c not in [
            "reportedTrait_clean",
            "efoTraits_clean",
            "sample_clean",
        ]
    ]

    df = df[final_columns + remaining_columns]

    df.to_csv(output_file, index=False)

    print("=" * 100)
    print(f"[DONE] Final rows saved: {len(df):,}")
    print(f"[SAVED] {output_file}")
    print("=" * 100)

    if len(df) > 0:
        preview_cols = [
            "Rank",
            "reportedTrait",
            "efoTraits",
            "initialSampleDescription",
            "CASES",
            "CONTROLS",
            "SAMPLES",
            "summaryStatistics",
            "FuzzySimilarity_reportedTrait",
            "FuzzySimilarity_mappedTrait",
            "PhenotypeRankScore",
        ]

        preview_cols = [c for c in preview_cols if c in df.columns]

        print("\n[PREVIEW]")
        print(df[preview_cols].head(30).to_string(index=False))

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Search GWAS Catalog. Filters summaryStatistics, population, and fuzzy score > 0 only."
    )

    parser.add_argument(
        "--phenotype",
        type=str,
        required=True,
        help="Phenotype name used for fuzzy ranking, e.g. migraine",
    )

    parser.add_argument(
        "--population",
        type=str,
        default="european",
        help="Population filter, e.g. european",
    )

    parser.add_argument(
        "--input",
        type=str,
        default="summary_statistics_table_export.tsv",
        help="GWAS Catalog summary statistics TSV file.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file.",
    )

    args = parser.parse_args()

    output_file = args.output
    if output_file is None:
        output_file = args.phenotype.replace(" ", "_").lower() + ".csv"

    search_phenotype_and_population(
        phenotype=args.phenotype,
        population=args.population,
        input_file=args.input,
        output_file=output_file,
    )


if __name__ == "__main__":
    main()