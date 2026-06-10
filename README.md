# GWASRanker

**A quality-aware framework for selecting GWAS summary-statistic files for reproducible polygenic risk score construction.**

GWASRanker evaluates, harmonises, scores, and ranks candidate GWAS summary-statistic files before downstream polygenic risk score (PRS) construction. The framework was developed to make GWAS input selection explicit, auditable, and reproducible, especially when multiple GWAS summary-statistic files are available for the same phenotype.

![GWASRanker workflow](Flowchart_page-0001.jpg)

---

## Key results from the manuscript

GWASRanker was evaluated across **284 candidate GWAS summary-statistic files** from **13 phenotypes**. Of these, **246 files produced successful PLINK PRS results** and were used for supervised model development.

Main findings:

- GWAS files for the same phenotype were **not interchangeable**; within-phenotype test PRS performance varied by up to approximately **0.40 AUC units**.
- The **target-aware model** using Features 1--8 produced the most consistent test-PRS ranking performance, with **Spearman rho approximately 0.631** under leave-one-phenotype-out validation.
- The **full pre-performance model** using Features 1--9 achieved the strongest top-file identification, including the highest **top-1 identification rate** and **NDCG@3**.
- The most important predictors of GWAS suitability followed a practical feature hierarchy:
  1. **Target-overlap success**
  2. **MAF concordance between GWAS and target genotype data**
  3. **GWAS signal strength and heritability-related features**
  4. **File-format, schema, and harmonisation quality**

These results show that GWAS input selection is a measurable source of variation in PRS workflows and should be treated as a formal preprocessing and ranking step rather than a manual choice based only on sample size or SNP count.

---

## Repository overview

This repository contains the GWASRanker code used to:

1. Search and download candidate GWAS summary-statistic files from the GWAS Catalog.
2. Inspect, normalise, and extract GWAS files.
3. Generate nine feature groups describing GWAS quality, harmonisation, heritability, target compatibility, PRS-readiness, clumping, pruning, and target Fisher concordance.
4. Calculate PRS using PLINK and PRSice-2.
5. Merge feature tables and PRS outputs.
6. Train nested GWASRanker models.
7. Generate final phenotype-specific GWAS rankings and performance summaries.

---

## Important data and software requirements

This repository does **not** include restricted genotype or phenotype data. The target genotype data used in the manuscript were derived from UK Biobank and must be accessed through the appropriate approved data-access route.

Before running the pipeline, make sure the following are available.

### Required software

Place the following executables in the working directory or make them available in your system `PATH`:

```text
plink
plink2
PRSice_linux or PRSice executable
Rscript
python
```

The manuscript used the following major tools:

- PLINK / PLINK2 for genotype processing, clumping, pruning, and scoring.
- PRSice-2 for PRS construction.
- GWASLab for harmonisation support.
- LDSC for heritability and summary-statistic signal features.
- Python packages for data processing and machine-learning model development.

### Required databases and reference resources

Download and configure all required databases before running the Variation scripts. These are needed for reference matching, harmonisation, heritability, target compatibility, and PRS-readiness calculations.

Typical required resources include:

```text
GWAS Catalog metadata and downloaded summary statistics
HapMap3 reference SNP set
dbSNP151 reference files
1000 Genomes European reference files
LDSC reference files and weights
PLINK BIM reference files
Target genotype BIM/FAM/BED files
Phenotype and covariate files
```

The exact location of each database should match the paths expected by the scripts, or the paths should be edited in the corresponding Python files before execution.

---

## Target genotype data organisation

The target genotype data must be organised phenotype-by-phenotype and fold-by-fold. This structure is required because the PRS scripts and several Variation scripts expect train/test genotype files inside phenotype-specific `Fold_0` to `Fold_4` directories.

For details on preparing genotype folds and PRS benchmarking data, refer to:

- https://github.com/MuhammadMuneeb007/Benchmarking-Heritability-Estimation-Strategies-Across-86-Configurations-and-Their-Downstream-Effect
- https://muhammadmuneeb007.github.io/PRSTools/Introduction.html

Example structure for one phenotype:

```text
body_mass_index_bmi
├── body_mass_index_bmi.bed
├── body_mass_index_bmi.bim
├── body_mass_index_bmi.cov
├── body_mass_index_bmi.covOLD
├── body_mass_index_bmi.fam
├── body_mass_index_bmi.gz
├── body_mass_index_bmi.height
├── body_mass_index_bmi.heightOLD
├── body_mass_index_bmi_QC.bed
├── body_mass_index_bmi_QC.bim
├── body_mass_index_bmi_QC.cov
├── body_mass_index_bmi_QC.fam
├── body_mass_index_bmi_QC.log
├── Fold_0
│   ├── test_data.bed
│   ├── test_data.bim
│   ├── test_data.cov
│   ├── test_data.fam
│   ├── test_data.log
│   ├── train_data.a1
│   ├── train_data.bed
│   ├── train_data.bim
│   ├── train_data.cov
│   ├── train_data.fam
│   ├── train_data.log
│   ├── train_data.mismatch
│   ├── train_data.QC.bed
│   ├── train_data.QC.bim
│   ├── train_data.QC.fam
│   ├── train_data.QC.het
│   ├── train_data.QC.log
│   ├── train_data.QC.rel.id
│   ├── train_data.QC.snplist
│   └── train_data.valid.sample
├── Fold_1
├── Fold_2
├── Fold_3
├── Fold_4
├── GCST90018947_buildGRCh37.tsv.gz
├── gwas.csv.modified
├── Output.py
└── PeopleWithPhenotype.txt
```

The project root may also contain shared genotype files and setup scripts:

```text
genotypes.bed
genotypes.bim19
genotypes.bim38
genotypes.fam
phenotype_file.txt
plink
plink2
Step1-MakeDirectories.py
Step2-PerformQualityControls.py
```

---

## Input phenotype files

The repository expects phenotype-specific CSV files. Example files include:

```text
asthma.csv
blood_pressure_medication.csv
body_mass_index_bmi.csv
cholesterol_lowering_medication.csv
depression.csv
gastro_oesophageal_reflux_gord_gastric_reflux.csv
hayfever_allergic_rhinitis.csv
high_cholesterol.csv
hypertension.csv
hypothyroidism_myxoedema.csv
irritable_bowel_syndrome.csv
migraine.csv
```

Each file should contain the phenotype name or search terms needed to retrieve candidate GWAS summary-statistic files from the GWAS Catalog.

---

# Pipeline from start to finish

The full workflow should be run in the following order.

---

## Step 1: Download and inspect GWAS files using Module 1 to Module 5

The first stage identifies candidate GWAS files from the GWAS Catalog, downloads them, extracts them, and lists PRS-compatible columns.

### Module 1: Search phenotype and population

```bash
python Module1-SearchPhenotypeandPopulation.py
```

Purpose:

- Search GWAS Catalog records for the selected phenotype.
- Collect phenotype-level and population-level metadata.
- Generate the initial candidate GWAS list.

Expected output:

```text
<phenotype>.csv
GWAS accession metadata
Candidate GWAS records
```

---

### Module 2: Search, poke, normalise, and scan

```bash
python Module2-Search_Poke_Normalize_Scan.py
```

Purpose:

- Inspect candidate GWAS links.
- Check whether files are downloadable.
- Normalise metadata and file naming.
- Identify likely summary-statistic files.

---

### Module 3: Download GWAS files

```bash
python Module3-DownloadGWAS.py
```

Purpose:

- Download candidate GWAS summary-statistic files.
- Store files in phenotype-specific directories.
- Preserve accession IDs and source metadata.

To list downloaded files:

```bash
python Module3.1-ListDownloadedGWASFiles.py
```

---

### Module 4: Extract GWAS archives

```bash
python Module4-ExtractGWAS.py
```

Purpose:

- Extract compressed GWAS files where needed.
- Detect file structure, archive members, delimiters, and headers.
- Prepare files for schema mapping and harmonisation.

---

### Module 5: List PRS columns

```bash
python Module5-ListPRSColumns.py
```

Purpose:

- Inspect available GWAS columns.
- Identify SNP, chromosome, position, allele, effect-size, standard-error, P-value, sample-size, and frequency fields.
- Check whether the minimum PRS columns are available.

---

## Step 2: Generate GWASRanker features using Variation 0 to Variation 9

The Variation scripts generate the nine GWASRanker feature groups. Run them after the GWAS files have been downloaded, extracted, and organised.

The scripts can usually be run one GWAS index at a time:

```bash
python Variation1.py <JOB_INDEX>
```

or using the supplied SLURM scripts where available:

```bash
sbatch variation.sh
sbatch variation1.sh
```

### Variation 0: Initial checks

```bash
python Variation0.py <JOB_INDEX>
```

Purpose:

- Prepare the selected GWAS job.
- Check paths and required input files.
- Confirm that phenotype-specific folders exist.

---

### Variation 1: Feature 1, raw file structure

```bash
python Variation1.py <JOB_INDEX>
python Variation1-AllFeatures.py
python Variation1-Summary.py
```

Generated output:

```text
Feature1_All.csv
```

Captures:

- File extension
- Compression type
- File size
- Delimiter
- Header status
- Sampled rows and columns
- Reader status

---

### Variation 2: Feature 2, schema mapping

```bash
python Variation2.py <JOB_INDEX>
python Variation2_Allfeatures.py
python Variation2-Summary.py
```

Generated output:

```text
Feature2_All.csv
```

Captures:

- Raw headers
- Normalised headers
- Mapped GWAS columns
- Missing/unmapped columns
- PRS minimum-column compatibility

---

### Variation 3: Feature 3, harmonisation status

```bash
python Variation3.py <JOB_INDEX>
python Variation3.1-Manualcheck.py
python Variation3-Allfeatures.py
```

Generated output:

```text
Feature3_All.csv
FinalGWAS.csv files inside phenotype/GWAS folders
```

Captures:

- GWASLab loading status
- Genome-build evidence
- Harmonised file status
- Standard column recovery
- Harmonised variant count
- QC removed variant count

---

### Variation 4: Feature 4, variant and statistical quality

```bash
python Variation4.py <JOB_INDEX>
python Variation4-Allfeatures.py
```

Generated output:

```text
Feature4_All.csv
```

Captures:

- Total variant count
- Missing SNP, P-value, effect-size, and allele percentages
- Duplicated SNP percentage
- Palindromic SNP percentage
- Non-ACGT allele percentage
- Genome-wide significant variant count
- Minimum and median P-value

---

### Variation 5: Feature 5, FinalGWAS quality

```bash
python Variation5.py <JOB_INDEX>
python Variation5-Allfeatures.py
```

Generated output:

```text
Feature5_All.csv
```

Captures:

- FinalGWAS row count
- File size
- Chromosome coverage
- Duplicated SNPID and position counts
- Allele completeness
- P-value validity
- Beta, odds-ratio, standard-error availability
- PRS-compatible variant count

---

### Variation 6: Feature 6, heritability and GWAS signal

```bash
python Variation6.py <JOB_INDEX>
python Variation6-Allfeatures.py
```

Generated output:

```text
Feature6_All.csv
```

Captures:

- LDSC heritability
- LDSC intercept
- LDSC ratio
- Genomic inflation factor
- Mean chi-square
- N x h2
- Effective sample-size summaries
- Heritability standard error

This step requires LDSC reference files and correct summary-statistic formatting.

---

### Variation 7: Feature 7, reference-panel overlap

```bash
python Variation7.py <JOB_INDEX>
python Variation7-Allfeatures.py
```

Generated output:

```text
Feature7_All.csv
```

Captures:

- HapMap3 overlap
- dbSNP151 overlap
- 1000 Genomes EUR overlap
- PLINK BIM reference overlap
- Reference allele-frequency concordance
- Reference-clumping readiness

Download and configure the required reference databases before this step.

---

### Variation 8: Feature 8, target-genotype compatibility

```bash
python Variation8.py <JOB_INDEX>
python Variation8-Allfeatures.py
```

Generated output:

```text
Feature8_All.csv
```

Captures:

- Target BIM variant count
- FinalGWAS variant count
- rsID overlap count
- Chromosome-position overlap count
- Direct allele match
- Reversed allele match
- Strand-compatible allele count
- Allele mismatch count
- Palindromic overlap percentage
- Final usable SNP count
- Target MAF concordance

This is one of the most important feature groups because target compatibility was the strongest predictor of downstream PRS suitability.

---

### Variation 9: Feature 9, PRS-readiness, clumping, pruning, and Fisher concordance

```bash
python Variation9.py <JOB_INDEX>
python Variation9-Allfeatures.py
python Variation9-CheckFiles.py
```

Generated output:

```text
Feature9_All.csv
```

Captures:

- Score-ready SNP count
- Clump-ready SNP count
- Valid SNPID fraction
- Valid P-value fraction
- Valid effect-size fraction
- Clumped variant counts across P-value thresholds
- LD prune-in and prune-out counts
- Pruning retention percentage
- Target Fisher beta concordance
- Target Fisher P-value concordance

This step requires PLINK/PLINK2, target genotype folds, phenotype files, and correctly harmonised GWAS files.

---

## Step 3: Calculate PRS using PLINK

After feature extraction and harmonisation, calculate PRS using PLINK.

```bash
python PRS1-Calculation-Plink.py
```

Additional normalisation scripts:

```bash
python PRS1-Normalization1.py
python PRS1-Normalization2-hg38.py
```

Purpose:

- Build PLINK-compatible score files.
- Perform clumping and thresholding.
- Calculate PRS across fold-specific train and test data.
- Evaluate pure PRS, null, and full covariate models.

Expected outputs include fold-level PRS scores and PRS performance summaries.

---

## Step 4: Calculate PRS using PRSice-2

PRSice-2 can also be used to calculate PRS results.

```bash
python PRS1-Calculation-PRSice-2.py
```

Additional scripts:

```bash
python PRS2-Calculation1.py
python PRS2-Calculation2-Verification.py
python PRS2-Calculation-Merge.py
python PRS2-Correlation.py
```

Purpose:

- Run PRSice-2 for candidate GWAS files.
- Verify PRSice outputs.
- Merge PRSice results.
- Compare PLINK and PRSice-derived PRS summaries where applicable.

---

## Step 5: Merge all features and PRS results

After all feature files and PRS outputs are available, merge them into a single modelling table.

```bash
python Analysis0.0-MergeAllFeatures.py
python AllFeatures-PRS-Merge.py
```

Expected output:

```text
AllFeatures_PRS_Merged.csv
```

This file is the main machine-readable GWASRanker dataset. It links each GWAS file to:

- Phenotype metadata
- GWAS accession information
- Feature 1--Feature 9 outputs
- PLINK PRS performance
- PRSice-2 summaries where available
- Derived modelling targets

---

## Step 6: Generate analysis datasets

The model-ready datasets are generated using Analysis19 scripts.

```bash
python Analysis19-GenerateData1.py
python Analysis19-GenerateData2.py
python Analysis19-GenerateData3.py
```

These scripts create the three nested modelling datasets:

| Dataset | Model | Feature groups | Interpretation |
|---|---|---|---|
| Dataset 1 | Model 1 | Features 1--6 | GWAS-intrinsic model |
| Dataset 2 | Model 2 | Features 1--8 | Target-aware model |
| Dataset 3 | Model 3 | Features 1--9 | Full pre-performance model |

The same supervised PRS-performance targets are used across all datasets. Leakage-prone columns, such as direct PRS performance fields, AUC values, R2 values, and downstream outcome summaries, are excluded from predictors.

---

## Step 7: Train GWASRanker models

Train the three nested models using the Analysis20 scripts.

```bash
python Analysis20-TrainModel1.py
python Analysis20-TrainModel2.py
python Analysis20-TrainModel3.py
```

Purpose:

- Train GWASRanker as a supervised ranking framework.
- Evaluate multiple regression algorithms.
- Use leave-one-phenotype-out cross-validation.
- Report Spearman rank correlation, Kendall tau, top-k overlap, NDCG, MAE, RMSE, and R2.
- Estimate uncertainty using bootstrap confidence intervals.
- Assess significance using permutation and across-fold tests.

---

## Step 8: Fit final all-data models and generate final rankings

After LOPO validation, final selected models are fitted using all available GWAS files.

```bash
python Analysis21-FinalRankerModel1.py
python Analysis21-FinalRankerModel2.py
python Analysis21-FinalRankerModel3.py
```

Purpose:

- Refit selected models using all model-eligible GWAS files.
- Save final feature weights or feature-importance values.
- Generate phenotype-specific GWAS rankings.
- Produce final files that can be used for prospective GWAS selection.

Important: LOPO validation performance and final all-data rankings should be reported separately. LOPO results estimate generalisation, while final all-data models provide the fitted models for practical ranking.

---

## Step 9: List model performance

Use the following script to summarise final performance tables.

```bash
python Analysis22-ListPerformance.py
```

Purpose:

- Print model performance summaries.
- Compare Model 1, Model 2, and Model 3.
- Summarise selected models per target.
- Display ranking metrics and final feature-importance summaries.

---

# Additional descriptive analysis scripts

The following scripts generate descriptive summaries and supplementary tables:

```bash
python Analysis1-SeeYourData.py
python Analysis2-FileSchemaVariation.py
python Analysis3-HarmonisationFinalGWASQuality.py
python Analysis4-GWASIntrinsicStatisticalQuality.py
python Analysis4-GWASIntrinsicStatisticalQuality2.py
python Analysis5-ReferenceHeritabilityFeatures.py
python Analysis6-TargetGenotypeCompatibility.py
python Analysis7-PRSReadinessClumpingPruning.py
python Analysis8-PRSPerformance.py
```

GWAS correlation and concordance scripts:

```bash
python Analysis12-Correlation.py
python Analysis13-Correlation.py
python Analysis14-GWAS_Correlation.py
bash Analysis14-GWAS_Correlation.sh
python Analysis15-GWAS_Correlation_Table.py
```

Model development and multi-target modelling scripts:

```bash
python Analysis17-ModelDevelopment.py
python Analysis18-MultiTarget-MultiModel.py
```

---

# Expected repository files

A typical working directory may contain:

```text
AllFeatures_PRS_Merged.csv
AllFeatures-PRS-Merge.py
Analysis0.0-MergeAllFeatures.py
Analysis12-Correlation.py
Analysis13-Correlation.py
Analysis14-GWAS_Correlation.py
Analysis14-GWAS_Correlation.sh
Analysis15-GWAS_Correlation_Table.py
Analysis17-ModelDevelopment.py
Analysis18-MultiTarget-MultiModel.py
Analysis19-GenerateData1.py
Analysis19-GenerateData2.py
Analysis19-GenerateData3.py
Analysis20-TrainModel1.py
Analysis20-TrainModel2.py
Analysis20-TrainModel3.py
Analysis21-FinalRankerModel1.py
Analysis21-FinalRankerModel2.py
Analysis21-FinalRankerModel3.py
Analysis22-ListPerformance.py
Analysis1-SeeYourData.py
Analysis2-FileSchemaVariation.py
Analysis3-HarmonisationFinalGWASQuality.py
Analysis4-GWASIntrinsicStatisticalQuality.py
Analysis4-GWASIntrinsicStatisticalQuality2.py
Analysis5-ReferenceHeritabilityFeatures.py
Analysis6-TargetGenotypeCompatibility.py
Analysis7-PRSReadinessClumpingPruning.py
Analysis8-PRSPerformance.py
Module1-SearchPhenotypeandPopulation.py
Module2-Search_Poke_Normalize_Scan.py
Module3-DownloadGWAS.py
Module3.1-ListDownloadedGWASFiles.py
Module4-ExtractGWAS.py
Module5-ListPRSColumns.py
PRS1-Calculation-Plink.py
PRS1-Calculation-PRSice-2.py
PRS1-Normalization1.py
PRS1-Normalization2-hg38.py
PRS2-Calculation1.py
PRS2-Calculation2-Verification.py
PRS2-Calculation-Merge.py
PRS2-Correlation.py
Variation0.py
Variation1.py
Variation1-AllFeatures.py
Variation1-Summary.py
Variation2.py
Variation2_Allfeatures.py
Variation2-Summary.py
Variation3.py
Variation3.1-Manualcheck.py
Variation3-Allfeatures.py
Variation4.py
Variation4-Allfeatures.py
Variation5.py
Variation5-Allfeatures.py
Variation6.py
Variation6-Allfeatures.py
Variation7.py
Variation7-Allfeatures.py
Variation8.py
Variation8-Allfeatures.py
Variation9.py
Variation9-Allfeatures.py
Variation9-CheckFiles.py
Feature1_All.csv
Feature2_All.csv
Feature3_All.csv
Feature4_All.csv
Feature5_All.csv
Feature6_All.csv
Feature7_All.csv
Feature8_All.csv
Feature9_All.csv
Flowchart.pdf
Flowchart_cropped.pdf
Flowchart_page-0001.jpg
Supplementary Material 1.xlsx
```

---

# Minimal run order

For a complete run from GWAS discovery to final ranking:

```bash
# 1. Search, download, inspect, and extract GWAS files
python Module1-SearchPhenotypeandPopulation.py
python Module2-Search_Poke_Normalize_Scan.py
python Module3-DownloadGWAS.py
python Module3.1-ListDownloadedGWASFiles.py
python Module4-ExtractGWAS.py
python Module5-ListPRSColumns.py

# 2. Generate GWASRanker features
python Variation0.py <JOB_INDEX>
python Variation1.py <JOB_INDEX>
python Variation2.py <JOB_INDEX>
python Variation3.py <JOB_INDEX>
python Variation4.py <JOB_INDEX>
python Variation5.py <JOB_INDEX>
python Variation6.py <JOB_INDEX>
python Variation7.py <JOB_INDEX>
python Variation8.py <JOB_INDEX>
python Variation9.py <JOB_INDEX>

# 3. Merge per-feature outputs
python Variation1-AllFeatures.py
python Variation2_Allfeatures.py
python Variation3-Allfeatures.py
python Variation4-Allfeatures.py
python Variation5-Allfeatures.py
python Variation6-Allfeatures.py
python Variation7-Allfeatures.py
python Variation8-Allfeatures.py
python Variation9-Allfeatures.py

# 4. Calculate PRS
python PRS1-Calculation-Plink.py
python PRS1-Calculation-PRSice-2.py
python PRS2-Calculation-Merge.py

# 5. Merge feature and PRS tables
python Analysis0.0-MergeAllFeatures.py
python AllFeatures-PRS-Merge.py

# 6. Generate model-ready datasets
python Analysis19-GenerateData1.py
python Analysis19-GenerateData2.py
python Analysis19-GenerateData3.py

# 7. Train LOPO models
python Analysis20-TrainModel1.py
python Analysis20-TrainModel2.py
python Analysis20-TrainModel3.py

# 8. Fit final rankers and generate final rankings
python Analysis21-FinalRankerModel1.py
python Analysis21-FinalRankerModel2.py
python Analysis21-FinalRankerModel3.py

# 9. List performance summaries
python Analysis22-ListPerformance.py
```

For HPC/SLURM execution, use the supplied shell scripts and array jobs where available:

```bash
sbatch variation.sh
sbatch variation1.sh
bash Analysis14-GWAS_Correlation.sh
```

---

# Output files

Important outputs include:

```text
Feature1_All.csv
Feature2_All.csv
Feature3_All.csv
Feature4_All.csv
Feature5_All.csv
Feature6_All.csv
Feature7_All.csv
Feature8_All.csv
Feature9_All.csv
AllFeatures_PRS_Merged.csv
Model1/
Model2/
Model3/
Final GWAS ranking tables
Feature-importance summaries
LOPO validation performance reports
```

---

# Interpreting the feature groups

The nine feature groups are designed to represent increasingly target-aware information.

| Feature group | Main purpose |
|---|---|
| F1 | Raw file structure |
| F2 | Schema mapping and PRS minimum-column compatibility |
| F3 | Harmonisation status and genome-build handling |
| F4 | Variant-level and intrinsic statistical quality |
| F5 | FinalGWAS.csv quality and PRS-compatible variant counts |
| F6 | LDSC heritability and GWAS signal |
| F7 | External reference-panel overlap |
| F8 | Target-genotype compatibility |
| F9 | PRS-readiness, clumping, pruning, and Fisher concordance |

Model interpretation:

- **Model 1** is useful when only GWAS summary statistics and reference resources are available.
- **Model 2** is the strongest general model when target genotype data are available.
- **Model 3** is useful when the goal is to identify the single best GWAS file after PRS-readiness, clumping, pruning, and target Fisher concordance features have been calculated.

---

# Practical checklist before running

Before running the full pipeline, confirm the following:

- [ ] Phenotype folders exist.
- [ ] Fold_0 to Fold_4 exist for each phenotype.
- [ ] Each fold contains `train_data.bed/bim/fam`, `test_data.bed/bim/fam`, and covariate files.
- [ ] `train_data.QC.bed`, `train_data.QC.bim`, and `train_data.QC.fam` exist where required.
- [ ] `plink` and `plink2` are present in the working directory or available in `PATH`.
- [ ] PRSice-2 executable is available.
- [ ] LDSC and reference files are downloaded.
- [ ] HapMap3, dbSNP151, and 1000 Genomes EUR resources are downloaded.
- [ ] GWAS files have been downloaded and extracted.
- [ ] The path variables inside the scripts match the local directory structure.
- [ ] Restricted genotype and phenotype data are not committed to the public repository.

---

# Data availability

The genotype and phenotype data used in the manuscript were accessed through UK Biobank and are subject to UK Biobank access restrictions. These data cannot be redistributed through this repository.

GWAS summary statistics were obtained from the GWAS Catalog. Users should download the relevant GWAS files from the original source links using the Module 1--5 workflow.

The repository is intended to provide the code and workflow needed to reproduce the GWASRanker analysis when users have appropriate access to the required genotype, phenotype, GWAS, and reference resources.

---

# Citation

If you use this code, please cite the GWASRanker manuscript:

```text
Muneeb M, Ascher DB. GWASRanker: a quality-aware framework for selecting GWAS summary statistics for reproducible polygenic score construction.
```

---

# Contact

Muhammad Muneeb  
School of Chemistry and Molecular Biosciences, The University of Queensland  
GitHub: https://github.com/MuhammadMuneeb007  
Repository: https://github.com/MuhammadMuneeb007/GWASRanker
