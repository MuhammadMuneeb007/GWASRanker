# Contributing to GWASRanker

Thank you for helping improve GWASRanker.

## Development setup

```bash
conda env create -f environment.yml
conda activate gwasranker
```

External tools such as PLINK, PLINK 2, PRSice-2, LDSC reference files, and
target genotype data are not distributed with this repository.

## Before submitting changes

1. Do not commit GWAS, genotype, reference, phenotype, or participant-level data.
2. Do not commit credentials, cookies, cluster paths, or external tool binaries.
3. Compile-check changed Python files:

```bash
python -m compileall -q .
```

4. Describe the affected pipeline stage and the verification performed.

## Reporting problems

Please include:

- the script and command used;
- the relevant traceback or tool log;
- the expected and observed behavior;
- sanitized input headers, without participant-level data.

