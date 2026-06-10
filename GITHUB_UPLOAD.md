# Publishing GWASRanker on GitHub

The GitHub repository already exists at:

```text
https://github.com/MuhammadMuneeb007/GWASRanker
```

## Recommended Repository Description

```text
A reproducible pipeline for harmonising, profiling, and ranking GWAS summary statistics for polygenic risk score analysis.
```

## Recommended GitHub Topics

```text
gwas
gwas-summary-statistics
polygenic-risk-score
prs
bioinformatics
genomics
machine-learning
plink
prsice-2
python
```

## Install and Authenticate GitHub CLI

From Windows PowerShell:

```powershell
winget install --id GitHub.cli
gh auth login
```

Choose:

```text
GitHub.com
HTTPS
Login with a web browser
```

Close and reopen PowerShell if `gh` is not found immediately after installation.

## Verify and Upload

Run these commands from the repository folder:

```powershell
git status
git remote -v
git log --oneline --decorate -5
git push -u origin main
```

## Set Description and Topics

```powershell
gh repo edit MuhammadMuneeb007/GWASRanker `
  --description "A reproducible pipeline for harmonising, profiling, and ranking GWAS summary statistics for polygenic risk score analysis." `
  --add-topic gwas `
  --add-topic gwas-summary-statistics `
  --add-topic polygenic-risk-score `
  --add-topic prs `
  --add-topic bioinformatics `
  --add-topic genomics `
  --add-topic machine-learning `
  --add-topic plink `
  --add-topic prsice-2 `
  --add-topic python
```

## Create the First Release

```powershell
git tag -a v0.1.0 -m "GWASRanker v0.1.0"
git push origin v0.1.0

gh release create v0.1.0 `
  --repo MuhammadMuneeb007/GWASRanker `
  --title "GWASRanker v0.1.0" `
  --notes "Initial public release of the GWASRanker GWAS harmonisation, feature-generation, PRS evaluation, and ranking pipeline."
```

## Optional Visibility Commands

Keep the repository private while reviewing it:

```powershell
gh repo edit MuhammadMuneeb007/GWASRanker `
  --visibility private `
  --accept-visibility-change-consequences
```

Make it public after the final review:

```powershell
gh repo edit MuhammadMuneeb007/GWASRanker `
  --visibility public `
  --accept-visibility-change-consequences
```

## Future Updates

```powershell
git status
git add -A
git commit -m "Describe the update"
git push
```

Never use `git add -f` for ignored GWAS, genotype, reference, phenotype, or
generated result files.

