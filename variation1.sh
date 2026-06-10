#!/bin/bash
#SBATCH --job-name=Feature9FailedGWAS
#SBATCH --nodes=1
#SBATCH --partition=general 
#SBATCH --time=24:00:00
#SBATCH --output=feature9_failed_gwas.%A_%a.out
#SBATCH --error=feature9_failed_gwas.%A_%a.err
#SBATCH --array=61,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,131,140,157,158,174,175,177,178,197,198,199,200,223,224,225,233,241,267,269,270,271,272,273,274,275,276,280,282
#SBATCH --mem=250G
#SBATCH --cpus-per-task=1

echo "=============================================="
echo "Running Variation9 for failed GWAS index: $SLURM_ARRAY_TASK_ID"
echo "=============================================="

# python Variation1.py "$SLURM_ARRAY_TASK_ID" --force
# python Variation2.py "$SLURM_ARRAY_TASK_ID" --force
# python Variation3.py "$SLURM_ARRAY_TASK_ID" --force
# python Variation3.1-Manualcheck.py "$SLURM_ARRAY_TASK_ID" --full-file
# python Variation4.py "$SLURM_ARRAY_TASK_ID" --force
# python Variation5.py "$SLURM_ARRAY_TASK_ID" --force
# python Variation6.py "$SLURM_ARRAY_TASK_ID" --force
# python Variation7.py "$SLURM_ARRAY_TASK_ID" --force
#
# python Variation8.py "$SLURM_ARRAY_TASK_ID"

python Variation9.py "$SLURM_ARRAY_TASK_ID"

# python PRS1-Normalization1.py "$SLURM_ARRAY_TASK_ID" --force
# python PRS1-Normalization2-hg38.py "$SLURM_ARRAY_TASK_ID" --force
#
# python PRS1-Calculation-Plink.py "$SLURM_ARRAY_TASK_ID"
# python PRS1-Calculation-PRSice-2.py "$SLURM_ARRAY_TASK_ID"