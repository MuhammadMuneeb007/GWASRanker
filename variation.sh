#!/bin/bash
#SBATCH --job-name=Feature1GWAS
#SBATCH --nodes=1
#SBATCH --partition=general 
#SBATCH --time=24:00:00
#SBATCH --output=feature1_gwas.%A_%a.out
#SBATCH --error=feature1_gwas.%A_%a.err
#SBATCH --array=1-285
#SBATCH --mem=250G
#SBATCH --cpus-per-task=1

echo "=============================================="
echo "Running Variation1 for GWAS index: $SLURM_ARRAY_TASK_ID"
echo "=============================================="

#python Variation1.py "$SLURM_ARRAY_TASK_ID" --force
#python Variation2.py "$SLURM_ARRAY_TASK_ID" --force
#python Variation3.py "$SLURM_ARRAY_TASK_ID" --force
#python Variation3.1-Manualcheck.py "$SLURM_ARRAY_TASK_ID" --full-file
#python Variation4.py "$SLURM_ARRAY_TASK_ID" --force
#python Variation5.py "$SLURM_ARRAY_TASK_ID" --force
#python Variation6.py "$SLURM_ARRAY_TASK_ID" --force
#python Variation7.py "$SLURM_ARRAY_TASK_ID" --force
#
#python Variation8.py "$SLURM_ARRAY_TASK_ID" 
python Variation9.py "$SLURM_ARRAY_TASK_ID"
 
#python PRS1-Normalization1.py "$SLURM_ARRAY_TASK_ID" --force
#python PRS1-Normalization2-hg38.py "$SLURM_ARRAY_TASK_ID"  --force
# 
#python PRS1-Calculation-Plink.py "$SLURM_ARRAY_TASK_ID" 
#python PRS1-Calculation-PRSice-2.py "$SLURM_ARRAY_TASK_ID" 



