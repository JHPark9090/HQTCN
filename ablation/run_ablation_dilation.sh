#!/bin/bash
#SBATCH --account=m4747_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --array=0-8
#SBATCH --job-name=hqtcn_dil
#SBATCH --chdir=/pscratch/sd/j/junghoon/HQTCN_Project
#SBATCH --output=/pscratch/sd/j/junghoon/HQTCN_Project/ablation_results/hqtcn_dilation_%A_%a.out
#SBATCH --error=/pscratch/sd/j/junghoon/HQTCN_Project/ablation_results/hqtcn_dilation_%A_%a.err

# =============================================================================
# HQTCN Ablation Study - DILATION ONLY
# =============================================================================
# Total: 9 jobs (d=1,2,4 × 3 seeds) - skip d=3 (baseline)
#
# Job mapping:
#   0-2: d=1 × seeds 2025,2026,2027
#   3-5: d=2 × seeds 2025,2026,2027
#   6-8: d=4 × seeds 2025,2026,2027
#
# Baseline (d=3, L=2) already exists from original experiments.
# Estimated time per job: ~10 hours (30 epochs × ~20 min/epoch)
# =============================================================================

set +x

# Configuration arrays
DILATIONS=(1 2 4)      # Skip d=3 (baseline)
SEEDS=(2025 2026 2027)

# Determine configuration based on array task ID
TASK_ID=$SLURM_ARRAY_TASK_ID
DIL_IDX=$((TASK_ID / 3))
SEED_IDX=$((TASK_ID % 3))
DILATION=${DILATIONS[$DIL_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=============================================="
echo "HQTCN Ablation Study - DILATION"
echo "=============================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $TASK_ID"
echo "Dilation: $DILATION"
echo "Depth: 2 (fixed)"
echo "Seed: $SEED"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=============================================="

# Activate Python environment (NERSC Perlmutter)
module load python
module load conda
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

# Set working directory
cd /pscratch/sd/j/junghoon/HQTCN_Project/scripts

# Print environment info
echo ""
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=============================================="
echo "Running DILATION Ablation: d=$DILATION, seed=$SEED"
echo "=============================================="

# Output directory
OUTPUT_DIR="/pscratch/sd/j/junghoon/HQTCN_Project/ablation_results/dilation"
mkdir -p $OUTPUT_DIR

# Run the ablation
python HQTCN_ablation_study.py \
    --freq 80 \
    --n-sample 50 \
    --batch-size 32 \
    --seeds $SEED \
    --num-epochs 30 \
    --ablation dilation \
    --single-dilation $DILATION \
    --output-dir $OUTPUT_DIR

echo ""
echo "=============================================="
echo "DILATION Ablation Complete: d=$DILATION, seed=$SEED"
echo "End Time: $(date)"
echo "=============================================="
