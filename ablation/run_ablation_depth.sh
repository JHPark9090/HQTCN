#!/bin/bash
#SBATCH --account=m4727_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --array=0-5
#SBATCH --job-name=hqtcn_dep
#SBATCH --chdir=/pscratch/sd/j/junghoon/HQTCN_Project
#SBATCH --output=/pscratch/sd/j/junghoon/HQTCN_Project/ablation_results/hqtcn_depth_%A_%a.out
#SBATCH --error=/pscratch/sd/j/junghoon/HQTCN_Project/ablation_results/hqtcn_depth_%A_%a.err

# =============================================================================
# HQTCN Ablation Study - DEPTH ONLY
# =============================================================================
# Total: 6 jobs (L=1,3 × 3 seeds) - skip L=2 (baseline)
#
# Job mapping:
#   0-2: L=1 × seeds 2025,2026,2027  (~5.5 hrs each)
#   3-5: L=3 × seeds 2025,2026,2027  (~15 hrs each)
#
# Baseline (d=3, L=2) already exists from original experiments.
# Estimated total GPU hours: ~61.5 hrs
# =============================================================================

set +x

# Configuration arrays
DEPTHS=(1 3)           # Skip L=2 (baseline)
SEEDS=(2025 2026 2027)

# Determine configuration based on array task ID
TASK_ID=$SLURM_ARRAY_TASK_ID
DEPTH_IDX=$((TASK_ID / 3))
SEED_IDX=$((TASK_ID % 3))
DEPTH=${DEPTHS[$DEPTH_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=============================================="
echo "HQTCN Ablation Study - DEPTH"
echo "=============================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $TASK_ID"
echo "Dilation: 3 (fixed)"
echo "Depth: $DEPTH"
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
echo "Running DEPTH Ablation: L=$DEPTH, seed=$SEED"
echo "=============================================="

# Output directory
OUTPUT_DIR="/pscratch/sd/j/junghoon/HQTCN_Project/ablation_results/depth"
mkdir -p $OUTPUT_DIR

# Run the ablation
python HQTCN_ablation_study.py \
    --freq 80 \
    --n-sample 50 \
    --batch-size 32 \
    --seeds $SEED \
    --num-epochs 30 \
    --ablation depth \
    --single-depth $DEPTH \
    --output-dir $OUTPUT_DIR

echo ""
echo "=============================================="
echo "DEPTH Ablation Complete: L=$DEPTH, seed=$SEED"
echo "End Time: $(date)"
echo "=============================================="
