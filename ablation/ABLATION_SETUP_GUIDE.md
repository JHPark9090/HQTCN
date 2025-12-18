# HQTCN Ablation Study Setup Guide

This guide helps you configure and run the HQTCN ablation study on NERSC Perlmutter.

## Overview

- **Colleague #1**: Runs dilation ablation (9 jobs, ~90 GPU hours)
- **Colleague #2**: Runs depth ablation (6 jobs, ~61.5 GPU hours)

---

## Files You Received

| File | Description |
|------|-------------|
| `run_ablation_dilation.sh` | SLURM batch script for dilation ablation |
| `run_ablation_depth.sh` | SLURM batch script for depth ablation |
| `HQTCN_ablation_study.py` | Main Python script for ablation experiments |
| `Load_PhysioNet_EEG.py` | Data loader for PhysioNet EEG dataset |

---

## Configuration Checklist

### 1. SLURM Account

**Current setting in batch scripts:**
```bash
#SBATCH --account=m4727_g
```

**Action:** Change to your SLURM account if different.

```bash
# Edit the batch script
sed -i 's/m4727_g/YOUR_ACCOUNT/g' run_ablation_dilation.sh
# or
sed -i 's/m4727_g/YOUR_ACCOUNT/g' run_ablation_depth.sh
```

---

### 2. Working Directory

**Current setting:**
```bash
#SBATCH --chdir=/pscratch/sd/j/junghoon/HQTCN_Project
```

**Action:** Update to your project directory.

Also update in the script body (line ~60):
```bash
cd /pscratch/sd/j/junghoon/HQTCN_Project/scripts
```

---

### 3. Output/Error Log Directory

**Current setting:**
```bash
#SBATCH --output=/pscratch/sd/j/junghoon/HQTCN_Project/ablation_results/hqtcn_dilation_%A_%a.out
#SBATCH --error=/pscratch/sd/j/junghoon/HQTCN_Project/ablation_results/hqtcn_dilation_%A_%a.err
```

**Action:** Update paths to your directory. Create the output directory before running:
```bash
mkdir -p /YOUR/PATH/ablation_results
```

---

### 4. Conda Environment

**Current setting:**
```bash
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg
```

**Action:** Either:
- **(Option A)** Use the existing environment if you have access
- **(Option B)** Create your own environment with required packages (see below)

#### Required Packages

```bash
# Create new environment
conda create -n qml_eeg python=3.11 -y
conda activate qml_eeg

# Install packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pennylane==0.41.1
pip install mne==1.9.0
pip install scikit-learn
pip install pandas
pip install numpy
pip install tqdm
pip install matplotlib
```

#### Verify Installation

```python
import torch
import pennylane as qml
import mne
import sklearn

print(f"PyTorch: {torch.__version__}")
print(f"PennyLane: {qml.__version__}")
print(f"MNE: {mne.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

### 5. Results Output Directory

**Current setting in Python script call:**
```bash
OUTPUT_DIR="/pscratch/sd/j/junghoon/HQTCN_Project/ablation_results/dilation"
# or
OUTPUT_DIR="/pscratch/sd/j/junghoon/HQTCN_Project/ablation_results/depth"
```

**Action:** Update to your preferred output directory.

---

### 6. EEG Dataset

The PhysioNet EEG dataset is **automatically downloaded** by `Load_PhysioNet_EEG.py` using MNE-Python.

**Default download location:** `~/mne_data/`

**If you want a different location**, set the environment variable before running:
```bash
export MNE_DATA=/your/preferred/path
```

Or add to the batch script:
```bash
export MNE_DATA=/your/preferred/path
```

**Note:** First run will download ~2GB of data. Subsequent runs use cached data.

---

## Summary of Lines to Edit

### In `run_ablation_dilation.sh` or `run_ablation_depth.sh`:

| Line | Parameter | Current Value | Change To |
|------|-----------|---------------|-----------|
| 2 | `--account` | `m4727_g` | Your account |
| 12 | `--chdir` | `/pscratch/sd/j/junghoon/HQTCN_Project` | Your project dir |
| 13 | `--output` | `/pscratch/sd/j/junghoon/HQTCN_Project/ablation_results/...` | Your output dir |
| 14 | `--error` | `/pscratch/sd/j/junghoon/HQTCN_Project/ablation_results/...` | Your error dir |
| 57 | `conda activate` | `/pscratch/sd/j/junghoon/conda-envs/qml_eeg` | Your conda env |
| 60 | `cd` | `/pscratch/sd/j/junghoon/HQTCN_Project/scripts` | Your scripts dir |
| 77-78 | `OUTPUT_DIR` | `/pscratch/sd/j/junghoon/HQTCN_Project/ablation_results/...` | Your results dir |

---

## Running the Experiments

### Before First Run

1. Complete all configuration changes above
2. Create output directories:
   ```bash
   mkdir -p /YOUR/PATH/ablation_results/dilation
   mkdir -p /YOUR/PATH/ablation_results/depth
   ```
3. Ensure Python scripts are in your scripts directory:
   ```bash
   ls /YOUR/PATH/scripts/
   # Should show: HQTCN_ablation_study.py, Load_PhysioNet_EEG.py
   ```

### Submit Jobs

**Colleague #1 (Dilation):**
```bash
cd /YOUR/PATH/batch_jobs
sbatch run_ablation_dilation.sh
```

**Colleague #2 (Depth):**
```bash
cd /YOUR/PATH/batch_jobs
sbatch run_ablation_depth.sh
```

### Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Check output logs
tail -f /YOUR/PATH/ablation_results/hqtcn_dilation_*.out
```

---

## Expected Output

Each job produces:
- CSV file with metrics: `ablation_results/{dilation,depth}/ablation_YYYYMMDD_HHMMSS/*.csv`
- SLURM log files: `hqtcn_{dilation,depth}_JOBID_TASKID.out/.err`

### Results Format

The CSV contains columns:
- `ablation_type`: "dilation" or "depth"
- `dilation`: dilation factor (1, 2, 3, or 4)
- `circuit_depth`: circuit depth (1, 2, or 3)
- `seed`: random seed (2025, 2026, or 2027)
- `test_auroc`: test set AUROC score
- `test_loss`: test set loss

---

## Troubleshooting

### "conda: command not found"
Add these lines before `conda activate`:
```bash
module load python
module load conda
```

### "scipy has no attribute 'constants'"
The Python script already includes the fix. If you see this error, ensure you're using the provided `HQTCN_ablation_study.py`.

### CUDA out of memory
Add to batch script:
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
```

### Job timeout
The default time limit is 24 hours. Depth=3 jobs may take ~15 hours each. If needed, increase:
```bash
#SBATCH --time=30:00:00
```

---

## Contact

After jobs complete, please send the following files back:
- All CSV files in `ablation_results/{dilation,depth}/`
- SLURM output logs (`.out` files) for verification

Thank you for your help!

