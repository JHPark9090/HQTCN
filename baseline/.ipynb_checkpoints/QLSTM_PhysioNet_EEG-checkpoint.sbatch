#!/bin/bash
#SBATCH --job-name=QLSTM_PhysioNet    # A descriptive name for your job
#SBATCH -A m4138
#SBATCH -q shared
#SBATCH -C cpu
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=0
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=pakmasha99@gmail.com
#SBATCH --output=/pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main/slurm/output_log/slurm_%j.out
#SBATCH --error=/pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main/slurm/error_log/slurm_%j.err

# --- Environment and Execution ---
cd /pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main/mbbn
source bin/activate
cd /pscratch/sd/p/pakmasha/QTCN/baseline
python -u QLSTM_v0_PhysioNet_EEG.py &> error_log_physionet.txt