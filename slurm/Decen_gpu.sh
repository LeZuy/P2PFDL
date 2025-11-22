#!/bin/bash

#SBATCH --job-name=P2PFDL
#
#SBATCH -N1
#SBATCH -n1
#SBATCH --account=pi_duc.tran
#SBATCH --time=00-20:00:00
#SBATCH --mem=80gb
#SBATCH --error=./slurm/P2PFDL.err
#SBATCH --output=./slurm/P2PFDL.out
#SBATCH --partition=DGXA100
#SBATCH --export=HOME

#
# specify gpu type and quantity.  Uncomment the option you
# want
##SBATCH --gres=gpu:h200:1
##SBATCH --gres=gpu:3g.71gb:1
##SBATCH --gres=gpu:1g.18gb:1
#SBATCH --gres=gpu:A100:4

# Optional - enforce GPU/CPU affinity
##SBATCH --gres-flags=enforce-binding

# Optional email alerts
##SBATCH --mail-type=ALL
##SBATCH --mail-user=duy.le004@umb.edu

# source the local environment if --export=HOME 
. /etc/profile

echo "Job ID:               $SLURM_JOB_ID"
echo "Job Name:             $SLURM_JOB_NAME"
echo "User:                 $SLURM_JOB_USER"
echo "Partition:            $SLURM_JOB_PARTITION"
echo "Node List:            $SLURM_JOB_NODELIST"
echo "Number of Nodes:      $SLURM_JOB_NUM_NODES"
echo "Tasks (ntasks):       $SLURM_NTASKS"
echo "CPUs per Task:        $SLURM_CPUS_PER_TASK"
echo "CPUs on Node:         $SLURM_CPUS_ON_NODE"
echo "CPUs allocated:       $SLURM_JOB_CPUS_PER_NODE"
echo "GPUs allocated:      $SLURM_JOB_GPUS"
echo "GPU devices:         $CUDA_VISIBLE_DEVICES"
echo "Memory allocated:     $SLURM_MEM_PER_NODE"
echo "Submit Directory:     $SLURM_SUBMIT_DIR"
echo "Hostname:             $(hostname)"
echo "Time:           $(date)"

hostname
sleep 0

source /share/apps/linux-ubuntu20.04-zen2/anaconda3-2021.05/etc/profile.d/conda.sh
conda activate p2pdfl
echo "Start nvidia-smi logger"  # để thấy trong stdout
LOGFILE="${SLURM_SUBMIT_DIR:-$PWD}/nvidia-smi.log"
( while true; do date; nvidia-smi; sleep 60; done ) >>"$LOGFILE" 2>&1 &
# export PYTHONPATH="${SLURM_SUBMIT_DIR:-$(pwd)}/src:${PYTHONPATH}"
python main.py --config configs/experiment.yaml

# Diagnostic/Logging Information
echo "Finish Run"
echo "end time is `date`"
