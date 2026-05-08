#!/bin/bash
#SBATCH -J scr_data
#SBATCH -p ghanem
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=62
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -o scr_data_%j.out
#SBATCH -e scr_data_%j.err

set -euo pipefail

echo "========================================"
echo "SLURM_JOB_ID        = ${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST  = ${SLURM_JOB_NODELIST}"
echo "SLURM_CPUS_ON_NODE  = ${SLURM_CPUS_ON_NODE:-}"
echo "SLURM_CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK}"
echo "========================================"

PROJECT_DIR="/home/ssariedd/AME508/hpc_scr_data_generation"
ENV_PATH="/home1/ssariedd/software/python_envs/narx_cpu"

cd "${PROJECT_DIR}"

echo "PWD         = $(pwd)"
echo "HOSTNAME    = $(hostname)"
echo "NPROC       = $(nproc)"
echo "PROJECT_DIR = ${PROJECT_DIR}"
echo "ENV_PATH    = ${ENV_PATH}"

module purge
module load conda/25.11.0

echo "PYTHON      = ${ENV_PATH}/bin/python"

"${ENV_PATH}/bin/python" - <<'PYEOF'
import sys
import numpy as np

print("python:", sys.version)
print("numpy:", np.__version__)

try:
    import scipy
    print("scipy:", scipy.__version__)
except Exception as e:
    print("scipy import failed:", repr(e))

try:
    import tqdm
    print("tqdm:", tqdm.__version__)
except Exception as e:
    print("tqdm import failed:", repr(e))
PYEOF

# 62 independent Python workers, each single-threaded
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "OMP_NUM_THREADS      = ${OMP_NUM_THREADS}"
echo "OPENBLAS_NUM_THREADS = ${OPENBLAS_NUM_THREADS}"
echo "MKL_NUM_THREADS      = ${MKL_NUM_THREADS}"
echo "NUMEXPR_NUM_THREADS  = ${NUMEXPR_NUM_THREADS}"

echo "Starting SCR model data generation at: $(date)"
"${ENV_PATH}/bin/python" generate_scr_model_data.py
echo "Finished SCR model data generation at: $(date)"
