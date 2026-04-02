#!/bin/bash
#SBATCH --job-name=NiT-imagenet9to4
#SBATCH --partition=rocky
#SBATCH --account=proj_1793
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --time=4-0:0
#SBATCH --output=/home/dgkagramanyan/Native-Resolution-Image-Synthesis-v2/logs/slurm_%j.out
#SBATCH --error=/home/dgkagramanyan/Native-Resolution-Image-Synthesis-v2/logs/slurm_%j.out
#SBATCH --constraint="type_e|type_h"

# ── Environment ────────────────────────────────────────────────────────────────
module purge
module load Python/Anaconda
module load CUDA/13.1

conda activate nit_rocky

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

nvidia-smi

which nvcc
nvcc --version

which gcc
gcc --version

which g++
g++ --version

conda list | grep -E "torch|cuda|cudnn|ninja"

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO


# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR=/home/dgkagramanyan/Native-Resolution-Image-Synthesis-v2
LOGDIR=$PROJECT_DIR/logs

mkdir -p "$LOGDIR"

# ── Info ───────────────────────────────────────────────────────────────────────
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPUs       : $CUDA_VISIBLE_DEVICES"
echo "Start time : $(date)"
echo "────────────────────────────────────"

# ── Train ──────────────────────────────────────────────────────────────────────
cd "$PROJECT_DIR"

torchrun \
    --nnodes 1 \
    --nproc_per_node 2 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:0 \
    projects/train/packed_trainer_c2i.py \
    --config datasets/imagenet_9to4_nit/train_config.yaml \
    --project_dir workdir/imagenet_9to4 \
    --seed 0

EXIT_CODE=$?

# ── Done ───────────────────────────────────────────────────────────────────────
echo "────────────────────────────────────"
echo "End time   : $(date)"
echo "Exit code  : $EXIT_CODE"

exit $EXIT_CODE
