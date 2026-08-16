#!/bin/bash
#SBATCH --job-name=p01-extract-__SHORT__
#SBATCH --output=/scratch/USER/results/01-acts/extract___SHORT__.log
#SBATCH --error=/scratch/USER/results/01-acts/extract___SHORT__.err
#SBATCH --partition=sporc
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
#SBATCH --mem=120G

# Env-fix stack (verbatim from notes/cluster-ops/README.md)
export PYTHONNOUSERSITE=1
export HF_HOME=/scratch/USER/hf
export TOKENIZERS_PARALLELISM=false
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export CUDA_HOME=/home/USER/miniforge3/envs/jrp/lib/python3.11/site-packages/nvidia/cu13
export PATH=/home/USER/miniforge3/envs/jrp/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/home/USER/miniforge3/envs/jrp/lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /scratch/USER/results/01-acts

cd /home/USER/jollen-rapid-project/projects/01-sandbagging-detection

echo "Extracting __MODEL__ on __EVALS__ at $(date)"

/home/USER/miniforge3/envs/jrp/bin/python extract_organism_acts.py \
    --model __MODEL__ \
    --evals "__EVALS__" \
    --out-dir /scratch/USER/results/01-acts

echo "Completed at $(date)"
