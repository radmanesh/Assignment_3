#!/bin/bash
### Some common partitions
##SBATCH --partition=gpu_a100
##SBATCH --partition=sooner_gpu_test_ada
##SBATCH --partition=sooner_gpu_test
##SBATCH --partition=gpu
#SBATCH --partition=debug_gpu # This partition is currently selected.
#
#SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=64
#SBATCH --cpus-per-task=20
##SBATCH --mem=32G
#SBATCH --mem=16G
##SBATCH --time=12:00:00
##SBATCH --time=00:20:00
#SBATCH --time=00:12:00
#SBATCH --job-name=nllm_training
#SBATCH --output=nllm_%j.out
#SBATCH --error=nllm_%j.err
#SBATCH --mail-user=radmanesh@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs529329/Assignment_3/src
#SBATCH --output=results/exp2_%j_stdout.txt
#SBATCH --error=results/exp2_%j_stderr.txt
##SBATCH --array=0-4    # the double ## means that this line is ignored

# Load required modules
export MAMBA_ROOT_PREFIX="/scratch/cs529329/micromamba"

MM="/scratch/cs529329/micromamba/bin/micromamba"

eval "$("$MM" shell hook -s bash)"

micromamba activate cs5293-3

# Activate your conda environment if you're using one
# source activate cs5293-3

export WANDB_API_KEY=""

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo ""

echo "Experiment 1: Model: DAN, Embedding: Random, Embedding Dim: 50, Hidden Dim: 128"
python sentiment_classifier.py --model dan --embedding_dim 50 --embedding random --hidden_size 128

echo "Experiment 2: Model: DAN, Embedding: GloVe, Embedding Dim: 50, Hidden Dim: 128"
python sentiment_classifier.py --model dan --embedding_dim 50 --embedding glove --hidden_size 128

echo "Experiment 3: Model: LSTM, Embedding: Random, Embedding Dim: 50, Hidden Dim: 128"
python sentiment_classifier.py --model lstm --embedding_dim 50 --embedding random --hidden_size 128

echo "Experiment 4: Model: LSTM, Embedding: GloVe, Embedding Dim: 50, Hidden Dim: 128"
python sentiment_classifier.py --model lstm --embedding_dim 50 --embedding glove --hidden_size 128

echo "Experiment 5: Model: LSTM, Embedding: GloVe, Embedding Dim: 300, Hidden Dim: 512"
python sentiment_classifier.py --model lstm --embedding_dim 300 --embedding glove --glove_path ../data/glove.840B.300d.txt --hidden_size 512

echo "Experiment 6: Model: LSTM, Embedding: Gensim, Embedding Dim: 300, Hidden Dim: 512"
python sentiment_classifier.py --model lstm --embedding_dim 300 --embedding gensim --hidden_size 512