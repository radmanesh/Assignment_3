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
#SBATCH --chdir=/home/cs529329/assignment_3
#SBATCH --output=results/exp2_%j_stdout.txt
#SBATCH --error=results/exp2_%j_stderr.txt
##SBATCH --array=0-4    # the double ## means that this line is ignored

# Load required modules
module load Python/3.10.4-GCCcore-11.3.0

# Activate your conda environment if you're using one
# source activate cs5293-3

# Change to the directory containing your script
cd $SLURM_SUBMIT_DIR

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo ""

# Run the neural n-gram language model with different hyperparameters
# Experiment 1: context_size=2, embedding_dim=10 (baseline from notebook)
echo "Experiment 1: Baseline (context=2, embed=10, hidden=128)"
python nllm.py ../data/training.txt ../data/test.txt ../data/seeds.txt \
    --context_size 2 \
    --embedding_dim 10 \
    --hidden_size 128 \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.01 \
    --seed 1

# Rename output files for this experiment
mv ngram-prob.trace ngram-prob_exp1.trace
mv ngram-gen.trace ngram-gen_exp1.trace
mv ngram_model.pth ngram_model_exp1.pth

# Experiment 2: Larger embeddings
echo ""
echo "Experiment 2: Larger embeddings (context=2, embed=50, hidden=128)"
python nllm.py ../data/training.txt ../data/test.txt ../data/seeds.txt \
    --context_size 2 \
    --embedding_dim 50 \
    --hidden_size 128 \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.01 \
    --seed 1

mv ngram-prob.trace ngram-prob_exp2.trace
mv ngram-gen.trace ngram-gen_exp2.trace
mv ngram_model.pth ngram_model_exp2.pth

# Experiment 3: Larger context
echo ""
echo "Experiment 3: Larger context (context=5, embed=50, hidden=128)"
python nllm.py ../data/training.txt ../data/test.txt ../data/seeds.txt \
    --context_size 5 \
    --embedding_dim 50 \
    --hidden_size 128 \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.01 \
    --seed 1

mv ngram-prob.trace ngram-prob_exp3.trace
mv ngram-gen.trace ngram-gen_exp3.trace
mv ngram_model.pth ngram_model_exp3.pth

# Experiment 4: Larger hidden layer
echo ""
echo "Experiment 4: Larger hidden layer (context=5, embed=50, hidden=256)"
python nllm.py ../data/training.txt ../data/test.txt ../data/seeds.txt \
    --context_size 5 \
    --embedding_dim 50 \
    --hidden_size 256 \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.01 \
    --seed 1

mv ngram-prob.trace ngram-prob_exp4.trace
mv ngram-gen.trace ngram-gen_exp4.trace
mv ngram_model.pth ngram_model_exp4.pth

# Experiment 5: More training epochs with lower learning rate
echo ""
echo "Experiment 5: More epochs, lower LR (context=5, embed=50, hidden=256, epochs=20, lr=0.005)"
python nllm.py ../data/training.txt ../data/test.txt ../data/seeds.txt \
    --context_size 5 \
    --embedding_dim 50 \
    --hidden_size 256 \
    --epochs 20 \
    --batch_size 32 \
    --lr 0.005 \
    --seed 1

mv ngram-prob.trace ngram-prob_exp5.trace
mv ngram-gen.trace ngram-gen_exp5.trace
mv ngram_model.pth ngram_model_exp5.pth

echo ""
echo "End time: $(date)"
echo "All experiments completed!"
