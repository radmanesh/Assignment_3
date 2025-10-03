# Neural N-gram Language Model (NLLM)

## Setup Environment

```bash
# Create environment from environment.yml
mamba env create -f environment.yml

# Activate environment
mamba activate cs5293-3
```

## Quick Start

### Run Locally
```bash
cd src
python nllm.py ../data/training.txt ../data/test.txt ../data/seeds.txt
```

### Run on OSCER
```bash
cd src
sbatch run_nllm.sbatch
```

## Usage

```bash
python nllm.py <training_file> <test_file> <seeds_file> [options]
```

**Common Options:**
- `--context_size 5` - Context window size (default: 2)
- `--embedding_dim 50` - Embedding dimension (default: 10)
- `--hidden_size 256` - Hidden layer size (default: 128)
- `--epochs 10` - Training epochs (default: 10)
- `--lr 0.01` - Learning rate (default: 0.01)

**Example:**
```bash
python nllm.py ../data/training.txt ../data/test.txt ../data/seeds.txt \
    --context_size 5 --embedding_dim 50 --epochs 10
```

## Output Files

- **`ngram-prob.trace`** - Log-probabilities for test sentences
- **`ngram-gen.trace`** - Generated text from seed words
- **`ngram_model.pth`** - Saved model weights
- **`vocab.pkl`** - Vocabulary
- **`word_to_ix.pkl`** - Word-to-index mapping