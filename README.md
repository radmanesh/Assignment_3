# NLP Assignment 3: Neural Language Models & Sentiment Classification

This repository contains implementations of:
1. **Neural N-gram Language Model (NLLM)** - Character-level language model
2. **Sentiment Classifier** - SST-5 sentiment analysis with DAN/LSTM models

## Setup Environment

```bash
# Create environment from environment.yml
mamba env create -f environment.yml

# Activate environment
mamba activate cs5293-3

# Install additional dependencies
pip install gensim datasets spacy scikit-learn wandb matplotlib seaborn

# Download spaCy model for sentiment classifier
python -m spacy download en_core_web_sm
```

## Quick Start

### Part 1: Neural N-gram Language Model (NLLM)

#### Run Locally
```bash
cd src
python nllm.py ../data/training.txt ../data/test.txt ../data/seeds.txt
```

#### Run on OSCER
```bash
cd src
sbatch run_nllm.sbatch
```

### Part 2: Sentiment Classification

#### Run Locally
```bash
cd src

# Train DAN with random embeddings
python sentiment_classifier.py --model dan --embedding random --epochs 10

# Train LSTM with GloVe embeddings
python sentiment_classifier.py --model lstm --embedding glove --glove_path ../data/glove.6B.300d-subset.txt --embedding_dim 300

# Train DAN with Gensim Word2Vec (Skip-gram)
python sentiment_classifier.py --model dan --embedding gensim --embedding_dim 100 --w2v_sg 1 --w2v_epochs 10

# Train LSTM with Gensim Word2Vec (CBOW)
python sentiment_classifier.py --model lstm --embedding gensim --embedding_dim 300 --w2v_sg 0 --w2v_epochs 20
```

#### Run on OSCER
```bash
cd src
sbatch run_sentiment.sbatch
```

## Usage

### Neural N-gram Language Model (NLLM)

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

### Sentiment Classifier

```bash
python sentiment_classifier.py [options]
```

**Model Options:**
- `--model {dan,lstm}` - Model architecture (default: dan)
- `--embedding {random,glove,gensim}` - Embedding type (default: random)
- `--embedding_dim 300` - Embedding dimension (default: 50)
- `--hidden_size 256` - Hidden layer size (default: 128)
- `--lstm_layers 2` - Number of LSTM layers (default: 1)

**Training Options:**
- `--epochs 20` - Number of training epochs (default: 10)
- `--batch_size 64` - Mini-batch size (default: 32)
- `--lr 0.001` - Learning rate (default: 0.001)
- `--dropout 0.3` - Dropout rate for LSTM (default: 0.3)
- `--max_len 200` - Maximum sequence length (default: 200)
- `--seed 42` - Random seed (default: 42)

**GloVe Embedding Options:**
- `--glove_path PATH` - Path to GloVe embeddings file (default: ../data/glove.6B.50d-subset.txt)

**Gensim Word2Vec Options:**
- `--w2v_window 5` - Word2Vec context window size (default: 5)
- `--w2v_min_count 2` - Word2Vec minimum word frequency (default: 2)
- `--w2v_sg {0,1}` - Word2Vec algorithm: 1=skip-gram, 0=CBOW (default: 1)
- `--w2v_epochs 10` - Word2Vec training epochs (default: 10)

**Examples:**

```bash
# DAN with random embeddings (baseline)
python sentiment_classifier.py --model dan --embedding random --hidden_size 256 --epochs 20

# LSTM with GloVe embeddings (300d)
python sentiment_classifier.py --model lstm --embedding glove \
    --glove_path ../data/glove.6B.300d-subset.txt \
    --embedding_dim 300 --hidden_size 256 --lstm_layers 2 --epochs 20

# DAN with Word2Vec Skip-gram (trained on SST-5)
python sentiment_classifier.py --model dan --embedding gensim \
    --embedding_dim 100 --w2v_window 5 --w2v_sg 1 --w2v_epochs 15

# LSTM with Word2Vec CBOW (trained on SST-5)
python sentiment_classifier.py --model lstm --embedding gensim \
    --embedding_dim 300 --w2v_window 7 --w2v_sg 0 --w2v_epochs 20 \
    --hidden_size 256 --lstm_layers 2 --dropout 0.5
```

## Output Files

### NLLM Outputs

- **`ngram-prob.trace`** - Log-probabilities for test sentences
- **`ngram-gen.trace`** - Generated text from seed words
- **`ngram_model.pth`** - Saved model weights
- **`vocab.pkl`** - Vocabulary
- **`word_to_ix.pkl`** - Word-to-index mapping

### Sentiment Classifier Outputs

- **Confusion Matrix PNG** - `confusion_matrix_h_TIMESTAMP.png` - Heatmap visualization
- **WandB Logs** - Training/evaluation metrics logged to Weights & Biases
  - Training Loss & Accuracy
  - Validation Loss & Accuracy
  - Test Loss & Accuracy
  - Confusion Matrix Heatmap
- **Console Output** - Classification report and confusion matrix

## Data Files

### NLLM Data
- `data/training.txt` - Training corpus for language model
- `data/test.txt` - Test sentences for perplexity evaluation
- `data/seeds.txt` - Seed words for text generation

### Sentiment Classifier Data
- `data/glove.6B.50d-subset.txt` - GloVe 50d embeddings (subset)
- `data/glove.6B.300d-subset.txt` - GloVe 300d embeddings (subset)
- SST-5 dataset automatically downloaded from HuggingFace (`SetFit/sst5`)

## Project Structure

```
Assignment_3/
├── README.md                    # This file
├── environment.yml              # Mamba/Conda environment specification
├── requirements.txt             # Python package requirements
├── instructions.txt             # Assignment instructions
├── data/                        # Data files
│   ├── training.txt            # NLLM training corpus
│   ├── test.txt                # NLLM test sentences
│   ├── seeds.txt               # NLLM seed words
│   ├── glove.6B.50d-subset.txt # GloVe embeddings (50d)
│   └── glove.6B.300d-subset.txt # GloVe embeddings (300d)
├── src/                         # Source code
│   ├── nllm.py                 # Neural N-gram Language Model
│   ├── sentiment_classifier.py # SST-5 Sentiment Classifier
│   ├── run_nllm.sbatch         # SLURM script for NLLM
│   └── run_sentiment.sbatch    # SLURM script for Sentiment Classifier
└── notebooks/                   # Jupyter notebooks
    └── Assignment_3.ipynb      # Analysis and experiments
```

## Models

### 1. Neural N-gram Language Model (NLLM)
- Character-level language model
- Predicts next character given context
- Uses embedding layer + feedforward neural network
- Evaluated on perplexity and text generation quality

### 2. Sentiment Classifier (SST-5)
Two architectures available:

#### Deep Averaging Network (DAN)
- Averages word embeddings
- Two hidden layers with ReLU activation
- Fast and simple baseline

#### LSTM Classifier
- Unidirectional LSTM encoder
- Pack padded sequences for efficiency
- Batch normalization and dropout regularization
- Multi-layer support

Three embedding options:
1. **Random** - Randomly initialized embeddings (trained from scratch)
2. **GloVe** - Pre-trained GloVe embeddings (frozen or fine-tuned)
3. **Gensim Word2Vec** - Train Word2Vec on SST-5 training data
   - Skip-gram or CBOW algorithms
   - Configurable window size and epochs

## Notes

### WandB Integration
The sentiment classifier uses Weights & Biases for experiment tracking:
- Set `WANDB_API_KEY` environment variable before running
- Or comment out wandb lines if not using

### OSCER Usage
- Adjust SLURM parameters in `.sbatch` files based on your needs
- Update `--chdir` path to your project directory
- Update `--mail-user` to your email
- Adjust partition based on availability (`gpu`, `gpu_a100`, etc.)

### Numpy Version Note
- Gensim requires numpy <2.0
- If you have numpy 2.x, run: `pip install "numpy<2.0"`
- See conversation history for detailed resolution steps

## License
Academic use only - for CS 5293 NLP course assignment.