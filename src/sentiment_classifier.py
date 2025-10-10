#!/usr/bin/env python3
"""
Sentiment Classifier Module for SST-5 Dataset

This script trains a Deep Averaging Network (DAN) or LSTM for sentiment classification
on the SST-5 dataset using spaCy tokenization. The dataset is automatically downloaded
from HuggingFace and processed entirely in memory - no local files are loaded or saved.

Supports three embedding types:
    1. random: Random initialization
    2. glove: Pre-trained GloVe embeddings (requires local file)
    3. gensim: Train Word2Vec embeddings on SST-5 training data using Gensim

Installation:
    pip install torch datasets spacy scikit-learn wandb gensim
    python -m spacy download en_core_web_sm

Usage:
    python sentiment_classifier.py [options]

Optional arguments:
    --model: Model architecture ('dan' or 'lstm') (default: dan)
    --embedding: Embedding type ('glove', 'random', or 'gensim') (default: random)
    --embedding_dim: Dimensionality of word embeddings (default: 50)
    --hidden_size: Hidden layer size (default: 128)
    --lstm_layers: Number of LSTM layers (default: 1)
    --epochs: Number of training epochs (default: 10)
    --batch_size: Mini-batch size (default: 32)
    --lr: Learning rate (default: 0.001)
    --dropout: Dropout rate for LSTM (default: 0.3)
    --max_len: Maximum sequence length (default: 200)
    --seed: Random seed (default: 42)
    --glove_path: Path to GloVe embeddings file (optional, for glove embedding)
    --w2v_window: Word2Vec context window size (default: 5, for gensim embedding)
    --w2v_min_count: Word2Vec minimum word frequency (default: 2, for gensim embedding)
    --w2v_sg: Word2Vec algorithm: 1=skip-gram, 0=CBOW (default: 1, for gensim embedding)
    --w2v_epochs: Word2Vec training epochs (default: 10, for gensim embedding)

Examples:
    # Train DAN with random embeddings
    python sentiment_classifier.py --model dan --embedding random --epochs 20

    # Train LSTM with GloVe embeddings
    python sentiment_classifier.py --model lstm --embedding glove --epochs 15

    # Train DAN with Gensim Word2Vec (skip-gram)
    python sentiment_classifier.py --model dan --embedding gensim --embedding_dim 100 --w2v_window 5 --w2v_sg 1

    # Train LSTM with Gensim Word2Vec (CBOW)
    python sentiment_classifier.py --model lstm --embedding gensim --embedding_dim 300 --w2v_sg 0 --w2v_epochs 20
"""

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import spacy
from datasets import load_dataset
import wandb
from gensim.models import Word2Vec

# Load spaCy model for tokenization
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
    nlp = spacy.load('en_core_web_sm')


class SSTDataset(Dataset):
    """
    Custom Dataset class for SST-5 sentiment classification.

    Uses spaCy for tokenization and handles word-to-index mapping.
    Works with in-memory data from HuggingFace datasets.
    """

    def __init__(self, data, word_to_ix, max_len=50):
        """
        Initialize SST-5 dataset from in-memory data.

        Args:
            data: HuggingFace dataset split (train/validation/test)
            word_to_ix: Dictionary mapping words to indices
            max_len: Maximum sequence length for padding/truncation
        """
        self.word_to_ix = word_to_ix
        self.max_len = max_len
        self.sentences = []  # List of tokenized sentences
        self.labels = []  # List of sentiment labels (0-4)

        # Process data from HuggingFace dataset
        for example in data:
            # SetFit/sst5 uses 'text' field for sentences
            sentence = example['text']
            label = example['label']

            # Tokenize sentence using spaCy
            doc = nlp(sentence.lower())
            tokens = [token.text for token in doc]

            self.sentences.append(tokens)
            self.labels.append(label)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.sentences)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (sentence_indices, label, length)
        """
        tokens = self.sentences[idx]
        label = self.labels[idx]

        # Convert tokens to indices, use <unk> for unknown words
        unk_idx = self.word_to_ix['<unk>']  # Will raise KeyError if <unk> not in vocab
        indices = [self.word_to_ix.get(token, unk_idx) for token in tokens]

        # Get actual length before padding
        length = len(indices)

        # Pad or truncate to max_len
        if len(indices) < self.max_len:
            # Pad with 0 (assuming 0 is <pad> index)
            indices = indices + [0] * (self.max_len - len(indices))
        else:
            # Truncate to max_len
            indices = indices[:self.max_len]
            length = self.max_len

        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long), length


class DeepAveragingNetwork(nn.Module):
    """
    Deep Averaging Network (DAN) for sentiment classification.

    Averages word embeddings and passes through feedforward layers.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes=5):
        """
        Initialize DAN model.

        Args:
            vocab_size: Number of unique words in vocabulary
            embedding_dim: Dimensionality of word embeddings
            hidden_size: Number of units in hidden layer
            num_classes: Number of output classes (5 for SST-5)
        """
        super(DeepAveragingNetwork, self).__init__()
        # Embedding layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # First hidden layer
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        # Second hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Output layer for classification (5 classes)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        # ReLU activation
        self.relu = nn.ReLU()

    def forward(self, inputs, lengths=None):
        """
        Forward pass through the DAN model.

        Args:
            inputs: Tensor of shape (batch_size, seq_len) containing word indices
            lengths: Tensor of shape (batch_size,) containing actual sequence lengths

        Returns:
            logits: Tensor of shape (batch_size, num_classes) with class scores
        """
        # Get embeddings: (batch_size, seq_len, embedding_dim)
        embeds = self.embeddings(inputs)

        # Average embeddings across sequence dimension
        # If lengths provided, use them for proper averaging (ignore padding)
        if lengths is not None:
            # Create mask for valid positions
            mask = torch.arange(inputs.size(1), device=inputs.device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
            # Masked average
            embeds = (embeds * mask).sum(dim=1) / lengths.unsqueeze(-1).float()
        else:
            # Simple average across all positions
            embeds = embeds.mean(dim=1)  # (batch_size, embedding_dim)

        # Pass through feedforward layers
        x = self.relu(self.fc1(embeds))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)  # (batch_size, 5)

        return logits


class LSTMClassifier(nn.Module):
    """
    Improved LSTM-based sentiment classifier.

    Uses unidirectional (left-to-right) LSTM to encode sequences with dropout regularization.
    Processes variable-length sequences efficiently using pack_padded_sequence.
    """

    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights=None, dropout=0.3, num_layers=1):
        """
        Initialize LSTM classifier with improved architecture.

        Args:
            batch_size: Number of samples per batch (used for hidden state initialization)
            output_size: Number of output classes (5 for SST-5)
            hidden_size: Number of units in LSTM hidden state
            vocab_size: Total number of unique words in vocabulary
            embedding_length: Dimensionality of word embeddings
            weights: Pre-trained embedding weights (optional, can be None for random init)
            dropout: Dropout rate for regularization (default: 0.3)
            num_layers: Number of stacked LSTM layers (default: 1)
        """
        super(LSTMClassifier, self).__init__()
        # Store hyperparameters as instance variables
        self.batch_size = batch_size  # Default batch size for hidden state initialization
        self.output_size = output_size  # Number of output classes
        self.hidden_size = hidden_size  # LSTM hidden state size
        self.vocab_size = vocab_size  # Vocabulary size
        self.embedding_length = embedding_length  # Embedding dimension
        self.num_layers = num_layers  # Number of LSTM layers
        self.dropout_rate = dropout  # Dropout rate for regularization

        # Word embedding layer: maps word indices to dense vectors
        # padding_idx=0 ensures padding tokens have zero embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length, padding_idx=0)

        # Initialize with pre-trained embeddings if provided
        if weights is not None:
            # Set embedding weights (frozen to prevent training)
            self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)

        # Dropout layer for embedding regularization
        self.embedding_dropout = nn.Dropout(dropout)

        # Multi-layer LSTM: processes sequences (seq_len, batch_size, embedding_length)
        # Note: batch_first=False, so input should be (seq_len, batch, feature)
        # dropout applies dropout between LSTM layers (only if num_layers > 1)
        self.lstm = nn.LSTM(
            embedding_length,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=False
        )

        # Dropout layer after LSTM for regularization
        self.dropout = nn.Dropout(dropout)

        # Additional fully connected layer for better representation learning
        self.fc = nn.Linear(hidden_size, hidden_size)

        # Batch normalization for stable training
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # ReLU activation function
        self.relu = nn.ReLU()

        # Output layer: maps LSTM hidden state to class scores
        self.label = nn.Linear(hidden_size, output_size)

    def forward(self, input_sentence, batch_size=None, lengths=None):
        """
        Forward pass through the improved LSTM classifier.

        Uses pack_padded_sequence for efficient processing of variable-length sequences.

        Args:
            input_sentence: Tensor of shape (batch_size, seq_len) containing word indices
            batch_size: Batch size (optional, uses default if None)
            lengths: Tensor of actual sequence lengths for efficient packing (optional)

        Returns:
            final_output: Tensor of shape (batch_size, output_size) with class scores
        """
        # Get word embeddings: (batch_size, seq_len, embedding_length)
        input_embeds = self.word_embeddings(input_sentence)

        # Apply dropout to embeddings for regularization
        input_embeds = self.embedding_dropout(input_embeds)

        # Determine current batch size (use provided or default)
        current_batch_size = batch_size if batch_size is not None else self.batch_size

        # Get device (CPU or CUDA) from input tensor
        device = input_sentence.device

        # Initialize hidden state: (num_layers, batch_size, hidden_size)
        h_0 = torch.zeros(self.num_layers, current_batch_size, self.hidden_size).to(device)

        # Initialize cell state: (num_layers, batch_size, hidden_size)
        c_0 = torch.zeros(self.num_layers, current_batch_size, self.hidden_size).to(device)

        # Use pack_padded_sequence for efficient processing if lengths are provided
        if lengths is not None:
            # Sort sequences by length (required for pack_padded_sequence)
            # This avoids processing padding tokens
            lengths_cpu = lengths.cpu()
            sorted_lengths, sorted_idx = lengths_cpu.sort(0, descending=True)

            # Reorder embeddings based on sorted indices
            input_embeds_sorted = input_embeds[sorted_idx]

            # Permute to (seq_len, batch_size, embedding_length) for LSTM
            # LSTM expects (seq_len, batch, feature) when batch_first=False
            input_embeds_sorted = input_embeds_sorted.permute(1, 0, 2)

            # Pack sequences: only processes actual tokens, skips padding
            packed_input = nn.utils.rnn.pack_padded_sequence(
                input_embeds_sorted,
                sorted_lengths.clamp(min=1),  # Ensure minimum length of 1
                batch_first=False
            )

            # LSTM forward pass on packed sequences
            # output: packed outputs at each time step
            # final_hidden_state: (num_layers, batch_size, hidden_size) - final hidden state
            # final_cell_state: (num_layers, batch_size, hidden_size) - final cell state
            packed_output, (final_hidden_state, final_cell_state) = self.lstm(packed_input, (h_0, c_0))

            # Restore original order of sequences
            _, unsorted_idx = sorted_idx.sort(0)
            final_hidden_state = final_hidden_state[:, unsorted_idx, :]

        else:
            # No lengths provided: process all tokens (including padding)
            # Permute to (seq_len, batch_size, embedding_length) for LSTM
            input_embeds = input_embeds.permute(1, 0, 2)

            # LSTM forward pass
            # output: (seq_len, batch_size, hidden_size) - outputs at each time step
            # final_hidden_state: (num_layers, batch_size, hidden_size) - final hidden state
            # final_cell_state: (num_layers, batch_size, hidden_size) - final cell state
            output, (final_hidden_state, final_cell_state) = self.lstm(input_embeds, (h_0, c_0))

        # Use final hidden state from the last LSTM layer for classification
        # final_hidden_state[-1]: (batch_size, hidden_size) - last layer's hidden state
        lstm_out = final_hidden_state[-1]  # (batch_size, hidden_size)

        # Apply dropout for regularization
        lstm_out = self.dropout(lstm_out)

        # Pass through additional fully connected layer
        fc_out = self.fc(lstm_out)  # (batch_size, hidden_size)

        # Apply batch normalization for stable training
        fc_out = self.batch_norm(fc_out)

        # Apply ReLU activation
        fc_out = self.relu(fc_out)

        # Apply dropout again
        fc_out = self.dropout(fc_out)

        # Final classification layer
        final_output = self.label(fc_out)  # (batch_size, output_size)

        return final_output


def load_glove_embeddings(glove_path, word_to_ix, embedding_dim):
    """
    Load pre-trained GloVe embeddings from local file or download.

    Args:
        glove_path: Path to GloVe embeddings file
        word_to_ix: Dictionary mapping words to indices
        embedding_dim: Dimensionality of embeddings

    Returns:
        embedding_matrix: NumPy array of shape (vocab_size, embedding_dim)
    """
    print(f"Initializing GloVe embeddings ({embedding_dim}d)...")
    # Initialize embedding matrix with random values
    vocab_size = len(word_to_ix)
    embedding_matrix = np.random.randn(vocab_size, embedding_dim).astype(np.float32) * 0.01


    # Try to load GloVe vectors if available
    try:
        found = 0
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                if word in word_to_ix:
                    vector = np.asarray(values[1:], dtype='float32')
                    embedding_matrix[word_to_ix[word]] = vector
                    found += 1
        print(f"Found {found}/{vocab_size} words in GloVe embeddings")
    except FileNotFoundError:
        print(f"GloVe file not found at {glove_path}, using random embeddings")

    return embedding_matrix


def train_word2vec_embeddings(train_data, embedding_dim=50, window=5, min_count=2, workers=4, sg=1, epochs=10):
    """
    Train Word2Vec embeddings using Gensim on SST-5 training data.

    Args:
        train_data: HuggingFace dataset containing training examples
        embedding_dim: Dimensionality of word embeddings (default: 50)
        window: Context window size - max distance between current and predicted word (default: 5)
        min_count: Minimum word frequency to be included in vocabulary (default: 2)
        workers: Number of CPU threads for training (default: 4)
        sg: Training algorithm - 1 for skip-gram, 0 for CBOW (default: 1 for skip-gram)
        epochs: Number of training epochs over the corpus (default: 10)

    Returns:
        word2vec_model: Trained Gensim Word2Vec model containing word embeddings
    """
    print(f"\nTraining Word2Vec embeddings on SST-5 training data...")
    print(f"Parameters: dim={embedding_dim}, window={window}, min_count={min_count}, sg={'skip-gram' if sg else 'CBOW'}, epochs={epochs}")

    # Prepare sentences for Word2Vec training
    # Word2Vec expects a list of tokenized sentences
    sentences = []  # List to store tokenized sentences
    for example in train_data:
        # Get text from the example and convert to lowercase
        text = example['text'].lower()
        # Tokenize using spaCy
        doc = nlp(text)
        # Extract token strings
        tokens = [token.text for token in doc]
        # Add tokenized sentence to the training corpus
        sentences.append(tokens)

    print(f"Training on {len(sentences)} sentences...")

    # Train Word2Vec model
    # sg=1 for skip-gram (predicts context from target word)
    # sg=0 for CBOW (predicts target word from context)
    word2vec_model = Word2Vec(
        sentences=sentences,  # List of tokenized sentences
        vector_size=embedding_dim,  # Dimensionality of word vectors
        window=window,  # Maximum distance between current and predicted word
        min_count=min_count,  # Ignore words with frequency less than this
        workers=workers,  # Number of CPU threads for parallel training
        sg=sg,  # Training algorithm: 1=skip-gram, 0=CBOW
        epochs=epochs,  # Number of iterations over the corpus
        seed=42  # Random seed for reproducibility
    )

    print(f"Word2Vec training completed!")
    print(f"Vocabulary size: {len(word2vec_model.wv)}")

    return word2vec_model


def load_word2vec_embeddings(word2vec_model, word_to_ix, embedding_dim):
    """
    Convert trained Word2Vec model to embedding matrix for PyTorch.

    Args:
        word2vec_model: Trained Gensim Word2Vec model
        word_to_ix: Dictionary mapping words to their indices in vocabulary
        embedding_dim: Dimensionality of word embeddings

    Returns:
        embedding_matrix: NumPy array of shape (vocab_size, embedding_dim) containing embeddings
    """
    print(f"\nLoading Word2Vec embeddings into embedding matrix...")

    # Get vocabulary size from word_to_ix dictionary
    vocab_size = len(word_to_ix)

    # Initialize embedding matrix with small random values
    # This ensures words not in Word2Vec vocabulary have some representation
    embedding_matrix = np.random.randn(vocab_size, embedding_dim).astype(np.float32) * 0.01

    # Set padding token (<pad>) embedding to zeros
    # Padding tokens should not contribute to gradient updates
    if '<pad>' in word_to_ix:
        embedding_matrix[word_to_ix['<pad>']] = np.zeros(embedding_dim)

    # Set unknown token (<unk>) embedding to zeros
    # Unknown words will use this zero vector
    if '<unk>' in word_to_ix:
        embedding_matrix[word_to_ix['<unk>']] = np.zeros(embedding_dim)

    # Copy Word2Vec embeddings for words that exist in both vocabularies
    found = 0  # Counter for words found in Word2Vec model
    for word, idx in word_to_ix.items():
        # Check if word exists in Word2Vec vocabulary
        if word in word2vec_model.wv:
            # Get the word vector from Word2Vec model
            embedding_matrix[idx] = word2vec_model.wv[word]
            found += 1

    # Print statistics about embedding coverage
    print(f"Found {found}/{vocab_size} words in Word2Vec embeddings ({100*found/vocab_size:.2f}%)")
    print(f"Remaining {vocab_size - found} words initialized randomly")

    return embedding_matrix


def build_vocab(train_data, dev_data):
    """
    Build vocabulary from training and development data in memory.

    Args:
        train_data: Training dataset from HuggingFace
        dev_data: Development dataset from HuggingFace

    Returns:
        word_to_ix: Dictionary mapping words to indices
        vocab: List of vocabulary words
    """
    vocab = set()

    # Read training data
    for example in train_data:
        # SetFit/sst5 uses 'text' field for sentences
        sentence = example['text'].lower()
        doc = nlp(sentence)
        tokens = [token.text for token in doc]
        vocab.update(tokens)

    # Read dev data
    for example in dev_data:
        # SetFit/sst5 uses 'text' field for sentences
        sentence = example['text'].lower()
        doc = nlp(sentence)
        tokens = [token.text for token in doc]
        vocab.update(tokens)

    # Add special tokens: <pad> at index 0, <unk> at index 1
    vocab_list = ['<pad>', '<unk>'] + sorted(list(vocab))
    word_to_ix = {word: idx for idx, word in enumerate(vocab_list)}

    return word_to_ix, vocab_list


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train model for one epoch.

    Args:
        model: The neural network model
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimization algorithm
        device: Device to run training on (cpu/cuda)

    Returns:
        avg_loss: Average loss over the epoch
        accuracy: Training accuracy
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels, lengths in dataloader:
        # Move data to device
        inputs = inputs.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass - pass batch_size and lengths for LSTM compatibility
        current_batch_size = inputs.size(0)
        if hasattr(model, 'word_embeddings'):  # LSTM model
            # Pass lengths to LSTM for efficient packed sequence processing
            logits = model(inputs, batch_size=current_batch_size, lengths=lengths)
        else:  # DAN model
            logits = model(inputs, lengths)

        # Compute loss
        loss = criterion(logits, labels)

        # Log training metrics to wandb
        wandb.log({"Training Loss": loss.item()})

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent exploding/vanishing gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        # Update parameters
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        # Log training accuracy to wandb
        batch_acc = (predictions == labels).float().mean().item()
        wandb.log({"Training Accuracy": batch_acc})

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on validation/test data.

    Args:
        model: The neural network model
        dataloader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to run evaluation on (cpu/cuda)

    Returns:
        avg_loss: Average loss over the dataset
        accuracy: Evaluation accuracy
        all_preds: All predictions (for metrics)
        all_labels: All true labels (for metrics)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels, lengths in dataloader:
            # Move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            # Forward pass - pass batch_size and lengths for LSTM compatibility
            current_batch_size = inputs.size(0)
            if hasattr(model, 'word_embeddings'):  # LSTM model
                # Pass lengths to LSTM for efficient packed sequence processing
                logits = model(inputs, batch_size=current_batch_size, lengths=lengths)
            else:  # DAN model
                logits = model(inputs, lengths)

            # Compute loss
            loss = criterion(logits, labels)

            # Log evaluation metrics to wandb
            wandb.log({"Evaluation Loss": loss.item()})

            # Track metrics
            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Log evaluation accuracy to wandb
            batch_acc = (predictions == labels).float().mean().item()
            wandb.log({"Evaluation Accuracy": batch_acc})

            # Store predictions and labels for detailed metrics
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy, all_preds, all_labels


def main():
    """Main function to train and evaluate sentiment classifier."""
    # Parse command-line arguments (only hyperparameters, no file paths)
    parser = argparse.ArgumentParser(
        description='SST-5 Sentiment Classifier with Word Embeddings',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', type=str, default='dan', choices=['dan', 'lstm'],
                       help='Model architecture (dan or lstm)')
    parser.add_argument('--embedding', type=str, default='random', choices=['glove', 'random', 'gensim'],
                       help='Embedding type: glove, random, or gensim (train Word2Vec on SST-5)')
    parser.add_argument('--embedding_dim', type=int, default=50,
                       help='Dimensionality of word embeddings')
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='Hidden layer size')
    parser.add_argument('--lstm_layers', type=int, default=1,
                       help='Number of LSTM layers (for lstm model)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Mini-batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate for LSTM regularization')
    parser.add_argument('--max_len', type=int, default=200,
                       help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--glove_path', type=str, default='../data/glove.6B.50d-subset.txt',
                       help='Path to GloVe embeddings file (optional)')
    parser.add_argument('--w2v_window', type=int, default=5,
                       help='Word2Vec context window size (for gensim embedding)')
    parser.add_argument('--w2v_min_count', type=int, default=2,
                       help='Word2Vec minimum word frequency (for gensim embedding)')
    parser.add_argument('--w2v_sg', type=int, default=1, choices=[0, 1],
                       help='Word2Vec algorithm: 1=skip-gram, 0=CBOW (for gensim embedding)')
    parser.add_argument('--w2v_epochs', type=int, default=10,
                       help='Word2Vec training epochs (for gensim embedding)')

    args = parser.parse_args()

    # Initialize wandb for experiment tracking
    # Note: Set WANDB_API_KEY environment variable before running for security
    wandb.init(project="sentiment-classification")

    # Log hyperparameters to wandb
    wandb.config.update(args)

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("="*80)
    print("SST-5 Sentiment Classification")
    print("="*80)
    print(f"Model: {args.model.upper()}")
    print(f"Embedding: {args.embedding}")
    if args.embedding == 'gensim':
        # Print Word2Vec training parameters
        print(f"  Word2Vec algorithm: {'Skip-gram' if args.w2v_sg else 'CBOW'}")
        print(f"  Word2Vec window size: {args.w2v_window}")
        print(f"  Word2Vec min count: {args.w2v_min_count}")
        print(f"  Word2Vec epochs: {args.w2v_epochs}")
    print(f"Embedding dimension: {args.embedding_dim}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Tokenization: spaCy")
    print("="*80)

    # Download SST-5 dataset from HuggingFace
    print("\nDownloading SST-5 dataset from HuggingFace...")
    dataset = load_dataset("SetFit/sst5")

    # Get train, validation, and test splits
    train_data = dataset['train']
    dev_data = dataset['validation']
    test_data = dataset['test']

    print(f"Downloaded {len(train_data)} training examples")
    print(f"Downloaded {len(dev_data)} validation examples")
    print(f"Downloaded {len(test_data)} test examples")

    # Build vocabulary
    print("\nBuilding vocabulary from downloaded data...")
    word_to_ix, vocab = build_vocab(train_data, dev_data)
    print(f"Vocabulary size: {len(vocab)}")

    # Create datasets
    print("\nCreating in-memory datasets...")
    train_dataset = SSTDataset(train_data, word_to_ix, args.max_len)
    dev_dataset = SSTDataset(dev_data, word_to_ix, args.max_len)
    test_dataset = SSTDataset(test_data, word_to_ix, args.max_len)

    print(f"Train size: {len(train_dataset)}")
    print(f"Dev size: {len(dev_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Initialize model
    print("\nInitializing model...")
    if args.model == 'dan':
        model = DeepAveragingNetwork(
            vocab_size=len(vocab),
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_classes=5
        )
    else:  # lstm
        model = LSTMClassifier(
            batch_size=args.batch_size,
            output_size=5,
            hidden_size=args.hidden_size,
            vocab_size=len(vocab),
            embedding_length=args.embedding_dim,
            weights=None,
            dropout=args.dropout,
            num_layers=args.lstm_layers
        )

    # Load pre-trained embeddings if specified
    if args.embedding == 'glove':
        # Load pre-trained GloVe embeddings from file
        embedding_matrix = load_glove_embeddings(args.glove_path, word_to_ix, args.embedding_dim)
        if args.model == 'dan':
            # Copy embeddings to DAN model
            model.embeddings.weight.data.copy_(torch.from_numpy(embedding_matrix))
        else:  # lstm
            # Copy embeddings to LSTM model
            model.word_embeddings.weight.data.copy_(torch.from_numpy(embedding_matrix))
        # Optionally freeze embeddings to prevent training
        # model.embeddings.weight.requires_grad = False
    elif args.embedding == 'gensim':
        # Train custom Word2Vec embeddings on SST-5 training data
        print("\n" + "="*80)
        print("Training Word2Vec embeddings using Gensim...")
        print("="*80)

        # Train Word2Vec model on training data
        word2vec_model = train_word2vec_embeddings(
            train_data,
            embedding_dim=args.embedding_dim,
            window=args.w2v_window,
            min_count=args.w2v_min_count,
            workers=4,  # Use 4 CPU threads for parallel training
            sg=args.w2v_sg,  # 1 for skip-gram, 0 for CBOW
            epochs=args.w2v_epochs
        )

        # Convert Word2Vec model to embedding matrix
        embedding_matrix = load_word2vec_embeddings(word2vec_model, word_to_ix, args.embedding_dim)

        # Copy embeddings to model
        if args.model == 'dan':
            # Copy embeddings to DAN model
            model.embeddings.weight.data.copy_(torch.from_numpy(embedding_matrix))
        else:  # lstm
            # Copy embeddings to LSTM model
            model.word_embeddings.weight.data.copy_(torch.from_numpy(embedding_matrix))

        print("Word2Vec embeddings loaded successfully!")
        print("="*80)

        # Optionally freeze embeddings to prevent fine-tuning during training
        # Uncomment the next line if you want to freeze the embeddings
        # model.embeddings.weight.requires_grad = False

    model = model.to(device)

    # Watch model for gradient and parameter tracking
    wandb.watch(model, log="all", log_freq=100)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params:,}")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print("\nTraining model...")
    best_dev_acc = 0.0
    best_model_state = None  # Store best model in memory
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate on dev set
        dev_loss, dev_acc, _, _ = evaluate(model, dev_loader, criterion, device)

        # Log epoch-level metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss_epoch": train_loss,
            "train_acc_epoch": train_acc,
            "dev_loss_epoch": dev_loss,
            "dev_acc_epoch": dev_acc
        })

        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Dev Loss: {dev_loss:.4f}, Dev Acc: {dev_acc:.4f}")

        # Save best model in memory
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  New best model! Dev Acc: {dev_acc:.4f}")

    print("\nTraining completed!")
    print(f"Best dev accuracy: {best_dev_acc:.4f}")

    # Load best model from memory and evaluate on test set
    print("\nEvaluating on test set...")
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

    # Log final test metrics to wandb
    wandb.log({
        "test_loss": test_loss,
        "test_acc": test_acc
    })

    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # Generate classification report
    print("\nClassification Report:")
    label_names = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    print(classification_report(test_labels, test_preds, target_names=label_names))

    # Generate confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds)
    print(cm)

    # Print summary
    print("\n" + "="*80)
    print("Training completed successfully!")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print("="*80)

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
