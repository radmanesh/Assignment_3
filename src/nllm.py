#!/usr/bin/env python3
"""
Neural N-gram Language Model (NLLM)
This script trains a neural n-gram language model using PyTorch,
calculates log-probabilities for test sentences, and generates text from seed words.

Usage:
    python nllm.py <training_file> <test_file> <seeds_file> [options]

Arguments:
    training_file: Path to the training data file
    test_file: Path to the test data file
    seeds_file: Path to the seeds file for text generation

Optional arguments:
    --context_size: Number of previous words to use as context (default: 2)
    --embedding_dim: Dimensionality of word embeddings (default: 10)
    --epochs: Number of training epochs (default: 10)
    --batch_size: Mini-batch size for training (default: 32)
    --lr: Learning rate (default: 0.01)
    --hidden_size: Hidden layer size (default: 128)
    --max_len: Maximum length of generated text (default: 40)
    --k: Top-k sampling parameter (default: 10)
    --seed: Random seed for reproducibility (default: 1)
"""

import argparse
import pickle
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class NGramLanguageModelerBatch(nn.Module):
    """
    Neural N-gram Language Model with batch processing support.

    This model uses word embeddings and a feedforward neural network
    to predict the next word given a context of previous words.
    """

    def __init__(self, vocab_size, embedding_dim, context_size, hidden_size=128):
        """
        Initialize the N-gram language model.

        Args:
            vocab_size: Number of unique words in vocabulary
            embedding_dim: Dimensionality of word embeddings
            context_size: Number of previous words to use as context
            hidden_size: Number of units in hidden layer (default: 128)
        """
        super(NGramLanguageModelerBatch, self).__init__()
        # Embedding layer to convert word indices to dense vectors
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # First linear layer: flattened context embeddings to hidden layer
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_size)
        # Second linear layer: hidden layer to vocabulary size (output layer)
        self.linear2 = nn.Linear(hidden_size, vocab_size)
        # Activation function
        self.relu = nn.ReLU()

        # Sequential model combining all layers
        self.model = nn.Sequential(
            self.embeddings,
            nn.Flatten(),
            self.linear1,
            self.relu,
            self.linear2,
            nn.LogSoftmax(dim=1)
        )

    def forward(self, inputs):
        """
        Forward pass through the model.

        Args:
            inputs: Tensor of shape (batch_size, context_size) containing word indices

        Returns:
            log_probs: Tensor of shape (batch_size, vocab_size) with log probabilities
        """
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


def extract_ngrams(context_size, sentence):
    """
    Extract n-grams from a sentence for training.

    Args:
        context_size: Number of previous words to use as context
        sentence: List of words in the sentence

    Returns:
        ngrams: List of tuples (context_words, target_word)
    """
    # Pad the beginning with <s> tokens to handle first few words
    padded_sentence = ["<s>"] * context_size + sentence

    # Create n-gram tuples: (context, target)
    ngrams = [
        (padded_sentence[i - context_size:i], padded_sentence[i])
        for i in range(context_size, len(padded_sentence))
    ]
    return ngrams


def train_model_mini_batch(model, ngrams, word_to_ix, loss_function, optimizer,
                          epochs=10, batch_size=32):
    """
    Train the neural n-gram language model using mini-batch gradient descent.

    Args:
        model: The NGramLanguageModelerBatch model
        ngrams: List of (context, target) tuples
        word_to_ix: Dictionary mapping words to indices
        loss_function: Loss criterion (e.g., NLLLoss)
        optimizer: Optimization algorithm (e.g., SGD)
        epochs: Number of training epochs
        batch_size: Size of mini-batches

    Returns:
        losses: List of total loss per epoch
    """
    # Prepare the dataset
    contexts = []
    targets = []
    for context, target in ngrams:
        # Convert words to indices
        context_idxs = [word_to_ix[w] for w in context]
        target_idx = word_to_ix[target]
        contexts.append(context_idxs)
        targets.append(target_idx)

    # Convert to tensors
    contexts_tensor = torch.tensor(contexts, dtype=torch.long)
    targets_tensor = torch.tensor(targets, dtype=torch.long)

    # Create DataLoader for mini-batch training
    dataset = TensorDataset(contexts_tensor, targets_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(epochs):
        total_loss = 0
        # Iterate through mini-batches
        for context_batch, target_batch in dataloader:
            # Zero out gradients from previous iteration
            optimizer.zero_grad()
            # Forward pass
            log_probs = model(context_batch)
            # Compute loss
            loss = loss_function(log_probs, target_batch)
            # Backward pass
            loss.backward()
            # Update parameters
            optimizer.step()
            # Accumulate loss
            total_loss += loss.item()

        losses.append(total_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    return losses


def sentence_log_probability(model, sentence, word_to_ix, context_size=2):
    """
    Calculate the log-probability of a sentence using the trained model.

    Args:
        model: Trained NGramLanguageModelerBatch model
        sentence: Input sentence as a string
        word_to_ix: Dictionary mapping words to indices
        context_size: Number of previous words to use as context

    Returns:
        log_prob: Total log-probability of the sentence
    """
    # Set model to evaluation mode
    model.eval()

    # Convert sentence to lowercase and split into words
    words = sentence.lower().strip().split()

    # Pad the beginning with <s> tokens
    padded_sentence = ["<s>"] * context_size + words

    # Initialize cumulative log probability
    log_prob = 0.0

    # Iterate through each word position
    with torch.no_grad():
        for i in range(context_size, len(padded_sentence)):
            # Extract context
            context = padded_sentence[i - context_size:i]
            # Get target word
            target = padded_sentence[i]

            # Convert to indices (use <s> for unknown words)
            context_idxs = [word_to_ix.get(w, word_to_ix["<s>"]) for w in context]
            target_idx = word_to_ix.get(target, word_to_ix["<s>"])

            # Create tensor with batch dimension
            context_tensor = torch.tensor([context_idxs], dtype=torch.long)

            # Get log-probabilities
            log_probs = model(context_tensor)

            # Accumulate log-probability of target word
            log_prob += log_probs[0, target_idx].item()

    return log_prob


def generate_text(model, start_words, word_to_ix, ix_to_word, context_size=2,
                 max_len=40, k=10):
    """
    Generate text using the trained model.

    Args:
        model: Trained NGramLanguageModelerBatch model
        start_words: Starting words as a string
        word_to_ix: Dictionary mapping words to indices
        ix_to_word: Dictionary mapping indices to words
        context_size: Number of previous words to use as context
        max_len: Maximum number of words to generate
        k: Top-k sampling parameter

    Returns:
        generated_text: Generated text as a string
    """
    # Set model to evaluation mode
    model.eval()

    # Prepare initial context
    words = start_words.lower().split()
    if len(words) < context_size:
        words = ["<s>"] * (context_size - len(words)) + words

    generated_words = words[:]

    with torch.no_grad():
        for _ in range(max_len):
            # Get the last context_size words
            context = generated_words[-context_size:]
            context_idxs = [word_to_ix.get(w, word_to_ix["<s>"]) for w in context]
            context_tensor = torch.tensor([context_idxs], dtype=torch.long)

            # Get log probabilities
            log_probs = model(context_tensor)
            # Convert to probabilities and remove batch dimension
            probs = torch.exp(log_probs).squeeze()

            # Get top k probabilities and their indices
            topk_probs, topk_indices = torch.topk(probs, k)
            # Normalize to get valid probability distribution
            topk_probs = topk_probs / torch.sum(topk_probs)

            # Sample one index from top k
            next_word_idx = torch.multinomial(topk_probs, 1).item()
            # Get corresponding word
            next_word = ix_to_word[topk_indices[next_word_idx].item()]

            generated_words.append(next_word)

            # Stop if we generate sentence-ending punctuation
            if next_word in {".", "?", "!"}:
                break

    # Return generated words excluding initial <s> tokens
    return ' '.join(generated_words[context_size:])


def main():
    """Main function to train model and generate outputs."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Neural N-gram Language Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('training_file', type=str, help='Path to training data file')
    parser.add_argument('test_file', type=str, help='Path to test data file')
    parser.add_argument('seeds_file', type=str, help='Path to seeds file')
    parser.add_argument('--context_size', type=int, default=2,
                       help='Number of previous words to use as context')
    parser.add_argument('--embedding_dim', type=int, default=10,
                       help='Dimensionality of word embeddings')
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='Hidden layer size')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Mini-batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--max_len', type=int, default=40,
                       help='Maximum length of generated text')
    parser.add_argument('--k', type=int, default=10,
                       help='Top-k sampling parameter')
    parser.add_argument('--seed', type=int, default=1,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("="*80)
    print("Neural N-gram Language Model Training")
    print("="*80)
    print(f"Context size: {args.context_size}")
    print(f"Embedding dimension: {args.embedding_dim}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("="*80)

    # Read training data
    print(f"\nReading training data from {args.training_file}...")
    with open(args.training_file, "r") as f:
        lines = f.readlines()

    # Generate ngrams from training data
    print("Extracting n-grams from training data...")
    ngrams = []
    vocab = set()
    for line in lines:
        words = line.lower().strip().split()
        vocab.update(words)
        ngrams.extend(extract_ngrams(args.context_size, words))

    # Add special token and create vocabulary list
    vocab.add("<s>")
    vocab = list(vocab)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of n-grams: {len(ngrams)}")

    # Create word-to-index mapping
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = {i: word for word, i in word_to_ix.items()}

    # Initialize model
    print("\nInitializing model...")
    model = NGramLanguageModelerBatch(
        len(vocab),
        args.embedding_dim,
        args.context_size,
        args.hidden_size
    )

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")

    # Define loss function and optimizer
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # Train the model
    print("\nTraining model...")
    losses = train_model_mini_batch(
        model, ngrams, word_to_ix, loss_function, optimizer,
        epochs=args.epochs, batch_size=args.batch_size
    )

    print("\nTraining completed!")

    # Save model and vocabulary
    print("\nSaving model and vocabulary...")
    torch.save(model.state_dict(), "ngram_model.pth")
    with open("word_to_ix.pkl", "wb") as f:
        pickle.dump(word_to_ix, f)
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print("Model saved to ngram_model.pth")

    # Calculate log-probabilities for test sentences
    print("\n" + "="*80)
    print("Calculating log-probabilities for test sentences...")
    print("="*80)

    with open(args.test_file, "r") as f:
        test_lines = f.readlines()

    # Open output file for probabilities
    with open("ngram-prob.trace", "w") as prob_file:
        for line in test_lines:
            log_prob = sentence_log_probability(
                model, line.strip(), word_to_ix, args.context_size
            )
            output = f"{log_prob:.6f}\t{line.strip()}\n"
            prob_file.write(output)
            print(output.strip())

    print(f"\nLog-probabilities saved to ngram-prob.trace")

    # Generate text from seed words
    print("\n" + "="*80)
    print("Generating text from seed words...")
    print("="*80)

    with open(args.seeds_file, "r") as f:
        seed_lines = f.readlines()

    # Open output file for generated text
    with open("ngram-gen.trace", "w") as gen_file:
        for line in seed_lines:
            seed = line.strip()
            generated_text = generate_text(
                model, seed, word_to_ix, ix_to_word,
                context_size=args.context_size,
                max_len=args.max_len,
                k=args.k
            )
            output = f"Seed: {seed}\nGenerated: {generated_text}\n\n"
            gen_file.write(output)
            print(output)

    print("Generated text saved to ngram-gen.trace")
    print("\n" + "="*80)
    print("All tasks completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
