#!/bin/bash
# Quick test script to verify nllm.py works correctly

echo "Testing nllm.py with minimal configuration..."
echo ""

# Run with minimal epochs for quick testing
python nllm.py ../data/training.txt ../data/test.txt ../data/seeds.txt \
    --epochs 2 \
    --context_size 2 \
    --embedding_dim 10 \
    --hidden_size 128 \
    --batch_size 32

echo ""
echo "Test completed! Check the output files:"
echo "  - ngram-prob.trace (log probabilities)"
echo "  - ngram-gen.trace (generated text)"
echo "  - ngram_model.pth (saved model)"
