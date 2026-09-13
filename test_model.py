import torch
import pickle
from model import SequenceRecTransformer

with open("vocab.pkl", "rb") as f:
    vocab_info = pickle.load(f)

vocab_size = vocab_info["vocab_size"]
print(f"Vocab size: {vocab_size}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SequenceRecTransformer(vocab_size=vocab_size, max_len=50).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Test a forward pass with a dummy batch
dummy_input = torch.randint(0, vocab_size, (64, 50)).to(device)
output = model(dummy_input)
print(f"Output shape: {output.shape}")  # should be (64, 50, vocab_size)