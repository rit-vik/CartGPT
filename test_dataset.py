import pickle
from dataset_module import SequenceRecDataset
from torch.utils.data import DataLoader

with open("tokenized_sequences.pkl", "rb") as f:
    tokenized_sequences = pickle.load(f)

train_ds = SequenceRecDataset(tokenized_sequences, mode="train")
val_ds = SequenceRecDataset(tokenized_sequences, mode="val")
test_ds = SequenceRecDataset(tokenized_sequences, mode="test")

print(f"Train dataset size: {len(train_ds)}")
print(f"Val dataset size: {len(val_ds)}")
print(f"Test dataset size: {len(test_ds)}")

# Peek at one training example
input_seq, target_seq = train_ds[0]
print(f"Input shape: {input_seq.shape}")
print(f"Target shape: {target_seq.shape}")
print(f"Input: {input_seq}")
print(f"Target: {target_seq}")

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
batch_input, batch_target = next(iter(train_loader))
print(f"Batch input shape: {batch_input.shape}")
print(f"Batch target shape: {batch_target.shape}")