import torch
from torch.utils.data import Dataset

PAD_TOKEN = 0
MAX_LEN = 50  # max sequence length the model will look at


class SequenceRecDataset(Dataset):
    """
    mode="train": input = tokens[:-2], predict next-item at every position (causal LM style)
    mode="val":   input = tokens[:-2] (same history as train), target = tokens[-2] (the val item)
    mode="test":  input = tokens[:-1] (history including val item), target = tokens[-1] (the test item)
    """

    def __init__(self, tokenized_sequences, mode="train", max_len=MAX_LEN):
        self.mode = mode
        self.max_len = max_len
        self.users = []
        self.sequences = []

        for user, tokens in tokenized_sequences.items():
            if len(tokens) < 3:
                continue  # need at least train + val + test items
            self.users.append(user)
            self.sequences.append(tokens)

    def __len__(self):
        return len(self.sequences)

    def _pad_left(self, seq):
        """Left-pad with PAD_TOKEN so the most recent items are right-aligned."""
        seq = seq[-self.max_len:]  # truncate to max_len if too long
        pad_len = self.max_len - len(seq)
        return [PAD_TOKEN] * pad_len + seq

    def __getitem__(self, idx):
        tokens = self.sequences[idx]

        if self.mode == "train":
            history = tokens[:-2]  # exclude val + test items entirely
            # Standard causal LM setup: input predicts the next token at every position
            input_seq = history[:-1]
            target_seq = history[1:]

            input_padded = self._pad_left(input_seq)
            target_padded = self._pad_left(target_seq)

            return (
                torch.tensor(input_padded, dtype=torch.long),
                torch.tensor(target_padded, dtype=torch.long),
            )

        elif self.mode == "val":
            history = tokens[:-2]          # same as train input history
            target = tokens[-2]            # the held-out val item
            input_padded = self._pad_left(history)
            return (
                torch.tensor(input_padded, dtype=torch.long),
                torch.tensor(target, dtype=torch.long),
            )

        elif self.mode == "test":
            history = tokens[:-1]          # includes val item now, excludes test item
            target = tokens[-1]            # the held-out test item
            input_padded = self._pad_left(history)
            return (
                torch.tensor(input_padded, dtype=torch.long),
                torch.tensor(target, dtype=torch.long),
            )

        else:
            raise ValueError(f"Unknown mode: {self.mode}")