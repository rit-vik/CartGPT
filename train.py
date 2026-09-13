import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import math
import os
from collections import Counter

from dataset_module import SequenceRecDataset
from model import SequenceRecTransformer

# ---- CHANGE THIS LINE PER CATEGORY ----
RESULTS_DIR = "results/office_products"
# -----------------------------------------

PAD_TOKEN = 0
MAX_LEN = 50
BATCH_SIZE = 128
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
D_FF = 256
DROPOUT = 0.1
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 1000
NUM_EPOCHS = 5
EVAL_EVERY_N_STEPS = 500
TOP_K = 10
NUM_NEGATIVES = 100
NEG_SAMPLING_POWER = 0.75  # classic word2vec-style smoothing of the frequency distribution


def build_negative_sampling_distribution(tokenized_sequences, vocab_size, power=0.75):
    """
    Build a probability distribution over items for negative sampling,
    weighted by (frequency ^ power) — the standard trick from word2vec,
    also widely used in recommender negative sampling.

    Sampling negatives proportional to popularity forces the model to learn
    to out-rank the items it will ACTUALLY compete against at inference time,
    rather than easily-distinguishable random noise.
    """
    freq = torch.zeros(vocab_size, dtype=torch.float)
    counter = Counter()
    for tokens in tokenized_sequences.values():
        if len(tokens) < 3:
            continue
        counter.update(tokens[:-2])  # only count training-visible items

    for item_id, count in counter.items():
        freq[item_id] = count

    freq[PAD_TOKEN] = 0.0
    freq = freq.clamp(min=0)
    smoothed = freq.pow(power)
    smoothed[PAD_TOKEN] = 0.0
    # Avoid an all-zero distribution edge case
    if smoothed.sum() == 0:
        smoothed = torch.ones(vocab_size)
        smoothed[PAD_TOKEN] = 0.0
    probs = smoothed / smoothed.sum()
    return probs


def get_lr_lambda(warmup_steps):
    def lr_lambda(step):
        step = max(step, 1)
        if step < warmup_steps:
            return step / warmup_steps
        return (warmup_steps ** 0.5) * (step ** -0.5)
    return lr_lambda


def sampled_softmax_loss(model, hidden, targets, neg_dist, num_negatives, device):
    """
    hidden: (N, d_model) hidden states at valid (non-pad) positions
    targets: (N,) true next-item IDs at those positions
    neg_dist: (vocab_size,) precomputed popularity-weighted sampling distribution
    Returns scalar loss.
    """
    N = hidden.size(0)

    # Sample negatives proportional to item popularity, not uniformly —
    # this forces the model to learn to out-rank REAL competitive items.
    neg_ids = torch.multinomial(neg_dist, num_negatives, replacement=True).to(device)

    scale = model.d_model ** 0.5
    pos_emb = model.token_emb(targets)               # (N, d_model)
    pos_scores = (hidden * pos_emb).sum(dim=-1, keepdim=True) / scale  # (N, 1)

    neg_scores = model.score_items(hidden, neg_ids)  # (N, num_negatives)

    logits = torch.cat([pos_scores, neg_scores], dim=1)  # (N, 1+num_negatives)
    labels = torch.zeros(N, dtype=torch.long, device=device)

    loss = nn.functional.cross_entropy(logits, labels)
    return loss


@torch.no_grad()
def evaluate(model, eval_loader, device, top_k=10):
    """Full-vocab ranking eval — fine here since we only score the FINAL
    position per example, not every position (small tensor, no OOM risk)."""
    model.eval()
    hits = 0
    ndcg_sum = 0.0
    total = 0

    for input_seq, target in eval_loader:
        input_seq = input_seq.to(device)
        target = target.to(device)

        hidden = model.get_hidden(input_seq)        # (B, T, d_model)
        last_hidden = hidden[:, -1, :]                # (B, d_model)
        scores = model.score_full_vocab(last_hidden)   # (B, vocab_size)

        topk_items = torch.topk(scores, k=top_k, dim=-1).indices  # (B, K)

        for i in range(target.size(0)):
            true_item = target[i].item()
            predicted_topk = topk_items[i].tolist()
            if true_item in predicted_topk:
                hits += 1
                rank = predicted_topk.index(true_item) + 1
                ndcg_sum += 1.0 / math.log2(rank + 1)
            total += 1

    model.train()
    hit_rate = hits / total
    ndcg = ndcg_sum / total
    return hit_rate, ndcg


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(os.path.join(RESULTS_DIR, "vocab.pkl"), "rb") as f:
        vocab_info = pickle.load(f)
    vocab_size = vocab_info["vocab_size"]

    with open(os.path.join(RESULTS_DIR, "tokenized_sequences.pkl"), "rb") as f:
        tokenized_sequences = pickle.load(f)

    print("Building popularity-weighted negative sampling distribution...")
    neg_dist = build_negative_sampling_distribution(
        tokenized_sequences, vocab_size, power=NEG_SAMPLING_POWER
    )

    train_ds = SequenceRecDataset(tokenized_sequences, mode="train", max_len=MAX_LEN)
    val_ds = SequenceRecDataset(tokenized_sequences, mode="val", max_len=MAX_LEN)
    test_ds = SequenceRecDataset(tokenized_sequences, mode="test", max_len=MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SequenceRecTransformer(
        vocab_size=vocab_size,
        max_len=MAX_LEN,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters (with weight tying): {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda(WARMUP_STEPS))

    best_hit_rate = -1.0
    step = 0
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        num_batches = 0

        for input_seq, target_seq in train_loader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            hidden = model.get_hidden(input_seq)  # (B, T, d_model)

            # Only train on non-pad target positions
            mask = target_seq != PAD_TOKEN
            valid_hidden = hidden[mask]        # (N_valid, d_model)
            valid_targets = target_seq[mask]   # (N_valid,)

            if valid_hidden.size(0) == 0:
                continue  # skip degenerate batches (shouldn't normally happen)

            loss = sampled_softmax_loss(
                model, valid_hidden, valid_targets, neg_dist, NUM_NEGATIVES, device
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            step += 1

            if step % EVAL_EVERY_N_STEPS == 0:
                avg_loss = total_loss / num_batches
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1} | Step {step} | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

        hit_rate, ndcg = evaluate(model, val_loader, device, top_k=TOP_K)
        print(f"=== Epoch {epoch+1} complete | Val Hit Rate@{TOP_K}: {hit_rate:.4f} | Val NDCG@{TOP_K}: {ndcg:.4f} ===")

        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"model_epoch_{epoch+1}.pt"))

        if hit_rate > best_hit_rate:
            best_hit_rate = hit_rate
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_model.pt"))
            print(f"    -> New best model saved (Val Hit Rate@{TOP_K}: {hit_rate:.4f})")

    print(f"\nLoading best model (Val Hit Rate@{TOP_K}: {best_hit_rate:.4f}) for final test evaluation...")
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "best_model.pt")))
    test_hit_rate, test_ndcg = evaluate(model, test_loader, device, top_k=TOP_K)
    print(f"=== FINAL TEST RESULTS (best checkpoint) | Hit Rate@{TOP_K}: {test_hit_rate:.4f} | NDCG@{TOP_K}: {test_ndcg:.4f} ===")


if __name__ == "__main__":
    main()