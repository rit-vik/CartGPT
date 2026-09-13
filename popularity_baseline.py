import pickle
import os
from collections import Counter

# ---- CHANGE THIS LINE PER CATEGORY ----
RESULTS_DIR = "results/office_products"
# -----------------------------------------

with open(os.path.join(RESULTS_DIR, "tokenized_sequences.pkl"), "rb") as f:
    tokenized_sequences = pickle.load(f)

# Count item frequency across ALL training histories (excluding val/test items,
# to avoid leaking future information into the baseline)
item_freq = Counter()
val_targets = []
test_targets = []

for user, tokens in tokenized_sequences.items():
    if len(tokens) < 3:
        continue
    train_part = tokens[:-2]
    val_target = tokens[-2]
    test_target = tokens[-1]

    item_freq.update(train_part)
    val_targets.append(val_target)
    test_targets.append(test_target)

TOP_K = 10
most_popular = [item for item, _ in item_freq.most_common(TOP_K)]
most_popular_set = set(most_popular)

def hit_rate_at_k(targets, top_k_set):
    hits = sum(1 for t in targets if t in top_k_set)
    return hits / len(targets)

val_hr = hit_rate_at_k(val_targets, most_popular_set)
test_hr = hit_rate_at_k(test_targets, most_popular_set)

print(f"Popularity baseline — Val Hit Rate@{TOP_K}: {val_hr:.4f}")
print(f"Popularity baseline — Test Hit Rate@{TOP_K}: {test_hr:.4f}")