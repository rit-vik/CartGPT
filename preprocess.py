import json
import pickle
import os
from collections import defaultdict

# ---- CHANGE THESE TWO LINES PER CATEGORY ----
CATEGORY_FILE = "data/Office_Products.jsonl"
OUTPUT_DIR = "results/office_products"
# -----------------------------------------------

MIN_INTERACTIONS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Processing {CATEGORY_FILE} -> {OUTPUT_DIR}")

# ---- Pass 1: count interactions per user/item ----
user_counts = defaultdict(int)
item_counts = defaultdict(int)

with open(CATEGORY_FILE, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        user_counts[r["user_id"]] += 1
        item_counts[r["parent_asin"]] += 1

valid_users = {u for u, c in user_counts.items() if c >= MIN_INTERACTIONS}
valid_items = {i for i, c in item_counts.items() if c >= MIN_INTERACTIONS}

print(f"Users passing filter: {len(valid_users)}")
print(f"Items passing filter: {len(valid_items)}")

# ---- Pass 2: build per-user sequences ----
user_sequences = defaultdict(list)

with open(CATEGORY_FILE, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["user_id"] in valid_users and r["parent_asin"] in valid_items:
            user_sequences[r["user_id"]].append((r["timestamp"], r["parent_asin"]))

for u in user_sequences:
    user_sequences[u].sort(key=lambda x: x[0])

print(f"Total sequences: {len(user_sequences)}")

with open(os.path.join(OUTPUT_DIR, "user_sequences.pkl"), "wb") as f:
    pickle.dump(dict(user_sequences), f)

# ---- Build vocabulary ----
all_items = set()
for seq in user_sequences.values():
    for _, item in seq:
        all_items.add(item)

item_to_id = {item: idx + 2 for idx, item in enumerate(sorted(all_items))}
id_to_item = {idx: item for item, idx in item_to_id.items()}
VOCAB_SIZE = len(item_to_id) + 2

print(f"Vocabulary size (including PAD/UNK): {VOCAB_SIZE}")

with open(os.path.join(OUTPUT_DIR, "vocab.pkl"), "wb") as f:
    pickle.dump({"item_to_id": item_to_id, "id_to_item": id_to_item, "vocab_size": VOCAB_SIZE}, f)

# ---- Tokenize sequences ----
tokenized_sequences = {}
for user, seq in user_sequences.items():
    tokenized_sequences[user] = [item_to_id[item] for _, item in seq]

usable = sum(1 for tokens in tokenized_sequences.values() if len(tokens) >= 3)
print(f"Users with usable sequences (>=3 interactions): {usable}")

with open(os.path.join(OUTPUT_DIR, "tokenized_sequences.pkl"), "wb") as f:
    pickle.dump(tokenized_sequences, f)

print(f"Done. All files saved in {OUTPUT_DIR}/")