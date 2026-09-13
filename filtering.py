import json
from collections import defaultdict

filename = "Grocery_and_Gourmet_Food.jsonl"
MIN_INTERACTIONS = 5

# Pass 1: count interactions per user and item
user_counts = defaultdict(int)
item_counts = defaultdict(int)

with open(filename, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        user_counts[r["user_id"]] += 1
        item_counts[r["parent_asin"]] += 1

valid_users = {u for u, c in user_counts.items() if c >= MIN_INTERACTIONS}
valid_items = {i for i, c in item_counts.items() if c >= MIN_INTERACTIONS}

print(f"Users passing filter: {len(valid_users)}")
print(f"Items passing filter: {len(valid_items)}")

# Pass 2: build per-user sequences using only valid users/items
user_sequences = defaultdict(list)

with open(filename, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["user_id"] in valid_users and r["parent_asin"] in valid_items:
            user_sequences[r["user_id"]].append((r["timestamp"], r["parent_asin"]))

# Sort each user's sequence by timestamp
for u in user_sequences:
    user_sequences[u].sort(key=lambda x: x[0])

print(f"Total sequences: {len(user_sequences)}")

# Sanity check: print one example
example_user = next(iter(user_sequences))
print(f"Example user: {example_user}")
print(f"Sequence: {user_sequences[example_user]}")

# Save to disk so we don't have to redo this every time
import pickle
with open("user_sequences.pkl", "wb") as f:
    pickle.dump(dict(user_sequences), f)
print("Saved to user_sequences.pkl")