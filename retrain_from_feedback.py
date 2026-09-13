import pickle
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
import db

# ---- CHANGE THIS LINE PER CATEGORY ----
CATEGORY = "grocery"
RESULTS_DIR = "results/grocery"
# -----------------------------------------

with open(os.path.join(RESULTS_DIR, "vocab.pkl"), "rb") as f:
    vocab_info = pickle.load(f)
item_to_id = vocab_info["item_to_id"]

with open(os.path.join(RESULTS_DIR, "tokenized_sequences.pkl"), "rb") as f:
    tokenized_sequences = pickle.load(f)

feedback_rows = db.get_feedback_for_retraining(CATEGORY)
print(f"Found {len(feedback_rows)} feedback rows for {CATEGORY}")

added = 0
skipped = 0

for i, (input_asins, chosen_asin) in enumerate(feedback_rows):
    # Convert ASINs to token IDs; skip any not in vocab
    token_ids = [item_to_id[a] for a in input_asins if a in item_to_id]

    if chosen_asin not in item_to_id:
        skipped += 1
        continue  # the chosen item must be a known vocab item to be useful

    if len(token_ids) == 0:
        skipped += 1
        continue

    # Append the chosen item as the new "next" token, forming a valid
    # (history, target) sequence just like the original training data.
    full_sequence = token_ids + [item_to_id[chosen_asin]]

    # Use a synthetic user key so it doesn't collide with real user IDs
    synthetic_key = f"feedback_user_{i}"
    tokenized_sequences[synthetic_key] = full_sequence
    added += 1

print(f"Added {added} new sequences from feedback, skipped {skipped} (unknown items)")

with open(os.path.join(RESULTS_DIR, "tokenized_sequences.pkl"), "wb") as f:
    pickle.dump(tokenized_sequences, f)

print(f"Updated tokenized_sequences.pkl in {RESULTS_DIR}/")
print("Re-run train.py on this category to retrain with the new data.")
