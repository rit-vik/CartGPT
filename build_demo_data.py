import pickle
import os
import random

# ---- CHANGE THIS LINE PER CATEGORY ----
RESULTS_DIR = "results/musical_instruments"
# -----------------------------------------

NUM_SAMPLE_USERS = 8
MIN_SEQ_LEN = 5
MAX_SEQ_LEN = 15

random.seed(42)  # reproducible sample selection

with open(os.path.join(RESULTS_DIR, "vocab.pkl"), "rb") as f:
    vocab_info = pickle.load(f)
item_to_id = vocab_info["item_to_id"]

with open(os.path.join(RESULTS_DIR, "tokenized_sequences.pkl"), "rb") as f:
    tokenized_sequences = pickle.load(f)

with open(os.path.join(RESULTS_DIR, "item_metadata.pkl"), "rb") as f:
    item_metadata = pickle.load(f)

id_to_item = vocab_info["id_to_item"]


def resolve(asin):
    info = item_metadata.get(asin)
    if info is None:
        return {"asin": asin, "title": asin, "thumbnail": None}
    return {"asin": asin, "title": info.get("title") or asin, "thumbnail": info.get("thumbnail")}


# ---- Build sample users ----
candidates = [
    (user, tokens) for user, tokens in tokenized_sequences.items()
    if MIN_SEQ_LEN <= len(tokens) <= MAX_SEQ_LEN
]
print(f"Candidate users with {MIN_SEQ_LEN}-{MAX_SEQ_LEN} interactions: {len(candidates)}")

selected = random.sample(candidates, min(NUM_SAMPLE_USERS, len(candidates)))

sample_users = []
for i, (user, token_ids) in enumerate(selected):
    asins = [id_to_item[t] for t in token_ids]
    history_asins = asins[:-1]   # everything except the true last item
    true_next_asin = asins[-1]

    sample_users.append({
        "sample_id": f"user_{i+1}",
        "history": [resolve(a) for a in history_asins],
        "true_next": resolve(true_next_asin),
    })

print(f"Built {len(sample_users)} sample users")
for su in sample_users:
    titles = [h["title"][:40] for h in su["history"]]
    print(f"  {su['sample_id']}: {titles} -> TRUE NEXT: {su['true_next']['title'][:40]}")

# ---- Build searchable item list (vocab-restricted, has metadata) ----
searchable_items = []
for asin in item_to_id:
    info = item_metadata.get(asin)
    if info and info.get("title"):
        searchable_items.append({"asin": asin, "title": info["title"], "thumbnail": info.get("thumbnail")})

print(f"\nSearchable items (in vocab AND has metadata): {len(searchable_items)}")

demo_data = {
    "sample_users": sample_users,
    "searchable_items": searchable_items,
}

with open(os.path.join(RESULTS_DIR, "demo_data.pkl"), "wb") as f:
    pickle.dump(demo_data, f)

print(f"Saved demo_data.pkl to {RESULTS_DIR}/")