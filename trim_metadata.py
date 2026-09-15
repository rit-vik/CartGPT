import pickle
import os

# ---- CHANGE THIS LINE PER CATEGORY ----
RESULTS_DIR = "backend/results/video_games"
# -----------------------------------------

with open(os.path.join(RESULTS_DIR, "vocab.pkl"), "rb") as f:
    vocab_info = pickle.load(f)
item_to_id = vocab_info["item_to_id"]

with open(os.path.join(RESULTS_DIR, "item_metadata.pkl"), "rb") as f:
    item_metadata = pickle.load(f)

before_count = len(item_metadata)
before_size_mb = os.path.getsize(os.path.join(RESULTS_DIR, "item_metadata.pkl")) / (1024 * 1024)

# Keep only vocab items, and only the fields actually used (title, thumbnail)
trimmed = {}
for asin in item_to_id:
    info = item_metadata.get(asin)
    if info:
        trimmed[asin] = {"title": info.get("title"), "thumbnail": info.get("thumbnail")}

with open(os.path.join(RESULTS_DIR, "item_metadata.pkl"), "wb") as f:
    pickle.dump(trimmed, f)

after_size_mb = os.path.getsize(os.path.join(RESULTS_DIR, "item_metadata.pkl")) / (1024 * 1024)

print(f"{RESULTS_DIR}:")
print(f"  Items: {before_count} -> {len(trimmed)}")
print(f"  Size: {before_size_mb:.1f} MB -> {after_size_mb:.1f} MB")

# Also strip the now-redundant searchable_items list from demo_data.pkl —
# the backend derives search results from item_metadata directly instead.
demo_data_path = os.path.join(RESULTS_DIR, "demo_data.pkl")
if os.path.exists(demo_data_path):
    demo_before_mb = os.path.getsize(demo_data_path) / (1024 * 1024)
    with open(demo_data_path, "rb") as f:
        demo_data = pickle.load(f)
    demo_data.pop("searchable_items", None)
    with open(demo_data_path, "wb") as f:
        pickle.dump(demo_data, f)
    demo_after_mb = os.path.getsize(demo_data_path) / (1024 * 1024)
    print(f"  demo_data.pkl: {demo_before_mb:.1f} MB -> {demo_after_mb:.1f} MB")
