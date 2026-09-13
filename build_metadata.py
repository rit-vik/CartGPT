import json
import pickle
import os

# ---- CHANGE THESE TWO LINES PER CATEGORY ----
META_FILE = "data/meta_Office_Products.jsonl"
OUTPUT_DIR = "results/office_products"
# -----------------------------------------------

print(f"Processing {META_FILE} -> {OUTPUT_DIR}/item_metadata.pkl")

item_metadata = {}

with open(META_FILE, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        asin = record.get("parent_asin")
        if not asin:
            continue

        title = record.get("title", "Unknown Product")

        # Grab a thumbnail image if available (nice touch for the demo UI)
        # Raw JSONL format for `images` varies: could be a list of image
        # objects, or a dict of lists — handle both defensively.
        thumbnail = None
        images = record.get("images")
        if isinstance(images, list) and len(images) > 0:
            first_img = images[0]
            if isinstance(first_img, dict):
                thumbnail = first_img.get("thumb") or first_img.get("large") or first_img.get("hi_res")
        elif isinstance(images, dict):
            thumbs = images.get("thumb")
            if thumbs and isinstance(thumbs, list) and thumbs[0]:
                thumbnail = thumbs[0]

        item_metadata[asin] = {
            "title": title,
            "thumbnail": thumbnail,
            "store": record.get("store"),
            "price": record.get("price"),
        }

print(f"Total items with metadata: {len(item_metadata)}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "item_metadata.pkl"), "wb") as f:
    pickle.dump(item_metadata, f)

print("Saved item_metadata.pkl")

# Quick sanity check: look up a few items from our existing vocab
vocab_path = os.path.join(OUTPUT_DIR, "vocab.pkl")
if os.path.exists(vocab_path):
    with open(vocab_path, "rb") as f:
        vocab_info = pickle.load(f)
    sample_asins = list(vocab_info["item_to_id"].keys())[:5]
    print("\nSample lookups:")
    for asin in sample_asins:
        info = item_metadata.get(asin, {"title": "NOT FOUND IN METADATA"})
        print(f"  {asin} -> {info.get('title')}")