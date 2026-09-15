import pickle
import os
import re
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from model import SequenceRecTransformer
import db

# ---- Model hyperparameters (must match what was used in training) ----
MAX_LEN = 50
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
D_FF = 256
DROPOUT = 0.1
PAD_TOKEN = 0
TOP_K = 10

CATEGORIES = {
    "grocery": {
        "display_name": "Grocery and Gourmet Food",
        "results_dir": "results/grocery",
    },
    "video_games": {
        "display_name": "Video Games",
        "results_dir": "results/video_games",
    },
    "musical_instruments": {
        "display_name": "Musical Instruments",
        "results_dir": "results/musical_instruments",
    },
    "office_products": {
        "display_name": "Office Products",
        "results_dir": "results/office_products",
    },
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load everything for every category ONCE at startup ----
loaded = {}


def load_category(key, info):
    results_dir = info["results_dir"]
    print(f"Loading category: {key} from {results_dir}")

    with open(os.path.join(results_dir, "vocab.pkl"), "rb") as f:
        vocab_info = pickle.load(f)

    with open(os.path.join(results_dir, "item_metadata.pkl"), "rb") as f:
        item_metadata = pickle.load(f)

    with open(os.path.join(results_dir, "demo_data.pkl"), "rb") as f:
        demo_data = pickle.load(f)

    model = SequenceRecTransformer(
        vocab_size=vocab_info["vocab_size"],
        max_len=MAX_LEN,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
    ).to(device)

    state_dict = torch.load(
        os.path.join(results_dir, "best_model.pt"),
        map_location=device,
    )
    model.load_state_dict(state_dict)
    model.eval()

    return {
        "display_name": info["display_name"],
        "item_to_id": vocab_info["item_to_id"],
        "id_to_item": vocab_info["id_to_item"],
        "vocab_size": vocab_info["vocab_size"],
        "item_metadata": item_metadata,
        "sample_users": demo_data["sample_users"],
        # Derived from item_metadata directly (already trimmed to vocab-only
        # items) instead of storing a duplicate copy on disk.
        "searchable_items_lower": [
            (info.get("title", "").lower(), {"asin": asin, "title": info.get("title"), "thumbnail": info.get("thumbnail")})
            for asin, info in item_metadata.items()
            if info.get("title")
        ],
        "model": model,
    }


app = FastAPI(title="Sequence Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend domain once deployed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    db.init_db()
    for key, info in CATEGORIES.items():
        loaded[key] = load_category(key, info)
    print("All categories loaded.")


def get_category_or_404(category: str):
    if category not in loaded:
        raise HTTPException(status_code=404, detail=f"Unknown category: {category}")
    return loaded[category]


def resolve_item(cat_data, asin):
    info = cat_data["item_metadata"].get(asin)
    if info is None:
        return {"asin": asin, "title": asin, "thumbnail": None}
    return {"asin": asin, "title": info.get("title") or asin, "thumbnail": info.get("thumbnail")}


# ---- Request/response models ----
class PredictRequest(BaseModel):
    items: List[str]  # list of ASINs, in chronological order (oldest first)
    mode: str = "custom"  # "sample" or "custom" — sample-user replays aren't logged for retraining


class FeedbackRequest(BaseModel):
    items: List[str]
    chosen_asin: str


class ItemRequestBody(BaseModel):
    url: str
    note: str = ""


# Matches the ASIN in common Amazon URL formats:
# /dp/ASIN, /gp/product/ASIN, /product/ASIN, with optional trailing path/query
ASIN_PATTERN = re.compile(r"/(?:dp|gp/product|product)/([A-Z0-9]{10})")


def extract_asin(url: str) -> str | None:
    match = ASIN_PATTERN.search(url)
    return match.group(1) if match else None


# ---- Endpoints ----

@app.get("/categories")
def list_categories():
    return [
        {"key": key, "display_name": data["display_name"]}
        for key, data in loaded.items()
    ]


@app.get("/categories/{category}/sample-users")
def get_sample_users(category: str):
    cat_data = get_category_or_404(category)
    return cat_data["sample_users"]


@app.get("/categories/{category}/search")
def search_items(category: str, q: str, limit: int = 20):
    cat_data = get_category_or_404(category)
    q_lower = q.lower().strip()
    if not q_lower:
        return []

    results = []
    for title_lower, item in cat_data["searchable_items_lower"]:
        if q_lower in title_lower:
            results.append(item)
            if len(results) >= limit:
                break
    return results


@app.post("/categories/{category}/predict")
def predict(category: str, request: PredictRequest):
    cat_data = get_category_or_404(category)
    model = cat_data["model"]
    item_to_id = cat_data["item_to_id"]

    if len(request.items) == 0:
        raise HTTPException(status_code=400, detail="At least one item is required.")

    # Convert ASINs to token IDs; skip any not in vocab
    token_ids = [item_to_id[asin] for asin in request.items if asin in item_to_id]

    if len(token_ids) == 0:
        raise HTTPException(
            status_code=400,
            detail="None of the provided items are recognized by this category's model.",
        )

    # Left-pad / truncate to MAX_LEN (same convention as training)
    token_ids = token_ids[-MAX_LEN:]
    pad_len = MAX_LEN - len(token_ids)
    padded = [PAD_TOKEN] * pad_len + token_ids

    input_tensor = torch.tensor([padded], dtype=torch.long, device=device)  # (1, MAX_LEN)

    with torch.no_grad():
        hidden = model.get_hidden(input_tensor)         # (1, MAX_LEN, d_model)
        last_hidden = hidden[:, -1, :]                    # (1, d_model)
        scores = model.score_full_vocab(last_hidden)       # (1, vocab_size)
        topk = torch.topk(scores, k=TOP_K, dim=-1)
        topk_ids = topk.indices[0].tolist()
        topk_scores = topk.values[0].tolist()

    id_to_item = cat_data["id_to_item"]
    predictions = []
    for tid, score in zip(topk_ids, topk_scores):
        asin = id_to_item.get(tid)
        if asin is None:
            continue  # PAD or UNK token, skip
        item = resolve_item(cat_data, asin)
        item["score"] = score
        predictions.append(item)

    # Log every prediction request. Sample-user replays are historical data
    # already used for training/eval, so only custom sequences are useful
    # signal for future retraining, but we log both for basic usage analytics.
    try:
        db.log_prediction(
            category=category,
            mode=request.mode,
            input_sequence=request.items,
            predicted_items=[p["asin"] for p in predictions],
        )
    except Exception as e:
        print(f"Logging failed (non-fatal): {e}")

    return {"predictions": predictions}


@app.post("/categories/{category}/feedback")
def submit_feedback(category: str, request: FeedbackRequest):
    """Records which prediction a visitor picked as their real guess.
    This is the actual usable signal for future retraining — an unlabeled
    custom sequence alone doesn't tell us what someone would really buy."""
    get_category_or_404(category)  # validates category exists
    try:
        db.log_feedback(
            category=category,
            input_sequence=request.items,
            chosen_asin=request.chosen_asin,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log feedback: {e}")
    return {"status": "ok"}


@app.post("/categories/{category}/request-item")
def request_missing_item(category: str, request: ItemRequestBody):
    """Logs a visitor-submitted product link for an item missing from the
    vocabulary. This does NOT add the item live — a new item needs a new
    embedding row and a retraining pass before the model knows anything
    about it. This just queues it for review."""
    get_category_or_404(category)
    asin = extract_asin(request.url)
    try:
        db.log_item_request(
            category=category,
            url=request.url,
            extracted_asin=asin,
            note=request.note,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log request: {e}")

    return {
        "status": "ok",
        "extracted_asin": asin,
        "note": "Item queued for review. It will not appear in predictions until added to the model in a future update.",
    }


@app.get("/")
def root():
    return {"status": "ok", "categories_loaded": list(loaded.keys())}
