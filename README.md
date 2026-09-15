# CartGPT

A GPT-style transformer, built from scratch in PyTorch, trained to predict what someone will buy next based on their purchase history. No pretrained weights, no fine-tuning: tokenization, embeddings, self-attention, and the training loop are all implemented from the ground up.

**Live demo:** https://red-wave-01c00a11e.5.azurestaticapps.net
**Backend API:** https://cartgpt-backend-ritvik.azurewebsites.net

## The idea

A user's purchase history is treated the same way a language model treats a sentence: an ordered sequence of tokens, except each token is a product instead of a word. The model learns to predict the next item in the sequence, the same next-token prediction objective that powers GPT-style language models.

## The finding

The obvious hypothesis going in was that repeat-purchase categories (like groceries) would benefit most from sequence modeling, since there's more temporal signal to learn from. That turned out to only be partly true.

| Category | Popularity baseline (Hit Rate@10) | Model (Hit Rate@10) | Relative improvement |
|---|---|---|---|
| Grocery and Gourmet Food | 0.0099 | 0.0135 | +36% |
| Video Games | 0.0215 | 0.0351 | +63% |
| Musical Instruments | 0.0226 | 0.0225 | ~0% |
| Office Products | 0.0078 | 0.0090 | +15% |

Video Games had the lowest repeat-purchase density of the four categories, yet the largest improvement. Musical Instruments had similar density to Video Games, and the model added nothing over a plain "recommend whatever's popular" baseline.

Repeat-purchase density alone doesn't explain this. What seems to matter more is whether a category has a learnable interest trajectory: genres, franchises, and sequels create a discoverable next-step pattern that grocery reordering and instrument accessories don't. Full writeup, including the debugging story behind two real training failures, is on the [Methodology page](https://red-wave-01c00a11e.5.azurestaticapps.net/methodology).

## Tech stack

- **Model**: PyTorch, decoder-only transformer (causal self-attention, weight-tied embeddings, sampled softmax with popularity-weighted negative sampling)
- **Backend**: FastAPI, serving predictions from all four trained models, with SQLite logging for feedback and item requests
- **Frontend**: Next.js (App Router, static export), Recharts
- **Deployment**: Azure App Service (backend), Azure Static Web Apps (frontend)
- **Data**: [McAuley Lab Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

## Project structure

```
├── backend/                 FastAPI app, serves trained models
│   ├── app.py
│   ├── db.py                 SQLite logging for feedback/item requests
│   └── model.py
├── frontend/                 Next.js site (findings, methodology, live demo)
│   └── src/
│       ├── app/
│       ├── components/
│       └── lib/
├── dataset_module.py         PyTorch Dataset for sequence data
├── model.py                  Transformer architecture
├── train.py                  Training loop with negative sampling
├── preprocess.py              Filtering + tokenization pipeline
├── build_metadata.py          Builds ASIN -> product title/image lookup
├── build_demo_data.py         Builds sample users for the live demo
├── trim_metadata.py           Trims metadata to vocab-only items (for deployment size)
├── retrain_from_feedback.py   Merges logged demo feedback into training data
└── popularity_baseline.py     Baseline comparison
```

## Reproducing this locally

Trained model weights aren't included in this repo (binary files, several hundred MB, excluded via `.gitignore`). To reproduce:

### 1. Get the data

Download review and metadata files per category from the [McAuley Lab dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw), place them in `data/`.

### 2. Preprocess and train

```bash
python preprocess.py        # edit CATEGORY_FILE / OUTPUT_DIR at the top per category
python train.py             # edit RESULTS_DIR at the top per category
python build_metadata.py    # edit META_FILE / OUTPUT_DIR
python build_demo_data.py   # edit RESULTS_DIR
```

### 3. Run the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
```

### 4. Run the frontend

```bash
cd frontend
npm install
# create .env.local with:
# NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
npm run dev
```

## Limitations

- Vocabulary size varies substantially across categories (55K to 246K items), a smaller catalog is a mechanically easier ranking problem independent of sequential structure, which is a potential confound in the cross-category comparison.
- The source dataset has occasional category mislabeling (some items filed under Video Games are, on inspection, books or unrelated media).
- Negative sampling uses random negatives per batch rather than per-example hard negatives.

## Attribution

This project is independent and not affiliated with, endorsed by, or connected to Amazon.com, Inc. in any way. Product data is drawn from the McAuley Lab's public academic dataset and used here for research and demonstration purposes only.
