import pickle

with open("user_sequences.pkl", "rb") as f:
    user_sequences = pickle.load(f)

# Build item vocabulary
# Reserve 0 = PAD, 1 = UNK (just in case)
all_items = set()
for seq in user_sequences.values():
    for _, item in seq:
        all_items.add(item)

item_to_id = {item: idx + 2 for idx, item in enumerate(sorted(all_items))}
id_to_item = {idx: item for item, idx in item_to_id.items()}
PAD_TOKEN = 0
UNK_TOKEN = 1
VOCAB_SIZE = len(item_to_id) + 2

print(f"Vocabulary size (including PAD/UNK): {VOCAB_SIZE}")

# Convert each user's sequence into a list of item token IDs, in time order
tokenized_sequences = {}
for user, seq in user_sequences.items():
    tokenized_sequences[user] = [item_to_id[item] for _, item in seq]

# Leave-one-out split:
# last item = test target, second-to-last = val target, everything before = train input
train_data = {}
val_data = {}
test_data = {}

MIN_SEQ_LEN = 3  # need at least train-input(1+) + val + test

for user, tokens in tokenized_sequences.items():
    if len(tokens) < MIN_SEQ_LEN:
        continue
    train_data[user] = tokens[:-2]        # everything except last 2
    val_data[user]   = tokens[:-1]        # everything except last 1 (input), target = tokens[-1]... wait we need target separately
    test_data[user]  = tokens             # full sequence, target = tokens[-1]

print(f"Users with usable sequences: {len(train_data)}")

# Save vocab + splits
with open("vocab.pkl", "wb") as f:
    pickle.dump({"item_to_id": item_to_id, "id_to_item": id_to_item, "vocab_size": VOCAB_SIZE}, f)

with open("tokenized_sequences.pkl", "wb") as f:
    pickle.dump(tokenized_sequences, f)

print("Saved vocab.pkl and tokenized_sequences.pkl")