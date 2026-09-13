import json
from collections import defaultdict, Counter

# ---- CHANGE THIS LINE PER CATEGORY ----
CATEGORY_FILE = "data/Musical_Instruments.jsonl"
# ----------------------------------------

user_counts = defaultdict(int)
item_counts = defaultdict(int)
total_lines = 0

with open(CATEGORY_FILE, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        user_counts[record["user_id"]] += 1
        item_counts[record["parent_asin"]] += 1
        total_lines += 1

print(f"File: {CATEGORY_FILE}")
print(f"Total reviews: {total_lines}")
print(f"Unique users: {len(user_counts)}")
print(f"Unique items: {len(item_counts)}")

count_dist = Counter(user_counts.values())
print("Users with exactly N reviews (N=1 to 10):")
for n in range(1, 11):
    print(f"  {n} review(s): {count_dist[n]} users")

MIN_INTERACTIONS = 5
users_ge5 = sum(c for u, c in [(u, user_counts[u]) for u in user_counts] if user_counts[u] >= MIN_INTERACTIONS)
num_users_ge5 = sum(1 for u in user_counts if user_counts[u] >= MIN_INTERACTIONS)
print(f"\nUsers with >= {MIN_INTERACTIONS} reviews: {num_users_ge5} ({num_users_ge5/len(user_counts)*100:.1f}% of users)")