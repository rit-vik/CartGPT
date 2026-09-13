import json
from collections import defaultdict

filename = "Grocery_and_Gourmet_Food.jsonl"

user_counts = defaultdict(int)
item_counts = defaultdict(int)
total_lines = 0

with open(filename, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        user_counts[record["user_id"]] += 1
        item_counts[record["parent_asin"]] += 1
        total_lines += 1

print(f"Total reviews: {total_lines}")
print(f"Unique users: {len(user_counts)}")
print(f"Unique items: {len(item_counts)}")

# Distribution check
from collections import Counter
count_dist = Counter(user_counts.values())
print("Users with exactly N reviews (N=1 to 10):")
for n in range(1, 11):
    print(f"  {n} review(s): {count_dist[n]} users")