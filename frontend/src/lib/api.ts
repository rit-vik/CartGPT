const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

export type Category = {
  key: string;
  display_name: string;
};

export type Item = {
  asin: string;
  title: string;
  thumbnail: string | null;
};

export type SampleUser = {
  sample_id: string;
  history: Item[];
  true_next: Item;
};

export type Prediction = Item & { score: number };

export async function getCategories(): Promise<Category[]> {
  const res = await fetch(`${API_BASE}/categories`);
  if (!res.ok) throw new Error("Failed to load categories");
  return res.json();
}

export async function getSampleUsers(category: string): Promise<SampleUser[]> {
  const res = await fetch(`${API_BASE}/categories/${category}/sample-users`);
  if (!res.ok) throw new Error("Failed to load sample users");
  return res.json();
}

export async function searchItems(
  category: string,
  query: string
): Promise<Item[]> {
  if (!query.trim()) return [];
  const res = await fetch(
    `${API_BASE}/categories/${category}/search?q=${encodeURIComponent(query)}&limit=15`
  );
  if (!res.ok) throw new Error("Search failed");
  return res.json();
}

export async function predict(
  category: string,
  items: string[],
  mode: "sample" | "custom" = "custom"
): Promise<Prediction[]> {
  const res = await fetch(`${API_BASE}/categories/${category}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items, mode }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || "Prediction failed");
  }
  const data = await res.json();
  return data.predictions;
}

export async function submitFeedback(
  category: string,
  items: string[],
  chosenAsin: string
): Promise<void> {
  await fetch(`${API_BASE}/categories/${category}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items, chosen_asin: chosenAsin }),
  }).catch(() => {
    // Feedback logging is best-effort; failures shouldn't disrupt the demo
  });
}

export async function requestMissingItem(
  category: string,
  url: string,
  note: string = ""
): Promise<{ status: string; extracted_asin: string | null; note: string }> {
  const res = await fetch(`${API_BASE}/categories/${category}/request-item`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, note }),
  });
  if (!res.ok) throw new Error("Could not submit item request");
  return res.json();
}
