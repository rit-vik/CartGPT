"use client";

import { useState, useEffect, useCallback } from "react";
import {
  getCategories,
  getSampleUsers,
  searchItems,
  predict,
  submitFeedback,
  requestMissingItem,
  type Category,
  type SampleUser,
  type Item,
  type Prediction,
} from "@/lib/api";

type Mode = "sample" | "custom";

export default function DemoClient() {
  const [categories, setCategories] = useState<Category[]>([]);
  const [category, setCategory] = useState<string>("");
  const [mode, setMode] = useState<Mode>("sample");

  const [sampleUsers, setSampleUsers] = useState<SampleUser[]>([]);
  const [selectedSample, setSelectedSample] = useState<SampleUser | null>(null);
  const [loadingSampleUsers, setLoadingSampleUsers] = useState(false);
  const [categoriesLoading, setCategoriesLoading] = useState(true);

  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<Item[]>([]);
  const [customSequence, setCustomSequence] = useState<Item[]>([]);
  const [searching, setSearching] = useState(false);

  const [predictions, setPredictions] = useState<Prediction[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [warmingUp, setWarmingUp] = useState(false);
  const [feedbackGiven, setFeedbackGiven] = useState<Set<string>>(new Set());
  const [requestUrl, setRequestUrl] = useState("");
  const [requestStatus, setRequestStatus] = useState<string | null>(null);
  const [searchedAtLeastOnce, setSearchedAtLeastOnce] = useState(false);

  // Load categories once
  useEffect(() => {
    getCategories()
      .then((cats) => {
        setCategories(cats);
        if (cats.length > 0) setCategory(cats[0].key);
      })
      .catch(() => setError("Could not reach the backend. It may still be waking up."))
      .finally(() => setCategoriesLoading(false));
  }, []);

  // Load sample users whenever category changes
  useEffect(() => {
    if (!category) return;
    setSelectedSample(null);
    setPredictions(null);
    setCustomSequence([]);
    setLoadingSampleUsers(true);
    getSampleUsers(category)
      .then(setSampleUsers)
      .catch(() => setSampleUsers([]))
      .finally(() => setLoadingSampleUsers(false));
  }, [category]);

  // Debounced search
  useEffect(() => {
    if (mode !== "custom" || !searchQuery.trim()) {
      setSearchResults([]);
      setSearchedAtLeastOnce(false);
      return;
    }
    setSearching(true);
    const handle = setTimeout(() => {
      searchItems(category, searchQuery)
        .then((results) => {
          setSearchResults(results);
          setSearchedAtLeastOnce(true);
        })
        .catch(() => setSearchResults([]))
        .finally(() => setSearching(false));
    }, 300);
    return () => clearTimeout(handle);
  }, [searchQuery, category, mode]);

  const currentSequence: Item[] =
    mode === "sample" ? selectedSample?.history ?? [] : customSequence;

  const runPrediction = useCallback(async () => {
    if (currentSequence.length === 0) return;
    setLoading(true);
    setError(null);
    setPredictions(null);
    setFeedbackGiven(new Set());

    const warmupTimer = setTimeout(() => setWarmingUp(true), 4000);

    try {
      const asins = currentSequence.map((item) => item.asin);
      const results = await predict(category, asins, mode);
      setPredictions(results);
    } catch {
      setError(
        "Couldn't get predictions right now. The backend may be waking up from idle. Try again in a moment."
      );
    } finally {
      clearTimeout(warmupTimer);
      setWarmingUp(false);
      setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [category, currentSequence]);

  function addToCustomSequence(item: Item) {
    if (customSequence.find((i) => i.asin === item.asin)) return;
    setCustomSequence((prev) => [...prev, item]);
    setSearchQuery("");
    setSearchResults([]);
  }

  function removeFromCustomSequence(asin: string) {
    setCustomSequence((prev) => prev.filter((i) => i.asin !== asin));
  }

  function handleFeedback(asin: string) {
    if (feedbackGiven.has(asin)) return; // no duplicate logging for the same item
    setFeedbackGiven((prev) => new Set(prev).add(asin));
    const asins = currentSequence.map((item) => item.asin);
    submitFeedback(category, asins, asin);
  }

  async function handleRequestItem() {
    if (!requestUrl.trim()) return;
    setRequestStatus("Submitting...");
    try {
      const result = await requestMissingItem(category, requestUrl.trim());
      setRequestStatus(
        result.extracted_asin
          ? "Added to the review queue. Thanks for the link."
          : "Submitted, though the product ID couldn't be read from that link. A reviewer can still check it manually."
      );
      setRequestUrl("");
    } catch {
      setRequestStatus("Couldn't submit that right now. Try again in a moment.");
    }
  }

  const trueNextWasPredicted =
    mode === "sample" &&
    selectedSample &&
    predictions?.some((p) => p.asin === selectedSample.true_next.asin);

  return (
    <div className="demo">
      {categoriesLoading && (
        <p className="cold-start-banner">
          Connecting to the model server. This app runs on a free tier that
          sleeps when idle, so the first load can take up to a minute.
        </p>
      )}

      {/* Category selector */}
      <div className="demo-categories">
        {categories.map((cat) => (
          <button
            key={cat.key}
            className={`category-pill ${category === cat.key ? "category-pill-active" : ""}`}
            onClick={() => setCategory(cat.key)}
          >
            {cat.display_name}
          </button>
        ))}
      </div>

      {/* Mode toggle */}
      <div className="demo-mode-toggle">
        <button
          className={mode === "sample" ? "mode-btn mode-btn-active" : "mode-btn"}
          onClick={() => setMode("sample")}
        >
          Pick a real shopper
        </button>
        <button
          className={mode === "custom" ? "mode-btn mode-btn-active" : "mode-btn"}
          onClick={() => setMode("custom")}
        >
          Build your own history
        </button>
      </div>

      <div className="demo-body">
        {/* Left: input builder */}
        <div className="demo-input-panel">
          {mode === "sample" ? (
            <>
              <h3>Choose a shopper</h3>
              {loadingSampleUsers ? (
                <p className="empty-state">
                  Waking up the model server, this can take up to a minute on
                  the first request after idle.
                </p>
              ) : sampleUsers.length === 0 ? (
                <p className="empty-state">
                  Couldn&apos;t load shoppers. Try switching categories or
                  reloading the page.
                </p>
              ) : (
                <div className="sample-user-grid">
                  {sampleUsers.map((su) => (
                    <button
                      key={su.sample_id}
                      className={`sample-user-card ${
                        selectedSample?.sample_id === su.sample_id ? "sample-user-card-active" : ""
                      }`}
                      onClick={() => {
                        setSelectedSample(su);
                        setPredictions(null);
                      }}
                    >
                      <span className="sample-user-count">{su.history.length} items</span>
                      <span className="sample-user-preview" title={su.history.map((h) => h.title).join(", ")}>
                        {su.history
                          .slice(0, 2)
                          .map((h) => h.title)
                          .join(", ")}
                        {su.history.length > 2 ? "..." : ""}
                      </span>
                    </button>
                  ))}
                </div>
              )}
            </>
          ) : (
            <>
              <h3>Search for items</h3>
              <input
                type="text"
                className="search-input"
                placeholder="e.g. coffee, controller, guitar strings"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
              {searching && <p className="search-hint">Searching...</p>}
              {searchResults.length > 0 && (
                <div className="search-results">
                  {searchResults.map((item) => (
                    <button
                      key={item.asin}
                      className="search-result-item"
                      onClick={() => addToCustomSequence(item)}
                    >
                      {item.title}
                    </button>
                  ))}
                </div>
              )}
              {searchedAtLeastOnce && !searching && searchResults.length === 0 && (
                <div className="request-item-box">
                  <p className="request-item-prompt">
                    Couldn&apos;t find that item. If you have a link to it on
                    Amazon, you can submit it for review.
                  </p>
                  <div className="request-item-form">
                    <input
                      type="text"
                      className="search-input"
                      placeholder="Paste an Amazon product link"
                      value={requestUrl}
                      onChange={(e) => setRequestUrl(e.target.value)}
                    />
                    <button
                      className="btn btn-ghost request-item-submit"
                      onClick={handleRequestItem}
                      disabled={!requestUrl.trim()}
                    >
                      Submit
                    </button>
                  </div>
                  {requestStatus && <p className="request-item-status">{requestStatus}</p>}
                </div>
              )}
            </>
          )}

          <button
            className="btn btn-primary predict-btn"
            onClick={runPrediction}
            disabled={currentSequence.length === 0 || loading}
          >
            {loading ? "Predicting..." : "Predict next item"}
          </button>

          {warmingUp && (
            <p className="warmup-note">
              The model server is waking up from idle. This can take up to a
              minute on the first request.
            </p>
          )}
          {error && <p className="error-note">{error}</p>}

          {/* Current sequence display */}
          <div className="current-sequence">
            <h4>
              {mode === "sample" ? "Purchase history" : "Your sequence"}
              {currentSequence.length > 0 && ` (${currentSequence.length})`}
            </h4>
            {currentSequence.length === 0 ? (
              <p className="empty-state">
                {mode === "sample"
                  ? "Select a shopper above to see their history."
                  : "Search and add items to build a sequence."}
              </p>
            ) : (
              <ol className="sequence-list">
                {currentSequence.map((item) => (
                  <li key={item.asin}>
                    <span className="truncate-2" title={item.title}>{item.title}</span>
                    {mode === "custom" && (
                      <button
                        className="remove-btn"
                        onClick={() => removeFromCustomSequence(item.asin)}
                        aria-label={`Remove ${item.title}`}
                      >
                        &times;
                      </button>
                    )}
                  </li>
                ))}
              </ol>
            )}
          </div>
        </div>

        {/* Right: results */}
        <div className="demo-results-panel">
          <h3>Top 10 predictions</h3>
          {!predictions && !loading && (
            <p className="empty-state">Predictions will appear here.</p>
          )}
          {loading && <p className="empty-state">Running the model...</p>}
          {predictions && (
            <>
              {mode === "sample" && selectedSample && (
                <p className={`true-next-note ${trueNextWasPredicted ? "true-next-hit" : ""}`}>
                  {trueNextWasPredicted
                    ? `The model's predictions included the real next purchase: "${selectedSample.true_next.title}"`
                    : `The real next purchase was "${selectedSample.true_next.title}", not in the top 10 this time.`}
                </p>
              )}
              {mode === "custom" && feedbackGiven.size === 0 && predictions.length > 0 && (
                <p className="feedback-prompt">
                  Which of these would you actually buy next? Pick as many as
                  apply, your choices help improve future versions of this
                  model.
                </p>
              )}
              <ol className="predictions-list">
                {predictions.map((p, i) => (
                  <li key={p.asin} className="prediction-item">
                    <span className="prediction-rank">{i + 1}</span>
                    <span className="prediction-title truncate-2" title={p.title}>{p.title}</span>
                    {mode === "custom" && (
                      <button
                        className={`feedback-btn ${feedbackGiven.has(p.asin) ? "feedback-btn-chosen" : ""}`}
                        onClick={() => handleFeedback(p.asin)}
                        disabled={feedbackGiven.has(p.asin)}
                      >
                        {feedbackGiven.has(p.asin) ? "Noted" : "I'd buy this"}
                      </button>
                    )}
                  </li>
                ))}
              </ol>
              {mode === "custom" && feedbackGiven.size > 0 && (
                <p className="feedback-thanks">
                  Thanks, that helps build a dataset for future retraining.
                </p>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
