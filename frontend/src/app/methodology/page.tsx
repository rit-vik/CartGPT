import ArchitectureDiagram from "@/components/ArchitectureDiagram";
import DebugTimeline from "@/components/DebugTimeline";

export const metadata = {
  title: "Methodology | CartGPT",
  description:
    "How the sequence recommender was built: data pipeline, transformer architecture, training, and the real bugs hit along the way.",
};

export default function Methodology() {
  return (
    <div className="method-page">
      <header className="method-header">
        <h1>How this was built</h1>
        <p>
          Every part of this, tokenization, embeddings, attention, the
          training loop, was written from scratch in PyTorch. No pretrained
          weights, no fine-tuning. This page walks through the pipeline,
          including the parts that broke along the way.
        </p>
      </header>

      <section className="method-section">
        <h2>1. Data</h2>
        <p>
          The source is the McAuley Lab's Amazon Reviews 2023 dataset,
          split across four categories: Grocery and Gourmet Food, Video
          Games, Musical Instruments, and Office Products. Each review
          record contributes a <code>user_id</code>, <code>parent_asin</code>{" "}
          (item), and <code>timestamp</code>.
        </p>
        <p>
          For each user, interactions are sorted by timestamp into a
          sequence, the same way a sentence is an ordered sequence of words.
          Users and items with fewer than 5 interactions are filtered out,
          since a sequence needs enough history to actually contain learnable
          structure.
        </p>
        <div className="method-stat-row">
          <div className="method-stat">
            <span className="method-stat-value">14.3M</span>
            <span className="method-stat-label">raw Grocery reviews</span>
          </div>
          <div className="method-stat">
            <span className="method-stat-value">493K</span>
            <span className="method-stat-label">usable Grocery sequences</span>
          </div>
          <div className="method-stat">
            <span className="method-stat-value">4</span>
            <span className="method-stat-label">categories compared</span>
          </div>
        </div>
      </section>

      <section className="method-section">
        <h2>2. Architecture</h2>
        <p>
          A decoder-only transformer, structurally the same family as GPT,
          applied to item IDs instead of subword tokens. Each product in the
          catalog is a vocabulary entry. A causal mask ensures the model only
          ever attends to earlier purchases when predicting the next one.
        </p>
        <ArchitectureDiagram />
        <p className="method-caption">
          Item embeddings and positional embeddings are summed, passed
          through 2 transformer blocks (causal self-attention plus
          feed-forward, each with residual connections and layer norm), then
          projected back to vocabulary space to produce a prediction at
          every position.
        </p>
      </section>

      <section className="method-section">
        <h2>3. Training</h2>
        <p>
          The objective is next-item prediction: given everything a user
          bought before, predict what they bought next. Optimized with
          AdamW, a warmup-then-decay learning rate schedule, and gradient
          clipping, the same recipe used for training small language models.
        </p>
        <p>
          One deliberate design choice: the output layer shares weights with
          the input embedding table (weight tying). This roughly halves
          parameter count and, more importantly, made the negative sampling
          approach below practical.
        </p>
      </section>

      <section className="method-section">
        <h2>4. What actually went wrong</h2>
        <p>
          Two real failures shaped the final design more than any
          architecture choice did. Both are worth walking through, since
          they explain design decisions that would otherwise look arbitrary.
        </p>
        <DebugTimeline />
      </section>

      <section className="method-section">
        <h2>5. Evaluation</h2>
        <p>
          Standard leave-one-out protocol from the sequential recommendation
          literature: for each user, the last item in their history is held
          out as the test target, the second-to-last as the validation
          target, and everything before that is visible to the model as
          input.
        </p>
        <p>
          The metric is Hit Rate@10, does the true next item land in the
          model's top 10 predictions, and NDCG@10, which additionally
          rewards ranking the correct item higher within that top 10. Every
          result is compared against a popularity baseline (always
          recommend the 10 most popular items) to check whether the model
          learned anything beyond what raw popularity already explains.
        </p>
      </section>

      <section className="method-section">
        <h2>6. Limitations</h2>
        <ul className="method-list">
          <li>
            Vocabulary size varies substantially across categories (55K to
            246K items). A smaller catalog is a mechanically easier ranking
            problem, independent of how much sequential structure exists,
            which is a potential confound in the cross-category comparison.
          </li>
          <li>
            The source dataset has occasional category mislabeling, some
            items filed under Video Games are, on inspection, books or
            unrelated media. This is a property of the underlying data, not
            the model.
          </li>
          <li>
            Negative sampling uses a fixed set of random negatives per batch
            rather than per-example hard negatives, which caps how finely
            the model can learn to discriminate between very similar items.
          </li>
        </ul>
      </section>
    </div>
  );
}
