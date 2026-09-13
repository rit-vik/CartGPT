const stages = [
  {
    number: "01",
    title: "Full-vocabulary softmax ran out of memory",
    problem:
      "Computing logits over a 246,452-item vocabulary for every position in every batch produced a tensor over 1.5 billion values, and crashed an 8GB GPU during the backward pass.",
    fix:
      "Switched to sampled softmax: score the true next item against a small set of sampled negatives instead of the entire catalog, the same technique word2vec uses for large vocabularies.",
  },
  {
    number: "02",
    title: "The model got worse the longer it trained",
    problem:
      "Validation Hit Rate@10 declined every epoch, from 1.01% down to 0.55%, while training loss kept improving. The final model performed worse than just recommending whatever was popular.",
    fix:
      "The negative samples were uniformly random, which made the training task too easy to distinguish real preferences from noise. Switching to popularity-weighted negative sampling forced the model to learn to outrank the items it would actually compete against.",
  },
  {
    number: "03",
    title: "Confirming the fix actually worked",
    problem:
      "A fix that reduces one number needs to be checked against the metric that actually matters, not just against the training loss.",
    fix:
      "After the change, Hit Rate@10 rose every single epoch instead of falling, and the final model beat the popularity baseline by 36% on held-out test data, the first result in the project that was clearly better than doing nothing clever.",
  },
];

export default function DebugTimeline() {
  return (
    <div className="debug-timeline">
      {stages.map((stage) => (
        <div key={stage.number} className="debug-stage">
          <div className="debug-number">{stage.number}</div>
          <div className="debug-content">
            <h3>{stage.title}</h3>
            <p>
              <span className="debug-tag debug-tag-problem">Problem</span>{" "}
              {stage.problem}
            </p>
            <p>
              <span className="debug-tag debug-tag-fix">Fix</span> {stage.fix}
            </p>
          </div>
        </div>
      ))}
    </div>
  );
}
