const history = [
  "Organic Turmeric Powder",
  "Frontier Co-op Ground Cinnamon",
  "Sprouted Organic Walnuts",
  "Raw Texas Pecans",
];

const predicted = "True Lime Black Cherry Limeade";

export default function SequenceHero() {
  return (
    <div className="sequence-hero" aria-hidden="true">
      <div className="sequence-row">
        {history.map((title, i) => (
          <div key={i} className="sequence-card">
            <span className="sequence-index">{i + 1}</span>
            <span className="sequence-title">{title}</span>
          </div>
        ))}
        <div className="sequence-arrow">&rarr;</div>
        <div className="sequence-card sequence-card-predicted">
          <span className="predicted-label">Model predicts</span>
          <span className="sequence-title">{predicted}</span>
        </div>
      </div>
      <p className="sequence-caption">
        An actual purchase sequence from the dataset. This was the model&apos;s
        #1 prediction, and the real next purchase.
      </p>
    </div>
  );
}
