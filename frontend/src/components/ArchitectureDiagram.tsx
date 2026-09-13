const stack = [
  { label: "Output projection", detail: "tied to item embeddings, scores next item" },
  { label: "Transformer block x2", detail: "causal self-attention + feed-forward" },
  { label: "Item + positional embeddings", detail: "summed, dropout applied" },
  { label: "Input sequence", detail: "item IDs, most recent purchases" },
];

export default function ArchitectureDiagram() {
  return (
    <div className="arch-diagram" aria-hidden="true">
      {stack.map((layer, i) => (
        <div key={i} className="arch-layer">
          <div className="arch-layer-label">{layer.label}</div>
          <div className="arch-layer-detail">{layer.detail}</div>
        </div>
      ))}
    </div>
  );
}
