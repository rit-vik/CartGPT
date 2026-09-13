import DemoClient from "@/components/DemoClient";

export const metadata = {
  title: "Live Demo | CartGPT",
  description:
    "Pick a real shopper or build your own purchase history, and see live predictions from a transformer trained from scratch on Amazon purchase sequences.",
};

export default function DemoPage() {
  return (
    <div className="demo-page">
      <div className="demo-page-header">
        <h1>Try it yourself</h1>
        <p>
          Pick a real shopper from the dataset, or build a custom purchase
          history, and see what the model predicts they will buy next.
        </p>
      </div>
      <DemoClient />
    </div>
  );
}
