import SequenceHero from "@/components/SequenceHero";
import FindingsChart from "@/components/FindingsChart";
import Link from "next/link";

export default function Home() {
  return (
    <div className="home">
      {/* ---- Hero ---- */}
      <section className="hero">
        <div className="hero-text">
          <h1>
            A transformer that predicts your next purchase.
          </h1>
          <p className="hero-sub">
            Built from scratch: tokenization, self-attention, training loop,
            trained on 30M+ Amazon purchase sequences across four categories.
            It beat a popularity baseline by 63% on Video Games. On Musical
            Instruments, it added nothing at all.
          </p>
          <div className="hero-ctas">
            <Link href="/demo" className="btn btn-primary">
              Try the live demo
            </Link>
            <Link href="/methodology" className="btn btn-ghost">
              How it works
            </Link>
          </div>
        </div>
        <SequenceHero />
      </section>

      {/* ---- Findings ---- */}
      <section className="findings">
        <h2>The result nobody predicted</h2>
        <p className="section-intro">
          The obvious hypothesis was that repeat-purchase categories, like
          groceries, would benefit most from sequence modeling. That&apos;s
          only partly true.
        </p>

        <FindingsChart />

        <div className="finding-callout">
          <p>
            <strong>Video Games</strong> had the lowest repeat-purchase
            density of all four categories, yet the model beat a plain
            popularity baseline by <strong>63%</strong>, the largest margin
            in the study. <strong>Musical Instruments</strong> had similar
            density, and the model added <strong>nothing</strong> over
            popularity.
          </p>
          <p>
            Repeat-purchase density predicts some of the variation, but not
            all of it. What seems to matter more is whether a category has a
            learnable <em>interest trajectory</em>: genres, franchises, and
            sequels create a discoverable next-step pattern that grocery
            reordering and instrument accessories don&apos;t.
          </p>
        </div>
      </section>
    </div>
  );
}
