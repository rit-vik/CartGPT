export const metadata = {
  title: "Terms & Conditions | CartGPT",
};

export default function Terms() {
  return (
    <div className="legal-page">
      <h1>Terms &amp; Conditions</h1>
      <p className="legal-updated">Last updated: September 2026</p>

      <p>
        This site is a personal research and portfolio project demonstrating
        a sequence-based recommendation model trained from scratch. By using
        this site, you agree to the following terms.
      </p>

      <h2>Not affiliated with Amazon</h2>
      <p>
        This project is independent and is not affiliated with, endorsed by,
        sponsored by, or in any way officially connected to Amazon.com, Inc.
        or any of its subsidiaries or affiliates. Product titles, images, and
        identifiers shown on this site originate from a public academic
        dataset and are used solely for research and demonstration purposes.
      </p>

      <h2>No purchasing functionality</h2>
      <p>
        This site does not sell products, process payments, or fulfill
        orders of any kind. Predictions shown in the live demo are model
        outputs generated for illustrative purposes and do not constitute
        purchase recommendations, financial advice, or an endorsement of any
        product.
      </p>

      <h2>Accuracy of predictions</h2>
      <p>
        The model powering this demo is a research prototype. Predictions
        are provided &quot;as is&quot; without any guarantee of accuracy,
        completeness, or fitness for any particular purpose. Results shown in
        the Findings section reflect evaluation on held-out historical data
        and are not a guarantee of future performance.
      </p>

      <h2>Use of this site</h2>
      <p>You agree not to use this site to:</p>
      <ul>
        <li>Attempt to disrupt or overload the backend service</li>
        <li>Scrape or bulk-extract data beyond normal browsing use</li>
        <li>Misrepresent the outputs of this project as your own work</li>
      </ul>

      <h2>Intellectual property</h2>
      <p>
        The code, design, and written analysis on this site are the
        author&apos;s own work unless otherwise noted, and are shared for
        portfolio purposes. The underlying dataset is subject to its own
        original license terms from the McAuley Lab.
      </p>

      <h2>Changes</h2>
      <p>
        These terms may be updated from time to time as the project evolves.
        Continued use of the site after changes constitutes acceptance of
        the updated terms.
      </p>
    </div>
  );
}
