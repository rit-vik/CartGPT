export const metadata = {
  title: "Privacy Policy | CartGPT",
};

export default function PrivacyPolicy() {
  return (
    <div className="legal-page">
      <h1>Privacy Policy</h1>
      <p className="legal-updated">Last updated: September 2026</p>

      <p>
        This site is an independent research and portfolio project. It is not
        affiliated with, endorsed by, or connected to Amazon.com, Inc. in any
        way. Product data used in the live demo comes from a public academic
        dataset (the McAuley Lab Amazon Reviews 2023 collection) and is used
        here for demonstration purposes only.
      </p>

      <h2>What this site collects</h2>
      <p>This site does not require an account and does not collect:</p>
      <ul>
        <li>Names, email addresses, or any other personal identifiers</li>
        <li>Payment information</li>
        <li>Cookies used for advertising or cross-site tracking</li>
      </ul>

      <h2>The live demo</h2>
      <p>
        When you use the live demo, the item sequence you select or build is
        sent to a backend server to generate predictions. This request is
        processed to return a result and is not linked to any personal
        identity. Request logs may be retained briefly by the hosting
        provider for operational purposes (e.g. debugging, abuse prevention),
        consistent with standard server logging practices.
      </p>

      <h2>Hosting and infrastructure</h2>
      <p>
        This site is hosted using third-party infrastructure providers (for
        the frontend and backend respectively). These providers may collect
        standard technical information, such as IP address and browser type,
        as part of normal web server operation. Refer to your hosting
        provider&apos;s own privacy documentation for details on their data
        handling.
      </p>

      <h2>Changes to this policy</h2>
      <p>
        As this is a personal project, this policy may be updated
        occasionally to reflect changes to the site. Material changes will be
        reflected by updating the date above.
      </p>

      <h2>Contact</h2>
      <p>
        Questions about this site can be directed via the contact information
        on the author&apos;s GitHub profile.
      </p>
    </div>
  );
}
