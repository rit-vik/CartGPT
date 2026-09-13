import type { Metadata } from "next";
import { Fraunces, Inter } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const fraunces = Fraunces({
  subsets: ["latin"],
  variable: "--font-display",
  weight: ["400", "500", "600", "700"],
  style: ["normal", "italic"],
});

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-body",
});

export const metadata: Metadata = {
  title: "CartGPT: A Transformer That Predicts Your Next Purchase",
  description:
    "A GPT-style transformer trained from scratch on Amazon purchase sequences across four categories, and what it reveals about which kinds of shopping behavior are actually predictable.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${fraunces.variable} ${inter.variable}`}>
        <header className="site-header">
          <Link href="/" className="site-logo">
            CartGPT
          </Link>
          <nav className="site-nav">
            <Link href="/">Findings</Link>
            <Link href="/methodology">Methodology</Link>
            <Link href="/demo">Live Demo</Link>
          </nav>
        </header>
        <main>{children}</main>
        <footer className="site-footer">
          <p>
            Built from scratch: tokenization, attention, training loop, and all.{" "}
            <a href="https://github.com/rit-vik/CartGPT" target="_blank" rel="noopener noreferrer">
              View the code
            </a>
          </p>
          <p className="footer-legal">
            <Link href="/privacy">Privacy Policy</Link>
            <Link href="/terms">Terms &amp; Conditions</Link>
          </p>
        </footer>
      </body>
    </html>
  );
}
