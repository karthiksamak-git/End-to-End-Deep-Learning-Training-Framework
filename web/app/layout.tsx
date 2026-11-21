import type { Metadata } from "next";
import type { ReactNode } from "react";

import { Nav } from "@/components/nav";

import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "dl-train-framework — experiment dashboard",
    template: "%s · dl-train-framework",
  },
  description:
    "Results from an end-to-end PyTorch training framework: training curves, cross-validation, hyper-parameter search, error analysis and in-browser ONNX inference.",
};

const themeScript = `
try {
  var stored = localStorage.getItem('theme');
  var dark = stored ? stored === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
  if (dark) document.documentElement.classList.add('dark');
} catch (e) {}
`;

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body className="flex min-h-screen flex-col">
        <Nav />
        <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8 sm:px-6 sm:py-10">
          {children}
        </main>
        <footer className="border-t border-ink-200/70 py-6 text-center text-xs text-ink-700/60 dark:border-white/10 dark:text-ink-100/50">
          Trained locally with PyTorch · results are real measured runs, not placeholders ·
          inference runs entirely in your browser with onnxruntime-web
        </footer>
      </body>
    </html>
  );
}
