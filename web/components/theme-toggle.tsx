"use client";

import { useEffect, useState } from "react";

type Theme = "light" | "dark";

function applyTheme(theme: Theme) {
  const root = document.documentElement;
  root.classList.toggle("dark", theme === "dark");
  try {
    window.localStorage.setItem("theme", theme);
  } catch {
    // Private-mode browsers can reject storage; the class change still applies.
  }
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("light");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const stored = window.localStorage.getItem("theme");
    const initial: Theme =
      stored === "light" || stored === "dark"
        ? stored
        : window.matchMedia("(prefers-color-scheme: dark)").matches
          ? "dark"
          : "light";
    setTheme(initial);
    setMounted(true);
  }, []);

  const toggle = () => {
    const next: Theme = theme === "dark" ? "light" : "dark";
    setTheme(next);
    applyTheme(next);
  };

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label="Toggle colour theme"
      className="flex items-center gap-2 rounded-lg border border-ink-200/80 px-3 py-1.5 text-xs font-medium transition hover:bg-ink-100 dark:border-white/10 dark:hover:bg-white/10"
    >
      <span aria-hidden="true">{mounted && theme === "dark" ? "🌙" : "☀️"}</span>
      {mounted ? (theme === "dark" ? "Dark" : "Light") : "Theme"}
    </button>
  );
}
