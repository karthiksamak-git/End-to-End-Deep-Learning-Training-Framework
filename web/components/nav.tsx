"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { ThemeToggle } from "@/components/theme-toggle";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/experiments", label: "Experiments" },
  { href: "/cross-validation", label: "Cross-validation" },
  { href: "/hpo", label: "HPO" },
  { href: "/error-analysis", label: "Error analysis" },
  { href: "/playground", label: "Playground" },
];

export function Nav() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-40 border-b border-ink-200/70 bg-ink-50/85 backdrop-blur-md dark:border-white/10 dark:bg-ink-950/80">
      <div className="mx-auto flex max-w-6xl flex-col gap-3 px-4 py-3 sm:px-6">
        <div className="flex items-center justify-between gap-4">
          <Link href="/" className="flex items-center gap-2.5">
            <span className="grid h-8 w-8 place-items-center rounded-lg bg-indigo-600 text-sm font-bold text-white">
              dl
            </span>
            <span className="text-sm font-semibold tracking-tight sm:text-base">
              dl-train-framework
            </span>
          </Link>
          <ThemeToggle />
        </div>
        <nav className="flex gap-1 overflow-x-auto pb-0.5 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          {LINKS.map((link) => {
            const active = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`whitespace-nowrap rounded-lg px-3 py-1.5 text-sm transition ${
                  active
                    ? "bg-indigo-600 text-white"
                    : "text-ink-700/80 hover:bg-ink-100 dark:text-ink-100/70 dark:hover:bg-white/10"
                }`}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
