import type { ReactNode } from "react";

export function Card({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`min-w-0 rounded-xl border border-ink-200/80 bg-white/85 p-5 shadow-sm backdrop-blur dark:border-white/10 dark:bg-white/[0.04] ${className}`}
    >
      {children}
    </div>
  );
}

export function SectionTitle({
  title,
  hint,
  action,
}: {
  title: string;
  hint?: string;
  action?: ReactNode;
}) {
  return (
    <div className="mb-4 flex flex-wrap items-end justify-between gap-2">
      <div>
        <h2 className="text-lg font-semibold tracking-tight">{title}</h2>
        {hint ? <p className="muted mt-1">{hint}</p> : null}
      </div>
      {action}
    </div>
  );
}

export function PageHeader({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description: string;
}) {
  return (
    <header className="mb-8">
      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-indigo-600 dark:text-indigo-400">
        {eyebrow}
      </p>
      <h1 className="mt-2 text-3xl font-bold tracking-tight sm:text-4xl">{title}</h1>
      <p className="mt-3 max-w-3xl text-sm leading-relaxed text-ink-700/80 dark:text-ink-100/70">
        {description}
      </p>
    </header>
  );
}

export function Stat({
  label,
  value,
  sub,
  accent = false,
}: {
  label: string;
  value: string;
  sub?: string;
  accent?: boolean;
}) {
  return (
    <Card className={accent ? "border-indigo-300/70 dark:border-indigo-400/30" : ""}>
      <p className="text-xs font-medium uppercase tracking-wide text-ink-700/60 dark:text-ink-100/50">
        {label}
      </p>
      <p className="mt-2 text-2xl font-semibold tabular-nums">{value}</p>
      {sub ? <p className="muted mt-1">{sub}</p> : null}
    </Card>
  );
}

export function Badge({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "good" | "warn" | "info";
}) {
  const tones: Record<string, string> = {
    neutral: "bg-ink-100 text-ink-700 dark:bg-white/10 dark:text-ink-100/80",
    good: "bg-emerald-100 text-emerald-800 dark:bg-emerald-400/15 dark:text-emerald-300",
    warn: "bg-amber-100 text-amber-800 dark:bg-amber-400/15 dark:text-amber-300",
    info: "bg-indigo-100 text-indigo-800 dark:bg-indigo-400/15 dark:text-indigo-300",
  };
  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium ${tones[tone]}`}
    >
      {children}
    </span>
  );
}

export function EmptyState({ title, detail }: { title: string; detail: string }) {
  return (
    <Card className="border-dashed">
      <p className="font-medium">{title}</p>
      <p className="muted mt-1">{detail}</p>
      <pre className="mt-4 overflow-x-auto rounded-lg bg-ink-100 p-3 text-xs dark:bg-black/30">
        make all
      </pre>
    </Card>
  );
}

export function DataTable({
  head,
  rows,
  caption,
  minWidth = 560,
}: {
  head: string[];
  rows: Array<Array<ReactNode>>;
  caption?: string;
  minWidth?: number;
}) {
  return (
    <div className="max-w-full overflow-x-auto">
      <table className="w-full border-collapse text-sm" style={{ minWidth }}>
        {caption ? <caption className="muted mb-2 text-left">{caption}</caption> : null}
        <thead>
          <tr className="border-b border-ink-200 text-left dark:border-white/10">
            {head.map((label) => (
              <th
                key={label}
                className="whitespace-nowrap px-2 py-2 text-xs font-semibold uppercase tracking-wide text-ink-700/60 dark:text-ink-100/50"
              >
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr
              key={rowIndex}
              className="border-b border-ink-100/80 last:border-0 dark:border-white/5"
            >
              {row.map((cell, cellIndex) => (
                <td key={cellIndex} className="whitespace-nowrap px-2 py-2 tabular-nums">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function formatPct(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatNumber(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

export function formatCount(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return value.toLocaleString("en-US");
}

export function formatSeconds(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  if (value < 60) return `${value.toFixed(1)} s`;
  const minutes = Math.floor(value / 60);
  return `${minutes} min ${Math.round(value % 60)} s`;
}
