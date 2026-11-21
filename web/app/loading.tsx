export default function Loading() {
  return (
    <div className="space-y-4" role="status" aria-live="polite">
      <div className="h-3 w-32 animate-pulse rounded bg-ink-200 dark:bg-white/10" />
      <div className="h-8 w-2/3 animate-pulse rounded bg-ink-200 dark:bg-white/10" />
      <div className="h-4 w-1/2 animate-pulse rounded bg-ink-200 dark:bg-white/10" />
      <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[0, 1, 2, 3].map((key) => (
          <div
            key={key}
            className="h-24 animate-pulse rounded-xl border border-ink-200/70 bg-white/60 dark:border-white/10 dark:bg-white/5"
          />
        ))}
      </div>
      <span className="sr-only">Loading dashboard data…</span>
    </div>
  );
}
