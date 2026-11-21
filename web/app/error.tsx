"use client";

import { useEffect } from "react";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="rounded-xl border border-rose-300/60 bg-rose-50 p-6 text-sm text-rose-900 dark:border-rose-400/30 dark:bg-rose-500/10 dark:text-rose-100">
      <h1 className="text-lg font-semibold">Something broke while rendering this page</h1>
      <p className="mt-2 break-words">{error.message}</p>
      <button type="button" onClick={reset} className="btn-ghost mt-4">
        Try again
      </button>
    </div>
  );
}
