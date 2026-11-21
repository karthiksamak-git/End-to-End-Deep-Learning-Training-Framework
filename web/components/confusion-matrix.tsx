"use client";

import { useMemo, useState } from "react";

export function ConfusionMatrix({
  labels,
  matrix,
}: {
  labels: string[];
  matrix: number[][];
}) {
  const [selected, setSelected] = useState<{ row: number; col: number } | null>(null);
  const max = useMemo(
    () => Math.max(1, ...matrix.flat()),
    [matrix],
  );
  const rowTotals = useMemo(
    () => matrix.map((row) => row.reduce((sum, value) => sum + value, 0)),
    [matrix],
  );

  const active = selected ? matrix[selected.row][selected.col] : null;
  const activeTrue = selected ? rowTotals[selected.row] : 0;

  return (
    <div>
      <div className="overflow-x-auto">
        <table className="border-separate border-spacing-1 text-xs">
          <thead>
            <tr>
              <th className="w-24" />
              {labels.map((label) => (
                <th
                  key={label}
                  className="h-16 min-w-[48px] align-bottom text-[10px] font-medium text-ink-700/70 dark:text-ink-100/60"
                >
                  <span className="block origin-bottom-left rotate-[-55deg] whitespace-nowrap">
                    {label}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, rowIndex) => (
              <tr key={labels[rowIndex]}>
                <th className="whitespace-nowrap pr-2 text-right text-[10px] font-medium text-ink-700/70 dark:text-ink-100/60">
                  {labels[rowIndex]}
                </th>
                {row.map((value, colIndex) => {
                  const intensity = value / max;
                  const isSelected =
                    selected?.row === rowIndex && selected?.col === colIndex;
                  return (
                    <td key={`${rowIndex}-${colIndex}`} className="p-0">
                      <button
                        type="button"
                        onClick={() => setSelected({ row: rowIndex, col: colIndex })}
                        aria-label={`True ${labels[rowIndex]}, predicted ${labels[colIndex]}: ${value} samples`}
                        className={`h-10 w-12 rounded text-[11px] font-medium tabular-nums transition ${
                          isSelected ? "ring-2 ring-indigo-500 ring-offset-1" : ""
                        } ${intensity > 0.45 ? "text-white" : "text-ink-900 dark:text-white"}`}
                        style={{
                          backgroundColor:
                            value === 0
                              ? "rgba(148,163,184,0.12)"
                              : `rgba(79,70,229,${(0.12 + intensity * 0.85).toFixed(3)})`,
                        }}
                      >
                        {value || ""}
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted mt-3">
        {selected && active !== null ? (
          <>
            <span className="font-medium text-ink-900 dark:text-ink-100">
              {labels[selected.row]} → {labels[selected.col]}
            </span>
            : {active} of {activeTrue} samples ({((active / Math.max(1, activeTrue)) * 100).toFixed(1)}%
            of that row)
          </>
        ) : (
          "Click any cell to inspect counts. Rows are the true class, columns the prediction."
        )}
      </p>
    </div>
  );
}
