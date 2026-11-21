"use client";

import { useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { Run } from "@/lib/types";

const tooltipStyle = {
  backgroundColor: "#0f172a",
  border: "1px solid #334155",
  borderRadius: 8,
  color: "#e2e8f0",
  fontSize: 12,
};

function CurveChart({
  run,
  dataKeyA,
  dataKeyB,
  labelA,
  labelB,
  colorA,
  colorB,
  domain,
}: {
  run: Run;
  dataKeyA: string;
  dataKeyB: string;
  labelA: string;
  labelB: string;
  colorA: string;
  colorB: string;
  domain?: [number | "auto", number | "auto"];
}) {
  return (
    <div className="h-72 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={run.history} margin={{ top: 8, right: 12, bottom: 4, left: -12 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#94a3b8" strokeOpacity={0.25} />
          <XAxis
            dataKey="epoch"
            stroke="#94a3b8"
            tick={{ fontSize: 12 }}
            label={{ value: "epoch", position: "insideBottomRight", offset: -2, fontSize: 11 }}
          />
          <YAxis stroke="#94a3b8" tick={{ fontSize: 12 }} domain={domain ?? ["auto", "auto"]} />
          <Tooltip contentStyle={tooltipStyle} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Line
            type="monotone"
            dataKey={dataKeyA}
            name={labelA}
            stroke={colorA}
            strokeWidth={2}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey={dataKeyB}
            name={labelB}
            stroke={colorB}
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ExperimentExplorer({ runs }: { runs: Run[] }) {
  const [selectedId, setSelectedId] = useState(runs[0]?.id ?? "");
  const selected = runs.find((run) => run.id === selectedId) ?? runs[0];

  if (!selected) {
    return <p className="muted">No runs available.</p>;
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-2">
        {runs.map((run) => {
          const active = run.id === selected.id;
          return (
            <button
              key={run.id}
              type="button"
              onClick={() => setSelectedId(run.id)}
              className={`rounded-lg border px-3 py-1.5 text-xs transition ${
                active
                  ? "border-indigo-500 bg-indigo-600 text-white"
                  : "border-ink-200/80 bg-white/60 hover:bg-ink-100 dark:border-white/10 dark:bg-white/5 dark:hover:bg-white/10"
              }`}
            >
              {run.name}
            </button>
          );
        })}
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-xl border border-ink-200/80 bg-white/85 p-5 shadow-sm dark:border-white/10 dark:bg-white/[0.04]">
          <h3 className="mb-1 font-semibold">Loss</h3>
          <p className="muted mb-3">
            Cross-entropy per epoch · {selected.history.length} epochs ·{" "}
            {selected.device.device ?? "cpu"}
          </p>
          <CurveChart
            run={selected}
            dataKeyA="train_loss"
            dataKeyB="val_loss"
            labelA="train"
            labelB="val"
            colorA="#6366f1"
            colorB="#f59e0b"
          />
        </div>
        <div className="rounded-xl border border-ink-200/80 bg-white/85 p-5 shadow-sm dark:border-white/10 dark:bg-white/[0.04]">
          <h3 className="mb-1 font-semibold">Accuracy</h3>
          <p className="muted mb-3">
            Best val accuracy{" "}
            {selected.best_record
              ? `${(selected.best_record.val_accuracy * 100).toFixed(2)}%`
              : "—"}{" "}
            at epoch {selected.best.epoch ?? "—"}
          </p>
          <CurveChart
            run={selected}
            dataKeyA="train_accuracy"
            dataKeyB="val_accuracy"
            labelA="train"
            labelB="val"
            colorA="#10b981"
            colorB="#ef4444"
            domain={[0, 1]}
          />
        </div>
      </div>

      <div className="rounded-xl border border-ink-200/80 bg-white/85 p-5 shadow-sm dark:border-white/10 dark:bg-white/[0.04]">
        <h3 className="mb-3 font-semibold">Learning rate schedule</h3>
        <div className="h-48 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={selected.history} margin={{ top: 4, right: 12, bottom: 0, left: -4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#94a3b8" strokeOpacity={0.25} />
              <XAxis dataKey="epoch" stroke="#94a3b8" tick={{ fontSize: 12 }} />
              <YAxis
                stroke="#94a3b8"
                tick={{ fontSize: 12 }}
                tickFormatter={(value: number) => value.toExponential(1)}
              />
              <Tooltip contentStyle={tooltipStyle} />
              <Line
                type="monotone"
                dataKey="lr"
                name="learning rate"
                stroke="#8b5cf6"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
