"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

import type { HpoFile } from "@/lib/types";

const tooltipStyle = {
  backgroundColor: "#0f172a",
  border: "1px solid #334155",
  borderRadius: 8,
  color: "#e2e8f0",
  fontSize: 12,
};

export function TrialScatter({ hpo }: { hpo: HpoFile }) {
  const data = hpo.trials.map((trial) => ({
    trial: trial.number,
    value: trial.value ?? 0,
    state: trial.state,
    hasValue: trial.value !== null,
  }));
  const completed = data.filter((row) => row.hasValue && row.state === "complete");

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: -12 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#94a3b8" strokeOpacity={0.25} />
          <XAxis
            dataKey="trial"
            name="trial"
            stroke="#94a3b8"
            tick={{ fontSize: 12 }}
            allowDecimals={false}
          />
          <YAxis
            dataKey="value"
            name={hpo.metric}
            stroke="#94a3b8"
            tick={{ fontSize: 12 }}
            domain={["auto", "auto"]}
          />
          <ZAxis range={[80, 80]} />
          <Tooltip
            contentStyle={tooltipStyle}
            formatter={(value: number) => Number(value).toFixed(4)}
            labelFormatter={(label) => `trial ${label}`}
          />
          <Scatter name="completed" data={completed} fill="#6366f1" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ImportanceChart({ hpo }: { hpo: HpoFile }) {
  const data = Object.entries(hpo.param_importance)
    .map(([name, value]) => ({ name, value: Number(value.toFixed(4)) }))
    .sort((a, b) => b.value - a.value);

  if (data.length === 0) {
    return (
      <p className="muted">
        Not enough completed trials to estimate parameter importance (needs at least two).
      </p>
    );
  }

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 32, bottom: 4, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#94a3b8" strokeOpacity={0.25} />
          <XAxis type="number" stroke="#94a3b8" tick={{ fontSize: 12 }} />
          <YAxis
            type="category"
            dataKey="name"
            stroke="#94a3b8"
            tick={{ fontSize: 12 }}
            width={100}
          />
          <Tooltip contentStyle={tooltipStyle} formatter={(value: number) => value.toFixed(4)} />
          <Bar dataKey="value" name="importance" radius={[0, 4, 4, 0]}>
            {data.map((entry, index) => (
              <Cell key={entry.name} fill={index === 0 ? "#6366f1" : "#94a3b8"} />
            ))}
            <LabelList
              dataKey="value"
              position="right"
              formatter={(value: number) => value.toFixed(3)}
              style={{ fontSize: 11, fill: "#94a3b8" }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
