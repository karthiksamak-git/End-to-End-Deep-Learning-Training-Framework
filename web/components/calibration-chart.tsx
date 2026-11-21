"use client";

import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { ErrorAnalysisFile } from "@/lib/types";

const tooltipStyle = {
  backgroundColor: "#0f172a",
  border: "1px solid #334155",
  borderRadius: 8,
  color: "#e2e8f0",
  fontSize: 12,
};

export function CalibrationChart({
  bins,
}: {
  bins: ErrorAnalysisFile["calibration"]["bins"];
}) {
  const data = bins.map((bin) => ({
    confidence: Number(bin.avg_confidence.toFixed(4)),
    accuracy: Number(bin.accuracy.toFixed(4)),
    count: bin.count,
  }));

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: -8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#94a3b8" strokeOpacity={0.25} />
          <XAxis
            type="number"
            dataKey="confidence"
            name="confidence"
            domain={[0, 1]}
            stroke="#94a3b8"
            tick={{ fontSize: 12 }}
            label={{
              value: "mean confidence",
              position: "insideBottom",
              offset: -4,
              fontSize: 11,
              fill: "#94a3b8",
            }}
          />
          <YAxis
            type="number"
            dataKey="accuracy"
            name="accuracy"
            domain={[0, 1]}
            stroke="#94a3b8"
            tick={{ fontSize: 12 }}
          />
          <Tooltip
            contentStyle={tooltipStyle}
            formatter={(value: number) => value.toFixed(3)}
            labelFormatter={() => ""}
          />
          <ReferenceLine
            segment={[
              { x: 0, y: 0 },
              { x: 1, y: 1 },
            ]}
            stroke="#94a3b8"
            strokeDasharray="4 4"
          />
          <Scatter data={data} fill="#6366f1" name="bins" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
