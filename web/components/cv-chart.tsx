"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { CvFold } from "@/lib/types";

const tooltipStyle = {
  backgroundColor: "#0f172a",
  border: "1px solid #334155",
  borderRadius: 8,
  color: "#e2e8f0",
  fontSize: 12,
};

export function CvChart({ folds }: { folds: CvFold[] }) {
  const data = folds.map((fold) => ({
    name: `fold ${fold.fold + 1}`,
    accuracy: Number(fold.accuracy.toFixed(4)),
    f1: Number(fold.f1_macro.toFixed(4)),
    auc: Number(fold.roc_auc.toFixed(4)),
  }));

  return (
    <div className="h-80 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: -16 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#94a3b8" strokeOpacity={0.25} />
          <XAxis dataKey="name" stroke="#94a3b8" tick={{ fontSize: 12 }} />
          <YAxis domain={[0.8, 1]} stroke="#94a3b8" tick={{ fontSize: 12 }} />
          <Tooltip contentStyle={tooltipStyle} formatter={(value: number) => value.toFixed(4)} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Bar dataKey="accuracy" name="accuracy" fill="#6366f1" radius={[4, 4, 0, 0]} />
          <Bar dataKey="f1" name="macro F1" fill="#10b981" radius={[4, 4, 0, 0]} />
          <Bar dataKey="auc" name="ROC-AUC" fill="#f59e0b" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
