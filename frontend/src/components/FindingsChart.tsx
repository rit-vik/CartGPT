"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

const data = [
  { category: "Grocery", baseline: 0.0099, model: 0.0135 },
  { category: "Video Games", baseline: 0.0215, model: 0.0351 },
  { category: "Musical Instruments", baseline: 0.0226, model: 0.0225 },
  { category: "Office Products", baseline: 0.0078, model: 0.009 },
];

export default function FindingsChart() {
  return (
    <div className="findings-chart">
      <ResponsiveContainer width="100%" height={360}>
        <BarChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" vertical={false} />
          <XAxis
            dataKey="category"
            tick={{ fontFamily: "var(--font-body)", fontSize: 13, fill: "#1a1a1a" }}
            axisLine={{ stroke: "#1a1a1a", opacity: 0.2 }}
            tickLine={false}
          />
          <YAxis
            tick={{ fontFamily: "var(--font-body)", fontSize: 12, fill: "#1a1a1a" }}
            axisLine={false}
            tickLine={false}
            width={50}
          />
          <Tooltip
            formatter={(value: number) => value.toFixed(4)}
            contentStyle={{
              fontFamily: "var(--font-body)",
              border: "1px solid #1a1a1a20",
              borderRadius: 8,
            }}
          />
          <Legend wrapperStyle={{ fontFamily: "var(--font-body)", fontSize: 13 }} />
          <Bar dataKey="baseline" name="Popularity baseline" fill="#d4d4d4" radius={[4, 4, 0, 0]} />
          <Bar dataKey="model" name="Trained model" fill="#ffcc00" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
      <p className="chart-caption">Hit Rate@10 on held-out purchases, by category</p>
    </div>
  );
}
