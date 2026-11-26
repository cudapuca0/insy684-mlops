"use client";

import { Bar, BarChart, CartesianGrid, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Model } from "@/lib/data";
import { axisTick, chart, legendProps, tooltipStyle } from "@/lib/chartTheme";

/** Predicted probabilities for bookings that were kept vs canceled (test period). */
export function ProbHistogram({ data, threshold }: { data: Model["prob_hist"]; threshold: number }) {
  return (
    <div className="h-64 w-full">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid vertical={false} stroke={chart.grid} />
          <XAxis dataKey="p" tick={axisTick} tickLine={false} axisLine={false} interval={4} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} />
          <YAxis tick={axisTick} tickLine={false} axisLine={false} width={44} tickFormatter={(v: number) => (v >= 1000 ? `${v / 1000}k` : String(v))} />
          <Tooltip {...tooltipStyle} cursor={{ fill: "rgba(255,255,255,0.04)" }}
            labelFormatter={(v) => `Predicted ${Math.round(Number(v) * 100)}–${Math.round(Number(v) * 100) + 5}%`} formatter={(v, n) => [Number(v).toLocaleString("en-US"), n]} />
          <Legend {...legendProps} formatter={(v) => <span style={{ color: chart.axisText }}>{v}</span>} />
          <ReferenceLine x={threshold} stroke="#ffffff" strokeOpacity={0.6} strokeDasharray="4 4" />
          <Bar dataKey="kept" name="Kept" stackId="a" fill={chart.kept} stroke={chart.surface} strokeWidth={2} maxBarSize={24} />
          <Bar dataKey="canceled" name="Canceled" stackId="a" fill={chart.cancel} stroke={chart.surface} strokeWidth={2} radius={[4, 4, 0, 0]} maxBarSize={24} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
