"use client";

import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Model } from "@/lib/data";
import { axisTick, chart, tooltipStyle } from "@/lib/chartTheme";

/** Predicted vs actual cancellation rate, in 10 equal-size groups. On the diagonal = well calibrated. */
export function CalibrationChart({ data }: { data: Model["calibration"] }) {
  return (
    <div className="h-64 w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 18, left: 0 }}>
          <CartesianGrid stroke={chart.grid} />
          <XAxis dataKey="predicted" type="number" domain={[0, 1]} ticks={[0, 0.5, 1]} tick={axisTick} tickLine={false} axisLine={false}
            tickFormatter={(v: number) => `${v * 100}%`} label={{ value: "Predicted chance of cancelling", position: "insideBottom", dy: 16, fill: chart.axisText, fontSize: 11 }} />
          <YAxis type="number" domain={[0, 1]} ticks={[0, 0.5, 1]} tick={axisTick} tickLine={false} axisLine={false} width={40} tickFormatter={(v: number) => `${v * 100}%`} />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={chart.context} strokeDasharray="4 4" />
          <Tooltip {...tooltipStyle} labelFormatter={(v) => `Predicted ${Math.round(Number(v) * 100)}%`} formatter={(v) => [`${Math.round(Number(v) * 100)}%`, "Actually canceled"]} />
          <Line dataKey="actual" stroke={chart.accent} strokeWidth={2} dot={{ r: 4, fill: chart.accent, stroke: chart.surface, strokeWidth: 2 }} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
