"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { axisTick, chart, tooltipStyle } from "@/lib/chartTheme";

const KINDS = {
  lead: {
    x: "days",
    tick: (v: number) => (v % 90 === 0 ? `${v}d` : ""),
    label: (v: number) => (v >= 570 ? "570+ days ahead" : `${v}–${v + 29} days ahead`),
  },
  nights: {
    x: "nights",
    tick: (v: number) => (v % 2 === 0 ? (v === 14 ? "14+" : String(v)) : ""),
    label: (v: number) => (v === 14 ? "14+ nights" : `${v} night${v === 1 ? "" : "s"}`),
  },
};

/** Simple count histogram of lead times or stay lengths. */
export function Histogram({ data, kind }: { data: { n: number; [k: string]: number }[]; kind: keyof typeof KINDS }) {
  const { x, tick, label } = KINDS[kind];
  return (
    <div className="h-56 w-full">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }} barCategoryGap={2}>
          <CartesianGrid vertical={false} stroke={chart.grid} />
          <XAxis dataKey={x} tick={axisTick} tickLine={false} axisLine={false} tickFormatter={tick} interval={0} />
          <YAxis tick={axisTick} tickLine={false} axisLine={false} tickFormatter={(v: number) => (v >= 1000 ? `${v / 1000}k` : String(v))} width={40} />
          <Tooltip {...tooltipStyle} cursor={{ fill: "rgba(255,255,255,0.04)" }} labelFormatter={(v) => label(Number(v))}
            formatter={(v) => [`${Number(v).toLocaleString("en-US")} bookings`, ""]} />
          <Bar dataKey="n" fill={chart.accent} radius={[4, 4, 0, 0]} maxBarSize={24} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
