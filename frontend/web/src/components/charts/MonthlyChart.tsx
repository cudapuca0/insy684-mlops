"use client";

import { useMemo, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Overview } from "@/lib/data";
import { axisTick, chart, HOTEL_COLOR, legendProps, tooltipStyle } from "@/lib/chartTheme";
import { monthShort } from "@/lib/format";

const METRICS = [
  { id: "bookings", label: "Bookings", fmt: (v: number) => v.toLocaleString("en-US") },
  { id: "cancel_rate", label: "Cancellation rate", fmt: (v: number) => `${Math.round(v * 100)}%` },
  { id: "adr", label: "Average daily rate", fmt: (v: number) => `€${Math.round(v)}` },
] as const;
type MetricId = (typeof METRICS)[number]["id"];

/** One line per hotel, by arrival month. Switch between bookings, cancellation rate and price. */
export function MonthlyChart({ data }: { data: Overview["monthly"] }) {
  const [metric, setMetric] = useState<MetricId>("bookings");
  const m = METRICS.find((x) => x.id === metric)!;
  const rows = useMemo(() => {
    const by: Record<string, Record<string, number | string>> = {};
    for (const d of data) (by[d.month] ??= { month: d.month })[d.hotel] = d[metric];
    return Object.values(by).sort((a, b) => String(a.month).localeCompare(String(b.month)));
  }, [data, metric]);
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">
        {METRICS.map((x) => (
          <button key={x.id} type="button" onClick={() => setMetric(x.id)}
            className={`rounded-full px-3 py-1.5 text-sm transition ${metric === x.id ? "bg-white text-black" : "bg-card-hover text-muted hover:text-foreground"}`}>
            {x.label}
          </button>
        ))}
      </div>
      <div className="h-72 w-full">
        <ResponsiveContainer>
          <LineChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid vertical={false} stroke={chart.grid} />
            <XAxis dataKey="month" tick={axisTick} tickLine={false} axisLine={false} interval={0}
              tickFormatter={(v: string) => (v.endsWith("-01") || v.endsWith("-07") ? monthShort(v) : "")} />
            <YAxis tick={axisTick} tickLine={false} axisLine={false} tickFormatter={(v: number) => m.fmt(v)} width={56} domain={[0, "auto"]} />
            <Tooltip {...tooltipStyle} labelFormatter={(v) => monthShort(String(v))} formatter={(v, n) => [m.fmt(Number(v)), n]} />
            <Legend {...legendProps} formatter={(v) => <span style={{ color: chart.axisText }}>{v}</span>} />
            {Object.keys(HOTEL_COLOR).map((h) => (
              <Line key={h} dataKey={h} stroke={HOTEL_COLOR[h]} strokeWidth={2} dot={false} type="monotone" />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
