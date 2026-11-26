"use client";

import { useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Model } from "@/lib/data";
import { axisTick, chart, legendProps, tooltipStyle } from "@/lib/chartTheme";

/** Move the cut-off: above it, a booking is flagged as likely to cancel. */
export function ThresholdExplorer({ sweep, chosen }: { sweep: Model["threshold_sweep"]; chosen: number }) {
  const [i, setI] = useState(sweep.findIndex((s) => Math.abs(s.t - chosen) < 1e-9));
  const s = sweep[i];
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-4">
        <label htmlFor="thr" className="text-sm text-muted">Cut-off</label>
        <input id="thr" type="range" min={0} max={sweep.length - 1} value={i} onChange={(e) => setI(Number(e.target.value))}
          className="w-56 accent-[#2dd4bf]" aria-valuetext={s.t.toFixed(2)} />
        <span className="text-2xl font-bold tabular-nums">{s.t.toFixed(2)}</span>
        {Math.abs(s.t - chosen) < 1e-9 && <span className="rounded-full bg-accent px-2 py-0.5 text-xs font-semibold text-black">chosen</span>}
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Metric label="Precision" value={s.precision} sub="flagged that really cancel" />
        <Metric label="Recall" value={s.recall} sub="cancellations caught" />
        <Metric label="F0.5" value={s.f05} sub="weights precision ×2" />
        <Metric label="Flagged" value={s.flagged} sub="of all bookings" />
      </div>
      <div className="h-56 w-full">
        <ResponsiveContainer>
          <LineChart data={sweep} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid vertical={false} stroke={chart.grid} />
            <XAxis dataKey="t" type="number" domain={[0.1, 0.9]} ticks={[0.1, 0.3, 0.5, 0.7, 0.9]} tick={axisTick} tickLine={false} axisLine={false} />
            <YAxis domain={[0, 1]} ticks={[0, 0.5, 1]} tick={axisTick} tickLine={false} axisLine={false} width={48} tickFormatter={(v: number) => `${v * 100}%`} />
            <Tooltip {...tooltipStyle} labelFormatter={(v) => `Cut-off ${Number(v).toFixed(2)}`} formatter={(v, n) => [`${Math.round(Number(v) * 100)}%`, n]} />
            <Legend {...legendProps} formatter={(v) => <span style={{ color: chart.axisText }}>{v}</span>} />
            <ReferenceLine x={s.t} stroke="#ffffff" strokeOpacity={0.5} />
            <Line dataKey="precision" name="Precision" stroke={chart.accent} strokeWidth={2} dot={false} isAnimationActive={false} />
            <Line dataKey="recall" name="Recall" stroke={chart.resort} strokeWidth={2} dot={false} isAnimationActive={false} />
            <Line dataKey="f05" name="F0.5" stroke="#c4c9d1" strokeWidth={2} strokeDasharray="5 4" dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function Metric({ label, value, sub }: { label: string; value: number; sub: string }) {
  return (
    <div className="rounded-xl bg-card-hover p-3">
      <p className="text-xs text-muted">{label}</p>
      <p className="text-2xl font-bold tabular-nums">{label === "F0.5" ? value.toFixed(2) : `${Math.round(value * 100)}%`}</p>
      <p className="text-xs text-muted">{sub}</p>
    </div>
  );
}
