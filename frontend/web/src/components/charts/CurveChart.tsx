"use client";

import { useState } from "react";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Model } from "@/lib/data";
import { axisTick, chart, tooltipStyle } from "@/lib/chartTheme";

/** ROC and precision-recall curves on the test period, with what a coin flip / guessing would give. */
export function CurveChart({ roc, pr, aucRoc, aucPr, base }: { roc: Model["roc"]; pr: Model["pr"]; aucRoc: number; aucPr: number; base: number }) {
  const [view, setView] = useState<"roc" | "pr">("roc");
  const isRoc = view === "roc";
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        {(["roc", "pr"] as const).map((v) => (
          <button key={v} type="button" onClick={() => setView(v)}
            className={`rounded-full px-3 py-1.5 text-sm transition ${view === v ? "bg-white text-black" : "bg-card-hover text-muted hover:text-foreground"}`}>
            {v === "roc" ? "ROC" : "Precision–recall"}
          </button>
        ))}
        <span className="ml-auto text-sm text-muted">area <span className="text-foreground tabular-nums">{(isRoc ? aucRoc : aucPr).toFixed(3)}</span></span>
      </div>
      <div className="h-64 w-full">
        <ResponsiveContainer>
          <LineChart data={(isRoc ? roc : pr) as Record<string, number>[]} margin={{ top: 8, right: 12, bottom: 18, left: 0 }}>
            <CartesianGrid stroke={chart.grid} />
            <XAxis dataKey={isRoc ? "fpr" : "recall"} type="number" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tick={axisTick} tickLine={false} axisLine={false}
              label={{ value: isRoc ? "Kept bookings wrongly flagged" : "Cancellations caught (recall)", position: "insideBottom", dy: 16, fill: chart.axisText, fontSize: 11 }} />
            <YAxis dataKey={isRoc ? "tpr" : "precision"} type="number" domain={[0, 1]} ticks={[0, 0.5, 1]} tick={axisTick} tickLine={false} axisLine={false} width={36} />
            {isRoc
              ? <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={chart.context} strokeDasharray="4 4" />
              : <ReferenceLine y={base} stroke={chart.context} strokeDasharray="4 4" label={{ value: "guessing", position: "insideBottomRight", fill: chart.axisText, fontSize: 11 }} />}
            <Tooltip {...tooltipStyle} labelFormatter={(v) => (isRoc ? `False alarm rate ${Number(v).toFixed(2)}` : `Recall ${Number(v).toFixed(2)}`)}
              formatter={(v) => [Number(v).toFixed(2), isRoc ? "Caught" : "Precision"]} />
            <Line dataKey={isRoc ? "tpr" : "precision"} stroke={chart.accent} strokeWidth={2} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
