import { pct } from "@/lib/format";

type Row = { key: string; n: number; cancel_rate: number };

/** Cancellation rate per group, with the overall rate marked. Bar length = rate. */
export function RateBars({ rows, overall, labels }: { rows: Row[]; overall: number; labels?: Record<string, string> }) {
  const max = Math.max(...rows.map((r) => r.cancel_rate), overall, 0.5);
  return (
    <ul className="space-y-2.5">
      {rows.map((r) => (
        <li key={r.key} className="text-sm" title={`${r.n.toLocaleString("en-US")} bookings · ${pct(r.cancel_rate, 1)} canceled`}>
          <div className="flex justify-between gap-3">
            <span>{labels?.[r.key] ?? r.key} <span className="text-xs text-muted">· {r.n.toLocaleString("en-US")}</span></span>
            <span className="tabular-nums">{pct(r.cancel_rate)}</span>
          </div>
          <div className="relative mt-1 h-2 rounded-full bg-white/5">
            <div className="h-2 rounded-full bg-cancel" style={{ width: `${(r.cancel_rate / max) * 100}%`, opacity: r.cancel_rate > overall ? 1 : 0.45 }} />
            <div className="absolute -top-1 h-4 w-px bg-white/60" style={{ left: `${(overall / max) * 100}%` }} title={`overall ${pct(overall, 1)}`} />
          </div>
        </li>
      ))}
    </ul>
  );
}
