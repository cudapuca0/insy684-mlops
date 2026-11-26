import type { ReactNode } from "react";

/** A titled block with an optional one-line subtitle. */
export function Section({ title, sub, children, right }: { title: string; sub?: string; children: ReactNode; right?: ReactNode }) {
  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-xl font-bold">{title}</h2>
          {sub && <p className="text-sm text-muted">{sub}</p>}
        </div>
        {right}
      </div>
      {children}
    </section>
  );
}
