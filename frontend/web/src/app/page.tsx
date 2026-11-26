import { Card } from "@/components/Card";
import { Section } from "@/components/Section";
import { StatCard } from "@/components/StatCard";
import { Histogram } from "@/components/charts/Histogram";
import { MonthlyChart } from "@/components/charts/MonthlyChart";
import { overview } from "@/lib/data";
import { dayLabel, formatNumber, pct } from "@/lib/format";
import { HOTEL_COLOR } from "@/lib/chartTheme";

export default function Bookings() {
  const s = overview.summary, f = overview.funnel;
  const maxCountry = overview.countries[0].n;
  return (
    <div className="space-y-10">
      <section>
        <p className="text-sm text-muted">Arrivals {dayLabel(s.first)} – {dayLabel(s.last)} · Portugal</p>
        <h1 className="mt-1 text-3xl font-bold sm:text-4xl">Two hotels, {formatNumber(s.bookings)} bookings</h1>
      </section>

      <div className="grid grid-cols-2 gap-3 sm:gap-4 lg:grid-cols-4">
        <StatCard label="Bookings" value={s.bookings} />
        <Card>
          <p className="text-sm text-muted">Canceled</p>
          <p className="mt-2 text-3xl font-bold tabular-nums text-cancel sm:text-4xl">{pct(s.cancel_rate, 1)}</p>
          <p className="mt-1 text-sm text-muted">of all bookings</p>
        </Card>
        <StatCard label="Median lead time" value={s.median_lead_time} unit="days" />
        <StatCard label="Guest countries" value={s.countries} />
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        {Object.entries(s.hotels).map(([h, v]) => (
          <Card key={h}>
            <p className="flex items-center gap-2 font-semibold"><span className="h-2.5 w-2.5 rounded-full" style={{ background: HOTEL_COLOR[h] }} />{h}</p>
            <div className="mt-3 grid grid-cols-4 gap-3 text-sm">
              <Fact v={formatNumber(v.bookings)} l="bookings" />
              <Fact v={pct(v.cancel_rate)} l="canceled" />
              <Fact v={`€${Math.round(v.adr)}`} l="per night" />
              <Fact v={v.nights.toFixed(1)} l="nights" />
            </div>
          </Card>
        ))}
      </div>

      <Section title="Month by month" sub="By arrival month · price = average daily rate of stays that happened">
        <Card><MonthlyChart data={overview.monthly} /></Card>
      </Section>

      <div className="grid gap-4 lg:grid-cols-2">
        <Section title="How far ahead people book">
          <Card>
            <Histogram data={overview.lead_time_hist} kind="lead" />
          </Card>
        </Section>
        <Section title="How long they stay">
          <Card>
            <Histogram data={overview.nights} kind="nights" />
          </Card>
        </Section>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Section title="Where guests come from" sub="Top 12 countries · bar = bookings, % = canceled">
          <Card>
            <ul className="space-y-2">
              {overview.countries.map((c) => (
                <li key={c.code} className="grid grid-cols-[44px_1fr_48px] items-center gap-3 text-sm" title={`${c.n.toLocaleString("en-US")} bookings`}>
                  <span className="text-muted">{c.code}</span>
                  <div className="h-2 rounded-full bg-white/5"><div className="h-2 rounded-full bg-accent" style={{ width: `${(c.n / maxCountry) * 100}%` }} /></div>
                  <span className="text-right tabular-nums">{pct(c.cancel_rate)}</span>
                </li>
              ))}
            </ul>
          </Card>
        </Section>
        <Section title="Cleaning the data" sub="Before any analysis or modelling">
          <Card>
            <ol className="space-y-3 text-sm">
              <Step n={f.raw} label="rows in the original file" />
              <Step n={-f.duplicates} label="exact duplicate rows removed" dim />
              <Step n={-f.no_guests_or_negative_adr} label="bookings with no guests or a negative price removed" dim />
              <Step n={f.clean} label="bookings analysed" strong />
            </ol>
            <p className="mt-4 text-xs text-muted">Same cleaning as the model’s training script, so every number on this site uses the same {formatNumber(f.clean)} bookings.</p>
          </Card>
        </Section>
      </div>
    </div>
  );
}

function Fact({ v, l }: { v: string; l: string }) {
  return <div><p className="text-lg font-bold tabular-nums">{v}</p><p className="text-xs text-muted">{l}</p></div>;
}

function Step({ n, label, dim, strong }: { n: number; label: string; dim?: boolean; strong?: boolean }) {
  return (
    <li className="flex items-baseline gap-3">
      <span className={`w-24 shrink-0 text-right text-lg font-bold tabular-nums ${dim ? "text-muted" : ""} ${strong ? "text-accent" : ""}`}>
        {n < 0 ? "−" : ""}{Math.abs(n).toLocaleString("en-US")}
      </span>
      <span className={dim ? "text-muted" : ""}>{label}</span>
    </li>
  );
}
