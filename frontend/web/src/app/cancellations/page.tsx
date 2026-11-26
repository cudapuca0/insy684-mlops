import { Card } from "@/components/Card";
import { RateBars } from "@/components/RateBars";
import { Section } from "@/components/Section";
import { overview } from "@/lib/data";
import { pct } from "@/lib/format";

export default function Cancellations() {
  const o = overview.summary.cancel_rate;
  const by = overview.by;
  const lead = by.lead_time;
  return (
    <div className="space-y-10">
      <section>
        <p className="text-sm text-muted">Share of bookings canceled · white tick = overall {pct(o, 1)}</p>
        <h1 className="mt-1 text-3xl font-bold sm:text-4xl">Who cancels?</h1>
      </section>

      <div className="grid gap-3 sm:grid-cols-3 sm:gap-4">
        <Headline big={`${pct(lead[0].cancel_rate)} → ${pct(lead[lead.length - 1].cancel_rate)}`} label="booked within a week vs over a year ahead" />
        <Headline big={pct(by.deposit_type.find((d) => d.key === "Non Refund")!.cancel_rate)} label="of non-refundable bookings canceled" />
        <Headline big={pct(by.previous_cancellations.find((d) => d.key === "1")!.cancel_rate)} label="when the guest canceled once before" />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Section title="Lead time" sub="Days between booking and arrival">
          <Card><RateBars rows={lead} overall={o} labels={{ "0-7": "0–7 days", "8-30": "8–30 days", "31-90": "1–3 months", "91-180": "3–6 months", "181-365": "6–12 months", "366+": "Over a year" }} /></Card>
        </Section>
        <Section title="Market segment">
          <Card><RateBars rows={by.market_segment} overall={o} /></Card>
        </Section>
        <Section title="Deposit">
          <Card>
            <RateBars rows={by.deposit_type} overall={o} labels={{ "No Deposit": "No deposit", "Non Refund": "Non-refundable", Refundable: "Refundable" }} />
            <p className="mt-3 text-xs text-muted">Counter-intuitive: only 1,038 such bookings, 91% of them groups or offline travel agencies, 95% from Portugal.</p>
          </Card>
        </Section>
        <Section title="Customer type">
          <Card><RateBars rows={by.customer_type} overall={o} /></Card>
        </Section>
        <Section title="Guest history">
          <Card className="space-y-5">
            <RateBars rows={by.repeated_guest} overall={o} />
            <RateBars rows={by.previous_cancellations} overall={o} labels={{ "0": "No past cancellation", "1": "1 past cancellation", "2+": "2 or more" }} />
          </Card>
        </Section>
        <Section title="Special requests" sub="Known only after booking, so the model can’t use it">
          <Card><RateBars rows={by.special_requests} overall={o} labels={{ "0": "None", "1": "1 request", "2": "2 requests", "3+": "3 or more" }} /></Card>
        </Section>
      </div>
    </div>
  );
}

function Headline({ big, label }: { big: string; label: string }) {
  return (
    <Card>
      <p className="text-4xl font-bold tracking-tight text-cancel tabular-nums">{big}</p>
      <p className="mt-2 text-sm text-muted">{label}</p>
    </Card>
  );
}
