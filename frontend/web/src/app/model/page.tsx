import { Card } from "@/components/Card";
import { Section } from "@/components/Section";
import { CalibrationChart } from "@/components/charts/CalibrationChart";
import { CurveChart } from "@/components/charts/CurveChart";
import { ProbHistogram } from "@/components/charts/ProbHistogram";
import { ThresholdExplorer } from "@/components/charts/ThresholdExplorer";
import { LABEL, model } from "@/lib/data";
import { dayLabel, pct } from "@/lib/format";

export default function ModelPage() {
  const m = model.metrics, c = model.confusion, sp = model.split;
  const total = sp.train[0] + sp.valid[0] + sp.test[0];
  const imp = model.importance.filter((x) => x.auc_drop > 0.001).slice(0, 12);
  const maxImp = imp[0].auc_drop;
  const groupLabel: Record<string, string> = { hotel: "Hotel", market_segment: "Segment", deposit_type: "Deposit", customer_type: "Customer" };

  return (
    <div className="space-y-12">
      <section>
        <p className="text-sm text-muted">Extra Trees + neural network, averaged · tested on the last {sp.test[0].toLocaleString("en-US")} bookings it never saw</p>
        <h1 className="mt-1 text-3xl font-bold sm:text-4xl">Predicting a cancellation at booking time</h1>
      </section>

      <div className="grid grid-cols-2 gap-3 sm:gap-4 lg:grid-cols-4">
        <Big v={m.roc_auc.toFixed(3)} l="ROC AUC" sub="0.5 = coin flip" />
        <Big v={pct(m.precision)} l="Precision" sub="flagged bookings that cancel" />
        <Big v={pct(m.recall)} l="Recall" sub="cancellations caught" />
        <Big v={m.brier.toFixed(3)} l="Brier score" sub={`vs ${m.brier_baseline.toFixed(3)} guessing the average`} />
      </div>

      <Section title="Trained on the past, tested on the future" sub="Split by arrival date, never shuffled">
        <Card>
          <div className="flex h-10 overflow-hidden rounded-xl text-xs font-semibold">
            {([["train", "Train", "bg-accent/80 text-black"], ["valid", "Validation", "bg-accent/40"], ["test", "Test", "bg-cancel/80 text-black"]] as const).map(([k, l, cls]) => (
              <div key={k} className={`flex items-center justify-center border-r-2 border-card ${cls}`} style={{ width: `${(sp[k][0] / total) * 100}%` }}>{l}</div>
            ))}
          </div>
          <div className="mt-3 grid grid-cols-3 gap-3 text-sm">
            {([["train", "Train"], ["valid", "Validation"], ["test", "Test"]] as const).map(([k, l]) => (
              <div key={k}>
                <p className="font-semibold">{l} · {sp[k][0].toLocaleString("en-US")}</p>
                <p className="text-xs text-muted">{dayLabel(sp[k][1])} – {dayLabel(sp[k][2])}</p>
              </div>
            ))}
          </div>
          <p className="mt-3 text-xs text-muted">The final model is refit on train + validation, then scored once on the test period.</p>
        </Card>
      </Section>

      <div className="grid gap-4 lg:grid-cols-2">
        <Section title="How well it ranks bookings">
          <Card><CurveChart roc={model.roc} pr={model.pr} aucRoc={m.roc_auc} aucPr={m.pr_auc} base={model.test_cancel_rate} /></Card>
        </Section>
        <Section title={`At the chosen cut-off (${model.threshold})`} sub="Test period">
          <Card>
            <div className="grid grid-cols-[auto_1fr_1fr] gap-1.5 text-center text-sm">
              <span />
              <span className="text-xs text-muted">Predicted: kept</span>
              <span className="text-xs text-muted">Predicted: cancels</span>
              <span className="self-center pr-2 text-right text-xs text-muted">Kept</span>
              <Cell n={c.tn} good />
              <Cell n={c.fp} />
              <span className="self-center pr-2 text-right text-xs text-muted">Canceled</span>
              <Cell n={c.fn} />
              <Cell n={c.tp} good />
            </div>
            <div className="mt-4 grid grid-cols-3 gap-3 text-sm">
              <Small v={pct(m.accuracy)} l={`accuracy (vs ${pct(m.baseline_accuracy)} saying “kept”)`} />
              <Small v={m.f1.toFixed(2)} l="F1" />
              <Small v={m.f05.toFixed(2)} l="F0.5" />
            </div>
          </Card>
        </Section>
      </div>

      <Section title="Choosing the cut-off" sub={`${model.threshold} gave the best F0.5 on the validation period (F0.5 counts precision twice as much as recall)`}>
        <Card><ThresholdExplorer sweep={model.threshold_sweep} chosen={model.threshold} /></Card>
      </Section>

      <div className="grid gap-4 lg:grid-cols-2">
        <Section title="Are the probabilities honest?" sub="Dots on the dashed line = well calibrated">
          <Card><CalibrationChart data={model.calibration} /></Card>
        </Section>
        <Section title="Scores for kept vs canceled bookings" sub="Dashed line = cut-off">
          <Card><ProbHistogram data={model.prob_hist} threshold={model.threshold} /></Card>
        </Section>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Section title="What the model relies on" sub="Drop in test AUC when one booking detail is shuffled">
          <Card>
            <ul className="space-y-2">
              {imp.map((x) => (
                <li key={x.feature} className="grid grid-cols-[130px_1fr_52px] items-center gap-3 text-sm">
                  <span className="truncate text-muted">{LABEL[x.feature] ?? x.feature}</span>
                  <div className="h-2 rounded-full bg-white/5"><div className="h-2 rounded-full bg-accent" style={{ width: `${(x.auc_drop / maxImp) * 100}%` }} /></div>
                  <span className="text-right tabular-nums">−{x.auc_drop.toFixed(3)}</span>
                </li>
              ))}
            </ul>
          </Card>
        </Section>
        <Section title="Left out on purpose" sub="Only known after booking, or would give the answer away">
          <Card className="space-y-4">
            <div className="flex flex-wrap gap-2">
              {model.excluded.map((e) => <span key={e} className="rounded-full bg-white/10 px-3 py-1 text-sm text-muted line-through decoration-cancel/70">{LABEL[e] ?? e}</span>)}
            </div>
            <div>
              <p className="mb-2 text-sm text-muted">Each model alone vs together (test AUC)</p>
              <div className="grid grid-cols-3 gap-3">
                {Object.entries(model.members_auc).map(([k, v]) => (
                  <div key={k} className={`rounded-xl p-3 ${k === "Ensemble" ? "bg-accent/20" : "bg-card-hover"}`}>
                    <p className="text-xs text-muted">{k === "Extra Trees V2" ? "Extra Trees" : k === "MLP V0" ? "Neural net" : "Average"}</p>
                    <p className="text-xl font-bold tabular-nums">{v.toFixed(3)}</p>
                  </div>
                ))}
              </div>
            </div>
          </Card>
        </Section>
      </div>

      <Section title="Does it work for everyone?" sub="Test period, groups with 200+ bookings">
        <Card className="overflow-x-auto">
          <table className="w-full min-w-[520px] text-sm">
            <thead className="text-left text-xs text-muted">
              <tr><th className="pb-3 font-normal">Group</th><th className="pb-3 text-right font-normal">Bookings</th>
                <th className="pb-3 text-right font-normal">Actually canceled</th><th className="pb-3 text-right font-normal">Predicted</th>
                <th className="pb-3 text-right font-normal">AUC</th></tr>
            </thead>
            <tbody>
              {model.groups.map((g) => (
                <tr key={g.by + g.key} className="border-t border-white/5">
                  <td className="py-2"><span className="text-muted">{groupLabel[g.by]}:</span> {g.key}</td>
                  <td className="text-right tabular-nums text-muted">{g.n.toLocaleString("en-US")}</td>
                  <td className="text-right tabular-nums">{pct(g.actual)}</td>
                  <td className={`text-right tabular-nums ${Math.abs(g.predicted - g.actual) > 0.08 ? "text-cancel" : ""}`}>{pct(g.predicted)}</td>
                  <td className="text-right tabular-nums">{g.auc.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-3 text-xs text-muted">Red = predicted rate more than 8 points off the real one.</p>
        </Card>
      </Section>
    </div>
  );
}

function Big({ v, l, sub }: { v: string; l: string; sub: string }) {
  return (
    <Card>
      <p className="text-sm text-muted">{l}</p>
      <p className="mt-2 text-3xl font-bold tabular-nums tracking-tight text-accent sm:text-4xl">{v}</p>
      <p className="mt-1 text-xs text-muted">{sub}</p>
    </Card>
  );
}
function Cell({ n, good }: { n: number; good?: boolean }) {
  return <div className={`rounded-xl py-4 text-2xl font-bold tabular-nums ${good ? "bg-accent/20" : "bg-white/5 text-muted"}`}>{n.toLocaleString("en-US")}</div>;
}
function Small({ v, l }: { v: string; l: string }) {
  return <div><p className="text-xl font-bold tabular-nums">{v}</p><p className="text-xs text-muted">{l}</p></div>;
}
