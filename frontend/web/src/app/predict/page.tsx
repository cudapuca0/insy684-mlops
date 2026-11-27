"use client";

import { useState, type FormEvent } from "react";

const sampleBooking = {
  hotel: "City Hotel",
  lead_time: 120,
  arrival_date_year: 2017,
  arrival_date_month: "August",
  arrival_date_day_of_month: 14,
  stays_in_weekend_nights: 1,
  stays_in_week_nights: 3,
  adults: 2,
  children: 0,
  babies: 0,
  meal: "BB",
  country: "PRT",
  market_segment: "Online TA",
  is_repeated_guest: 0,
  previous_cancellations: 0,
  previous_bookings_not_canceled: 0,
  reserved_room_type: "A",
  deposit_type: "No Deposit",
  agent: 9,
  company: null,
  customer_type: "Transient",
  adr: 110.0,
};

type Prediction = {
  cancel_probability: number;
  label: string;
};

type ValidationIssue = {
  loc: (string | number)[];
  msg: string;
};

export default function PredictPage() {
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function predict(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const form = new FormData(event.currentTarget);
    const booking = Object.fromEntries(
      Object.entries(sampleBooking).map(([name, example]) => {
        const value = String(form.get(name) ?? "").trim();

        if (name === "agent" || name === "company") {
          return [name, value === "" ? null : Number(value)];
        }

        return [name, typeof example === "number" ? Number(value) : value];
      }),
    );

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(booking),
      });
      const data = await response.json();

      if (!response.ok) {
        if (response.status === 422 && Array.isArray(data.detail)) {
          const details = data.detail.map(
            (issue: ValidationIssue) =>
              `${issue.loc.slice(1).join(" ").replaceAll("_", " ")}: ${issue.msg}`,
          );
          setError(details.join(". "));
        } else {
          setError(data.error ?? "The prediction service could not score this booking.");
        }
        return;
      }

      setResult(data);
    } catch {
      setError("Could not get a prediction. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="mx-auto max-w-4xl px-6 py-10">
      <h1 className="mb-2 text-3xl font-semibold">Predict a cancellation</h1>
      <p className="mb-6">
        Use the sample booking below or edit its details before submitting.
      </p>

      <form onSubmit={predict}>
        <div className="grid gap-4 sm:grid-cols-2">
          {Object.entries(sampleBooking).map(([name, example]) => {
            const optional = name === "agent" || name === "company";
            const numeric = typeof example === "number" || optional;

            return (
              <label key={name} className="grid gap-1">
                <span className="capitalize">{name.replaceAll("_", " ")}</span>
                {name === "hotel" ? (
                  <select
                    name={name}
                    defaultValue={String(example)}
                    className="rounded border bg-transparent p-2"
                  >
                    <option>City Hotel</option>
                    <option>Resort Hotel</option>
                  </select>
                ) : (
                  <input
                    name={name}
                    type={numeric ? "number" : "text"}
                    step={name === "adr" ? "any" : undefined}
                    defaultValue={example ?? ""}
                    required={!optional}
                    className="rounded border bg-transparent p-2"
                  />
                )}
              </label>
            );
          })}
        </div>

        <button
          type="submit"
          disabled={loading}
          className="mt-6 rounded bg-blue-600 px-4 py-2 text-white disabled:opacity-60"
        >
          {loading ? "Predicting…" : "Get prediction"}
        </button>
      </form>

      {error && <p role="alert" className="mt-4">{error}</p>}

      {result && (
        <div role="status" className="mt-6 rounded border p-4">
          <p className="text-xl font-semibold">
            {(result.cancel_probability * 100).toFixed(1)}% cancellation probability
          </p>
          <p>{result.label}</p>
        </div>
      )}
    </main>
  );
}
