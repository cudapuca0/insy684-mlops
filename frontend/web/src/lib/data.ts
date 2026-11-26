// Typed access to the JSON written by pipeline/build_data.py (copied into src/data by scripts/sync-data.mjs).
import overviewJson from "@/data/overview.json";
import modelJson from "@/data/model.json";

export type Rate = { key: string; n: number; cancel_rate: number };
export type Overview = {
  funnel: { raw: number; after_duplicates: number; clean: number; duplicates: number; no_guests_or_negative_adr: number };
  summary: {
    bookings: number; cancel_rate: number; first: string; last: string; median_lead_time: number; countries: number; adr: number;
    hotels: Record<string, { bookings: number; cancel_rate: number; adr: number; nights: number }>;
  };
  monthly: { month: string; hotel: string; bookings: number; canceled: number; cancel_rate: number; adr: number }[];
  seasonal: { month: number; hotel: string; adr: number; cancel_rate: number }[];
  by: Record<"lead_time" | "deposit_type" | "market_segment" | "customer_type" | "repeated_guest" | "previous_cancellations" | "special_requests" | "hotel", Rate[]>;
  countries: { code: string; n: number; cancel_rate: number }[];
  nights: { nights: number; n: number }[];
  lead_time_hist: { days: number; n: number }[];
};
export type Model = {
  split: Record<"train" | "valid" | "test", [number, string, string]>;
  threshold: number;
  test_cancel_rate: number;
  metrics: Record<"roc_auc" | "pr_auc" | "brier" | "brier_baseline" | "precision" | "recall" | "f1" | "f05" | "accuracy" | "baseline_accuracy", number>;
  confusion: { tn: number; fp: number; fn: number; tp: number };
  members_auc: Record<string, number>;
  roc: { fpr: number; tpr: number }[];
  pr: { recall: number; precision: number }[];
  calibration: { predicted: number; actual: number }[];
  threshold_sweep: { t: number; precision: number; recall: number; f1: number; f05: number; flagged: number }[];
  groups: { by: string; key: string; n: number; actual: number; predicted: number; auc: number }[];
  importance: { feature: string; auc_drop: number }[];
  prob_hist: { p: number; kept: number; canceled: number }[];
  excluded: string[];
};

export const overview = overviewJson as unknown as Overview;
export const model = modelJson as unknown as Model;

// Readable names for the dataset's columns
export const LABEL: Record<string, string> = {
  hotel: "Hotel", lead_time: "Lead time", country: "Country", agent: "Travel agent", market_segment: "Market segment",
  customer_type: "Customer type", deposit_type: "Deposit type", previous_cancellations: "Past cancellations",
  previous_bookings_not_canceled: "Past stays", adr: "Daily rate", children: "Children", adults: "Adults", babies: "Babies",
  stays_in_week_nights: "Week nights", stays_in_weekend_nights: "Weekend nights", is_repeated_guest: "Repeat guest",
  company: "Company booking", meal: "Meal plan", reserved_room_type: "Room type", distribution_channel: "Channel",
  arrival_date: "Arrival date", reservation_status: "Reservation status", reservation_status_date: "Status date",
  assigned_room_type: "Assigned room", booking_changes: "Booking changes", days_in_waiting_list: "Days on waiting list",
  required_car_parking_spaces: "Parking spaces", total_of_special_requests: "Special requests",
};
