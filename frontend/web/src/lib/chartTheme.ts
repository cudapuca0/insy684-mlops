// Shared chart colours. Text never uses data colours.
export const chart = {
  city: "#2dd4bf",
  resort: "#f5a524",
  cancel: "#f2617a",
  kept: "#5b6573",
  accent: "#2dd4bf",
  context: "#4a5260",
  grid: "rgba(255,255,255,0.06)",
  axisText: "#9ba3ae",
  tooltipBg: "#262c35",
  surface: "#171b21",
};
export const HOTEL_COLOR: Record<string, string> = { "City Hotel": chart.city, "Resort Hotel": chart.resort };

export const axisTick = { fill: chart.axisText, fontSize: 12 };

export const tooltipStyle = {
  contentStyle: { background: chart.tooltipBg, border: "none", borderRadius: 12, color: "#fff" },
  labelStyle: { color: chart.axisText },
  itemStyle: { color: "#fff" },
};

export const legendProps = {
  wrapperStyle: { fontSize: 12 },
  iconType: "circle" as const,
};
