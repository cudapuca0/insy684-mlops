export const MONTH_SHORT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
export const monthShort = (m: string) => `${MONTH_SHORT[Number(m.slice(5, 7)) - 1]} ${m.slice(0, 4)}`;
export const dayLabel = (d: string) => `${MONTH_SHORT[Number(d.slice(5, 7)) - 1]} ${Number(d.slice(8, 10))}, ${d.slice(0, 4)}`;
export const formatNumber = (n: number) => Math.round(n).toLocaleString("en-US");
export const pct = (x: number, digits = 0) => `${(x * 100).toFixed(digits)}%`;
