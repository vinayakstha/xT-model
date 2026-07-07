export const W = 720;
export const H = 466;
export const SX = W / 105;
export const SY = H / 68;
export const TEAM_COLORS = ["#60a5fa", "#f59e0b"];

export const METRICS: [string, string][] = [
  ["xt", "xT"],
  ["pv", "PV"],
  ["vaep_per_action", "VAEP per-action"],
  ["vaep_state_delta", "VAEP state-delta"],
  ["pv_concede", "PV concede"],
];

// green -> yellow gradient by value / max
export function heat(t: number): string {
  t = Math.max(0, Math.min(1, t));
  return `hsl(${140 - 80 * t}, 52%, ${26 + 20 * t}%)`;
}

export function fmtNum(v: number, dp = 2): { text: string; cls: string } {
  const n = Number(v);
  const s = (n >= 0 ? "" : "-") + Math.abs(n).toFixed(dp);
  const cls = n > 0.0001 ? "pos" : n < -0.0001 ? "neg" : "";
  return { text: s, cls };
}

export type ArrowKind = "pos" | "neg" | "zero";

export function arrowKind(v: number): ArrowKind {
  return v > 0.0015 ? "pos" : v < -0.0015 ? "neg" : "zero";
}

export const KIND_COLOR: Record<ArrowKind, string> = {
  pos: "#34d399",
  neg: "#f87171",
  zero: "#9aa4b2",
};

export function zoneCenter(zx: number, zy: number): [number, number] {
  return [((zx + 0.5) / 12) * 105, ((zy + 0.5) / 8) * 68];
}
