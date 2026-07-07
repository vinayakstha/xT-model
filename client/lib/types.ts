export interface TeamRow {
  team: string;
  xt: number;
  pv: number;
  vaep_per_action: number;
  vaep_state_delta: number;
  pv_concede: number;
  events?: number;
  passes?: number;
  carries?: number;
  dribbles?: number;
}

export interface PlayerRow {
  team: string;
  player: string;
  xt: number;
  pv: number;
  vaep_per_action: number;
  vaep_state_delta: number;
  pv_concede: number;
}

export interface EventRow {
  type: string;
  team: string;
  player: string;
  sx: number;
  sy: number;
  ex: number;
  ey: number;
  xt: number;
  pv: number;
  vaep: number;
  goal: boolean | null;
  xg: number | null;
}

export interface TopEventRow {
  team: string;
  player: string;
  type: string;
  minute: number | null;
  sx: number;
  sy: number;
  ex: number;
  ey: number;
  xt: number;
  pv: number;
  vaep: number;
}

export interface Margin {
  a: string;
  b: string;
  xt: number;
  pv: number;
  vaep_per_action: number;
  vaep_state_delta: number;
  pv_concede: number;
}

export interface Totals {
  events: number;
  passes: number;
  carries: number;
  dribbles: number;
}

export interface Distributions {
  xt: [number, number, number];
  pv: [number, number, number];
  vaep_per_action: [number, number, number];
  vaep_state_delta: [number, number, number];
}

export interface AnalyzeResult {
  n_events: number;
  teams: TeamRow[];
  team_names: string[];
  totals: Totals;
  margin: Margin | null;
  by_xt: PlayerRow[];
  top_per_action: PlayerRow[];
  top_state: PlayerRow[];
  bottom_per_action: PlayerRow[];
  bottom_state: PlayerRow[];
  top_events: TopEventRow[];
  distributions: Distributions;
  events: EventRow[];
}

export interface GridResult {
  zones: number[][];
  max_value: number;
  mode: string;
  period: string;
  label: string;
}

export interface MoveResult {
  zone_based_xt: number;
  start_zone_xt: number;
  end_zone_xt: number;
  start_zone: [number, number];
  end_zone: [number, number];
  model_pv: number;
  pv_concede: number;
  vaep_per_action: number;
  success: boolean;
}

export type MetricKey =
  | "xt"
  | "pv"
  | "vaep_per_action"
  | "vaep_state_delta"
  | "pv_concede";
export type GridMode = "xt" | "xg" | "pv";
export type Period = "combined" | "p1" | "p2" | "et";
export type EventType = "pass" | "carry" | "dribble" | "shot";
export type EventMetricKey = "xt" | "pv" | "vaep";
