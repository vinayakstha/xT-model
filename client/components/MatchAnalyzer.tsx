"use client";

import { useState, useRef, useCallback } from "react";
import PitchMarkings from "./Pitch";
import {
  W,
  H,
  SX,
  SY,
  TEAM_COLORS,
  METRICS,
  heat,
  heatTextColor,
  fmtNum,
  arrowKind,
  KIND_COLOR,
} from "@/lib/utils";
import type {
  AnalyzeResult,
  PlayerRow,
  TopEventRow,
  GridMode,
  MetricKey,
  EventType,
} from "@/lib/types";

type MapType = EventType | "all" | "top";

interface Grids {
  xt?: { z: number[][]; max: number };
  xg?: { z: number[][]; max: number };
  pv?: { z: number[][]; max: number };
}

function NumCell({ v, dp = 2 }: { v: number; dp?: number }) {
  const { text, cls } = fmtNum(v, dp);
  return <span className={cls}>{text}</span>;
}

function PlayerTable({
  rows,
  cols,
  headers,
  dps,
}: {
  rows: PlayerRow[];
  cols: MetricKey[];
  headers: string[];
  dps: number[];
}) {
  return (
    <table>
      <thead>
        <tr>
          <th>Team</th>
          <th>Player</th>
          {headers.map((h) => (
            <th key={h}>{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i}>
            <td>{r.team}</td>
            <td>{r.player}</td>
            {cols.map((c, j) => (
              <td key={c}>
                <NumCell v={r[c]} dp={dps[j]} />
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function TopEventsTable({ rows }: { rows: TopEventRow[] }) {
  return (
    <table>
      <thead>
        <tr>
          <th>Player</th>
          <th>Type</th>
          <th>Move</th>
          <th>xT</th>
          <th>PV</th>
          <th>VAEP</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i}>
            <td>{r.player}</td>
            <td>{r.type}</td>
            <td>
              ({r.sx.toFixed(0)},{r.sy.toFixed(0)})&rarr;({r.ex.toFixed(0)},
              {r.ey.toFixed(0)})
            </td>
            <td>
              <NumCell v={r.xt} dp={3} />
            </td>
            <td>{(r.pv * 100).toFixed(2)}%</td>
            <td>
              <NumCell v={r.vaep} dp={3} />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function MatchAnalyzer() {
  const [status, setStatus] = useState<{ msg: string; err: boolean }>({
    msg: "",
    err: false,
  });
  const [data, setData] = useState<AnalyzeResult | null>(null);
  const [mtype, setMtype] = useState<MapType>("all");
  type EventMetricKey = "xt" | "pv" | "vaep";
  const [mcol, setMcol] = useState<EventMetricKey>("xt");
  const [mgrid, setMgrid] = useState<GridMode | "off">("xt");
  const [mteam, setMteam] = useState<string>("both");
  const gridsRef = useRef<Grids>({});
  const [gridsLoaded, setGridsLoaded] = useState(false);

  const loadGridsOnce = useCallback(async () => {
    if (gridsLoaded) return;
    for (const g of ["xt", "xg", "pv"] as GridMode[]) {
      const res = await fetch(`/api/grid?mode=${g}`);
      const d = await res.json();
      gridsRef.current[g] = { z: d.zones, max: d.max_value || 1 };
    }
    setGridsLoaded(true);
  }, [gridsLoaded]);

  const onFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setStatus({ msg: `Scoring ${f.name} ...`, err: false });
    await loadGridsOnce();
    const fd = new FormData();
    fd.append("file", f);
    try {
      const r = await fetch("/api/analyze", { method: "POST", body: fd });
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        throw new Error(body.detail || r.statusText);
      }
      const d: AnalyzeResult = await r.json();
      setData(d);
      setMteam("both");
      setStatus({ msg: `Analyzed ${f.name}`, err: false });
    } catch (err: any) {
      setStatus({ msg: err.message, err: true });
    }
  };

  const teamNames = data
    ? data.team_names || data.teams.map((t) => t.team)
    : [];
  const colorOf = (t: string) =>
    TEAM_COLORS[Math.max(0, teamNames.indexOf(t))] || "#94a3b8";

  let events = data?.events ?? [];
  if (mteam !== "both") events = events.filter((e) => e.team === mteam);
  if (mtype === "top") {
    events = [...events]
      .sort((a, b) => Math.abs(b[mcol] as number) - Math.abs(a[mcol] as number))
      .slice(0, 20);
  } else if (mtype !== "all") {
    events = events.filter((e) => e.type === mtype);
  }

  const grid = mgrid !== "off" ? gridsRef.current[mgrid] : undefined;
  const cw = W / 12,
    ch = H / 8;

  return (
    <div>
      <section className="panel">
        <div className="controls">
          <label className="upload">
            Upload a match CSV
            <input type="file" accept=".csv" hidden onChange={onFile} />
          </label>
          <span className="or">
            upload a match event CSV to build the threat map
          </span>
          <span className={"status" + (status.err ? " err" : "")}>
            {status.msg}
          </span>
        </div>
        <p className="hint">
          CSV needs:{" "}
          <code>
            start_x start_y end_x end_y event_type team period minute second
          </code>{" "}
          (optional: <code>success player under_pressure counter_attack</code>).
        </p>
      </section>

      {data && (
        <section id="results">
          <div id="summary">
            <div className="summary-cards">
              {data.teams.slice(0, 2).map((t, i) => {
                const sign = (x: number) => (x >= 0 ? "+" : "") + x.toFixed(2);
                return (
                  <div
                    key={t.team}
                    className={"tcard " + (i === 0 ? "a" : "b")}
                  >
                    <div className="tname">{t.team}</div>
                    <div className="tmetrics">
                      <span>
                        <b>{t.xt.toFixed(2)}</b>
                        <em>xT</em>
                      </span>
                      <span>
                        <b>{t.pv.toFixed(2)}</b>
                        <em>PV</em>
                      </span>
                      <span>
                        <b>{sign(t.vaep_per_action)}</b>
                        <em>VAEP</em>
                      </span>
                    </div>
                    <div className="tcounts">
                      Events {t.events} &middot; Passes {t.passes} &middot;
                      Carries {t.carries} &middot; Dribbles {t.dribbles}
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="totrow">
              <span>
                Total events <b>{data.totals.events}</b>
              </span>
              <span>
                Passes <b>{data.totals.passes}</b>
              </span>
              <span>
                Carries <b>{data.totals.carries}</b>
              </span>
              <span>
                Dribbles <b>{data.totals.dribbles}</b>
              </span>
            </div>
          </div>

          <h2>
            Threat map <span className="tag">{events.length} events</span>
          </h2>
          <div className="controls wrap threatctl">
            <span className="ctl">
              <b>Type</b>
              {(
                ["all", "pass", "carry", "dribble", "shot", "top"] as MapType[]
              ).map((t) => (
                <button
                  key={t}
                  className={"seg" + (mtype === t ? " active" : "")}
                  onClick={() => setMtype(t)}
                >
                  {t === "all"
                    ? "All"
                    : t === "top"
                      ? "Top 20"
                      : t[0].toUpperCase() +
                        t.slice(1) +
                        (t !== "shot" ? "es" : "s")}
                </button>
              ))}
            </span>
            <span className="ctl">
              <b>Colour</b>
              {(["xt", "pv", "vaep"] as EventMetricKey[]).map((c) => (
                <button
                  key={c}
                  className={"seg" + (mcol === c ? " active" : "")}
                  onClick={() => setMcol(c)}
                >
                  {c === "vaep" ? "VAEP" : c.toUpperCase()}
                </button>
              ))}
            </span>
            <span className="ctl">
              <b>Grid</b>
              {(["xt", "xg", "pv", "off"] as (GridMode | "off")[]).map((g) => (
                <button
                  key={g}
                  className={"seg" + (mgrid === g ? " active" : "")}
                  onClick={() => setMgrid(g)}
                >
                  {g === "off" ? "Off" : g.toUpperCase()}
                </button>
              ))}
            </span>
            <span className="ctl">
              <b>Team</b>
              <button
                className={"seg" + (mteam === "both" ? " active" : "")}
                onClick={() => setMteam("both")}
              >
                Both
              </button>
              {teamNames.map((n) => (
                <button
                  key={n}
                  className={"seg" + (mteam === n ? " active" : "")}
                  onClick={() => setMteam(n)}
                >
                  {n}
                </button>
              ))}
            </span>
          </div>

          <div className="pitchwrap">
            <svg viewBox={`0 0 ${W} ${H}`}>
              <defs>
                {(["pos", "neg", "zero"] as const).map((k) => (
                  <marker
                    key={k}
                    id={`ah-${k}`}
                    markerWidth={8}
                    markerHeight={8}
                    refX={7}
                    refY={4}
                    orient="auto"
                    markerUnits="userSpaceOnUse"
                  >
                    <path d="M0,0 L8,4 L0,8 Z" fill={KIND_COLOR[k]} />
                  </marker>
                ))}
              </defs>

              {grid &&
                Array.from({ length: 12 }).map((_, zx) =>
                  Array.from({ length: 8 }).map((_, zy) => {
                    const v = grid.z[zx][zy];
                    const x = zx * cw,
                      y = zy * ch;
                    return (
                      <g key={`${zx}-${zy}`}>
                        <rect
                          x={x + 0.5}
                          y={y + 0.5}
                          width={cw - 1}
                          height={ch - 1}
                          fill={heat(v / grid.max)}
                          className="cell"
                          stroke="rgba(0,0,0,0.25)"
                          strokeWidth={1}
                        />
                        <text
                          x={x + cw / 2}
                          y={y + ch / 2 + 1}
                          className="cellval"
                          fill={heatTextColor(v / grid.max)}
                        >
                          {(v * 100).toFixed(2)}%
                        </text>
                        <text
                          x={x + cw / 2}
                          y={y + ch / 2 + 14}
                          className="cellidx"
                          fill={heatTextColor(v / grid.max)}
                        >
                          ({zx},{zy})
                        </text>
                      </g>
                    );
                  }),
                )}

              <PitchMarkings />

              {events.map((e, i) => {
                if (e.type === "shot") {
                  const r = Math.min(4 + (e.xg || 0.03) * 24, 20);
                  return (
                    <circle
                      key={i}
                      cx={e.sx * SX}
                      cy={e.sy * SY}
                      r={r}
                      fill={colorOf(e.team)}
                      fillOpacity={e.goal ? 0.95 : 0.4}
                      stroke={e.goal ? "#fff" : "#0b1220"}
                      strokeWidth={e.goal ? 2 : 1}
                      className="shot"
                    >
                      <title>{`${e.player} — ${e.goal ? "GOAL" : "shot"} · xG ${e.xg == null ? "n/a" : e.xg.toFixed(3)} · PV ${(e.pv * 100).toFixed(2)}%`}</title>
                    </circle>
                  );
                }
                const v = e[mcol] as number;
                const k = arrowKind(v);
                return (
                  <line
                    key={i}
                    x1={e.sx * SX}
                    y1={e.sy * SY}
                    x2={e.ex * SX}
                    y2={e.ey * SY}
                    stroke={KIND_COLOR[k]}
                    strokeWidth={1 + Math.min(Math.abs(v) * 45, 3)}
                    strokeOpacity={0.65}
                    markerEnd={`url(#ah-${k})`}
                    className="arrow"
                  >
                    <title>{`${e.player} ${e.type} · xT ${(e.xt * 100).toFixed(2)}% · PV ${(e.pv * 100).toFixed(2)}% · VAEP ${(e.vaep * 100).toFixed(2)}%`}</title>
                  </line>
                );
              })}
            </svg>
          </div>
          <div className="legend">
            <span>
              <span className="sw" style={{ background: KIND_COLOR.pos }} />
              {mcol.toUpperCase()} &gt; 0 (progression)
            </span>
            <span>
              <span className="sw" style={{ background: KIND_COLOR.neg }} />
              {mcol.toUpperCase()} &lt; 0 (regression)
            </span>
            <span>
              <span className="sw" style={{ background: KIND_COLOR.zero }} />
              &asymp; 0
            </span>
            <span>shots = dots (size = xG, filled = goal)</span>
          </div>

          <div className="grid2">
            <div>
              <h2>Player xT rankings</h2>
              <div className="tablewrap tall">
                <PlayerTable
                  rows={data.by_xt}
                  cols={["xt", "pv", "vaep_per_action", "vaep_state_delta"]}
                  headers={["xT", "PV", "VAEP/act", "VAEP state"]}
                  dps={[4, 2, 2, 2]}
                />
              </div>
            </div>
            <div>
              <h2>Top events by |xT|</h2>
              <div className="tablewrap tall">
                <TopEventsTable rows={data.top_events} />
              </div>
            </div>
          </div>

          <div className="metricbar">
            {[
              ["Events", String(data.n_events), ""],
              [
                "xT mean",
                data.distributions.xt[0].toFixed(5),
                `max ${data.distributions.xt[1].toFixed(4)}`,
              ],
              [
                "PV mean",
                data.distributions.pv[0].toFixed(5),
                `max ${data.distributions.pv[1].toFixed(4)}`,
              ],
              [
                "VAEP/action mean",
                data.distributions.vaep_per_action[0].toFixed(5),
                `max ${data.distributions.vaep_per_action[1].toFixed(4)}`,
              ],
              [
                "VAEP state mean",
                data.distributions.vaep_state_delta[0].toFixed(5),
                `max ${data.distributions.vaep_state_delta[1].toFixed(4)}`,
              ],
            ].map(([k, v, r]) => (
              <div className="m" key={k}>
                <div className="k">{k}</div>
                <div className="v">{v}</div>
                <div className="r">{r}</div>
              </div>
            ))}
          </div>

          <h2>Team totals</h2>
          <div className="tablewrap">
            <table>
              <thead>
                <tr>
                  <th>Team</th>
                  {METRICS.map(([, l]) => (
                    <th key={l}>{l}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.teams.map((t) => (
                  <tr key={t.team}>
                    <td>{t.team}</td>
                    {METRICS.map(([c]) => (
                      <td key={c}>
                        <NumCell v={t[c as MetricKey]} />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="note">
            {data.margin ? (
              <>
                Margin ({data.margin.a} &minus; {data.margin.b}):{" "}
                {METRICS.map(([c, l]) => (
                  <span key={c}>
                    {l} <NumCell v={data.margin![c as MetricKey]} />{" "}
                    &middot;{" "}
                  </span>
                ))}
              </>
            ) : (
              ""
            )}
          </p>

          <div className="grid2">
            <div>
              <h2>
                The goalkeeper / CB paradox{" "}
                <span className="tag">the flip</span>
              </h2>
              <div className="tablewrap">
                <PlayerTable
                  rows={data.bottom_per_action.slice(0, 6)}
                  cols={["vaep_per_action", "vaep_state_delta", "pv_concede"]}
                  headers={["VAEP/act", "VAEP state", "PV concede"]}
                  dps={[2, 2, 2]}
                />
              </div>
              <p className="note">
                Bottom of per-action VAEP is keepers &amp; centre-backs
                (structural end-coordinate concede bias). Canonical{" "}
                <b>state-delta</b> credits their recoveries &mdash; watch the
                sign flip.
              </p>
            </div>
            <div>
              <h2>
                Top 5 by VAEP <span className="tag">state-delta</span>
              </h2>
              <div className="tablewrap">
                <PlayerTable
                  rows={data.top_state.slice(0, 5)}
                  cols={["vaep_state_delta", "vaep_per_action"]}
                  headers={["VAEP state", "VAEP/act"]}
                  dps={[2, 2]}
                />
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
