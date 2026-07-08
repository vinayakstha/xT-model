"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import PitchMarkings from "./Pitch";
import { W, H, heat, heatTextColor, zoneCenter } from "@/lib/utils";
import type { GridMode, Period, EventType, MoveResult } from "@/lib/types";

export default function ThreatGridExplorer({ active }: { active: boolean }) {
  const [grid, setGrid] = useState<GridMode>("xt");
  const [period, setPeriod] = useState<Period>("combined");
  const [type, setType] =
    useState<Extract<EventType, "pass" | "carry" | "dribble">>("pass");
  const [success, setSuccess] = useState(true);
  const [pressure, setPressure] = useState(false);
  const [counter, setCounter] = useState(false);
  const [start, setStart] = useState<[number, number] | null>(null);
  const [end, setEnd] = useState<[number, number] | null>(null);
  const [zones, setZones] = useState<number[][] | null>(null);
  const [maxVal, setMaxVal] = useState(1);
  const [move, setMove] = useState<MoveResult | null>(null);
  const loadedRef = useRef(false);

  const loadGrid = useCallback(async () => {
    const d = await (
      await fetch(`/api/grid?mode=${grid}&period=${period}`)
    ).json();
    setZones(d.zones);
    setMaxVal(d.max_value || 1);
  }, [grid, period]);

  useEffect(() => {
    if (active && !loadedRef.current) {
      loadedRef.current = true;
      loadGrid();
    }
  }, [active, loadGrid]);

  useEffect(() => {
    if (loadedRef.current) loadGrid();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [grid, period]);

  const computeMove = useCallback(
    async (s: [number, number], e: [number, number]) => {
      const [sx, sy] = zoneCenter(...s);
      const [ex, ey] = zoneCenter(...e);
      const body = {
        start_x: sx,
        start_y: sy,
        end_x: ex,
        end_y: ey,
        event_type: type,
        success,
        under_pressure: pressure,
        counter_attack: counter,
        period,
      };
      try {
        const d: MoveResult = await (
          await fetch("/api/move", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          })
        ).json();
        setMove(d);
      } catch (e) {
        setMove(null);
      }
    },
    [type, success, pressure, counter, period],
  );

  useEffect(() => {
    if (start && end) computeMove(start, end);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [type, success, pressure, counter]);

  const clickZone = (zx: number, zy: number) => {
    if (!start || (start && end)) {
      setStart([zx, zy]);
      setEnd(null);
      setMove(null);
    } else {
      setEnd([zx, zy]);
      setMove(null);
      computeMove(start, [zx, zy]);
    }
  };

  const reset = () => {
    setStart(null);
    setEnd(null);
    setMove(null);
  };

  const cw = W / 12,
    ch = H / 8;

  const sign = (x: number) => (x >= 0 ? "+" : "") + x.toFixed(4);

  return (
    <div>
      <section className="panel">
        <div className="controls wrap">
          <span className="ctl">
            <b>Grid</b>
            {(["xt", "xg", "pv"] as GridMode[]).map((g) => (
              <button
                key={g}
                className={"seg" + (grid === g ? " active" : "")}
                onClick={() => setGrid(g)}
              >
                {g.toUpperCase()}
              </button>
            ))}
          </span>
          {grid === "xt" && (
            <span className="ctl">
              <b>Period</b>
              {(
                [
                  ["combined", "Full"],
                  ["p1", "1st"],
                  ["p2", "2nd"],
                  ["et", "ET"],
                ] as [Period, string][]
              ).map(([p, label]) => (
                <button
                  key={p}
                  className={"seg" + (period === p ? " active" : "")}
                  onClick={() => setPeriod(p)}
                >
                  {label}
                </button>
              ))}
            </span>
          )}
          <span className="sep" />
          <span className="ctl">
            <b>Type</b>
            {(["pass", "carry", "dribble"] as const).map((t) => (
              <button
                key={t}
                className={"seg" + (type === t ? " active" : "")}
                onClick={() => setType(t)}
              >
                {t[0].toUpperCase() + t.slice(1)}
              </button>
            ))}
          </span>
          <label className="chk">
            <input
              type="checkbox"
              checked={success}
              onChange={(e) => setSuccess(e.target.checked)}
            />{" "}
            Success
          </label>
          <label className="chk">
            <input
              type="checkbox"
              checked={pressure}
              onChange={(e) => setPressure(e.target.checked)}
            />{" "}
            Pressure
          </label>
          <label className="chk">
            <input
              type="checkbox"
              checked={counter}
              onChange={(e) => setCounter(e.target.checked)}
            />{" "}
            Counter
          </label>
          <button className="ghost" onClick={reset}>
            Reset
          </button>
        </div>
        <p className="hint">
          Each cell is the value of being in that zone.{" "}
          <b>Click a cell to set the start</b>, then a second cell for the end
          &mdash; you get the move's xT (end &minus; start) and the model PV of
          that exact action.
        </p>
      </section>

      <section>
        <div className="pitchwrap">
          <svg viewBox={`0 0 ${W} ${H}`}>
            {zones &&
              Array.from({ length: 12 }).map((_, zx) =>
                Array.from({ length: 8 }).map((_, zy) => {
                  const v = zones[zx][zy];
                  const x = zx * cw,
                    y = zy * ch;
                  const isStart = start && start[0] === zx && start[1] === zy;
                  const isEnd = end && end[0] === zx && end[1] === zy;
                  return (
                    <g key={`${zx}-${zy}`}>
                      <rect
                        x={x + 0.5}
                        y={y + 0.5}
                        width={cw - 1}
                        height={ch - 1}
                        fill={heat(v / maxVal)}
                        className={
                          "cell" +
                          (isStart ? " start" : "") +
                          (isEnd ? " end" : "")
                        }
                        stroke="rgba(0,0,0,0.25)"
                        strokeWidth={1}
                        onClick={() => clickZone(zx, zy)}
                      />
                      <text
                        x={x + cw / 2}
                        y={y + ch / 2 + 1}
                        className="cellval"
                        fill={heatTextColor(v / maxVal)}
                      >
                        {(v * 100).toFixed(2)}%
                      </text>
                      <text
                        x={x + cw / 2}
                        y={y + ch / 2 + 14}
                        className="cellidx"
                        fill={heatTextColor(v / maxVal)}
                      >
                        ({zx},{zy})
                      </text>
                    </g>
                  );
                }),
              )}

            <PitchMarkings />

            {start && (
              <>
                <defs>
                  <marker
                    id="gm-arrow"
                    markerWidth={9}
                    markerHeight={9}
                    refX={8}
                    refY={4.5}
                    orient="auto"
                    markerUnits="userSpaceOnUse"
                  >
                    <path d="M0,0 L9,4.5 L0,9 Z" fill="#ffffff" />
                  </marker>
                </defs>
                {(() => {
                  const ctr = (p: [number, number]): [number, number] => [
                    p[0] * cw + cw / 2,
                    p[1] * ch + ch / 2,
                  ];
                  const [sx, sy] = ctr(start);
                  return (
                    <>
                      {end &&
                        (() => {
                          const [ex, ey] = ctr(end);
                          const dx = ex - sx,
                            dy = ey - sy;
                          const len = Math.hypot(dx, dy) || 1;
                          const ux = dx / len,
                            uy = dy / len;
                          return (
                            <>
                              <line
                                x1={sx + ux * 8}
                                y1={sy + uy * 8}
                                x2={ex - ux * 11}
                                y2={ey - uy * 11}
                                stroke="#ffffff"
                                strokeWidth={2.5}
                                markerEnd="url(#gm-arrow)"
                              />
                              {move &&
                                (() => {
                                  const mx = (sx + ex) / 2,
                                    my = (sy + ey) / 2;
                                  const txt =
                                    "PV: " +
                                    (move.model_pv * 100).toFixed(2) +
                                    "%";
                                  const tw = txt.length * 6.2 + 12;
                                  return (
                                    <>
                                      <rect
                                        x={mx - tw / 2}
                                        y={my - 10}
                                        width={tw}
                                        height={17}
                                        rx={4}
                                        fill="#0b1220"
                                        fillOpacity={0.92}
                                        stroke="#34d399"
                                      />
                                      <text
                                        x={mx}
                                        y={my + 2.5}
                                        textAnchor="middle"
                                        fill="#34d399"
                                        fontSize={10.5}
                                        fontWeight={700}
                                      >
                                        {txt}
                                      </text>
                                    </>
                                  );
                                })()}
                              <circle
                                cx={ex}
                                cy={ey}
                                r={8}
                                fill="none"
                                stroke="#3b82f6"
                                strokeWidth={3}
                              />
                              <circle cx={ex} cy={ey} r={3.5} fill="#ffffff" />
                              <text
                                x={ex}
                                y={ey - 13}
                                textAnchor="middle"
                                fill="#ffffff"
                                fontSize={9.5}
                                fontWeight={700}
                              >
                                END
                              </text>
                            </>
                          );
                        })()}
                      <circle
                        cx={sx}
                        cy={sy}
                        r={7}
                        fill="#3b82f6"
                        stroke="#ffffff"
                        strokeWidth={2}
                      />
                      <text
                        x={sx}
                        y={sy - 12}
                        textAnchor="middle"
                        fill="#ffffff"
                        fontSize={9.5}
                        fontWeight={700}
                      >
                        START
                      </text>
                    </>
                  );
                })()}
              </>
            )}
          </svg>
        </div>

        <div className="movecard">
          {!start && "xT: click the pitch to begin."}
          {start && !end && zones && (
            <>
              Start zone{" "}
              <b>
                ({start[0]},{start[1]})
              </b>{" "}
              = {(zones[start[0]][start[1]] * 100).toFixed(2)}%. Now click the
              end zone.
            </>
          )}
          {start && end && move && (
            <>
              <b>{type}</b> from ({start.join(",")}) &rarr; ({end.join(",")}),{" "}
              {success ? "success" : "failure"} &nbsp;|&nbsp; zone xT = end
              &minus; start = {(move.end_zone_xt * 100).toFixed(2)}% &minus;{" "}
              {(move.start_zone_xt * 100).toFixed(2)}% ={" "}
              <b>{sign(move.zone_based_xt * 100)}%</b>
              {!success && " \u00A0(0 on failure — Karun convention)"}
              <br />
              model PV of this action ={" "}
              <b>{(move.model_pv * 100).toFixed(2)}%</b> &nbsp;|&nbsp; PV
              concede = {(move.pv_concede * 100).toFixed(2)}% &nbsp;|&nbsp; VAEP
              (per-action) = <b>{sign(move.vaep_per_action * 100)}%</b>
            </>
          )}
        </div>
      </section>
    </div>
  );
}
