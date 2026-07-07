"use client";

import { W, H, SX, SY } from "@/lib/utils";

export default function PitchMarkings() {
  const boxH = 40.3 * SY;
  const boxY = (34 - 20.15) * SY;
  const boxD = 16.5 * SX;
  const sixH = 18.3 * SY;
  const sixY = (34 - 9.15) * SY;
  const sixD = 5.5 * SX;

  return (
    <>
      <rect x={1} y={1} width={W - 2} height={H - 2} className="pline" />
      <line x1={W / 2} y1={0} x2={W / 2} y2={H} className="pline" />
      <circle cx={W / 2} cy={H / 2} r={9.15 * SX} className="pline" />
      <circle cx={W / 2} cy={H / 2} r={2} fill="rgba(255,255,255,0.55)" />
      <rect x={0} y={boxY} width={boxD} height={boxH} className="pline" />
      <rect
        x={W - boxD}
        y={boxY}
        width={boxD}
        height={boxH}
        className="pline"
      />
      <rect x={0} y={sixY} width={sixD} height={sixH} className="pline" />
      <rect
        x={W - sixD}
        y={sixY}
        width={sixD}
        height={sixH}
        className="pline"
      />
      <circle cx={11 * SX} cy={H / 2} r={2} fill="rgba(255,255,255,0.55)" />
      <circle cx={W - 11 * SX} cy={H / 2} r={2} fill="rgba(255,255,255,0.55)" />
    </>
  );
}
