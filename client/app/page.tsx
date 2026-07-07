"use client";

import { useState } from "react";
import MatchAnalyzer from "@/components/MatchAnalyzer";
import ThreatGridExplorer from "@/components/ThreatGridExplorer";

type Tab = "analyzer" | "explorer";

export default function Home() {
  const [tab, setTab] = useState<Tab>("analyzer");

  return (
    <>
      <header>
        <h1>Flow of Threat</h1>
        <p className="sub">
          xT &middot; PV &middot; VAEP &mdash; the production Markov + XGBoost
          pipeline, runnable on any match
        </p>
        <nav className="tabs">
          <button
            className={"tab" + (tab === "analyzer" ? " active" : "")}
            onClick={() => setTab("analyzer")}
          >
            Match Analyzer
          </button>
          <button
            className={"tab" + (tab === "explorer" ? " active" : "")}
            onClick={() => setTab("explorer")}
          >
            Threat Grid Explorer
          </button>
        </nav>
      </header>

      <div style={{ display: tab === "analyzer" ? "block" : "none" }}>
        <MatchAnalyzer />
      </div>
      <div style={{ display: tab === "explorer" ? "block" : "none" }}>
        <ThreatGridExplorer active={tab === "explorer"} />
      </div>

      <footer>
        <p>
          Markov xT (12&times;8) &middot; per-action PV / concede XGBoost
          &middot; calibrated xG &middot; canonical Decroos state-delta VAEP.
        </p>
      </footer>
    </>
  );
}
