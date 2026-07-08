"use client";

import { useState } from "react";
import MatchAnalyzer from "@/components/MatchAnalyzer";
import ThreatGridExplorer from "@/components/ThreatGridExplorer";

type Tab = "analyzer" | "explorer";

export default function Home() {
  const [tab, setTab] = useState<Tab>("explorer");

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <h1>Flow of Threat</h1>
          <p className="sub">
            xT &middot; PV &middot; VAEP
          </p>
        </div>
        <nav className="sidebar-nav">
          <button
            className={"snav-item" + (tab === "explorer" ? " active" : "")}
            onClick={() => setTab("explorer")}
          >
            <span className="snav-icon">▦</span>
            Threat Grid Explorer
          </button>
          <button
            className={"snav-item" + (tab === "analyzer" ? " active" : "")}
            onClick={() => setTab("analyzer")}
          >
            <span className="snav-icon">⚡</span>
            Match Analyzer
          </button>
        </nav>
      </aside>

      <main className="main-content">
        <div style={{ display: tab === "explorer" ? "block" : "none" }}>
          <ThreatGridExplorer active={tab === "explorer"} />
        </div>
        <div style={{ display: tab === "analyzer" ? "block" : "none" }}>
          <MatchAnalyzer />
        </div>
        <footer>
          <p>
            Markov xT (12&times;8) &middot; per-action PV / concede XGBoost
            &middot; calibrated xG &middot; canonical Decroos state-delta VAEP.
          </p>
        </footer>
      </main>
    </div>
  );
}
