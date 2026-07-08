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
          <p className="sub">xT &middot; PV &middot; VAEP</p>
        </div>
        <nav className="sidebar-nav">
          <button
            className={"snav-item" + (tab === "explorer" ? " active" : "")}
            onClick={() => setTab("explorer")}
          >
            <svg
              className="snav-icon"
              viewBox="0 0 20 20"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <rect x="3" y="3" width="6" height="6" />
              <rect x="11" y="3" width="6" height="6" />
              <rect x="3" y="11" width="6" height="6" />
              <rect x="11" y="11" width="6" height="6" />
            </svg>
            xT Grid
          </button>
          <button
            className={"snav-item" + (tab === "analyzer" ? " active" : "")}
            onClick={() => setTab("analyzer")}
          >
            <svg
              className="snav-icon"
              viewBox="0 0 20 20"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M13 2.5L7 10.5H10.5L8.5 17.5L15 9H11.5L13 2.5Z" />
            </svg>
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
