"use client";

import { useState } from "react";
import MatchAnalyzer from "@/components/MatchAnalyzer";
import ThreatGridExplorer from "@/components/ThreatGridExplorer";

type Tab = "analyzer" | "explorer";

export default function Home() {
  const [tab, setTab] = useState<Tab>("explorer");

  return (
    <div className="app-layout">
      <nav className="navbar">
        <div className="navbar-brand">
          <h1>xT-ENGINE</h1>
          <p className="navbar-sub">
            <small>x</small>T &middot; PV &middot; VAEP
          </p>
        </div>
        <div className="navbar-actions">
          <button
            className={"navbar-btn" + (tab === "explorer" ? " active" : "")}
            onClick={() => setTab("explorer")}
          >
            <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="navbar-icon">
              <rect x="3" y="3" width="6" height="6" />
              <rect x="11" y="3" width="6" height="6" />
              <rect x="3" y="11" width="6" height="6" />
              <rect x="11" y="11" width="6" height="6" />
            </svg>
            <span><small>x</small>T Grid</span>
          </button>
          <button
            className={"navbar-btn" + (tab === "analyzer" ? " active" : "")}
            onClick={() => setTab("analyzer")}
          >
            <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="navbar-icon">
              <path d="M13 2.5L7 10.5H10.5L8.5 17.5L15 9H11.5L13 2.5Z" />
            </svg>
            <span>Match Analyzer</span>
          </button>
        </div>
      </nav>

      <div className="app-body">
        <main className="main-content">
          <div style={{ display: tab === "explorer" ? "block" : "none" }}>
            <ThreatGridExplorer active={tab === "explorer"} />
          </div>
          <div style={{ display: tab === "analyzer" ? "block" : "none" }}>
            <MatchAnalyzer />
          </div>
          <footer>
            <p>
              Markov <small>x</small>T (12&times;8) &middot; per-action PV / concede XGBoost
              &middot; calibrated <small>x</small>G &middot; canonical Decroos state-delta VAEP.
            </p>
          </footer>
        </main>
      </div>
    </div>
  );
}
