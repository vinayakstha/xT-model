"use client";

import { useState } from "react";
import MatchAnalyzer from "@/components/MatchAnalyzer";
import ThreatGridExplorer from "@/components/ThreatGridExplorer";

type Tab = "analyzer" | "explorer";

export default function Home() {
  const [tab, setTab] = useState<Tab>("explorer");
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="app-layout">
      <aside className={"sidebar" + (collapsed ? " collapsed" : "")}>
        <div className="sidebar-inner">
          <div className="sidebar-brand">
            <h1>Flow of Threat</h1>
            <button
              className="sidebar-toggle"
              onClick={() => setCollapsed(!collapsed)}
              title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            >
              <svg
                viewBox="0 0 16 16"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                {collapsed ? (
                  <path d="M10 3L6 8L10 13" />
                ) : (
                  <path d="M6 3L10 8L6 13" />
                )}
              </svg>
            </button>
          </div>
          <p className={"sub" + (collapsed ? " hidden" : "")}>
            <small>x</small>T &middot; PV &middot; VAEP
          </p>
          <nav className="sidebar-nav">
            <button
              className={"snav-item" + (tab === "explorer" ? " active" : "")}
              onClick={() => setTab("explorer")}
              title="Threat Grid Explorer"
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
              <span className={collapsed ? "hidden" : ""}><small>x</small>T Grid</span>
            </button>
            <button
              className={"snav-item" + (tab === "analyzer" ? " active" : "")}
              onClick={() => setTab("analyzer")}
              title="Match Analyzer"
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
              <span className={collapsed ? "hidden" : ""}>Match Analyzer</span>
            </button>
          </nav>
        </div>
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
            Markov <small>x</small>T (12&times;8) &middot; per-action PV / concede XGBoost
            &middot; calibrated <small>x</small>G &middot; canonical Decroos state-delta VAEP.
          </p>
        </footer>
      </main>
    </div>
  );
}
