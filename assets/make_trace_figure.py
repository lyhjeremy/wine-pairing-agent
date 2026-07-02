"""Render a saved agent trace (traces/*.json) as a node-timeline figure.

Run the agent once with --trace to produce a trace, then:
    python assets/make_trace_figure.py traces/last_run.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

COLORS = {
    "parse": "#7C1E2B", "strategize": "#48283C", "retrieve": "#C8963A",
    "recommend": "#5C3A52", "critique": "#8B2E3F", "bump": "#B0A08F",
}


def main(path: str) -> None:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    steps = data["steps"]

    fig, ax = plt.subplots(figsize=(9.5, 0.62 * len(steps) + 1.4))
    fig.patch.set_facecolor("#FBF4EA")
    ax.set_facecolor("#FBF4EA")

    t = 0.0
    for i, s in enumerate(steps):
        dur = s["ms"] / 1000.0
        y = len(steps) - i
        ax.barh(y, dur, left=t, height=0.62, color=COLORS.get(s["node"], "#888888"),
                edgecolor="white")
        summary = s["summary"][:46] + ("…" if len(s["summary"]) > 46 else "")
        ax.text(t + dur + 0.15, y, f"{s['node']} · {summary}", va="center", fontsize=9,
                color="#2A2028")
        t += dur

    ax.set_xlim(0, t * 1.65 if t else 1)
    ax.set_ylim(0.3, len(steps) + 0.9)
    ax.set_yticks([])
    ax.set_xlabel("seconds")
    ax.set_title(f"Agent run trace — {len(steps)} nodes, {data['total_ms']/1000:.1f}s total",
                 fontsize=12, fontweight="bold", color="#5A1E28")
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()

    out = ROOT / "assets" / "trace_timeline.png"
    fig.savefig(out, dpi=150, facecolor="#FBF4EA")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "traces" / "last_run.json"))
