"""Lightweight tracing / observability for the agent.

Runs the LangGraph agent via ``.stream()`` so every node fire is captured with a
wall-clock duration and a one-line summary, then renders a readable trace tree
and can export the whole run as JSON (LangSmith-style, but self-contained — no
account or network needed). Enable it from the CLI with ``--trace``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACE_DIR = ROOT / "traces"


def _summarize(node: str, delta: dict) -> str:
    if node == "parse":
        d = delta.get("dish", {})
        return f"{d.get('main', '?')} · {d.get('body', '?')} body"
    if node == "strategize":
        s = delta.get("strategy", {})
        return f"style → {s.get('target_style', '?')}"
    if node == "retrieve":
        return f"{len(delta.get('candidates', []))} candidate wines (tool: search_wines)"
    if node == "recommend":
        return "drafted pairing with citations"
    if node == "critique":
        c = delta.get("critique", {})
        return "sound ✓" if c.get("ok") else f"{len(c.get('issues', []))} issue(s) → retry"
    if node == "bump":
        return f"retry #{delta.get('attempts', '?')}"
    return ""


@dataclass
class Event:
    step: int
    node: str
    summary: str
    ms: float


@dataclass
class RunTrace:
    events: list[Event] = field(default_factory=list)

    def record(self, node: str, summary: str, ms: float) -> None:
        self.events.append(Event(len(self.events) + 1, node, summary, ms))

    @property
    def total_ms(self) -> float:
        return sum(e.ms for e in self.events)

    def render(self) -> str:
        width = max((len(e.node) for e in self.events), default=6)
        lines = ["", "AGENT TRACE  (node · time · detail)", "─" * 52]
        for e in self.events:
            lines.append(f"  ● {e.node:<{width}}  {e.ms/1000:5.1f}s   {e.summary}")
        lines.append("─" * 52)
        lines.append(f"  {len(self.events)} steps · {self.total_ms/1000:.1f}s total")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "total_ms": round(self.total_ms, 1),
            "steps": [
                {"step": e.step, "node": e.node, "ms": round(e.ms, 1), "summary": e.summary}
                for e in self.events
            ],
        }

    def save(self, name: str = "run") -> Path:
        TRACE_DIR.mkdir(exist_ok=True)
        path = TRACE_DIR / f"{name}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def run_traced(agent, inputs: dict, config: dict | None = None):
    """Invoke the agent while timing each node. Returns (final_state, RunTrace)."""
    trace = RunTrace()
    state: dict = {}
    last = time.perf_counter()
    for step in agent.stream(inputs, config or {"recursion_limit": 50}):
        now = time.perf_counter()
        for node, delta in step.items():
            state.update(delta)
            trace.record(node, _summarize(node, delta), (now - last) * 1000)
        last = now
    return state, trace
