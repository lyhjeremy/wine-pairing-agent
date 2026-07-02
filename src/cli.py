"""Recommend a wine for your meal with the LangGraph pairing agent.

    python -m src.cli "grilled ribeye with peppercorn sauce" --budget 40
    python -m src.cli "spicy Thai green curry" --budget 25 --color white
    python -m src.cli                      # interactive

Build the wine index first:  python fetch_data.py && python -m src.ingest
"""

from __future__ import annotations

import argparse
import sys

from .graph import build_agent

# Windows consoles/pipes default to cp1252, which can't encode the → and emoji in
# our output; force UTF-8 so the CLI prints cleanly on every platform.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _print(result: dict) -> None:
    print("\n" + "=" * 64)
    print("PAIRING")
    print("=" * 64)
    print(result.get("recommendation", "").strip())

    cands = result.get("candidates", [])
    if cands:
        print("\n— Candidate wines (retrieved from 130k reviews) —")
        for i, h in enumerate(cands, 1):
            print(f"  [{i}] {h.citation()}")

    print("\n" + "=" * 64)
    print("AGENT TRACE")
    print("=" * 64)
    for step in result.get("log", []):
        print(f"  • {step}")
    c = result.get("critique", {})
    print("  ✔ pairing passed review" if c.get("ok")
          else f"  ⚠ finalized with notes: {c.get('issues')}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Wine Pairing Agent (LangGraph)")
    ap.add_argument("meal", nargs="?", help="the dish, e.g. 'grilled ribeye with pepper sauce'")
    ap.add_argument("--budget", type=float, default=None, help="max $ per bottle")
    ap.add_argument("--color", default=None, help="red | white | rosé | sparkling (optional)")
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--trace", action="store_true",
                    help="print a timed node-by-node trace and save it to traces/")
    args = ap.parse_args()

    meal = args.meal or input("What are you eating? ").strip()
    budget = args.budget
    if args.meal is None and budget is None:
        raw = input("Budget per bottle (blank for any) $").strip()
        budget = float(raw) if raw else None

    request = {"meal": meal, "budget": budget, "color_pref": args.color}
    agent = build_agent()
    print("\n…running agent (parse → strategize → retrieve → recommend → critique)…")
    inputs = {"request": request, "max_attempts": args.max_attempts}
    if args.trace:
        from .trace import run_traced

        result, trace = run_traced(agent, inputs)
        _print(result)
        print(trace.render())
        print(f"\n(trace saved to {trace.save(name='last_run')})")
    else:
        result = agent.invoke(inputs, {"recursion_limit": 50})
        _print(result)


if __name__ == "__main__":
    main()
