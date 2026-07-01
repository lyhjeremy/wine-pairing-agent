"""Recommend a wine for your meal with the LangGraph pairing agent.

    python -m src.cli "grilled ribeye with peppercorn sauce" --budget 40
    python -m src.cli "spicy Thai green curry" --budget 25 --color white
    python -m src.cli                      # interactive

Build the wine index first:  python fetch_data.py && python -m src.ingest
"""

from __future__ import annotations

import argparse

from .graph import build_agent


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
    args = ap.parse_args()

    meal = args.meal or input("What are you eating? ").strip()
    budget = args.budget
    if args.meal is None and budget is None:
        raw = input("Budget per bottle (blank for any) $").strip()
        budget = float(raw) if raw else None

    request = {"meal": meal, "budget": budget, "color_pref": args.color}
    agent = build_agent()
    print("\n…running agent (parse → strategize → retrieve → recommend → critique)…")
    result = agent.invoke(
        {"request": request, "max_attempts": args.max_attempts},
        {"recursion_limit": 50},
    )
    _print(result)


if __name__ == "__main__":
    main()
