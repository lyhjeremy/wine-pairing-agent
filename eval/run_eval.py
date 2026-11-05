"""Score the pairing agent on the frozen dish set and measure cost and latency.

    python -m eval.run_eval

For each dish in eval/dishes.json:
  * baseline: the same search tool queried with the raw dish text (no parse or
    strategize step), top hit taken as the pick;
  * agent: the full LangGraph agent, unchanged, with its default 2 attempts.
A pick matches when its Wine Enthusiast variety is in the dish's accepted list.

The Claude CLI is called with the agent's own prompts and flags plus
`--output-format json`, which adds `total_cost_usd` (the API-equivalent cost;
the CLI runs on a subscription) and `duration_ms` to each reply.
Writes eval/runs.json, eval/calls.csv and eval/summary.json.

    python -m eval.run_eval --only risotto     # run named dishes, keep earlier runs
    python -m eval.run_eval --baseline-only    # score the retrieval baseline on all dishes
    python -m eval.run_eval --summary-only     # rewrite summary.json from saved runs

Every pairing makes up to 10 CLI calls (see the README), so a full pass over the
set costs about $12 at API prices; runs.json is resumable dish by dish.
"""

from __future__ import annotations

import csv
import inspect
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

from src import llm as llm_mod
from src.graph import build_agent
from src.tools import search_wines

EVAL = Path(__file__).resolve().parent
CALLS: list[dict] = []
CURRENT = {"dish": None}


def _node_name() -> str:
    for fr in inspect.stack():
        if fr.filename.endswith("graph.py") and fr.function not in {"_llm"}:
            return fr.function
    return "?"


def _complete_cli_json(self, prompt, system):
    cmd = llm_mod._claude_command() + ["-p"]
    if system:
        cmd += ["--append-system-prompt", system]
    if llm_mod.DEFAULT_CLI_MODEL:
        cmd += ["--model", llm_mod.DEFAULT_CLI_MODEL]
    cmd += ["--output-format", "json"]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=self.timeout)
    wall = time.perf_counter() - t0
    if res.returncode != 0:
        raise llm_mod.LLMError(f"claude CLI failed: {res.stderr.strip()}")
    out = json.loads(res.stdout)
    u = out.get("usage", {})
    row = {
        "dish": CURRENT["dish"], "node": _node_name(),
        "wall_s": round(wall, 2), "duration_ms": out.get("duration_ms"),
        "cost_usd": out.get("total_cost_usd"),
        "input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens"),
        "cache_read": u.get("cache_read_input_tokens"),
        "cache_write": u.get("cache_creation_input_tokens"),
        "models": "+".join(sorted(out.get("modelUsage", {}).keys())),
    }
    CALLS.append(row)
    path = EVAL / "calls.csv"
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    return (out.get("result") or "").strip()


llm_mod.LLM._complete_cli = _complete_cli_json


def _wine(h) -> dict:
    m = h.meta
    return {"title": m.get("title"), "variety": m.get("variety"),
            "price": m.get("price"), "points": m.get("points")}


def _pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def baseline_all(spec):
    rows = []
    for d in spec["dishes"]:
        base = search_wines(d["meal"], max_price=d["budget"], k=6)
        pick = _wine(base[0]) if base else None
        rows.append({"id": d["id"], "meal": d["meal"], "budget": d["budget"],
                     "pick": pick, "top6": [_wine(h) for h in base],
                     "match": bool(pick and pick["variety"] in set(d["accept"]))})
    (EVAL / "baseline.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"baseline: {sum(r['match'] for r in rows)}/{len(rows)} top picks match")
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    spec = json.loads((EVAL / "dishes.json").read_text())
    base_rows = baseline_all(spec)
    if args.baseline_only:
        return
    agent = build_agent()
    runs_path = EVAL / "runs.json"
    runs = json.loads(runs_path.read_text()) if runs_path.exists() else []
    done = {r["id"] for r in runs}
    for d in ([] if args.summary_only else spec["dishes"]):
        if d["id"] in done or (args.only and d["id"] not in args.only):
            continue
        accept = set(d["accept"])
        base = search_wines(d["meal"], max_price=d["budget"], k=6)
        base_pick = _wine(base[0]) if base else None

        CURRENT["dish"] = d["id"]
        n0 = len(CALLS)
        t0 = time.perf_counter()
        err = None
        try:
            out = agent.invoke({"request": {"meal": d["meal"], "budget": d["budget"],
                                            "color_pref": None},
                                "max_attempts": 2}, {"recursion_limit": 50})
        except Exception as e:  # keep the panel going
            out, err = {}, f"{type(e).__name__}: {e}"
        wall = time.perf_counter() - t0
        calls = CALLS[n0:]
        cands = [_wine(h) for h in out.get("candidates", [])]
        rec = out.get("recommendation", "")
        cited = []
        for n in re.findall(r"\[(\d+)\]", rec):
            n = int(n)
            if n not in cited:
                cited.append(n)
        picks = [cands[n - 1] for n in cited if 1 <= n <= len(cands)]
        runs.append({
            "id": d["id"], "meal": d["meal"], "budget": d["budget"],
            "baseline_pick": base_pick,
            "baseline_match": bool(base_pick and base_pick["variety"] in accept),
            "strategy": out.get("strategy"), "candidates": cands,
            "recommendation": rec, "cited": cited,
            "invalid_citations": [n for n in cited if not 1 <= n <= len(cands)],
            "agent_pick": picks[0] if picks else None,
            "agent_match": bool(picks and picks[0]["variety"] in accept),
            "all_picks_match": bool(picks) and all(p["variety"] in accept for p in picks),
            "over_budget": [p["title"] for p in picks if (p["price"] or 0) > d["budget"]],
            "critique": out.get("critique"), "attempts": out.get("attempts", 0),
            "log": out.get("log", []), "error": err,
            "llm_calls": len(calls), "wall_s": round(wall, 2),
            "cost_usd": round(sum(c["cost_usd"] or 0 for c in calls), 5),
        })
        r = runs[-1]
        print(f"{d['id']:<14} base={r['baseline_match']!s:<5} agent={r['agent_match']!s:<5} "
              f"calls={r['llm_calls']} {r['wall_s']:.1f}s ${r['cost_usd']:.4f}", flush=True)
        (EVAL / "runs.json").write_text(json.dumps(runs, indent=2, ensure_ascii=False))

    ok = [r for r in runs if not r["error"]]
    walls = [r["wall_s"] for r in ok]
    costs = [r["cost_usd"] for r in ok]
    import platform
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                          capture_output=True, text=True).stdout.strip()
    summary = {
        "date": time.strftime("%Y-%m-%d"), "machine": chip or platform.machine(),
        "n_dishes": len(runs), "n_completed": len(ok),
        "baseline_matches_all_dishes": f'{sum(b["match"] for b in base_rows)} of {len(base_rows)}',
        "baseline_matches_on_agent_dishes": sum(r["baseline_match"] for r in runs),
        "agent_matches": sum(r["agent_match"] for r in runs),
        "agent_all_picks_match": sum(r["all_picks_match"] for r in runs),
        "critique_passed": sum(bool(r["critique"] and r["critique"].get("ok")) for r in ok),
        "runs_with_retry": sum(r["attempts"] > 0 for r in ok),
        "llm_calls": sum(r["llm_calls"] for r in runs),
        "llm_calls_per_pairing": sorted(r["llm_calls"] for r in runs),
        "cost_per_pairing_median": round(statistics.median(costs), 4),
        "cost_per_pairing_mean": round(statistics.mean(costs), 4),
        "latency_median_s": round(statistics.median(walls), 1),
        "latency_p95_s": round(_pct(walls, 0.95), 1),
        "models": sorted({row["models"] for row in csv.DictReader(open(EVAL / "calls.csv"))}),
        "cost_note": "total_cost_usd from claude -p --output-format json: API-equivalent, "
                     "nothing billed (Claude subscription)",
    }
    (EVAL / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
