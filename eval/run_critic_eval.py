"""Measure the rubric critique on the frozen 10-dish set.

    python -m eval.run_critic_eval              # 1. agent with the rubric critic, all dishes
    python -m eval.run_critic_eval --judges     # 2. re-judge each first-round draft
    python -m eval.run_critic_eval --summary    # 3. write eval/critic/summary.json

Step 1 runs the full LangGraph agent (parse, strategize, retrieve, recommend,
critique; max 2 retries) on every dish in eval/dishes.json with the Claude CLI on
Opus, the model the 2026-11-05 run used. Each run is streamed node by node and
saved as a trace in eval/traces/<dish>.json: node timings, every LLM call's
API-equivalent cost and duration as reported by `claude -p --output-format json`,
the search_wines tool call with its arguments and results, and the critique JSON.

Step 2 takes the first draft of each dish (the same dish, candidates and
recommendation the critic saw in round 1) and grades it three more ways:
  * legacy:  the original free-form critique prompt (Opus), as in src/graph.py
             critique_legacy, to compare pass rates on identical drafts;
  * second:  the same rubric on Sonnet, a different model tier, for agreement;
  * repeat:  the rubric on Opus again, for run-to-run stability.

Resumable: finished dishes and judgments are skipped. Writes under eval/critic/.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import time
from pathlib import Path

from src import graph as G
from src import llm as llm_mod
from src import rubric
from src.util import json_complete

EVAL = Path(__file__).resolve().parent
OUT = EVAL / "critic"
TRACES = EVAL / "traces"
SYSTEM_MODEL = "opus"
SECOND_MODEL = "sonnet"
CALLS: list[dict] = []
CTX = {"dish": None, "tag": None}


def _complete_cli_json(self, prompt, system):
    model = getattr(self, "model", SYSTEM_MODEL)
    cmd = llm_mod._claude_command() + ["-p", "--model", model, "--output-format", "json"]
    if system:
        cmd += ["--append-system-prompt", system]
    t0 = time.perf_counter()
    for attempt in range(3):
        res = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=self.timeout)
        if res.returncode == 0:
            break
        time.sleep(20 * (attempt + 1))
    else:
        raise llm_mod.LLMError(f"claude CLI failed: {res.stderr.strip()[:300]}")
    wall = time.perf_counter() - t0
    out = json.loads(res.stdout)
    u = out.get("usage", {})
    row = {
        "dish": CTX["dish"], "tag": CTX["tag"], "model_flag": model,
        "models": "+".join(sorted(out.get("modelUsage", {}).keys())),
        "wall_s": round(wall, 2), "duration_ms": out.get("duration_ms"),
        "cost_usd": out.get("total_cost_usd"),
        "input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens"),
        "cache_read": u.get("cache_read_input_tokens"),
        "cache_write": u.get("cache_creation_input_tokens"),
    }
    CALLS.append(row)
    OUT.mkdir(exist_ok=True)
    path = OUT / "calls.csv"
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    return (out.get("result") or "").strip()


llm_mod.LLM._complete_cli = _complete_cli_json


def _llm(model: str):
    x = llm_mod.LLM(backend="cli")
    x.model = model
    return x


def _wine(h, snippet=True) -> dict:
    m = h.meta
    d = {"title": m.get("title"), "variety": m.get("variety"), "price": m.get("price"),
         "points": m.get("points"), "country": m.get("country")}
    if snippet:
        note = h.doc.split("\n", 1)[-1]
        d["note"] = note if len(note) <= 220 else note[:217].rsplit(" ", 1)[0] + "..."
    return d


def _call_view(c: dict) -> dict:
    return {k: c[k] for k in ("models", "wall_s", "duration_ms", "cost_usd",
                              "input_tokens", "output_tokens", "cache_read", "cache_write")}


# --------------------------------------------------------------------------- #
# Step 1: agent runs with traces
# --------------------------------------------------------------------------- #
def run_agent(dishes, only=None):
    OUT.mkdir(exist_ok=True)
    TRACES.mkdir(exist_ok=True)
    runs_path = OUT / "runs.json"
    runs = json.loads(runs_path.read_text()) if runs_path.exists() else []
    done = {r["id"] for r in runs}
    G._LLM = _llm(SYSTEM_MODEL)
    agent = G.build_agent(critic="rubric")
    for d in dishes:
        if d["id"] in done or (only and d["id"] not in only):
            continue
        CTX.update(dish=d["id"], tag="agent")
        accept = set(d["accept"])
        inputs = {"request": {"meal": d["meal"], "budget": d["budget"], "color_pref": None},
                  "max_attempts": 2}
        state: dict = {}
        steps, rounds = [], []
        t_run = time.perf_counter()
        last = t_run
        n_calls = len(CALLS)
        err = None
        try:
            for upd in agent.stream(inputs, {"recursion_limit": 50}, stream_mode="updates"):
                now = time.perf_counter()
                for node, delta in upd.items():
                    prev = dict(state)
                    state.update(delta)
                    calls = [_call_view(c) for c in CALLS[n_calls:]]
                    n_calls = len(CALLS)
                    step = {"step": len(steps) + 1, "node": node,
                            "t_start_ms": round((last - t_run) * 1000),
                            "ms": round((now - last) * 1000), "llm_calls": calls}
                    if node == "parse":
                        step["input"] = {"meal": d["meal"], "budget": d["budget"]}
                        step["output"] = delta["dish"]
                    elif node == "strategize":
                        fb = prev.get("critique")
                        step["input"] = {"feedback": fb["issues"] if fb and not fb["ok"] else None}
                        step["output"] = delta["strategy"]
                    elif node == "retrieve":
                        step["tool_call"] = {"name": G.wine_search_tool.name,
                                             "args": {"query": state["strategy"]["search_query"],
                                                      "max_price": d["budget"], "k": 6}}
                        step["output"] = [_wine(h) for h in delta["candidates"]]
                    elif node == "recommend":
                        step["output"] = delta["recommendation"]
                    elif node == "critique":
                        c = delta["critique"]
                        step["output"] = {"ok": c["ok"], "criteria": c["criteria"],
                                          "schema_valid": c["schema_valid"],
                                          "repaired": c["repaired"]}
                        rounds.append({
                            "strategy": state["strategy"],
                            "candidates": [_wine(h, snippet=False) for h in state["candidates"]],
                            "candidates_text": G.candidates_text(state["candidates"]),
                            "recommendation": state["recommendation"],
                            "critique": c,
                        })
                    elif node == "bump":
                        step["output"] = {"attempts": delta["attempts"]}
                    steps.append(step)
                last = now
        except Exception as e:  # keep the panel going
            err = f"{type(e).__name__}: {e}"
        wall = time.perf_counter() - t_run
        cands = [_wine(h, snippet=False) for h in state.get("candidates", [])]
        refs = rubric.cited(state.get("recommendation", ""))
        picks = [cands[n - 1] for n in refs if 1 <= n <= len(cands)]
        all_calls = [c for s in steps for c in s["llm_calls"]]
        cost = round(sum(c["cost_usd"] or 0 for c in all_calls), 5)
        trace = {"dish": d["id"], "meal": d["meal"], "budget": d["budget"],
                 "date": time.strftime("%Y-%m-%d"), "machine": "Apple M4",
                 "system_model": SYSTEM_MODEL, "critic": "rubric",
                 "total_ms": round(wall * 1000), "cost_usd": cost,
                 "llm_calls": len(all_calls), "steps": steps,
                 "final": {"recommendation": state.get("recommendation"),
                           "critique_ok": bool(state.get("critique", {}).get("ok")),
                           "attempts": state.get("attempts", 0)},
                 "error": err}
        (TRACES / f"{d['id']}.json").write_text(json.dumps(trace, indent=1, ensure_ascii=False))
        runs.append({
            "id": d["id"], "meal": d["meal"], "budget": d["budget"],
            "dish": state.get("dish"), "rounds": rounds,
            "critique_rounds": len(rounds),
            "passed": bool(state.get("critique", {}).get("ok")),
            "passed_round": next((i + 1 for i, r in enumerate(rounds) if r["critique"]["ok"]), None),
            "agent_pick": picks[0] if picks else None,
            "agent_match": bool(picks and picks[0]["variety"] in accept),
            "all_picks_match": bool(picks) and all(p["variety"] in accept for p in picks),
            "over_budget": [p["title"] for p in picks if (p["price"] or 0) > d["budget"]],
            "llm_calls": len(all_calls), "wall_s": round(wall, 2), "cost_usd": cost,
            "error": err,
        })
        r = runs[-1]
        print(f"{d['id']:<14} passed={r['passed']!s:<5} rounds={r['critique_rounds']} "
              f"match={r['agent_match']!s:<5} calls={r['llm_calls']} {r['wall_s']:.1f}s "
              f"${r['cost_usd']:.3f} {err or ''}", flush=True)
        runs_path.write_text(json.dumps(runs, indent=1, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# Step 2: re-judge first-round drafts
# --------------------------------------------------------------------------- #
class _Hit:
    def __init__(self, w):
        self.price = w.get("price")


def legacy_verdict(llm, dish, draft, budget) -> dict:
    """The original critique: same code checks and same prompt as critique_legacy."""
    issues = []
    if len(draft["candidates"]) < 2:
        issues.append("too few candidate wines were found")
    if budget and draft["candidates"] and not any((w["price"] or 0) <= budget
                                                   for w in draft["candidates"]):
        issues.append(f"no candidates within the ${budget:g} budget")
    review = json_complete(
        llm,
        f"Dish: {dish}\nProposed pairing:\n{draft['recommendation']}\n\n"
        "Is this a genuinely good pairing for the dish (right body, acid/tannin logic)? "
        'Return JSON: {"issues": [short strings], "ok": bool}.',
        system="You are a critical sommelier reviewing a pairing.",
    )
    issues += [i for i in review.get("issues", []) if isinstance(i, str)]
    return {"ok": not issues, "issues": issues, "llm_ok_field": review.get("ok")}


def rubric_verdict(llm, dish, draft, budget) -> dict:
    hits = [_Hit(w) for w in draft["candidates"]]
    code = rubric.code_checks(hits, draft["recommendation"], budget)
    res = rubric.judge(llm, dish, draft["candidates_text"], draft["recommendation"])
    c = rubric.decide(code, res)
    return {"ok": c["ok"], "issues": c["issues"], "criteria": c["criteria"],
            "schema_valid": c["schema_valid"], "repaired": c["repaired"]}


def run_judges(which):
    runs = json.loads((OUT / "runs.json").read_text())
    path = OUT / "judges.json"
    judges = json.loads(path.read_text()) if path.exists() else {}
    plan = {"legacy": (SYSTEM_MODEL, legacy_verdict), "second": (SECOND_MODEL, rubric_verdict),
            "repeat": (SYSTEM_MODEL, rubric_verdict)}
    for name in which:
        model, fn = plan[name]
        llm = _llm(model)
        judges.setdefault(name, {})
        for r in runs:
            if r["id"] in judges[name] or not r["rounds"]:
                continue
            CTX.update(dish=r["id"], tag=f"judge-{name}")
            n0 = len(CALLS)
            t0 = time.perf_counter()
            v = fn(llm, r["dish"], r["rounds"][0], r["budget"])
            v["model"] = model
            v["wall_s"] = round(time.perf_counter() - t0, 2)
            v["cost_usd"] = round(sum(c["cost_usd"] or 0 for c in CALLS[n0:]), 5)
            v["calls"] = len(CALLS) - n0
            judges[name][r["id"]] = v
            print(f"{name:<7} {r['id']:<14} ok={v['ok']}", flush=True)
            path.write_text(json.dumps(judges, indent=1, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# Step 3: summary
# --------------------------------------------------------------------------- #
def _pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def kappa(a: list[bool], b: list[bool]):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return {"n": n, "agree": sum(x == y for x, y in zip(a, b)),
            "percent": round(100 * po, 1), "kappa": None if pe == 1 else round((po - pe) / (1 - pe), 3)}


AGE_WORDS = ("vintage", "years old", "past its peak", "past peak", "drinking window",
             "too old", "aged ", "release price", "2026")


def summary():
    runs = json.loads((OUT / "runs.json").read_text())
    judges = json.loads((OUT / "judges.json").read_text()) if (OUT / "judges.json").exists() else {}
    old = json.loads((EVAL / "summary.json").read_text())
    base = {b["id"]: b for b in json.loads((EVAL / "baseline.json").read_text())}
    ok = [r for r in runs if not r["error"]]
    walls = [r["wall_s"] for r in ok]
    costs = [r["cost_usd"] for r in ok]
    first = {r["id"]: r["rounds"][0]["critique"] for r in ok if r["rounds"]}
    all_crit = [rd["critique"] for r in ok for rd in r["rounds"]]
    age_fails = [i for c in all_crit for i in c["issues"] if any(w in i.lower() for w in AGE_WORDS)]
    fails_by_criterion: dict[str, int] = {}
    for c in all_crit:
        for k, v in c["criteria"].items():
            if not v["pass"]:
                fails_by_criterion[k] = fails_by_criterion.get(k, 0) + 1
    calls = list(csv.DictReader(open(OUT / "calls.csv")))
    agent_calls = [c for c in calls if c["tag"] == "agent"]

    out = {
        "date": time.strftime("%Y-%m-%d"), "machine": "Apple M4",
        "system_model": SYSTEM_MODEL, "second_judge_model": SECOND_MODEL,
        "models_seen": sorted({c["models"] for c in calls}),
        "n_dishes": len(runs), "n_completed": len(ok),
        "pass_rate": f"{sum(r['passed'] for r in ok)} of {len(ok)}",
        "passed_first_round": sum(r["passed_round"] == 1 for r in ok),
        "critique_rounds": sorted(r["critique_rounds"] for r in ok),
        "critique_rounds_mean": round(statistics.mean(r["critique_rounds"] for r in ok), 2),
        "criterion_failures_all_rounds": fails_by_criterion,
        "issues_mentioning_age": len(age_fails),
        "schema_valid_first_try": sum(c["schema_valid"] and not c["repaired"] for c in all_crit),
        "schema_repaired": sum(c["repaired"] for c in all_crit),
        "schema_invalid": sum(not c["schema_valid"] for c in all_crit),
        "critique_calls_total": len(all_crit),
        "agent_matches": f"{sum(r['agent_match'] for r in ok)} of {len(ok)}",
        "agent_all_picks_match": f"{sum(r['all_picks_match'] for r in ok)} of {len(ok)}",
        "over_budget_picks": sum(len(r["over_budget"]) for r in ok),
        "baseline_matches": f"{sum(base[r['id']]['match'] for r in ok)} of {len(ok)}",
        "legacy_run_2026_11_05": {"critique_passed": f"{old['critique_passed']} of {old['n_completed']}",
                                  "agent_matches": f"{old['agent_matches']} of {old['n_completed']}",
                                  "cost_per_pairing_median": old["cost_per_pairing_median"],
                                  "latency_median_s": old["latency_median_s"],
                                  "llm_calls_per_pairing": 10},
        "llm_calls_per_pairing": sorted(r["llm_calls"] for r in ok),
        "cost_per_pairing_median": round(statistics.median(costs), 4),
        "cost_per_pairing_p95": round(_pct(costs, 0.95), 4),
        "latency_median_s": round(statistics.median(walls), 1),
        "latency_p95_s": round(_pct(walls, 0.95), 1),
        "per_call_cost_median": round(statistics.median(float(c["cost_usd"]) for c in agent_calls), 4),
        "per_call_wall_median_s": round(statistics.median(float(c["wall_s"]) for c in agent_calls), 2),
        "cost_note": "total_cost_usd from claude -p --output-format json: API-equivalent cost "
                     "reported by the Claude CLI, nothing billed (Claude subscription)",
    }
    ids = [r["id"] for r in ok if r["rounds"]]
    if "legacy" in judges:
        lj = [judges["legacy"][i] for i in ids if i in judges["legacy"]]
        out["first_drafts_legacy_critic_pass"] = f"{sum(v['ok'] for v in lj)} of {len(lj)}"
        out["first_drafts_legacy_issues_mentioning_age"] = sum(
            any(any(w in i.lower() for w in AGE_WORDS) for i in v["issues"]) for v in lj)
        out["first_drafts_rubric_critic_pass"] = f"{sum(first[i]['ok'] for i in ids)} of {len(ids)}"
    for name in ("second", "repeat"):
        if name not in judges:
            continue
        have = [i for i in ids if i in judges[name]]
        a = [first[i]["ok"] for i in have]
        b = [judges[name][i]["ok"] for i in have]
        per = {}
        for k in rubric.LLM_CRITERIA:
            pa = [first[i]["criteria"][k]["pass"] for i in have if k in judges[name][i]["criteria"]]
            pb = [judges[name][i]["criteria"][k]["pass"] for i in have if k in judges[name][i]["criteria"]]
            per[k] = kappa(pa, pb)
        pooled_a = [first[i]["criteria"][k]["pass"] for i in have for k in rubric.LLM_CRITERIA]
        pooled_b = [judges[name][i]["criteria"][k]["pass"] for i in have for k in rubric.LLM_CRITERIA]
        out[f"agreement_{name}"] = {
            "model": judges[name][have[0]]["model"] if have else None,
            "overall_verdict": kappa(a, b), "pooled_criteria": kappa(pooled_a, pooled_b),
            "per_criterion": per, "judge_pass": f"{sum(b)} of {len(b)}",
            "cost_median": round(statistics.median(judges[name][i]["cost_usd"] for i in have), 4),
            "wall_median_s": round(statistics.median(judges[name][i]["wall_s"] for i in have), 1),
        }
    (OUT / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--judges", nargs="*", help="legacy second repeat (default: all three)")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    dishes = json.loads((EVAL / "dishes.json").read_text())["dishes"]
    if a.summary:
        return summary()
    if a.judges is not None:
        return run_judges(a.judges or ["legacy", "second", "repeat"])
    run_agent(dishes, a.only)


if __name__ == "__main__":
    main()
