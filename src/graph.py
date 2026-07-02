"""The wine-pairing agent, built as a LangGraph state machine.

    parse ─▶ strategize ─▶ retrieve ─▶ recommend ─▶ critique ─┬─ clean ─▶ END
                  ▲                                           │
                  └──────────────── issues & budget ──────────┘

`parse` turns a free-text meal into structured features; `strategize` reasons
from pairing principles to a target wine style + a search query; `retrieve`
calls the wine-review search **tool** (RAG over 130k reviews) for real bottles;
`recommend` picks and explains with citations; `critique` checks the result
(enough candidates, budget, coherence) and, if it's weak, loops back to
`strategize` with feedback to try a different style — up to a budget of attempts.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from .llm import LLM
from .pairing import PAIRING_PRINCIPLES
from .tools import wine_search_tool
from .util import json_complete


class PairState(TypedDict, total=False):
    request: dict          # {meal, budget, color_pref}
    dish: dict             # parsed dish features
    strategy: dict         # {target_style, search_query, rationale}
    candidates: list        # retrieved WineHit objects
    recommendation: str
    critique: dict         # {issues: [...], ok: bool}
    attempts: int
    max_attempts: int
    log: list


_LLM: LLM | None = None


def _llm() -> LLM:
    global _LLM
    if _LLM is None:
        _LLM = LLM()
    return _LLM


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def parse_dish(state: PairState) -> PairState:
    meal = state["request"]["meal"]
    prompt = (
        f'Analyze this meal for wine pairing: "{meal}".\n'
        "Return JSON: {\"main\": str, \"body\": \"light|medium|rich\", "
        "\"sauce\": str, \"cuisine\": str, \"key_flavors\": [str], "
        "\"cooking_method\": str}."
    )
    dish = json_complete(_llm(), prompt, system="You are a sommelier analyzing a dish.")
    log = state.get("log", []) + [f"parse → {dish.get('main')} ({dish.get('body')} body)"]
    return {"dish": dish, "log": log, "attempts": 0}


def strategize(state: PairState) -> PairState:
    dish, req = state["dish"], state["request"]
    feedback = ""
    if state.get("critique") and not state["critique"].get("ok"):
        feedback = (
            "\n\nA previous attempt was weak: "
            + "; ".join(state["critique"]["issues"])
            + ".\nChoose a DIFFERENT target style or broaden the search query."
        )
    constraints = []
    if req.get("budget"):
        constraints.append(f"budget about ${req['budget']:g} per bottle")
    if req.get("color_pref"):
        constraints.append(f"the guest prefers {req['color_pref']} wine")
    con = ("Constraints: " + ", ".join(constraints) + ".") if constraints else ""
    prompt = (
        f"{PAIRING_PRINCIPLES}\n\n"
        f"Dish (JSON): {dish}\n{con}\n\n"
        "Decide the ideal wine style for this dish, then write a short natural-language "
        "search query (grape/style/flavour words) to find matching real wines in a "
        "review database. Return JSON: {\"target_style\": str, \"search_query\": str, "
        "\"rationale\": one sentence}."
        f"{feedback}"
    )
    strat = json_complete(_llm(), prompt, system="You are a sommelier planning a pairing.")
    log = state.get("log", []) + [
        f"strategize → {strat.get('target_style')} | query: \"{strat.get('search_query')}\""
    ]
    return {"strategy": strat, "log": log}


def retrieve(state: PairState) -> PairState:
    """The retrieval TOOL: semantic search over 130k real wine reviews."""
    req = state["request"]
    # Invoke the retrieval as a named LangChain tool (see src/tools.py) rather
    # than calling the function directly — same abstraction the LLM would use.
    hits = wine_search_tool.invoke({
        "query": state["strategy"]["search_query"],
        "max_price": req.get("budget"),
        "k": 6,
    })
    log = state.get("log", []) + [f"retrieve → {len(hits)} candidate wines (tool: search_wines)"]
    return {"candidates": hits, "log": log}


def recommend(state: PairState) -> PairState:
    dish, strat = state["dish"], state["strategy"]
    hits = state["candidates"]
    context = "\n\n".join(
        f"[{i}] ({h.citation()}) {h.doc}" for i, h in enumerate(hits, 1)
    ) or "(no wines found)"
    prompt = (
        f"Dish: {dish}\nTarget style: {strat.get('target_style')} "
        f"({strat.get('rationale')})\n\n"
        f"Candidate wines from the review database (use ONLY these):\n{context}\n\n"
        "Recommend the best 1–2 wines for this dish. For each, name it with its score "
        "and price, cite it as [n], and explain in one or two sentences WHY it pairs "
        "(acid/tannin/body/flavour logic). Be warm and concise."
    )
    text = _llm().complete(
        prompt,
        system="You are a WSET-trained sommelier pairing wine to food. Recommend only "
        "from the provided wines and cite each with [n].",
    )
    log = state.get("log", []) + ["recommend → drafted pairing with citations"]
    return {"recommendation": text, "log": log}


def critique(state: PairState) -> PairState:
    hits = state.get("candidates", [])
    req = state["request"]
    issues = []
    # Deterministic checks.
    if len(hits) < 2:
        issues.append("too few candidate wines were found — the search was too narrow")
    if req.get("budget") and hits:
        under = [h for h in hits if (h.price or 0) <= req["budget"]]
        if not under:
            issues.append(f"no candidates within the ${req['budget']:g} budget")
    # LLM coherence check.
    review = json_complete(
        _llm(),
        f"Dish: {state['dish']}\nProposed pairing:\n{state.get('recommendation','')}\n\n"
        "Is this a genuinely good pairing for the dish (right body, acid/tannin logic)? "
        'Return JSON: {"issues": [short strings], "ok": bool}.',
        system="You are a critical sommelier reviewing a pairing.",
    )
    issues += [i for i in review.get("issues", []) if isinstance(i, str) and i not in issues]

    ok = len(issues) == 0
    log = state.get("log", []) + [
        f"critique → {'sound' if ok else str(len(issues)) + ' issue(s)'}"
        + (f": {issues[0]}" if issues else "")
    ]
    return {"critique": {"issues": issues, "ok": ok}, "log": log}


def _route(state: PairState) -> str:
    if state["critique"]["ok"]:
        return "done"
    if state.get("attempts", 0) >= state.get("max_attempts", 2):
        return "done"
    return "retry"


def _bump_attempt(state: PairState) -> PairState:
    return {"attempts": state.get("attempts", 0) + 1}


def build_agent():
    g = StateGraph(PairState)
    g.add_node("parse", parse_dish)
    g.add_node("strategize", strategize)
    g.add_node("retrieve", retrieve)
    g.add_node("recommend", recommend)
    g.add_node("critique", critique)
    g.add_node("bump", _bump_attempt)   # counts a retry before re-strategizing

    g.set_entry_point("parse")
    g.add_edge("parse", "strategize")
    g.add_edge("strategize", "retrieve")
    g.add_edge("retrieve", "recommend")
    g.add_edge("recommend", "critique")
    g.add_conditional_edges("critique", _route, {"retry": "bump", "done": END})
    g.add_edge("bump", "strategize")
    return g.compile()
