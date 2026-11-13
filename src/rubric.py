"""The critique as a structured rubric.

The first critique asked an LLM "is this a good pairing?" and passed a draft
only when the reply listed no issues. On 2026-11-05 it rejected 12 of 12 drafts,
7 of them first for the wine's age: the reviews were written in 2017 or
earlier and the critic judged them as of 2026. No retry can fix that.

This version splits the judgment into named criteria. Three are checked in code
(citations, budget, candidate count). Four go to the LLM, which must return JSON
matching ``RubricVerdict``: one pass/fail and a reason per criterion. The reply is
validated with pydantic; an invalid reply gets one repair attempt with the
validation error attached. The pass decision is made here, in code: a draft
passes when every criterion passes. The LLM never sets an overall "ok".

Out of scope, stated in the prompt: vintage age, drinking windows, current
prices and availability (the index is a fixed snapshot of reviews), and
preferring a different grape when the chosen one already fits.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .util import parse_json

LLM_CRITERIA = {
    "body_match": "The recommended wine's weight and intensity suit the dish's weight "
                  "(light with light, rich with rich).",
    "structure_logic": "The acid, tannin and sweetness reasoning is sound for the dish's fat, "
                       "protein, salt, spice and sweetness (for example no big tannic red "
                       "with very hot chili or delicate raw fish).",
    "flavor_link": "There is at least one plausible flavor bridge or deliberate contrast "
                   "between wine and dish.",
    "claims_supported": "What the recommendation says about each wine is consistent with "
                        "that wine's review text and is not invented.",
}
CODE_CRITERIA = {
    "grounded": "Every [n] citation points at a retrieved candidate, and at least one wine is cited.",
    "within_budget": "Every cited wine is at or under the budget.",
    "enough_candidates": "Retrieval returned at least 2 candidates.",
}
OUT_OF_SCOPE = (
    "vintage age, drinking windows or whether the wine is past its peak; current "
    "market price or availability (the review index is a fixed snapshot, so judge "
    "each wine as its review describes it); preferring a different grape or style "
    "when the chosen one already satisfies the criteria; wording or tone"
)


class Criterion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    passed: bool = Field(alias="pass")
    reason: str = Field(min_length=3, max_length=500)


class RubricVerdict(BaseModel):
    """What the LLM must return. Unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid")
    body_match: Criterion
    structure_logic: Criterion
    flavor_link: Criterion
    claims_supported: Criterion


def json_schema() -> dict:
    return RubricVerdict.model_json_schema(by_alias=True)


def rubric_prompt(dish: dict, candidates_text: str, recommendation: str) -> str:
    crit = "\n".join(f'- "{k}": {v}' for k, v in LLM_CRITERIA.items())
    return (
        f"Dish (JSON): {json.dumps(dish, ensure_ascii=False)}\n\n"
        f"Retrieved candidate wines (the only wines the recommender could use):\n"
        f"{candidates_text}\n\n"
        f"Proposed pairing:\n{recommendation}\n\n"
        "Grade the proposed pairing on each criterion below. Fail a criterion only for a "
        "real problem a sommelier would act on, and say what it is in the reason.\n"
        f"{crit}\n\n"
        f"Out of scope, never a reason to fail: {OUT_OF_SCOPE}.\n\n"
        "Return only JSON matching this schema, with a short reason for every criterion:\n"
        f"{json.dumps(json_schema())}\n"
        'Example shape: {"body_match": {"pass": true, "reason": "..."}, '
        '"structure_logic": {...}, "flavor_link": {...}, "claims_supported": {...}}'
    )


RUBRIC_SYSTEM = (
    "You are a strict but fair sommelier grading a wine pairing against a fixed rubric. "
    "Respond with ONLY valid JSON, no prose, no markdown fences."
)


def validate(raw: str) -> RubricVerdict:
    """Parse and validate an LLM reply; raises ValueError or ValidationError."""
    return RubricVerdict.model_validate(parse_json(raw))


def judge(llm, dish: dict, candidates_text: str, recommendation: str) -> dict:
    """Ask ``llm`` for a rubric verdict, validate it, repair once if needed.

    Returns {"verdict": dict | None, "valid": bool, "repaired": bool, "raw": [str]}.
    """
    prompt = rubric_prompt(dish, candidates_text, recommendation)
    raws = []
    for attempt in range(2):
        raw = llm.complete(prompt, system=RUBRIC_SYSTEM)
        raws.append(raw)
        try:
            v = validate(raw)
            return {"verdict": v.model_dump(by_alias=True), "valid": True,
                    "repaired": attempt == 1, "raw": raws}
        except (ValueError, ValidationError) as e:
            prompt = (prompt + "\n\nYour previous reply failed validation:\n"
                      f"{str(e)[:600]}\nReturn corrected JSON only.")
    return {"verdict": None, "valid": False, "repaired": False, "raw": raws}


_CITE = re.compile(r"\[(\d+)\]")


def cited(recommendation: str) -> list[int]:
    out = []
    for n in _CITE.findall(recommendation or ""):
        n = int(n)
        if n not in out:
            out.append(n)
    return out


def code_checks(candidates: list, recommendation: str, budget: float | None) -> dict:
    """The deterministic criteria. ``candidates`` are objects with ``.price``."""
    refs = cited(recommendation)
    bad = [n for n in refs if not 1 <= n <= len(candidates)]
    grounded_ok = bool(refs) and not bad
    over = []
    if budget:
        over = [n for n in refs if 1 <= n <= len(candidates)
                and (candidates[n - 1].price or 0) > budget]
    return {
        "grounded": {"pass": grounded_ok,
                     "reason": "cites " + ", ".join(f"[{n}]" for n in refs) if grounded_ok
                     else (f"citations out of range: {bad}" if bad else "no wine cited")},
        "within_budget": {"pass": not over,
                          "reason": "all cited wines within budget" if not over
                          else f"over budget: {over}"},
        "enough_candidates": {"pass": len(candidates) >= 2,
                              "reason": f"{len(candidates)} candidates retrieved"},
    }


def decide(code: dict, llm_result: dict) -> dict:
    """Combine code checks and the validated LLM verdict into one critique.

    An LLM reply that never validated counts as a failed critique with an
    explicit issue, so the loop retries instead of passing on bad JSON.
    """
    criteria = dict(code)
    issues = [f"{k}: {v['reason']}" for k, v in code.items() if not v["pass"]]
    if llm_result.get("valid"):
        for k, v in llm_result["verdict"].items():
            criteria[k] = v
            if not v["pass"]:
                issues.append(f"{k}: {v['reason']}")
    else:
        issues.append("rubric: the critic's reply did not match the schema after one repair")
    return {"ok": not issues, "issues": issues, "criteria": criteria,
            "schema_valid": bool(llm_result.get("valid")),
            "repaired": bool(llm_result.get("repaired"))}
