"""Fast tests for the rubric critique: no LLM, no index."""

import json
from dataclasses import dataclass

import pytest

from src import rubric
from src.graph import _route


@dataclass
class Hit:
    price: float | None


GOOD = {k: {"pass": True, "reason": "fine for this dish"} for k in rubric.LLM_CRITERIA}


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, prompt, system=None):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def test_valid_verdict_parses_with_fences():
    v = rubric.validate("```json\n" + json.dumps(GOOD) + "\n```")
    assert v.body_match.passed is True


def test_missing_and_extra_keys_are_rejected():
    bad = dict(GOOD)
    bad.pop("flavor_link")
    with pytest.raises(Exception):
        rubric.validate(json.dumps(bad))
    extra = dict(GOOD, ok=True)
    with pytest.raises(Exception):
        rubric.validate(json.dumps(extra))


def test_repair_step_retries_once_with_the_error():
    llm = FakeLLM(['{"body_match": {"pass": "maybe"}}', json.dumps(GOOD)])
    out = rubric.judge(llm, {"main": "x"}, "[1] wine", "Pick [1].")
    assert out["valid"] and out["repaired"]
    assert "failed validation" in llm.prompts[1]


def test_two_invalid_replies_fail_the_critique():
    llm = FakeLLM(["not json", "still not json"])
    out = rubric.judge(llm, {}, "", "")
    crit = rubric.decide(rubric.code_checks([Hit(10), Hit(12)], "Pick [1].", 20), out)
    assert not out["valid"] and not crit["ok"]
    assert any("schema" in i for i in crit["issues"])


def test_code_checks_catch_bad_citation_and_budget():
    hits = [Hit(15), Hit(45)]
    c = rubric.code_checks(hits, "Try [2] or [3].", 30)
    assert not c["grounded"]["pass"] and not c["within_budget"]["pass"]
    c = rubric.code_checks(hits, "Try [1].", 30)
    assert c["grounded"]["pass"] and c["within_budget"]["pass"]
    assert not rubric.code_checks(hits, "no citation", 30)["grounded"]["pass"]
    assert not rubric.code_checks([Hit(5)], "[1]", 30)["enough_candidates"]["pass"]


def test_pass_is_decided_in_code_not_by_the_llm():
    llm_result = {"valid": True, "verdict": dict(GOOD, structure_logic={"pass": False, "reason": "tannin vs chili"})}
    crit = rubric.decide(rubric.code_checks([Hit(10), Hit(12)], "[1]", 20), llm_result)
    assert not crit["ok"] and crit["issues"] == ["structure_logic: tannin vs chili"]
    llm_result = {"valid": True, "verdict": GOOD}
    assert rubric.decide(rubric.code_checks([Hit(10), Hit(12)], "[1]", 20), llm_result)["ok"]


def test_prompt_puts_vintage_age_out_of_scope():
    p = rubric.rubric_prompt({"main": "steak"}, "[1] JD 2007", "Pick [1].")
    assert "vintage age" in p and "fixed snapshot" in p


def test_route_stops_on_pass_or_attempt_limit():
    assert _route({"critique": {"ok": True}, "attempts": 0}) == "done"
    assert _route({"critique": {"ok": False}, "attempts": 0, "max_attempts": 2}) == "retry"
    assert _route({"critique": {"ok": False}, "attempts": 2, "max_attempts": 2}) == "done"
