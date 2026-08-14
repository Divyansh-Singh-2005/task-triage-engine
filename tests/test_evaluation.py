"""Tests for the evaluation schema, response parsing and providers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.enums import Complexity
from core.evaluation import (
    TaskEvaluation,
    build_user_prompt,
    extract_json_object,
    parse_evaluation,
)
from core.exceptions import EvaluationError
from core.profile import SkillProfile
from core.schemas import Task
from integrations.llm.base import TextLLMProvider
from integrations.llm.heuristic import HeuristicProvider

PROFILE = SkillProfile(
    display_name="Tester",
    skills=["Python", "RDF", "graph algorithms"],
    preferred_technologies=["rdflib"],
    avoid=["Unity"],
    weekly_hours=15,
)

GOOD_JSON = """{"score": 88, "skill_match": 90, "deadline_feasibility": 80,
"estimated_hours": 6, "complexity": "medium", "missing_skills": [],
"risks": ["scope"], "reason": "Strong overlap."}"""


def make_task(**kw) -> Task:
    base = {
        "task_id": "T-1",
        "project_name": "Project Dynamo",
        "title": "Build an RDF graph task",
        "description": "Write a Python task using rdflib and graph algorithms." * 3,
        "requirements": ["Python", "RDF"],
    }
    base.update(kw)
    return Task(**base)


# ------------------------------------------------------------------ parsing


def test_parses_a_plain_json_object():
    evaluation = parse_evaluation(GOOD_JSON)
    assert evaluation.score == 88
    assert evaluation.complexity is Complexity.MEDIUM


def test_parses_json_inside_code_fences():
    assert parse_evaluation(f"```json\n{GOOD_JSON}\n```").score == 88


def test_parses_json_surrounded_by_prose():
    raw = f"Sure, here is my analysis:\n{GOOD_JSON}\nHope that helps!"
    assert parse_evaluation(raw).score == 88


def test_brace_scanner_handles_nested_and_quoted_braces():
    raw = '{"score": 50, "skill_match": 50, "reason": "uses {braces} inside"}'
    assert extract_json_object(raw)["reason"] == "uses {braces} inside"


def test_empty_response_is_rejected():
    with pytest.raises(EvaluationError):
        parse_evaluation("   ")


def test_response_without_json_is_rejected():
    with pytest.raises(EvaluationError):
        parse_evaluation("I am unable to evaluate this task.")


def test_malformed_json_is_rejected():
    with pytest.raises(EvaluationError):
        parse_evaluation('{"score": 88,,}')


def test_out_of_range_score_is_rejected():
    with pytest.raises(EvaluationError):
        parse_evaluation('{"score": 150, "skill_match": 50}')


def test_missing_required_field_is_rejected():
    with pytest.raises(EvaluationError):
        parse_evaluation('{"skill_match": 50}')


def test_extra_keys_are_tolerated():
    assert parse_evaluation('{"score": 70, "skill_match": 60, "decision": "SKIP"}').score == 70


def test_string_where_list_expected_is_coerced():
    evaluation = parse_evaluation('{"score": 70, "skill_match": 60, "risks": "one risk"}')
    assert evaluation.risks == ["one risk"]


# ----------------------------------------------------------------- prompting


def test_prompt_contains_task_and_profile_details():
    prompt = build_user_prompt(make_task(), PROFILE, capacity_note="1 of 2 active tasks")
    assert "Project Dynamo" in prompt
    assert "graph algorithms" in prompt
    assert "1 of 2 active tasks" in prompt


def test_prompt_handles_a_task_with_no_requirements():
    assert "(none listed)" in build_user_prompt(make_task(requirements=[]), PROFILE)


# ------------------------------------------------------------ text provider


class ScriptedProvider(TextLLMProvider):
    """Returns canned responses in order, for exercising retry behaviour."""

    name = "scripted"
    model = "scripted-1"

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, system: str, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else ""


def test_text_provider_stamps_its_identity():
    evaluation = ScriptedProvider([GOOD_JSON]).evaluate_task(make_task(), PROFILE)
    assert evaluation.provider == "scripted"
    assert evaluation.model == "scripted-1"


def test_text_provider_retries_once_with_a_repair_instruction():
    provider = ScriptedProvider(["not json at all", GOOD_JSON])
    evaluation = provider.evaluate_task(make_task(), PROFILE)
    assert evaluation.score == 88
    assert len(provider.prompts) == 2
    assert "could not be parsed" in provider.prompts[1]


def test_text_provider_gives_up_after_the_retry():
    provider = ScriptedProvider(["nope", "still nope"])
    with pytest.raises(EvaluationError):
        provider.evaluate_task(make_task(), PROFILE)


# ------------------------------------------------------- heuristic provider


def test_heuristic_scores_a_matching_task_higher_than_a_mismatched_one():
    provider = HeuristicProvider()
    good = provider.evaluate_task(make_task(), PROFILE)
    bad = provider.evaluate_task(
        make_task(title="Salesforce admin work", description="", requirements=["Salesforce"]),
        PROFILE,
    )
    assert good.score > bad.score


def test_heuristic_penalizes_avoided_topics():
    provider = HeuristicProvider()
    neutral = provider.evaluate_task(make_task(title="Python graph work"), PROFILE)
    avoided = provider.evaluate_task(make_task(title="Python graph work in Unity"), PROFILE)
    assert avoided.score < neutral.score


def test_heuristic_reports_uncovered_requirements():
    evaluation = HeuristicProvider().evaluate_task(
        make_task(requirements=["Python", "Kubernetes"]), PROFILE
    )
    assert "Kubernetes" in evaluation.missing_skills


def test_heuristic_treats_a_passed_deadline_as_infeasible():
    past = datetime.now(timezone.utc) - timedelta(days=1)
    evaluation = HeuristicProvider().evaluate_task(make_task(deadline=past), PROFILE)
    assert evaluation.deadline_feasibility == 0


def test_heuristic_gives_full_feasibility_without_a_deadline():
    assert HeuristicProvider().evaluate_task(make_task(), PROFILE).deadline_feasibility == 100


def test_heuristic_flags_a_thin_description():
    evaluation = HeuristicProvider().evaluate_task(
        make_task(description="", requirements=[]), PROFILE
    )
    assert any("scope" in r.lower() or "thin" in r.lower() for r in evaluation.risks)


def test_heuristic_output_validates_against_the_schema():
    evaluation = HeuristicProvider().evaluate_task(make_task(), PROFILE)
    assert isinstance(evaluation, TaskEvaluation)
    assert 0 <= evaluation.score <= 100


def test_heuristic_is_deterministic():
    provider = HeuristicProvider()
    task = make_task()
    assert provider.evaluate_task(task, PROFILE).score == provider.evaluate_task(task, PROFILE).score
