"""The decision engine: deterministic rules over the AI's analysis.

Pure function of (evaluation, rules, workload). No I/O, no model calls, no
hidden state - so the reason a task was skipped is always reconstructible from
three inputs, and the rules can be changed without re-running any evaluations.

Design stance: the engine can only ever *lower* a decision, never raise it. A
model that returns score 100 still cannot produce an auto-approval when the
workload is full or the deadline has passed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.workload_manager import WorkloadStatus
from core.enums import Decision, TaskStatus
from core.evaluation import TaskEvaluation
from core.schemas import TaskRules

#: Below this, a deadline is treated as a real obstacle rather than a note.
FEASIBILITY_FLOOR = 50

_ORDER = {Decision.SKIP: 0, Decision.REVIEW_REQUIRED: 1, Decision.HIGH_MATCH: 2}


@dataclass
class EngineDecision:
    """What to do with a task, and every reason that fed into it."""

    decision: Decision
    next_status: TaskStatus
    reasons: list[str] = field(default_factory=list)
    #: True when the score alone would have justified a stronger outcome.
    downgraded: bool = False

    @property
    def needs_approval(self) -> bool:
        return self.next_status is TaskStatus.REVIEW_REQUIRED

    def summary(self) -> str:
        return f"{self.decision.value} -> {self.next_status.value}: " + "; ".join(self.reasons)


def _cap(current: Decision, ceiling: Decision) -> Decision:
    return current if _ORDER[current] <= _ORDER[ceiling] else ceiling


def decide(
    evaluation: TaskEvaluation,
    rules: TaskRules,
    workload: WorkloadStatus | None = None,
) -> EngineDecision:
    """Turn one evaluation into a workflow decision."""
    reasons: list[str] = []

    # 1. Score bands.
    if evaluation.score < rules.review_threshold:
        base = Decision.SKIP
        reasons.append(
            f"score {evaluation.score} is below the review threshold {rules.review_threshold}"
        )
    elif evaluation.score >= rules.minimum_ai_score:
        base = Decision.HIGH_MATCH
        reasons.append(f"score {evaluation.score} meets the high-match bar {rules.minimum_ai_score}")
    else:
        base = Decision.REVIEW_REQUIRED
        reasons.append(
            f"score {evaluation.score} sits between {rules.review_threshold} "
            f"and {rules.minimum_ai_score}"
        )

    decision = base

    # 2. Deadline. A dead deadline is disqualifying; a tight one is reviewable
    #    but never auto-approvable.
    if evaluation.deadline_feasibility == 0:
        decision = _cap(decision, Decision.SKIP)
        reasons.append("deadline has passed or leaves no usable time")
    elif evaluation.deadline_feasibility < FEASIBILITY_FLOOR:
        decision = _cap(decision, Decision.REVIEW_REQUIRED)
        reasons.append(
            f"deadline feasibility {evaluation.deadline_feasibility} is below {FEASIBILITY_FLOOR}"
        )

    # 3. Workload. At capacity a task can still be worth a look, but it must
    #    not slip into the approved pile automatically.
    if workload is not None and not workload.has_capacity:
        decision = _cap(decision, Decision.REVIEW_REQUIRED)
        reasons.append(workload.note())

    # 4. Human approval gate. Only a HIGH_MATCH with approval disabled may
    #    reach APPROVED without a person in the loop.
    if decision is Decision.SKIP:
        next_status = TaskStatus.SKIPPED
    elif decision is Decision.REVIEW_REQUIRED:
        next_status = TaskStatus.REVIEW_REQUIRED
    elif rules.require_human_approval:
        next_status = TaskStatus.REVIEW_REQUIRED
        reasons.append("human approval is required before claiming")
    else:
        next_status = TaskStatus.APPROVED
        reasons.append("human approval is disabled; auto-approved")

    return EngineDecision(
        decision=decision,
        next_status=next_status,
        reasons=reasons,
        downgraded=_ORDER[decision] < _ORDER[base],
    )
