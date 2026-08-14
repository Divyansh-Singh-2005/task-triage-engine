"""Evaluation service: score discovered tasks and act on the result.

Per-task failure isolation is the same contract as ingestion - one task whose
evaluation blows up leaves the rest of the queue intact and gets recorded as
FAILED rather than silently vanishing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from agent.decision_engine import decide
from agent.workload_manager import WorkloadManager
from core.enums import Decision, TaskStatus
from core.exceptions import EvaluationError
from core.profile import SkillProfile
from core.schemas import AgentConfig, Task
from database.models import TaskRecord
from database.repositories import EvaluationRepository, EventRepository, TaskRepository
from integrations.llm.base import LLMProvider


@dataclass
class EvaluationRun:
    """Outcome of evaluating one batch of tasks."""

    evaluated: int = 0
    skipped: int = 0
    review_required: int = 0
    approved: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        return (
            f"evaluated {self.evaluated}: skipped {self.skipped}, "
            f"review {self.review_required}, approved {self.approved}, "
            f"failed {self.failed}"
        )


def record_to_task(record: TaskRecord) -> Task:
    """Rehydrate the value object the evaluator expects from a stored row."""
    return Task(
        task_id=record.external_task_id,
        project_name=record.project.name,
        title=record.title,
        description=record.description,
        requirements=list(record.requirements or []),
        deadline=record.deadline,
        reward=record.reward,
        url=record.url,
        source=record.source,
        discovered_at=record.discovered_at,
        status=record.status,
    )


class EvaluationService:
    def __init__(self, session: Session, provider: LLMProvider) -> None:
        self.session = session
        self.provider = provider
        self.tasks = TaskRepository(session)
        self.evaluations = EvaluationRepository(session)
        self.events = EventRepository(session)
        self.workload = WorkloadManager(self.tasks)

    def run(self, config: AgentConfig, profile: SkillProfile, limit: int | None = None) -> EvaluationRun:
        """Evaluate every DISCOVERED task and apply the decision to each."""
        pending = list(self.tasks.list_by_status(TaskStatus.DISCOVERED))
        if limit is not None:
            pending = pending[:limit]

        run = EvaluationRun()
        for record in pending:
            try:
                self._evaluate_one(record, config, profile, run)
            except Exception as exc:
                run.failed += 1
                run.errors.append(f"{record.external_task_id}: {exc}")
                self.tasks.set_status(record, TaskStatus.FAILED)
                self.events.log(
                    "evaluation_failed",
                    message=str(exc),
                    task_id=record.id,
                    project_name=record.project.name,
                )

        self.events.log(
            "evaluation_batch_completed",
            message=run.summary(),
            payload={
                "provider": self.provider.describe(),
                "evaluated": run.evaluated,
                "skipped": run.skipped,
                "review_required": run.review_required,
                "approved": run.approved,
                "failed": run.failed,
            },
        )
        return run

    def _evaluate_one(
        self,
        record: TaskRecord,
        config: AgentConfig,
        profile: SkillProfile,
        run: EvaluationRun,
    ) -> None:
        workload = self.workload.status(config.task_rules)

        self.tasks.set_status(record, TaskStatus.EVALUATING)
        task = record_to_task(record)

        try:
            evaluation = self.provider.evaluate_task(
                task, profile, capacity_note=workload.note()
            )
        except EvaluationError:
            raise
        except Exception as exc:  # a provider bug must look like a provider failure
            raise EvaluationError(f"{self.provider.name} raised {type(exc).__name__}: {exc}") from exc

        outcome = decide(evaluation, config.task_rules, workload)

        self.evaluations.add(
            record,
            score=evaluation.score,
            decision=outcome.decision,
            reason=evaluation.reason,
            risks=list(evaluation.risks),
            estimated_hours=evaluation.estimated_hours,
            skill_match=evaluation.skill_match,
            deadline_feasibility=evaluation.deadline_feasibility,
            complexity=evaluation.complexity,
            model=evaluation.model or evaluation.provider,
        )
        self.tasks.set_status(record, outcome.next_status)

        run.evaluated += 1
        if outcome.next_status is TaskStatus.SKIPPED:
            run.skipped += 1
        elif outcome.next_status is TaskStatus.APPROVED:
            run.approved += 1
        else:
            run.review_required += 1

        self.events.log(
            "task_evaluated",
            message=outcome.summary(),
            task_id=record.id,
            project_name=record.project.name,
            payload={
                "score": evaluation.score,
                "decision": outcome.decision.value,
                "status": outcome.next_status.value,
                "downgraded": outcome.downgraded,
                "provider": evaluation.provider,
            },
        )

        if outcome.next_status is TaskStatus.REVIEW_REQUIRED and outcome.decision is Decision.HIGH_MATCH:
            self.events.log(
                "approval_requested",
                message=f"{record.title} scored {evaluation.score}",
                task_id=record.id,
                project_name=record.project.name,
            )
