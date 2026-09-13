"""Streamlit control dashboard.

All state lives in the config file and the database; this module only reads
and issues commands. Nothing here holds application state in ``st.session_state``
beyond transient UI concerns, so a browser refresh can never desynchronize the
dashboard from what the agent actually believes.
"""

from __future__ import annotations

import json

import streamlit as st

from core.enums import AgentStatus, MonitoringMode, TaskStatus
from core.exceptions import AgentError
from agent.sources.manual import ManualTaskSource
from services.agent_context import AgentContext
from services.approval_service import ApprovalService
from services.board import BoardSnapshot, TaskView, build_snapshot
from services.evaluation_service import EvaluationService
from services.task_service import TaskIngestionService

MODE_LABELS = {
    MonitoringMode.ACTIVE_PROJECT: "Active project",
    MonitoringMode.SELECTED_PROJECTS: "Selected projects",
    MonitoringMode.ANY_PROJECT: "Any project",
}

STATUS_TABS = [
    ("Needs review", (TaskStatus.REVIEW_REQUIRED,)),
    ("Approved", (TaskStatus.APPROVED, TaskStatus.CLAIM_PENDING, TaskStatus.CLAIMED)),
    ("New", (TaskStatus.DISCOVERED, TaskStatus.EVALUATING)),
    ("Skipped", (TaskStatus.SKIPPED,)),
    ("Failed", (TaskStatus.FAILED,)),
]


@st.cache_resource
def get_context() -> AgentContext:
    """Built once per server process; the engine is expensive to rebuild."""
    return AgentContext()


def notify(message: str, level: str = "success") -> None:
    """Queue a message to show after the rerun that follows an action."""
    st.session_state["flash"] = (level, message)


def drain_flash() -> None:
    flash = st.session_state.pop("flash", None)
    if flash:
        level, message = flash
        getattr(st, level, st.info)(message)


# --------------------------------------------------------------------- sidebar


def render_sidebar(ctx: AgentContext) -> None:
    config = ctx.config()

    with st.sidebar:
        st.subheader("Agent")
        running = config.agent_status is AgentStatus.RUNNING
        st.markdown(f"Status: {'Running' if running else 'Stopped'}")

        start, stop = st.columns(2)
        if start.button("Start", disabled=running, width="stretch"):
            ctx.manager.set_agent_status(AgentStatus.RUNNING)
            notify("Agent started")
            st.rerun()
        if stop.button("Stop", disabled=not running, width="stretch"):
            ctx.manager.set_agent_status(AgentStatus.STOPPED)
            notify("Agent stopped", "warning")
            st.rerun()

        st.divider()
        st.subheader("Monitoring mode")
        modes = list(MODE_LABELS)
        chosen = st.radio(
            "Mode",
            options=modes,
            index=modes.index(config.monitoring_mode),
            format_func=lambda m: MODE_LABELS[m],
            label_visibility="collapsed",
        )
        if chosen is not config.monitoring_mode:
            try:
                ctx.manager.set_monitoring_mode(chosen)
                notify(f"Mode set to {MODE_LABELS[chosen]}")
                st.rerun()
            except AgentError as exc:
                st.error(str(exc))

        st.divider()
        st.subheader("Projects")
        known = config.known_projects or ([config.active_project] if config.active_project else [])

        if config.monitoring_mode is MonitoringMode.ACTIVE_PROJECT and known:
            index = known.index(config.active_project) if config.active_project in known else 0
            picked = st.selectbox("Active project", known, index=index)
            if picked != config.active_project:
                ctx.manager.set_active_project(picked)
                notify(f"Active project: {picked}")
                st.rerun()

        if config.monitoring_mode is MonitoringMode.SELECTED_PROJECTS:
            picked = st.multiselect("Selected projects", known, default=config.selected_projects)
            if set(picked) != set(config.selected_projects):
                try:
                    for name in set(config.selected_projects) - set(picked):
                        ctx.manager.remove_selected_project(name)
                    for name in set(picked) - set(config.selected_projects):
                        ctx.manager.add_selected_project(name)
                    notify("Selection updated")
                    st.rerun()
                except AgentError as exc:
                    st.error(str(exc))

        if config.monitoring_mode is MonitoringMode.ANY_PROJECT:
            st.caption("No project restriction. Every discovered task is evaluated.")

        new_project = st.text_input("Register a project", placeholder="Project Beta")
        if st.button("Add project", width="stretch") and new_project.strip():
            ctx.manager.register_project(new_project)
            notify(f"Registered {new_project.strip()}")
            st.rerun()

        st.divider()
        st.subheader("Task rules")
        rules = config.task_rules
        review = st.slider("Review threshold", 0, 100, rules.review_threshold)
        minimum = st.slider("High-match score", 0, 100, rules.minimum_ai_score)
        maximum = st.number_input("Max active tasks", 0, 50, rules.maximum_active_tasks)
        approval = st.checkbox("Require human approval", rules.require_human_approval)

        changed = (
            review != rules.review_threshold
            or minimum != rules.minimum_ai_score
            or maximum != rules.maximum_active_tasks
            or approval != rules.require_human_approval
        )
        if st.button("Save rules", disabled=not changed, width="stretch"):
            try:
                ctx.manager.set_task_rules(
                    review_threshold=review,
                    minimum_ai_score=minimum,
                    maximum_active_tasks=int(maximum),
                    require_human_approval=approval,
                )
                notify("Rules saved")
                st.rerun()
            except AgentError as exc:
                st.error(str(exc))

        st.divider()
        st.caption(f"Evaluator: {ctx.provider.describe()}")
        if ctx.provider_is_fallback:
            st.warning(
                f"{ctx.settings.llm_provider} was requested but no API key is set; "
                "using the offline heuristic baseline."
            )


# ------------------------------------------------------------------ main panes


def render_metrics(snapshot: BoardSnapshot) -> None:
    columns = st.columns(5)
    pairs = [
        ("New", snapshot.count(TaskStatus.DISCOVERED)),
        ("Needs review", snapshot.count(TaskStatus.REVIEW_REQUIRED)),
        ("Approved", snapshot.count(TaskStatus.APPROVED)),
        ("Skipped", snapshot.count(TaskStatus.SKIPPED)),
        ("Failed", snapshot.count(TaskStatus.FAILED)),
    ]
    for column, (label, value) in zip(columns, pairs):
        column.metric(label, value)

    if snapshot.workload:
        status = snapshot.workload
        bar = min(1.0, status.active / status.maximum) if status.maximum else 1.0
        st.progress(bar, text=f"Workload: {status.note()}")


def render_intake(ctx: AgentContext) -> None:
    """Manual task entry.

    The only task source wired in today. Paste listings as JSON; the same
    pipeline runs regardless of where tasks eventually come from.
    """
    config = ctx.config()
    stopped = config.agent_status is not AgentStatus.RUNNING

    with st.expander("Add tasks", expanded=False):
        st.caption(
            "Paste a JSON list of tasks. Required per task: project_name, title. "
            "Optional: task_id, description, requirements, deadline, reward, url."
        )
        raw = st.text_area(
            "Task JSON",
            height=180,
            placeholder='[{"task_id": "D-9", "project_name": "Project Dynamo", '
            '"title": "Example task", "requirements": ["Python"]}]',
        )
        if st.button("Ingest and evaluate", type="primary", disabled=stopped):
            if stopped:
                return
            try:
                payload = json.loads(raw) if raw.strip() else []
            except json.JSONDecodeError as exc:
                st.error(f"That is not valid JSON: {exc}")
                return
            if not isinstance(payload, list) or not payload:
                st.error("Provide a non-empty JSON list of task objects.")
                return

            try:
                with ctx.session() as session:
                    ingested = TaskIngestionService(session).ingest(
                        ManualTaskSource(payload), config
                    )
                with ctx.session() as session:
                    evaluated = EvaluationService(session, ctx.provider).run(
                        config, ctx.profile
                    )
            except AgentError as exc:
                st.error(str(exc))
                return

            notify(f"{ingested.summary()} | {evaluated.summary()}")
            st.rerun()

    if stopped:
        st.info("The agent is stopped. Start it from the sidebar to ingest or evaluate.")


def render_task(ctx: AgentContext, task: TaskView) -> None:
    header = f"{task.title} - score {task.score_text}"
    with st.expander(header, expanded=False):
        left, right = st.columns([2, 1])
        with left:
            st.markdown(f"**Project:** {task.project}")
            st.markdown(f"**Task id:** {task.external_id} (source: {task.source})")
            if task.reason:
                st.markdown(f"**Why:** {task.reason}")
            if task.risks:
                st.markdown("**Risks:**")
                for risk in task.risks:
                    st.markdown(f"- {risk}")
            if task.url:
                st.markdown(f"[Open listing]({task.url})")
        with right:
            st.markdown(f"**Status:** {task.status.value}")
            if task.decision:
                st.markdown(f"**Decision:** {task.decision}")
            if task.estimated_hours:
                st.markdown(f"**Estimate:** {task.estimated_hours:g} h")
            if task.complexity:
                st.markdown(f"**Complexity:** {task.complexity}")
            if task.deadline:
                st.markdown(f"**Deadline:** {task.deadline:%Y-%m-%d %H:%M} UTC")
            if task.reward:
                st.markdown(f"**Reward:** {task.reward}")

        approve, skip, reset = st.columns(3)
        rules = ctx.config().task_rules

        if task.status is TaskStatus.REVIEW_REQUIRED:
            if approve.button("Approve", key=f"approve-{task.id}", type="primary"):
                _act(ctx, lambda s: ApprovalService(s).approve(task.id, rules), "Approved")
            if skip.button("Skip", key=f"skip-{task.id}"):
                _act(ctx, lambda s: ApprovalService(s).reject(task.id), "Skipped", "warning")
        elif task.status in (TaskStatus.SKIPPED, TaskStatus.FAILED):
            if reset.button("Re-evaluate", key=f"reset-{task.id}"):
                _act(ctx, lambda s: ApprovalService(s).reset(task.id), "Queued for re-evaluation")

        if task.status is TaskStatus.APPROVED:
            st.caption(
                "Approved locally. No claim action is wired up - see the README on "
                "authorized platform integration."
            )


def _act(ctx: AgentContext, operation, message: str, level: str = "success") -> None:
    try:
        with ctx.session() as session:
            operation(session)
    except AgentError as exc:
        st.error(str(exc))
        return
    notify(message, level)
    st.rerun()


def render_board(ctx: AgentContext, snapshot: BoardSnapshot) -> None:
    tabs = st.tabs([f"{label} ({len(snapshot.by_status(*statuses))})" for label, statuses in STATUS_TABS])
    for tab, (_, statuses) in zip(tabs, STATUS_TABS):
        with tab:
            tasks = snapshot.by_status(*statuses)
            if not tasks:
                st.caption("Nothing here.")
                continue
            for task in tasks:
                render_task(ctx, task)


def render_events(snapshot: BoardSnapshot) -> None:
    with st.expander("Recent activity", expanded=False):
        if not snapshot.events:
            st.caption("No events yet.")
            return
        st.dataframe(
            [
                {
                    "when": f"{event.created_at:%Y-%m-%d %H:%M:%S}",
                    "event": event.event_type,
                    "project": event.project or "",
                    "detail": event.message,
                }
                for event in snapshot.events
            ],
            width="stretch",
            hide_index=True,
        )


# ------------------------------------------------------------------ entrypoint


def render() -> None:
    st.set_page_config(page_title="Handshake Task Agent", layout="wide")
    ctx = get_context()
    config = ctx.config()

    st.title("Handshake Task Agent")
    st.caption(
        f"Mode: {MODE_LABELS[config.monitoring_mode]} - "
        f"Active project: {config.active_project or 'none'}"
    )

    drain_flash()
    render_sidebar(ctx)

    with ctx.session() as session:
        snapshot = build_snapshot(session, config.task_rules)

    render_metrics(snapshot)
    render_intake(ctx)
    render_board(ctx, snapshot)
    render_events(snapshot)
