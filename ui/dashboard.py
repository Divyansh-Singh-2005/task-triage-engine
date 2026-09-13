"""Streamlit control dashboard.

All state lives in the config file and the database; this module only reads
and issues commands. Nothing here holds application state in
``st.session_state`` beyond transient UI concerns, so a browser refresh can
never desynchronize the dashboard from what the agent actually believes.
"""

from __future__ import annotations

import json

import streamlit as st

from agent.orchestrator import AgentOrchestrator
from agent.sources.manual import ManualTaskSource
from agent.sources.registry import build_sources
from core.enums import AgentStatus, MonitoringMode, TaskStatus
from core.exceptions import AgentError
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

SORTS = {
    "Score (high first)": lambda t: (-(t.score if t.score is not None else -1),),
    "Score (low first)": lambda t: (t.score if t.score is not None else 999,),
    "Newest first": lambda t: (-(t.discovered_at.timestamp() if t.discovered_at else 0),),
    "Project": lambda t: (t.project.casefold(), t.title.casefold()),
}


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


def _act(ctx: AgentContext, operation, message: str, level: str = "success") -> None:
    try:
        with ctx.session() as session:
            operation(session)
    except AgentError as exc:
        st.error(str(exc))
        return
    notify(message, level)
    st.rerun()


# --------------------------------------------------------------------- sidebar


def render_sidebar(ctx: AgentContext) -> None:
    config = ctx.config()

    with st.sidebar:
        st.subheader("Agent")
        running = config.agent_status is AgentStatus.RUNNING
        st.markdown(f"Status: **{'Running' if running else 'Stopped'}**")

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
        if st.button("Add project", width="stretch"):
            # Say something either way. A button that silently does nothing on
            # an empty field reads as a bug.
            if not new_project.strip():
                st.warning("Type a project name first.")
            else:
                ctx.manager.register_project(new_project)
                notify(f"Registered {new_project.strip()}")
                st.rerun()

        st.divider()
        render_rules(ctx)

        st.divider()
        st.caption(f"Evaluator: {ctx.provider.describe()}")
        if ctx.provider_is_fallback:
            st.warning(
                f"{ctx.settings.llm_provider} was requested but no API key is set; "
                "using the offline heuristic baseline."
            )


def render_rules(ctx: AgentContext) -> None:
    """Task rules, with an explicit unsaved state.

    Streamlit widgets hold their value locally until something writes it back,
    so without this the sidebar can show one number while the agent uses
    another - which is exactly as confusing as it sounds.
    """
    rules = ctx.config().task_rules

    st.subheader("Task rules")
    review = st.slider("Review threshold", 0, 100, rules.review_threshold)
    minimum = st.slider("High-match score", 0, 100, rules.minimum_ai_score)
    maximum = st.number_input("Max active tasks", 0, 50, rules.maximum_active_tasks)
    approval = st.checkbox("Require human approval", rules.require_human_approval)

    pending = {
        "review_threshold": review,
        "minimum_ai_score": minimum,
        "maximum_active_tasks": int(maximum),
        "require_human_approval": approval,
    }
    saved = {
        "review_threshold": rules.review_threshold,
        "minimum_ai_score": rules.minimum_ai_score,
        "maximum_active_tasks": rules.maximum_active_tasks,
        "require_human_approval": rules.require_human_approval,
    }
    changed = pending != saved

    if changed:
        differences = ", ".join(
            f"{key.replace('_', ' ')} {saved[key]} to {value}"
            for key, value in pending.items()
            if value != saved[key]
        )
        st.warning(f"Unsaved: {differences}")

    save, revert = st.columns(2)
    if save.button("Save rules", disabled=not changed, type="primary", width="stretch"):
        try:
            ctx.manager.set_task_rules(**pending)
            notify("Rules saved")
            st.rerun()
        except AgentError as exc:
            st.error(str(exc))
    if revert.button("Revert", disabled=not changed, width="stretch"):
        st.rerun()


# ------------------------------------------------------------------ main panes


def render_header(ctx: AgentContext, snapshot: BoardSnapshot) -> None:
    config = ctx.config()
    st.title("Handshake Task Agent")

    bits = [
        f"Mode: {MODE_LABELS[config.monitoring_mode]}",
        f"Active project: {config.active_project or 'none'}",
    ]
    if snapshot.last_cycle:
        bits.append(f"Last cycle: {snapshot.last_cycle:%Y-%m-%d %H:%M} UTC")
    st.caption(" | ".join(bits))
    if snapshot.last_cycle_summary:
        st.caption(snapshot.last_cycle_summary)


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


def render_controls(ctx: AgentContext) -> None:
    """Run a cycle, refresh, and drop tasks in by hand."""
    config = ctx.config()
    stopped = config.agent_status is not AgentStatus.RUNNING

    run, refresh, _ = st.columns([1, 1, 3])
    if run.button("Run cycle now", type="primary", disabled=stopped, width="stretch"):
        orchestrator = AgentOrchestrator(ctx, build_sources(ctx.settings))
        result = orchestrator.run_once()
        level = "success" if result.ok else "warning"
        notify(result.summary(), level)
        st.rerun()
    if refresh.button("Refresh", width="stretch"):
        st.rerun()

    if stopped:
        st.info("The agent is stopped. Start it from the sidebar to ingest or evaluate.")

    with st.expander("Add tasks by hand", expanded=False):
        st.caption(
            f"The agent also polls {ctx.settings.inbox_path} on every cycle. "
            "Paste a JSON list below for a one-off. Required per task: "
            "project_name, title."
        )
        raw = st.text_area(
            "Task JSON",
            height=160,
            placeholder='[{"task_id": "D-9", "project_name": "Project Dynamo", '
            '"title": "Example task", "requirements": ["Python"]}]',
        )
        if st.button("Ingest and evaluate", disabled=stopped):
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
                    evaluated = EvaluationService(session, ctx.provider).run(config, ctx.profile)
            except AgentError as exc:
                st.error(str(exc))
                return
            notify(f"{ingested.summary()} | {evaluated.summary()}")
            st.rerun()


def render_task(ctx: AgentContext, task: TaskView, snapshot: BoardSnapshot) -> None:
    header = f"{task.title} - score {task.score_text} - {task.project}"
    with st.expander(header, expanded=False):
        left, right = st.columns([2, 1])
        with left:
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

        rules = ctx.config().task_rules
        at_capacity = snapshot.workload is not None and not snapshot.workload.has_capacity

        if task.status is TaskStatus.REVIEW_REQUIRED:
            force = False
            if at_capacity:
                st.warning(f"Workload is full ({snapshot.workload.note()}).")
                force = st.checkbox("Approve anyway, over capacity", key=f"force-{task.id}")
            approve, skip, _ = st.columns([1, 1, 2])
            if approve.button("Approve", key=f"approve-{task.id}", type="primary",
                              disabled=at_capacity and not force, width="stretch"):
                _act(ctx, lambda s: ApprovalService(s).approve(task.id, rules, force=force),
                     "Approved")
            if skip.button("Skip", key=f"skip-{task.id}", width="stretch"):
                _act(ctx, lambda s: ApprovalService(s).reject(task.id), "Skipped", "warning")

        elif task.status in (TaskStatus.SKIPPED, TaskStatus.FAILED):
            reset, _ = st.columns([1, 3])
            if reset.button("Re-evaluate", key=f"reset-{task.id}", width="stretch"):
                _act(ctx, lambda s: ApprovalService(s).reset(task.id),
                     "Queued for re-evaluation")

        elif task.status is TaskStatus.APPROVED:
            undo, _ = st.columns([1, 3])
            if undo.button("Undo approval", key=f"undo-{task.id}", width="stretch"):
                _act(ctx, lambda s: ApprovalService(s).reject(task.id, reason="approval undone"),
                     "Approval undone", "warning")
            st.caption(
                "Approved locally. No claim action is wired up - see the README on "
                "authorized platform integration."
            )


def render_board(ctx: AgentContext, snapshot: BoardSnapshot) -> None:
    st.subheader("Tasks")

    project_filter, sort_choice, _ = st.columns([2, 2, 3])
    projects = ["All projects", *sorted(snapshot.projects)]
    chosen_project = project_filter.selectbox("Project", projects, label_visibility="collapsed")
    chosen_sort = sort_choice.selectbox("Sort", list(SORTS), label_visibility="collapsed")

    def visible(statuses):
        tasks = snapshot.by_status(*statuses)
        if chosen_project != "All projects":
            tasks = [t for t in tasks if t.project == chosen_project]
        return sorted(tasks, key=SORTS[chosen_sort])

    tabs = st.tabs([f"{label} ({len(visible(statuses))})" for label, statuses in STATUS_TABS])
    for tab, (label, statuses) in zip(tabs, STATUS_TABS):
        with tab:
            tasks = visible(statuses)
            if not tasks:
                st.caption(
                    "No tasks here yet."
                    if chosen_project == "All projects"
                    else f"No {label.lower()} tasks for {chosen_project}."
                )
                continue
            for task in tasks:
                render_task(ctx, task, snapshot)


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

    with ctx.session() as session:
        snapshot = build_snapshot(session, config.task_rules)

    render_header(ctx, snapshot)
    drain_flash()
    render_sidebar(ctx)
    render_metrics(snapshot)
    render_controls(ctx)
    render_board(ctx, snapshot)
    render_events(snapshot)
