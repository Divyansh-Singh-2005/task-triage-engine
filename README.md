# Handshake Task Agent

A configurable, multi-project task monitoring and assistance system. It ingests
task listings, filters them by project, evaluates each one against a skill
profile, applies deterministic decision rules, and surfaces what deserves your
attention on a dashboard.

The project name is never hard-coded. Monitoring scope is configuration.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Config system, SQLite persistence, task model, project filter, tests | Done |
| 2 | LLM provider abstraction, validated evaluation, decision engine | Done |
| 3 | Streamlit dashboard, human approval workflow | Done |
| 4 | Structured logging, scheduler loop, dashboard polish | Done |
| 5 | Authorized platform integration and claim workflow | **Not built - see below** |

## On platform integration

The system deliberately contains **no browser automation and no claim action.**

Automated claiming on a work platform is the piece most likely to violate terms
of service, regardless of how politely the automation behaves - polling and
clicking at machine speed beats humans to the same queue. That question is
unresolved, so the integration does not exist rather than existing quietly.

Everything else works without it. Tasks enter through the `TaskSource`
interface, and the whole pipeline downstream is provider-agnostic. If a
sanctioned feed turns out to be available, it is one new subclass in
`agent/sources/` registered in `agent/sources/registry.py`, and nothing else
changes.

Approving a task marks intent locally. Acting on that intent is out of scope.

## Architecture

```
  task source --> project filter --> duplicate guard --> database
                                                             |
                                                             v
                                        LLM provider --> evaluation (validated)
                                                             |
                                                             v
                                          decision engine (deterministic rules)
                                                             |
                                     +-----------------------+--------------+
                                     v                       v              v
                                  SKIPPED            REVIEW_REQUIRED     APPROVED
                                                             |
                                                             v
                                                     human approval
```

Two ideas carry most of the design:

**The model analyses; code decides.** `TaskEvaluation` has no decision field.
The model returns score, estimated hours, deadline feasibility and risks; the
decision engine maps that to SKIP / REVIEW_REQUIRED / HIGH_MATCH using rules
that are unit-tested and reconstructible months later without re-running a
model.

**Constraints can only lower an outcome, never raise it.** A perfect score with
a passed deadline is still SKIP. A perfect score at full workload cannot reach
APPROVED.

### Layout

```
app.py                      Streamlit entrypoint
config/       settings.py   env-derived paths and secrets
              defaults.json seed agent config
              profile.json  the skill profile scoring is measured against
core/         schemas, enums, evaluation contract, logging, naming
agent/        project_manager, task_filter, decision_engine,
              workload_manager, orchestrator, sources/
database/     engine, ORM models, repositories
integrations/ llm/ provider abstraction (heuristic, anthropic)
services/     ingestion, evaluation, approval, board, agent_context
ui/           dashboard
scripts/      run_agent, demo_ingest, demo_evaluate
tests/        unit tests
```

## Setup

Requires Python 3.12+.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
pytest -q
```

## Running

**Dashboard** - agent control, project switching, task board, approvals:

```powershell
streamlit run app.py
```

**Monitoring loop** - polls the configured sources on a timer:

```powershell
python -m scripts.run_agent --interval 300
python -m scripts.run_agent --once      # single cycle
python -m scripts.run_agent --force     # single cycle even if the agent is stopped
```

The loop honours the agent status set in the dashboard: while stopped, cycles
are skipped rather than the process exiting, so you can leave it running and
toggle the agent from the UI. Config is re-read every cycle - mode changes take
effect without a restart.

## Feeding it tasks

The only source wired up today is `data/inbox.json`, polled every cycle. Drop a
JSON list in:

```json
[
  {
    "task_id": "D-1",
    "project_name": "Project Dynamo",
    "title": "Verifier-backed terminal task",
    "description": "Exact-arithmetic scoring with a holdout re-execution test.",
    "requirements": ["Python", "pytest", "Docker"],
    "deadline": "2026-10-01T12:00:00Z",
    "reward": "$150",
    "url": "https://example.com/task/D-1"
  }
]
```

Only `project_name` and `title` are required. A missing `task_id` is synthesized
from the title. The dashboard also takes a one-off paste under "Add tasks by
hand".

Duplicates are guarded twice: a unique constraint on `(source, external_task_id)`,
and a content fingerprint for sources with no stable id. Re-ingesting the same
file stores nothing.

## Monitoring modes

| Mode | Behaviour |
|---|---|
| `ACTIVE_PROJECT` | Only tasks from the one selected project |
| `SELECTED_PROJECTS` | Only tasks from the chosen set |
| `ANY_PROJECT` | No project restriction |

Project names compare case- and whitespace-insensitively, so `"project  dynamo"`
from a scraped label matches `"Project Dynamo"`.

## Evaluation

Two providers:

- **`heuristic`** - offline keyword overlap against `config/profile.json`. No key,
  no network, deterministic. Used by the test suite.
- **`anthropic`** - a real model. Falls back to the heuristic baseline if
  `ANTHROPIC_API_KEY` is blank, so a missing key degrades scoring rather than
  killing the loop.

Model output is never trusted: fenced JSON, JSON buried in prose, and a string
where a list belongs all parse; out-of-range scores and missing fields are
rejected. One retry with a repair instruction, then the task is marked FAILED -
one bad evaluation never aborts the batch.

Score quality tracks `config/profile.json` directly. Edit it.

## Configuration

Behaviour lives in `data/agent_config.json`, owned solely by `ProjectManager`
(validated, atomic writes). Paths and secrets live in `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Root log level |
| `LOG_FILE` | *(none)* | Also write logs to this file |
| `DATA_DIR` | `./data` | Database, config, inbox |
| `DATABASE_URL` | *(SQLite)* | Set a DSN to move to PostgreSQL |
| `POLL_INTERVAL_SECONDS` | `300` | Default loop interval |
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `heuristic` |
| `LLM_MODEL` | `claude-sonnet-4-5` | Model id |
| `ANTHROPIC_API_KEY` | *(blank)* | Never commit this |

## Logging

One line per event, context as `key=value`:

```
2026-09-13T18:41:08 INFO  agent.orchestrator ingested event=ingest source=inbox stored=1
```

A redacting filter sits on the handler, so anything resembling a credential is
scrubbed no matter which module logged it - including third-party libraries.
That is a safety net, not a licence: do not log secrets deliberately.

Every significant action is also written to the `agent_events` table, which is
what the dashboard activity panel reads.

## Testing

```powershell
pytest -q
```

No network calls, no API key required.

## Database

SQLite by default at `data/agent.db`, with foreign keys and WAL enabled. Tables:
`projects`, `tasks`, `evaluations` (append-only, so re-scoring stays auditable),
`agent_events`. Schema is created on startup via `create_all`; swap to Alembic
once it starts changing under real data.
