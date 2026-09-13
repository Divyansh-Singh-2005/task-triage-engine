"""Where task sources get wired up.

One function builds the list of sources the scheduler polls, so adding a
source later means editing this file and nothing else.
"""

from __future__ import annotations

from agent.sources.base import TaskSource
from agent.sources.inbox import InboxSource
from config.settings import Settings


def build_sources(settings: Settings) -> list[TaskSource]:
    """Every source the agent polls on each cycle.

    Today that is the watched inbox file alone. An authorized platform source
    would be appended here once its terms are confirmed; nothing downstream
    needs to change.
    """
    return [InboxSource(settings.inbox_path)]
