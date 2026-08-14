"""The task source boundary.

Everything upstream of this interface is "where tasks come from"; everything
downstream is provider-agnostic. Monitoring, filtering, evaluation, scoring,
approval and persistence all operate on :class:`Task` objects and neither know
nor care whether those arrived from a pasted listing, a file, an authorized
API, or anything else.

Practical consequence: a future authorized integration is a new subclass and
zero changes anywhere else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.schemas import Task


class TaskSource(ABC):
    """Produces normalized :class:`Task` objects from somewhere."""

    #: Short stable identifier, stored on every task row. Changing it for an
    #: existing source orphans that source's duplicate history, so don't.
    name: str = "abstract"

    @abstractmethod
    def fetch(self) -> list[Task]:
        """Return the tasks currently visible to this source.

        Implementations return everything they can see, including tasks that
        have been returned before. De-duplication is the ingestion service's
        job, not the source's.

        Raise :class:`core.exceptions.TaskSourceError` on failure rather than
        returning an empty list, so a broken source is never mistaken for a
        quiet one.
        """

    def describe(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
