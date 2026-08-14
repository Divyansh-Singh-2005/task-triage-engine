"""Canonical handling of project names.

Project names reach this application from several places: the config file,
the dashboard, and whatever task source is plugged in. They rarely agree on
casing or whitespace ("Project Dynamo", "project  dynamo", " Project Dynamo ").

Every comparison in the system goes through ``project_key`` so that a stray
space in a scraped label can never silently cause a task to be ignored.
"""

from __future__ import annotations


def normalize_project_name(name: str) -> str:
    """Collapse internal whitespace and strip the ends, preserving case.

    This is the *display* form: what gets stored in config and shown in the UI.
    """
    if not isinstance(name, str):
        raise TypeError(f"project name must be a string, got {type(name).__name__}")
    cleaned = " ".join(name.split())
    if not cleaned:
        raise ValueError("project name must not be empty")
    return cleaned


def project_key(name: str) -> str:
    """Case-insensitive comparison key for a project name.

    This is the *identity* form: never displayed, only compared.
    """
    return normalize_project_name(name).casefold()


def dedupe_projects(names: list[str]) -> list[str]:
    """Normalize a list of names, dropping case-insensitive duplicates.

    The first occurrence wins, so the display casing the user typed first is
    the one that survives. Order is otherwise preserved.
    """
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        display = normalize_project_name(name)
        key = project_key(display)
        if key not in seen:
            seen.add(key)
            result.append(display)
    return result
