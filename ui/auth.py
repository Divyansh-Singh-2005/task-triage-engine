"""Optional passphrase gate for the dashboard.

Be clear about what this is: a speed bump, not authentication. There are no
accounts, no sessions, no rate limiting, and the passphrase sits in ``.env`` in
plain text. It exists so that a dashboard deliberately exposed beyond loopback
is not wide open to anyone who guesses the port.

The real control is ``.streamlit/config.toml`` binding to 127.0.0.1. Leave
``DASHBOARD_PASSPHRASE`` blank and this is a no-op.
"""

from __future__ import annotations

import hmac

import streamlit as st

_UNLOCKED = "dashboard_unlocked"


def require_passphrase(expected: str) -> bool:
    """Return True when the dashboard may render.

    A blank ``expected`` disables the gate entirely.
    """
    if not expected:
        return True
    if st.session_state.get(_UNLOCKED):
        return True

    st.title("Handshake Task Agent")
    entered = st.text_input("Passphrase", type="password")
    if st.button("Unlock", type="primary"):
        # Constant-time comparison, so response timing says nothing useful.
        if hmac.compare_digest(entered, expected):
            st.session_state[_UNLOCKED] = True
            st.rerun()
        else:
            st.error("Incorrect passphrase.")
    st.caption(
        "Set DASHBOARD_PASSPHRASE in .env to change this, or leave it blank to "
        "disable the prompt."
    )
    return False
