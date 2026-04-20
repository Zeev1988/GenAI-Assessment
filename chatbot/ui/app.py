"""
Streamlit frontend — Israeli HMO Chatbot
=========================================

Architecture
------------
All session state lives in ``st.session_state`` (client-side).  On every user
turn the full conversation history + user info is sent to the FastAPI backend,
keeping the backend completely stateless.

Phase flow
----------
1. **collection** — LLM collects member info conversationally.
   When the backend signals ``transition=True`` the frontend switches to Q&A.
2. **qa** — LLM answers health-fund questions using the member's profile.
   Member info is shown in the sidebar.

Async note
----------
Streamlit's execution model is synchronous.  The API is called with the
standard ``requests`` library (blocking).  This is fine because:
  • Each Streamlit user session runs in its own thread.
  • Concurrency is handled by the FastAPI backend, not the frontend.
  • A spinner provides feedback while waiting for the LLM.
"""

from __future__ import annotations

import os
import uuid

import requests
import streamlit as st

# ── Page config — must be the very first Streamlit call ──────────────────────
st.set_page_config(
    page_title="HMO Chatbot",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Configuration ─────────────────────────────────────────────────────────────

API_BASE_URL: str = os.environ.get("CHATBOT_API_URL", "http://localhost:8000")

# ── CSS — the only styling we actually need is Hebrew/English direction ───────
# Streamlit doesn't auto-detect RTL for mixed-language content, so we force
# per-element direction based on the first strong character.
_CSS = """
<style>
.stChatMessage p,
.stChatMessage li,
.stMarkdown p {
    direction: auto;
    text-align: start;
    unicode-bidi: plaintext;
}
.stChatInput textarea {
    direction: auto;
    unicode-bidi: plaintext;
}
</style>
"""


# ── Session-state helpers ──────────────────────────────────────────────────────

def _init_state() -> None:
    """Initialise all session-state keys on first run."""
    defaults: dict = {
        "phase": "collection",
        "messages": [],          # list[dict] with keys "role" and "content"
        "user_info": None,       # dict | None
        "session_id": str(uuid.uuid4()),
        "greeted": False,        # True once the opening LLM greeting has been fetched
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _reset() -> None:
    """Wipe session state and start over."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    _init_state()


# ── API client ─────────────────────────────────────────────────────────────────

def _call_api(extra_user_message: str | None = None) -> dict:
    """POST to /api/v1/chat and return the JSON response dict.

    If *extra_user_message* is provided it is appended to the payload messages
    (used for the initial greeting trigger where we don't want to show the
    hidden message in the UI).

    Returns ``{"error": "<reason>"}`` on any failure so the caller can display
    a user-friendly error without crashing.
    """
    messages_payload = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]
    if extra_user_message is not None:
        messages_payload.append({"role": "user", "content": extra_user_message})

    payload = {
        "phase": st.session_state.phase,
        "messages": messages_payload,
        "user_info": st.session_state.user_info,
        "request_id": str(uuid.uuid4()),
    }

    try:
        resp = requests.post(
            f"{API_BASE_URL}/api/v1/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        return {"error": "The request timed out. Please try again."}
    except requests.exceptions.ConnectionError:
        return {
            "error": (
                "Cannot reach the API server. "
                f"Is it running at `{API_BASE_URL}`?"
            )
        }
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        return {"error": f"Server error: {detail or str(exc)}"}
    except Exception as exc:
        return {"error": f"Unexpected error: {exc}"}


# ── Phase-transition handler ───────────────────────────────────────────────────

def _apply_api_response(api_resp: dict) -> bool:
    """Process an API response dict, updating session state.

    Returns True if the page should be rerun.
    """
    if "error" in api_resp:
        st.error(api_resp["error"])
        return False

    assistant_msg: str = api_resp.get("message", "")
    transition: bool = api_resp.get("transition", False)

    if transition:
        # Collection complete → switch to Q&A with a fresh message history.
        st.session_state.phase = "qa"
        st.session_state.user_info = api_resp.get("extracted_user_info")
        st.session_state.messages = [
            {"role": "assistant", "content": assistant_msg}
        ]
    else:
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_msg}
        )

    return True


# ── Sidebar ────────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## HMO Chatbot")
        st.caption("Medical services Q&A  ·  שירותי קופות חולים")
        st.divider()

        if st.session_state.phase == "qa" and st.session_state.user_info:
            u: dict = st.session_state.user_info
            st.markdown("**Member profile**")
            st.markdown(
                f"{u.get('first_name', '')} {u.get('last_name', '')}  \n"
                f"HMO: **{u.get('hmo_name', '')}**  \n"
                f"Tier: **{u.get('insurance_tier', '')}**  \n"
                f"Age: {u.get('age', '')}  ·  Gender: {u.get('gender', '')}"
            )
            st.divider()

        if st.button("Start over / התחל מחדש", use_container_width=True):
            _reset()
            st.rerun()

        st.divider()
        phase_label = (
            "Registration" if st.session_state.phase == "collection" else "Q&A"
        )
        st.caption(f"Phase: {phase_label}")
        st.caption(f"Session: `{st.session_state.session_id[:8]}…`")


# ── Header ─────────────────────────────────────────────────────────────────────

def _render_header() -> None:
    if st.session_state.phase == "collection":
        st.markdown("## Health Fund Chatbot")
        st.caption(
            "Step 1 of 2 — Registration.  "
            "The assistant will guide you through registration in a friendly "
            "conversation.  You may reply in Hebrew or English."
        )
    else:
        u = st.session_state.user_info or {}
        hmo = u.get("hmo_name", "")
        tier = u.get("insurance_tier", "")
        st.markdown("## Medical Services Q&A")
        st.caption(f"Step 2 of 2 — Ask about your {hmo} {tier}-tier benefits.")

    st.divider()


# ── Chat history ───────────────────────────────────────────────────────────────

def _render_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ── Main app ───────────────────────────────────────────────────────────────────

def main() -> None:
    _init_state()
    st.markdown(_CSS, unsafe_allow_html=True)

    _render_sidebar()
    _render_header()

    # ── Opening greeting (collection phase, first load) ────────────────────────
    # Trigger the LLM greeting automatically so the user doesn't have to type
    # first.  A hidden "[SESSION_START]" is sent to the API but NOT stored in
    # the visible history.
    if not st.session_state.greeted and st.session_state.phase == "collection":
        st.session_state.greeted = True
        with st.spinner("Connecting…"):
            api_resp = _call_api(extra_user_message="[SESSION_START]")
        _apply_api_response(api_resp)
        st.rerun()

    _render_history()

    # ── Chat input ─────────────────────────────────────────────────────────────
    placeholder = (
        "Type your answer… / הקלד תשובה…"
        if st.session_state.phase == "collection"
        else "Ask a question… / שאל שאלה…"
    )

    if prompt := st.chat_input(placeholder):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                api_resp = _call_api()

        if "error" in api_resp:
            # Roll back the user message if we couldn't process it.
            st.session_state.messages.pop()
            st.error(api_resp["error"])
        else:
            _apply_api_response(api_resp)

        st.rerun()


if __name__ == "__main__":
    main()
