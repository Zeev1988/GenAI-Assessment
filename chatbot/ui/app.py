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
        "pending_confirmation": None,  # dict | None
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

def _call_api(
    extra_user_message: str | None = None,
    *,
    user_confirmed: bool = False,
    confirmed_data: dict | None = None,
) -> dict:
    """POST to /api/v1/chat and return the JSON response dict.

    If *extra_user_message* is provided it is appended to the payload messages
    (used for the initial greeting trigger where we don't want to show the
    hidden message in the UI).

    *user_confirmed* / *confirmed_data* carry the typed-confirmation channel.
    Set them only in response to an explicit user action (clicking the
    "Confirm" button on the pending-info dialog).  The backend prefers this
    over classifying the latest user turn as an affirmative phrase.

    Returns ``{"error": "<reason>"}`` on any failure so the caller can display
    a user-friendly error without crashing.
    """
    messages_payload = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]
    if extra_user_message is not None:
        messages_payload.append({"role": "user", "content": extra_user_message})

    payload: dict = {
        "phase": st.session_state.phase,
        "messages": messages_payload,
        "user_info": st.session_state.user_info,
        "request_id": str(uuid.uuid4()),
    }
    if user_confirmed:
        payload["user_confirmed"] = True
        if confirmed_data is not None:
            payload["confirmed_data"] = confirmed_data

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
    confirmation_pending: bool = api_resp.get("confirmation_pending", False)
    pending_user_info = api_resp.get("pending_user_info")

    if transition:
        # Collection complete → switch to Q&A with a fresh message history.
        st.session_state.phase = "qa"
        st.session_state.user_info = api_resp.get("extracted_user_info")
        st.session_state.messages = [
            {"role": "assistant", "content": assistant_msg}
        ]
        st.session_state.pending_confirmation = None
    else:
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_msg}
        )
        # Stash the pending snapshot when the backend asks for confirmation,
        # clear it otherwise so a stale dialog can't linger across turns.
        st.session_state.pending_confirmation = (
            pending_user_info if confirmation_pending else None
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


# ── Confirmation dialog ───────────────────────────────────────────────────────

# Order matters — this is the sequence the dialog shows, matching the order
# the LLM collected the fields in.  Keeping it a module-level constant makes
# the ordering explicit rather than depending on dict-iteration quirks.
_CONFIRMATION_FIELD_ORDER: tuple[tuple[str, str], ...] = (
    ("first_name", "First name / שם פרטי"),
    ("last_name", "Last name / שם משפחה"),
    ("id_number", "ID number / ת.ז."),
    ("gender", "Gender / מין"),
    ("age", "Age / גיל"),
    ("hmo_name", "HMO / קופה"),
    ("hmo_card_number", "HMO card / מס' כרטיס"),
    ("insurance_tier", "Tier / מסלול"),
)


def _format_pending_summary(pending: dict) -> str:
    """Render the pending snapshot as a markdown key/value list."""
    lines: list[str] = []
    for key, label in _CONFIRMATION_FIELD_ORDER:
        if key in pending and pending[key] not in (None, ""):
            lines.append(f"- **{label}:** {pending[key]}")
    # Fall back to any fields we didn't know about so the user always sees
    # everything the backend is about to submit.
    known = {k for k, _ in _CONFIRMATION_FIELD_ORDER}
    for key, value in pending.items():
        if key in known or value in (None, ""):
            continue
        lines.append(f"- **{key}:** {value}")
    return "\n".join(lines)


# Synthetic user turn recorded in the visible chat history when the user
# clicks Confirm.  The backend decides on the typed flag, not on this text —
# but without a user turn the transcript would have a visible gap between
# "please confirm…" and "welcome to Q&A", which reads oddly.
_CONFIRM_USER_MESSAGE = "✅ אישרתי את הפרטים / Confirmed"


def _render_confirmation_dialog() -> bool:
    """Render the Confirm / Edit dialog.  Returns True if a rerun is needed.

    Called only when ``st.session_state.pending_confirmation`` is set, i.e.,
    the backend's most recent response had ``confirmation_pending=True``.
    Clicking Confirm relays the typed confirmation to the backend; clicking
    Edit clears the pending snapshot and lets the user type a correction in
    the normal chat input.
    """
    pending = st.session_state.pending_confirmation
    if not pending:
        return False

    with st.container(border=True):
        st.markdown("**Please confirm your details / בבקשה אשר/י את הפרטים:**")
        summary = _format_pending_summary(pending)
        if summary:
            st.markdown(summary)

        col_confirm, col_edit = st.columns(2)
        confirm_clicked = col_confirm.button(
            "Confirm / אישור",
            type="primary",
            use_container_width=True,
            key="confirm_pending_btn",
        )
        edit_clicked = col_edit.button(
            "Edit / תיקון",
            use_container_width=True,
            key="edit_pending_btn",
        )

    if confirm_clicked:
        # Snapshot the pending data before we clear it — the backend uses
        # this to cross-check against the LLM's submit_user_info arguments.
        confirmed_data = dict(pending)
        st.session_state.messages.append(
            {"role": "user", "content": _CONFIRM_USER_MESSAGE}
        )
        st.session_state.pending_confirmation = None
        with st.spinner("Finalising / מעבד…"):
            api_resp = _call_api(
                user_confirmed=True,
                confirmed_data=confirmed_data,
            )
        _apply_api_response(api_resp)
        return True

    if edit_clicked:
        # Just drop the pending snapshot; the user types their correction
        # normally in the next turn and the LLM will call
        # request_user_confirmation again with the updated data.
        st.session_state.pending_confirmation = None
        return True

    return False


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

    # ── Confirmation dialog (only in collection phase, when the backend
    #    signalled confirmation_pending=True on the last response) ────────────
    if (
        st.session_state.phase == "collection"
        and st.session_state.pending_confirmation
    ):
        if _render_confirmation_dialog():
            st.rerun()
        return

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
