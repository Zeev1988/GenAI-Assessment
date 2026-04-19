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
   Member info card is shown in the sidebar.

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
    page_title="HMO Chatbot | צ'אטבוט קופות חולים",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Configuration ─────────────────────────────────────────────────────────────

API_BASE_URL: str = os.environ.get("CHATBOT_API_URL", "http://localhost:8000")

TIER_ICON: dict[str, str] = {"זהב": "🥇", "כסף": "🥈", "ארד": "🥉"}
HMO_ICON: dict[str, str] = {"מכבי": "💙", "מאוחדת": "💚", "כללית": "❤️"}
GENDER_ICON: dict[str, str] = {"זכר": "👨", "נקבה": "👩", "אחר": "🧑"}

# ── Custom CSS ─────────────────────────────────────────────────────────────────

_CSS = """
<style>
/* Auto-detect text direction per element (Hebrew ↔ English) */
.stChatMessage p,
.stChatMessage li,
.stMarkdown p {
    direction: auto;
    text-align: start;
    unicode-bidi: plaintext;
}

/* Slightly larger chat bubbles */
div[data-testid="stChatMessageContent"] {
    font-size: 1.02rem;
    line-height: 1.6;
}

/* Input box auto-direction */
.stChatInput textarea {
    direction: auto;
    unicode-bidi: plaintext;
}

/* Softer progress bar */
div[data-testid="stProgressBar"] > div {
    background: linear-gradient(90deg, #1a73e8 0%, #34a853 100%);
}

/* Sidebar member card */
.member-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 12px 16px;
    border-left: 4px solid #1a73e8;
    margin-bottom: 8px;
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
    """
    POST to /api/v1/chat and return the JSON response dict.

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
        return {"error": "⏱️ The request timed out. Please try again."}
    except requests.exceptions.ConnectionError:
        return {
            "error": (
                "🔌 Cannot reach the API server. "
                f"Is it running at `{API_BASE_URL}`?"
            )
        }
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        return {"error": f"❌ Server error: {detail or str(exc)}"}
    except Exception as exc:
        return {"error": f"❌ Unexpected error: {exc}"}


# ── Phase-transition handler ───────────────────────────────────────────────────

def _apply_api_response(api_resp: dict, user_message: str | None = None) -> bool:
    """
    Process an API response dict, updating session state.

    Returns True if the page should be rerun, False if we already handle it.
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
        st.markdown("## 🏥 HMO Chatbot")
        st.caption("Medical Services Q&A · שירותי קופות חולים")
        st.divider()

        if st.session_state.phase == "qa" and st.session_state.user_info:
            u: dict = st.session_state.user_info
            hmo = u.get("hmo_name", "")
            tier = u.get("insurance_tier", "")
            gender = u.get("gender", "")

            st.markdown("### 👤 Member Profile")
            st.markdown(
                f"<div class='member-card'>"
                f"<b>{u.get('first_name','')} {u.get('last_name','')}</b><br>"
                f"{HMO_ICON.get(hmo,'🏥')} <b>{hmo}</b> &nbsp;|&nbsp; "
                f"{TIER_ICON.get(tier,'⭐')} {tier}<br>"
                f"{GENDER_ICON.get(gender,'🧑')} {gender} &nbsp;|&nbsp; גיל {u.get('age','')}"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.divider()
            st.markdown("**Available topics / נושאים זמינים:**")
            for topic in [
                "💊 רפואה משלימה · Alternative medicine",
                "🦷 שיניים · Dental services",
                "👁️ אופטומטריה · Optometry",
                "🤰 הריון · Pregnancy",
                "🏋️ סדנאות · Health workshops",
                "🗣️ מרפאות תקשורת · Comm. clinics",
            ]:
                st.caption(topic)
            st.divider()

        # Always show the reset button.
        if st.button("🔄 Start Over / התחל מחדש", use_container_width=True):
            _reset()
            st.rerun()

        st.divider()
        phase_label = (
            "📋 Registration"
            if st.session_state.phase == "collection"
            else "💬 Q&A"
        )
        st.caption(f"Phase: {phase_label}")
        st.caption(f"Session: `{st.session_state.session_id[:8]}…`")


# ── Header ─────────────────────────────────────────────────────────────────────

def _render_header() -> None:
    if st.session_state.phase == "collection":
        st.markdown("## 🏥 Health Fund Chatbot")
        st.markdown(
            "*שלב 1 — הרשמה | Step 1 — Registration*\n\n"
            "The assistant will guide you through registration in a friendly conversation. "
            "You may reply in **Hebrew or English**."
        )
        st.progress(0.3, text="Step 1 of 2: Registration")
    else:
        u = st.session_state.user_info or {}
        hmo = u.get("hmo_name", "")
        tier = u.get("insurance_tier", "")
        name = u.get("first_name", "")
        st.markdown(
            f"## {HMO_ICON.get(hmo, '🏥')} Medical Services Q&A"
        )
        st.markdown(
            f"*שלום {name}! אתה במסלול {TIER_ICON.get(tier,'')} {tier} ב{hmo}. "
            f"שאל אותי על שירותי הקופה שלך.*"
        )
        st.progress(1.0, text="Step 2 of 2: Q&A")

    st.divider()


# ── Chat history ───────────────────────────────────────────────────────────────

def _render_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ── Main app ───────────────────────────────────────────────────────────────────

def main() -> None:
    _init_state()

    # Inject custom CSS.
    st.markdown(_CSS, unsafe_allow_html=True)

    _render_sidebar()
    _render_header()

    # ── Opening greeting (collection phase, first load) ────────────────────────
    # We trigger the LLM greeting automatically so the user doesn't have to
    # type first.  A hidden "[SESSION_START]" message is sent to the API but
    # NOT stored in the visible message history.
    if not st.session_state.greeted and st.session_state.phase == "collection":
        st.session_state.greeted = True
        with st.spinner("Connecting… / מתחבר…"):
            api_resp = _call_api(extra_user_message="[SESSION_START]")
        _apply_api_response(api_resp)
        st.rerun()

    _render_history()

    # ── Chat input ─────────────────────────────────────────────────────────────
    placeholder = (
        "הקלד תשובה... / Type your answer…"
        if st.session_state.phase == "collection"
        else "שאל שאלה... / Ask a question…"
    )

    if prompt := st.chat_input(placeholder):
        # 1. Show the user's message immediately.
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Append to history so it is included in the API payload.
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 3. Call the backend.
        with st.chat_message("assistant"):
            with st.spinner("חושב... / Thinking…"):
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
