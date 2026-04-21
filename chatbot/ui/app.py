"""Streamlit frontend for the HMO Chatbot.

All state lives in st.session_state. Each turn sends the full conversation
history + user_info to the stateless API.
"""

from __future__ import annotations

import os
import uuid

import requests
import streamlit as st

st.set_page_config(page_title="HMO Chatbot", layout="centered", initial_sidebar_state="expanded")

API_BASE_URL: str = os.environ.get("CHATBOT_API_URL", "http://localhost:8000")

# Force per-element direction so Hebrew and English render correctly.
_CSS = """
<style>
.stChatMessage p, .stChatMessage li, .stMarkdown p {
    direction: auto;
    text-align: start;
    unicode-bidi: plaintext;
}
.stChatInput textarea { direction: auto; unicode-bidi: plaintext; }
</style>
"""


def _init_state() -> None:
    defaults = {
        "phase": "collection",
        "messages": [],
        "user_info": None,
        "session_id": str(uuid.uuid4()),
        "greeted": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _reset() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    _init_state()


def _call_api(extra_user_message: str | None = None) -> dict:
    """POST to /api/v1/chat. Returns {'error': ...} on failure."""
    messages_payload = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
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
        resp = requests.post(f"{API_BASE_URL}/api/v1/chat", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        return {"error": "The request timed out."}
    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot reach the API server at `{API_BASE_URL}`."}
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            detail = ""
        return {"error": f"Server error: {detail or str(exc)}"}


def _apply_api_response(api_resp: dict) -> None:
    if "error" in api_resp:
        st.error(api_resp["error"])
        return

    assistant_msg = api_resp.get("message", "")

    if api_resp.get("transition"):
        st.session_state.phase = "qa"
        st.session_state.user_info = api_resp.get("extracted_user_info")
        st.session_state.messages = [{"role": "assistant", "content": assistant_msg}]
    else:
        st.session_state.messages.append({"role": "assistant", "content": assistant_msg})


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## HMO Chatbot")
        st.caption("Medical services Q&A · שירותי קופות חולים")
        st.divider()

        if st.session_state.phase == "qa" and st.session_state.user_info:
            u = st.session_state.user_info
            st.markdown("**Member profile**")
            st.markdown(
                f"{u.get('first_name', '')} {u.get('last_name', '')}  \n"
                f"HMO: **{u.get('hmo_name', '')}**  \n"
                f"Tier: **{u.get('insurance_tier', '')}**  \n"
                f"Age: {u.get('age', '')} · Gender: {u.get('gender', '')}"
            )
            st.divider()

        if st.button("Start over / התחל מחדש", use_container_width=True):
            _reset()
            st.rerun()

        st.divider()
        phase_label = "Registration" if st.session_state.phase == "collection" else "Q&A"
        st.caption(f"Phase: {phase_label}")


def _render_header() -> None:
    if st.session_state.phase == "collection":
        st.markdown("## Health Fund Chatbot")
        st.caption("Step 1 of 2 — Registration. Reply in Hebrew or English.")
    else:
        u = st.session_state.user_info or {}
        st.markdown("## Medical Services Q&A")
        st.caption(
            f"Step 2 of 2 — Ask about your {u.get('hmo_name', '')} "
            f"{u.get('insurance_tier', '')}-tier benefits."
        )
    st.divider()


def main() -> None:
    _init_state()
    st.markdown(_CSS, unsafe_allow_html=True)

    _render_sidebar()
    _render_header()

    # Trigger the opening LLM greeting on first load.
    if not st.session_state.greeted and st.session_state.phase == "collection":
        st.session_state.greeted = True
        with st.spinner("Connecting…"):
            api_resp = _call_api(extra_user_message="[SESSION_START]")
        _apply_api_response(api_resp)
        st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

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
            st.session_state.messages.pop()
            st.error(api_resp["error"])
        else:
            _apply_api_response(api_resp)

        st.rerun()


if __name__ == "__main__":
    main()
