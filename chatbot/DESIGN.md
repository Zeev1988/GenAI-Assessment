# Design Notes — HMO Chatbot

## Stateless API

The assignment requires statelessness, and it keeps the service trivially scalable: every request carries the full conversation history and the confirmed `user_info`. No server-side session store.

## Two-phase flow

The chatbot operates in two phases: **collection** (gathering member details) and **qa** (answering benefit questions).

In collection, the LLM exposes a `submit_user_info` tool. The system prompt instructs it to summarise the collected fields and wait for the user to confirm (e.g. "כן"/"yes") before calling the tool. When the tool is called, the backend transitions the session to Q&A.

The tool's JSON schema (with enum constraints on gender / HMO / tier and regex on the ID / card numbers) enforces the contract at the API level.

**Collected fields:** first name, last name, ID number (9 digits), gender (זכר/נקבה/אחר), age (0–120), HMO name (מכבי/מאוחדת/כללית), HMO card number (9 digits), insurance tier (זהב/כסף/ארד).

## Knowledge base & retrieval

The knowledge base is 6 HTML files containing service tables. Each file is split into one chunk per `<tr>` plus one intro chunk for the non-table content. Chunking by row preserves the column-to-HMO alignment — a fixed-size splitter would slice through cells.

Chunks are embedded once at startup with `text-embedding-ada-002`. For each Q&A turn the last user message is embedded and the top-k chunks are ranked by cosine similarity (in-memory linear scan — the corpus is ~42 chunks, so a vector DB would be overkill).

## Language handling

The system prompt tells the LLM to detect the user's language per-message and reply in kind (Hebrew or English). No hardcoded localisation.

## Logging

Console + rotating file (`chatbot.log`). Requests are tagged with a `request_id` and token usage is logged. The shared `common/logger.py` is used by both parts.
