# Architecture & Design Notes: HMO Chatbot

This document details the engineering decisions, state management logic, and retrieval strategies behind the HMO Chatbot microservice.

---

## 🧠 State & Phase Management

### 1. Completely Stateless API
The brief mandates statelessness, but there's also a pragmatic reason: any server-side session store introduces sticky routing and a new failure mode for multi-instance deployments. Passing the full conversation history client-side on every request means the API is infinitely horizontally scalable, and a Container App restart can't lose any user's context.

### 2. Tool-Calling for Phase Transitions
The chatbot operates in two distinct phases: **"collection"** (gathering user details) and **"qa"** (answering benefit questions). 

Using an explicit `submit_user_info` OpenAI function-call lets the LLM signal "collection complete" with a strongly-typed payload. The alternative — parsing a keyword like `[DONE]` in the reply text or making a separate classifier call — is less reliable and not self-documenting. The JSON schema (with `enum` constraints for gender / HMO / tier) *is* the strict contract.

**Data Collected & Validated by LLM:**
* First & Last name
* ID number (exactly 9 digits)
* Gender (זכר / נקבה / אחר)
* Age (0–120)
* HMO name (מכבי / מאוחדת / כללית)
* HMO card number (exactly 9 digits)
* Insurance tier (זהב / כסף / ארד)

---

## 🔍 Knowledge Base & Retrieval Layer

### 1. Chunking by `<tr>` (Table Rows)
The raw data consists of 6 HTML files containing service tables. A standard, fixed-size text chunker would cut indiscriminately through table cells, losing the vital column-to-HMO alignment. 

Instead, the backend splits every HTML file structurally: one chunk for the `<h2>` intro narrative, and **one chunk per service table row (`<tr>`)**. This ensures every chunk answers a complete semantic question ("How much does X cost at tier Y for HMO Z?") while remaining fully self-contained. Each chunk is also prepended with its parent topic title.

### 2. Why Retrieval? (Instead of Prompt Stuffing)
The six HTML files total ~43 KB. This comfortably fits inside GPT-4o's context window, meaning prompt-stuffing would technically work. 

We use Retrieval-Augmented Generation (RAG) anyway because:
1.  **Production Readiness:** A real HMO service catalogue contains thousands of rows. The retrieval path scales; stuffing does not.
2.  **Accuracy:** Narrower context measurably reduces mix-ups between tiers and HMOs during spot-checks.

### 3. Why No Vector Database?
The total corpus generates ~42 chunks. A linear cosine similarity scan using standard math libraries over ~42 vectors executes in the microsecond range. 

Adding a vector database (like Chroma, FAISS, or pgvector) would only add a heavy dependency and a persistence management burden in exchange for performance improvements the user cannot perceive. Relying purely on the native Azure OpenAI SDK and in-memory math perfectly aligns with the "no frameworks" approach, keeping the Docker image and runtime extremely lean.

---

## 🛠️ Additional Design Choices

### Multi-Language Mirroring
The LLM dynamically detects if the user is typing in Hebrew or English per-message and mirrors the language back automatically without requiring hardcoded localization files.

### PII & Logging
Logs go to both the console and a rotating file (`chatbot.log`). Every request is logged with its `request_id`, phase, token usage, and response time. The shared `common/logger.py` ensures consistent logging structures across both the extraction pipeline and the chatbot, while sensitive `user_info` payloads are omitted from standard INFO logs.
