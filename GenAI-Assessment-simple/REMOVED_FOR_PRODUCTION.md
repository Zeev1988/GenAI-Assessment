# What the lean version dropped — and when to bring it back

A fair question on the retry layer reminded me that "not needed for a take-home"
is not the same as "wrong for production." This document walks through every
significant piece of scaffolding I stripped out of the original and assigns it
a rank so you can tell at a glance which parts were over-engineering, which
parts were genuinely broken, and which parts are table stakes once you leave
the prototype stage.

## Verified facts about SDK retries

Before ranking anything retry-related, the empirical baseline:

- **OpenAI Python SDK** (including `AzureOpenAI`): `max_retries=2` by default.
  Retries 408, 409, 429, ≥500, and connection errors with short exponential
  backoff. Tunable per client (`AzureOpenAI(max_retries=5)`) or per request
  (`client.with_options(max_retries=5).chat.completions.create(...)`).
- **azure-core** (used by `azure-ai-documentintelligence`): the default
  `RetryPolicy` retries up to 10 total (3 connection / 3 read / 3 status),
  exponential backoff factor 0.8 capped at 120 s, retries 408/429/500/502/503/504.

So the SDKs handle the common transient-failure cases without help. A custom
retry wrapper on top adds observability (logs of each attempt) and a unified
budget across both SDKs — real value, but not free.

## Ranking rubric

| Rank | Label | Meaning |
|------|-------|---------|
| 5    | Keep for any real service | Table stakes; the lean version is weaker without it once the pipeline is not a single-user POC. |
| 4    | Strong production value | Worth porting when you move off Streamlit / add a second service / onboard on-call. |
| 3    | Depends on context | Useful for some production settings (cost optimization, compliance, scale), neutral for others. |
| 2    | Over-engineering in most contexts | Adds surface area without clear ROI; only defensible under specific constraints. |
| 1    | Actively harmful | Contradicts the spec, invents data, hides errors, or overfits tests to samples. Would be a bug in production. |

## Verdict table

| # | Piece | Rank | One-line verdict |
|---|-------|------|------------------|
| 1 | Typed exception hierarchy (`common/errors.py`) | 4 | Clean mapping of internal failures → HTTP codes; bring back with FastAPI. |
| 2 | Input validation (MIME sniff + size limit) | 5 | Mandatory the moment uploads hit a real HTTP endpoint. |
| 3 | PII redaction helpers | 4 | Good code; original bug was that it was never actually called. |
| 4 | Structured JSON logging + correlation IDs | 4 | Table stakes for any service behind a load balancer. |
| 5 | Stage timings + cache-hit flags in result | 4 | Direct ops value when you need to know where latency lives. |
| 6 | Tenacity retry wrapper | 3 | Observability win, but must disable SDK retries or budgets compound. |
| 7 | Persistent disk cache (diskcache) | 3 | Real savings when the same file is re-uploaded; overkill for single-user. |
| 8 | LLM-as-judge evaluator | 3 | Solid for sampled accuracy monitoring; "disabled by default" is the smell. |
| 9 | Async pipeline + sync wrapper | 2 | Only pays off if you add FastAPI and concurrent uploads. |
| 10 | Multi-layer guardrail + targeted re-ask | 2 | Doubles LLM cost for marginal gains; prefer human-in-the-loop. |
| 11 | ID-digit / phone-separator stripping | 4 | Information-preserving normalization; safe to keep. |
| 12 | Gender canonicalization ("M" → "male") | 3 | Helps downstream consumers; trivially defensible. |
| 13 | Luhn checksum on Israeli ID | 3 | Catches real typos; fair warning-level signal. |
| 14 | Cross-date ordering rules | 3 | Genuine data-quality check in production intake. |
| 15 | Docker / Makefile / pre-commit / mypy-strict | 3 | Normal production hygiene; excessive for a 4-day take-home. |
| 16 | Phone "6 → 0" / "8 → 0" rewrite | **1** | **Invents data.** Contradicts the system prompt. |
| 17 | `enforce_strict_format` clearing malformed values | **1** | **Destructive.** Real OCR'd values get silently replaced with "". |
| 18 | Mixed-script name leak rule | 2 | Overfits to a specific OCR pathology on the supplied samples. |
| 19 | Traffic-term cross-check against description | 2 | Heuristic overfit to the three filled forms; test comments admit it. |
| 20 | Accident-location anchor whitelist | 2 | Same — reverse-engineered from samples, not from the form spec. |
| 21 | Synthetic PDF generator (as implemented) | **1** | Flat `label: value` sheets don't exercise OCR layout/checkboxes, so green tests give false confidence. |
| 22 | 120-year age plausibility rule | 2 | Defensible as a warning, but strictly outside the assignment. |
| 23 | Name-must-appear-in-signature cross-check | 2 | Overfits to a specific observed failure. |

---

## Code that was removed, with rationale per rank

The snippets below are drop-in portable. If you ever promote this to a
FastAPI microservice with a real SLO, pull these into `common/` in the lean
version and wire them up deliberately, one at a time.

### Rank 5: keep for any real service

#### Input validation (MIME sniffing + size limit)

```python
# common/security.py
from __future__ import annotations
from typing import Any
from form_extraction.errors import InputError  # see typed errors below

_ALLOWED = frozenset({"application/pdf", "image/jpeg", "image/png"})
_PDF, _JPEG, _PNG = b"%PDF-", b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n"


def sniff_mime(data: bytes, filename: str | None = None) -> str:
    if not data:
        raise InputError("Uploaded file is empty.")
    if data.startswith(_PDF):
        return "application/pdf"
    if data.startswith(_JPEG):
        return "image/jpeg"
    if data.startswith(_PNG):
        return "image/png"
    if filename:
        lower = filename.lower()
        if lower.endswith(".pdf"):
            return "application/pdf"
        if lower.endswith((".jpg", ".jpeg")):
            return "image/jpeg"
        if lower.endswith(".png"):
            return "image/png"
    raise InputError("Unsupported file type.")


def validate_upload(data: bytes, filename: str | None, max_bytes: int) -> str:
    if len(data) > max_bytes:
        raise InputError("File too large.", details={"size": len(data), "max": max_bytes})
    mime = sniff_mime(data, filename)
    if mime not in _ALLOWED:
        raise InputError("Unsupported MIME type.", details={"mime": mime})
    return mime
```

**Why rank 5:** Streamlit's `type=[...]` filter is client-side only. The
moment this pipeline is exposed via HTTP, you need a magic-byte sniff (trust
nothing you didn't parse) and a size cap (prevent DoS / runaway Azure bills).

### Rank 4: strong production value

#### Typed exception hierarchy

```python
# common/errors.py
from __future__ import annotations
from typing import Any


class AppError(Exception):
    code: str = "app_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class InputError(AppError):       code = "input_error"
class AzureAuthError(AppError):   code = "azure_auth_error"
class OCRError(AppError):         code = "ocr_error"
class ExtractionError(AppError):  code = "extraction_error"
```

**Why rank 4:** In a real service you want `except AppError as e: return
JSONResponse(e.to_dict(), status_code=map_code(e.code))`. The lean version
raises bare `RuntimeError` which is fine for Streamlit but will bite you on
day one of FastAPI.

#### PII redaction

```python
# common/pii.py
from __future__ import annotations
import re
from typing import Any

_ID_RE = re.compile(r"\b\d{9}\b")
_PHONE_RE = re.compile(r"\b0\d{1,2}[-\s]?\d{7}\b")

_SENSITIVE = frozenset({"idNumber", "mobilePhone", "landlinePhone", "signature"})


def mask_value(v: str) -> str:
    return "**" if len(v) <= 2 else "*" * (len(v) - 2) + v[-2:]


def redact_text(t: str) -> str:
    t = _ID_RE.sub("[REDACTED_ID]", t)
    t = _PHONE_RE.sub("[REDACTED_PHONE]", t)
    return t


def redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: (mask_value(v) if k in _SENSITIVE and isinstance(v, str) and v else redact(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(x) for x in obj]
    if isinstance(obj, str):
        return redact_text(obj)
    return obj
```

**Why rank 4:** Israeli ID numbers, phones, and signatures are PII. In
production you call `redact(payload)` *before* it reaches any log sink. The
original defined this but never called it — that was the bug.

#### Structured JSON logging + correlation IDs

```python
# common/logging_setup.py
from __future__ import annotations
import logging, sys, uuid
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterator
import structlog

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="-")
_configured = False


def configure_logging(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=level.upper())
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            lambda _l, _m, d: {**d, "correlation_id": _correlation_id.get()},
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )
    _configured = True


@contextmanager
def correlation_scope(cid: str | None = None) -> Iterator[str]:
    token = _correlation_id.set(cid or uuid.uuid4().hex)
    try:
        yield _correlation_id.get()
    finally:
        _correlation_id.reset(token)
```

**Why rank 4:** Without a correlation ID threaded through every log line you
cannot answer "what happened during the upload that just failed for customer
X at 3:14." Structured JSON is what log aggregators (Datadog, Splunk, CloudWatch)
actually index.

#### Stage timings and cache-hit metadata

```python
# form_extraction/pipeline.py (extended version)
@dataclass
class PipelineResult:
    form: ExtractedForm
    report: ValidationReport
    ocr_text: str
    timings_ms: dict[str, float]
    from_cache: dict[str, bool]


def run(data: bytes) -> PipelineResult:
    t0 = time.perf_counter()
    ocr_text = run_ocr(data)
    t1 = time.perf_counter()
    form = extract(ocr_text)
    t2 = time.perf_counter()
    report = validate(form)
    t3 = time.perf_counter()
    return PipelineResult(
        form=form, report=report, ocr_text=ocr_text,
        timings_ms={"ocr_ms": (t1 - t0) * 1000, "extract_ms": (t2 - t1) * 1000, "validate_ms": (t3 - t2) * 1000},
        from_cache={"ocr": False, "extract": False},
    )
```

**Why rank 4:** In production you alert on p95 latency per stage, not overall.
Without stage breakdowns you cannot tell if OCR or the LLM is the slow path,
and you can't prove to Azure support whose service is degraded.

#### ID digit stripping / phone separator normalization (safe variant)

```python
# form_extraction/normalize.py
import re

_NON_DIGIT = re.compile(r"\D+")


def strip_id_digits(value: str) -> str:
    """Strip separators from an ID. Leave garbage untouched."""
    if not value:
        return value
    digits = _NON_DIGIT.sub("", value)
    return digits if len(digits) == 9 else value


def strip_phone_separators(value: str) -> str:
    """Strip spaces/hyphens/parens. Never rewrite leading digits."""
    if not value:
        return value
    return _NON_DIGIT.sub("", value)
```

**Why rank 4:** Pure separator removal is information-preserving —
`"123-456-789"` → `"123456789"` is the same ID. This is the *safe* subset of
the original sanitizer. Deliberately excludes the `6→0`/`8→0` rewrite (see
rank 1 below).

### Rank 3: depends on context

#### Tenacity retry wrapper (with logged attempts)

```python
# common/retry.py
from __future__ import annotations
import httpx
from typing import Any
from azure.core.exceptions import HttpResponseError, ServiceRequestError, ServiceResponseError
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from tenacity import AsyncRetrying, RetryCallState, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from common.logging_setup import configure_logging
import structlog

_log = structlog.get_logger(__name__)

_RETRYABLE = (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError,
              ServiceRequestError, ServiceResponseError,
              httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, _RETRYABLE):
        return True
    if isinstance(exc, HttpResponseError):
        return (exc.status_code or 0) >= 500
    return False


def _log_retry(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    _log.warning("retry", attempt=state.attempt_number,
                 next_sleep_s=round(state.next_action.sleep, 3) if state.next_action else None,
                 exc_type=type(exc).__name__ if exc else None)


async def with_retry(func: Any, *args: Any, max_attempts: int = 5, **kwargs: Any) -> Any:
    async for attempt in AsyncRetrying(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=0.5, max=20.0, jitter=0.5),
        retry=retry_if_exception(_retryable),
        before_sleep=_log_retry,
    ):
        with attempt:
            return await func(*args, **kwargs)
```

**Why rank 3:** The win is *logged* retries — Azure and OpenAI SDKs both
retry silently, which is maddening during an incident. The trap: if you
wrap the SDK without lowering the SDK's own budget, you get
`tenacity_attempts × sdk_max_retries` actual requests, which can blow through
rate limits faster than not retrying at all. If you port this, also set
`AzureOpenAI(max_retries=0)` and pass `retry_total=0` to the Azure DI client.

#### Persistent result cache

```python
# common/cache.py
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any
from diskcache import Cache

_TTL_S = {"ocr": 30 * 24 * 3600, "extract": 30 * 24 * 3600}


class ResultCache:
    def __init__(self, cache_dir: Path) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = Cache(directory=str(cache_dir))

    @staticmethod
    def fingerprint(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def get(self, stage: str, fp: str) -> Any | None:
        return self._cache.get(f"{stage}:{fp}")

    def set(self, stage: str, fp: str, value: Any) -> None:
        self._cache.set(f"{stage}:{fp}", value, expire=_TTL_S.get(stage))
```

**Why rank 3:** Document Intelligence charges per page. If your users
re-upload the same PDF (e.g., an intake worker verifies extraction, then
re-runs after fixing the filename), a fingerprint-keyed cache cuts Azure
spend directly. For single-user Streamlit it's ceremony.

#### LLM-as-judge evaluator

```python
# form_extraction/judge.py
from __future__ import annotations
import json
from openai import AzureOpenAI
from form_extraction.schemas import ExtractedForm

_SYS = "Score how faithfully the extracted JSON represents the OCR. 0-100."
_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"score": {"type": "integer"}, "comments": {"type": "array", "items": {"type": "string"}}},
    "required": ["score", "comments"],
}


def judge(ocr: str, form: ExtractedForm, client: AzureOpenAI, deployment: str) -> tuple[int, list[str]]:
    r = client.chat.completions.create(
        model=deployment, temperature=0.0,
        response_format={"type": "json_schema", "json_schema": {"name": "Judge", "schema": _SCHEMA, "strict": True}},
        messages=[
            {"role": "system", "content": _SYS},
            {"role": "user", "content": f"OCR:\n{ocr}\n\nJSON:\n{json.dumps(form.model_dump(), ensure_ascii=False)}"},
        ],
    )
    data = json.loads(r.choices[0].message.content or "{}")
    return max(0, min(100, int(data.get("score", 0)))), [str(c) for c in data.get("comments", []) if c]
```

**Why rank 3:** Excellent pattern for continuous accuracy monitoring — run
the judge on a 1% sample in production, alert when the rolling mean score
drops. The original hurt itself by shipping it disabled by default; either
use it (on a sample) or remove it.

#### Luhn checksum on Israeli ID

```python
def israeli_id_valid(value: str) -> bool:
    if not value.isdigit() or len(value) != 9:
        return False
    total = 0
    for i, ch in enumerate(value):
        d = int(ch) * (1 if i % 2 == 0 else 2)
        total += d if d < 10 else d - 9
    return total % 10 == 0
```

**Why rank 3:** Catches single-digit typos that the 9-digit length check
misses. Legitimate production-grade data-quality signal. As a *warning*, not
an error — Teudat Zehut with invalid checksums do exist in older records.

#### Cross-date ordering rules

```python
from datetime import date
from form_extraction.schemas import DatePart

def parse(p: DatePart) -> date | None:
    if not (p.day and p.month and p.year):
        return None
    try:
        return date(int(p.year), int(p.month), int(p.day))
    except ValueError:
        return None


def check_date_order(form) -> list[str]:
    issues: list[str] = []
    birth = parse(form.dateOfBirth)
    injury = parse(form.dateOfInjury)
    filing = parse(form.formFillingDate)
    receipt = parse(form.formReceiptDateAtClinic)
    if birth and injury and injury < birth:
        issues.append("dateOfInjury precedes dateOfBirth")
    if injury and filing and filing < injury:
        issues.append("formFillingDate precedes dateOfInjury")
    if filing and receipt and receipt < filing:
        issues.append("formReceiptDateAtClinic precedes formFillingDate")
    return issues
```

**Why rank 3:** These are genuine data-quality checks. They don't overfit to
a specific OCR pathology — they test logical invariants of any real filled
form. The original was right to include them; it was just more than the
brief asked for. Bring them back when you wire this into a real intake pipeline.

### Rank 2: over-engineering in most contexts

#### Multi-layer guardrail / targeted re-ask

The full orchestrator is in the original at
`form_extraction/backend/extraction/extractor.py`. Skeleton:

```python
async def extract_and_validate(ocr):
    form = await extract_fields(ocr)      # LLM call 1
    form = sanitize(form)                  # deterministic fixes
    report = build_report(form)            # run rules
    bad = recoverable_error_fields(report.issues)
    if bad:
        corrected = await targeted_reask(ocr, form, report.issues)  # LLM call 2
        if corrected:
            form = sanitize(corrected)
            report = build_report(form)
    form = enforce_strict_format(form)     # destructive clear (see rank 1)
    return form, report
```

**Why rank 2:** Doubles LLM cost on any form that trips a rule. The better
production pattern is an *asynchronous* review loop: surface low-confidence
extractions to a human queue, and retrain the prompt on the corrections.
Automatic re-ask tends to mask rather than fix the underlying prompt bug.

#### Async pipeline + sync wrapper

```python
# Original pattern:
async def run_pipeline(data: bytes) -> PipelineResult: ...

def run_pipeline_sync(data: bytes) -> PipelineResult:
    return asyncio.run(run_pipeline(data))  # blows up inside an existing loop
```

**Why rank 2:** Makes sense with FastAPI handling concurrent uploads. Pure
cost for Streamlit, which is single-threaded and synchronous.

### Rank 1: actively harmful

#### Phone "6 → 0" / "8 → 0" leading-digit rewrite

```python
# DO NOT PORT — this is the broken one.
_PHONE_LEADING_FIX = {"6": "0", "8": "0"}

def sanitize_phone(value: str) -> str:
    digits = re.sub(r"\D+", "", value)
    if len(digits) == 10 and digits[0] in _PHONE_LEADING_FIX:
        return _PHONE_LEADING_FIX[digits[0]] + digits[1:]  # <-- invents data
    return digits
```

**Why rank 1:** The system prompt tells the LLM "never invent, guess, or
paraphrase." Then the sanitizer silently rewrites the LLM's output based on
the theory that a leading `6` or `8` is a misread `0`. If the OCR actually
saw a `6`, the form either has a real `6` (you've just corrupted it) or the
OCR is wrong (you should log the discrepancy, not silently "fix" it). Keep
`strip_phone_separators` (rank 4) instead.

#### `enforce_strict_format` clearing malformed values

```python
# DO NOT PORT — destroys evidence.
def enforce_strict_format(form):
    data = form.model_dump()
    if data["idNumber"] and not re.fullmatch(r"\d{9}", data["idNumber"]):
        data["idNumber"] = ""   # <-- real OCR'd value is lost
    ...
    return ExtractedForm.model_validate(data)
```

**Why rank 1:** This silently replaces a field that *had something on the
form* with `""`, which is indistinguishable from "the field was blank on the
form." Downstream systems cannot tell those two cases apart. In production,
preserve the raw value, flag it as invalid, and let a human decide — never
erase.

#### Synthetic PDF generator (as implemented)

```python
# scripts/synth_form.py — renders "label: value" rows onto a blank A4
# instead of overlaying values onto the real 283_raw.pdf template.
c.drawRightString(540, y, _shape(f"{label}: {value}"))
```

**Why rank 1:** The generated PDFs don't exercise OCR's table detection,
checkbox/selection-mark handling, or RTL column layout — none of which
appear in a `label: value` listing. Tests on this data are green when the
real form still fails, which is worse than having no synthetic tests. If
you want synthetic data, overlay values onto `283_raw.pdf` at calibrated
coordinates (painful but correct) or use document-synth tools that preserve
layout. For now, the three real samples in `phase1_data/` are enough.

## How to use this doc

Treat it as a porting checklist. When you need to promote this from a
Streamlit POC to a production microservice, move items in rank order:

1. First port **rank 5** (`validate_upload`, typed errors) — without these,
   you have security holes.
2. Then port **rank 4** (logging, correlation IDs, PII redaction, stage
   timings, safe normalizers) — these are what your on-call engineer will
   ask for at 3 a.m.
3. Port **rank 3** selectively based on actual cost / accuracy signals.
4. Leave **rank 2** alone unless you have a measured reason.
5. **Never port rank 1.** Those were genuine defects, not opinionated choices.
