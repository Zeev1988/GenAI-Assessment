"""One-time calibration tool for Form 283 field coordinates.

Run this script ONCE on the blank (empty) Form 283 PDF to discover the
exact Azure DI word coordinates for every label and input area on the form.
Use the printed output to fill in the regions in field_regions.py.

Usage
-----
    python calibrate.py phase1_data/283_raw.pdf

Output
------
  • Console: every word on every page printed as:
        PAGE 1  [x0=0.312  y0=0.541  x1=1.234  y1=0.721]  'תאריך'
  • calibration_words.json: machine-readable list of all words + coordinates
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parent))
from form_extraction.core.config import get_settings


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _poly_bbox(polygon: list[float]) -> tuple[float, float, float, float]:
    """Convert a flat polygon list to (x_min, y_min, x_max, y_max) in inches."""
    xs = polygon[0::2]
    ys = polygon[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def _poly_center(polygon: list[float]) -> tuple[float, float]:
    xs = polygon[0::2]
    ys = polygon[1::2]
    return sum(xs) / len(xs), sum(ys) / len(ys)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(pdf_path: str) -> None:
    s = get_settings()
    client = DocumentIntelligenceClient(
        endpoint=s.azure_doc_intelligence_endpoint,
        credential=AzureKeyCredential(s.azure_doc_intelligence_key.get_secret_value()),
    )

    data = Path(pdf_path).read_bytes()
    print(f"Uploading {Path(pdf_path).name} ({len(data):,} bytes) to Azure DI …")

    with client:
        poller = client.begin_analyze_document(
            model_id="prebuilt-layout",
            body=AnalyzeDocumentRequest(bytes_source=data),
            locale="he",
        )
        result = poller.result()

    print(f"Pages returned: {len(result.pages or [])}\n")

    all_pages: list[dict] = []

    for page_idx, page in enumerate(result.pages or []):
        page_num = page_idx + 1
        words_data: list[dict] = []

        print(f"{'=' * 70}")
        print(f"PAGE {page_num}  ({len(page.words or [])} words)")
        print(f"{'=' * 70}")

        for word in sorted(page.words or [], key=lambda w: (
            _poly_center(w.polygon)[1],   # sort top-to-bottom
            -_poly_center(w.polygon)[0],  # then right-to-left (Hebrew)
        ) if getattr(w, "polygon", None) else (0, 0)):

            poly = getattr(word, "polygon", None)
            if not poly or len(poly) < 4:
                continue

            x0, y0, x1, y1 = _poly_bbox(poly)
            cx, cy = _poly_center(poly)
            text = (word.content or "").strip()

            print(
                f"  PAGE {page_num}  "
                f"[x0={x0:.3f}  y0={y0:.3f}  x1={x1:.3f}  y1={y1:.3f}]  "
                f"cx={cx:.3f}  cy={cy:.3f}  '{text}'"
            )

            words_data.append({
                "text": text,
                "x0": round(x0, 4),
                "y0": round(y0, 4),
                "x1": round(x1, 4),
                "y1": round(y1, 4),
                "cx": round(cx, 4),
                "cy": round(cy, 4),
            })

        all_pages.append({"page": page_num, "words": words_data})
        print()

    out = Path("calibration_words.json")
    out.write_text(json.dumps(all_pages, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved → {out.resolve()}")
    print()
    print("Next step: open calibration_words.json (or read the output above),")
    print("find the bounding box of each field's INPUT AREA (not the label),")
    print("and paste the (x0, y0, x1, y1) values into form_extraction/core/field_regions.py.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calibrate.py <path_to_blank_283.pdf>")
        sys.exit(1)
    main(sys.argv[1])
