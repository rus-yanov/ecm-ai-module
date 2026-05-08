"""Build ground truth from text-based real PDF documents.

Walks three real-document directories, checks each PDF for extractable text
(fitz word count > 20), runs regex extraction, and saves records with at least
2 non-None fields to experiment/real_ground_truth.json.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).parent.parent
EXPERIMENT_DIR = ROOT / "experiment"
OUT_PATH = EXPERIMENT_DIR / "real_ground_truth.json"

REAL_DOCS = Path("/Users/rustamakhmedzianov/Downloads/мага/вкр/ВКР/образцы документов")

_DIRS: dict[str, tuple[str, Path]] = {
    "ACT":     ("ACT",     REAL_DOCS / "акт"),
    "WAYBILL": ("WAYBILL", REAL_DOCS / "требование-накладная"),
    "ORDER":   ("ORDER",   REAL_DOCS / "приказ"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(pdf_path: Path) -> str | None:
    """Return full text if document has >20 words on any page, else None."""
    try:
        doc = fitz.open(str(pdf_path))
        pages_text = [page.get_text() for page in doc]
        doc.close()
    except Exception as exc:
        print(f"    [fitz error] {pdf_path.name}: {exc}")
        return None

    full = "\n".join(pages_text)
    if len(full.split()) > 20:
        return full
    return None


def _to_iso(date_str: str) -> str:
    """Convert DD.MM.YYYY → YYYY-MM-DD."""
    parts = date_str.split(".")
    if len(parts) == 3:
        return f"{parts[2]}-{parts[1]}-{parts[0]}"
    return date_str


# ---------------------------------------------------------------------------
# Extractors per document type
# ---------------------------------------------------------------------------

def _extract_act(text: str) -> dict:
    # Use pipeline field names: executor / executor_inn / receiver_name / total_cost_with_vat

    act_number = None
    m = re.search(r'№\s*(\S+)', text[:200])
    if m:
        act_number = m.group(1).strip(".,;")

    act_date = None
    m = re.search(r'(\d{1,2}[.]\d{2}[.]\d{4})', text[:400])
    if m:
        act_date = _to_iso(m.group(1))

    # executor / receiver_name may have value on the next line after the label;
    # trim at first comma to exclude INN/KPP appended on the same line
    lines = text.splitlines()
    executor = None
    for i, line in enumerate(lines):
        if "Исполнитель" in line and ":" in line:
            val = line.split(":", 1)[1].strip()
            if not val and i + 1 < len(lines):
                val = lines[i + 1].strip()
            if val:
                executor = val.split(",")[0].strip()
                break

    executor_inn = None
    m = re.search(r'ИНН[:\s]+(\d{10,12})', text)
    if m:
        executor_inn = m.group(1)

    receiver_name = None
    for i, line in enumerate(lines):
        if "Заказчик" in line and ":" in line:
            val = line.split(":", 1)[1].strip()
            if not val and i + 1 < len(lines):
                val = lines[i + 1].strip()
            if val:
                receiver_name = val.split(",")[0].strip()
                break

    # total_cost_with_vat: handle "55 550,00\nруб" format (amount and unit on separate lines)
    total_cost_with_vat = None
    matches = re.findall(r'(\d[\d\s]*(?:[,.]\d+)?)\s*\n?\s*(?:руб|₽)', text, re.IGNORECASE)
    if matches:
        raw = matches[-1].strip().replace(" ", "").replace(",", ".")
        try:
            total_cost_with_vat = str(round(float(raw), 2))
        except ValueError:
            total_cost_with_vat = raw

    return {
        "act_number":        act_number,
        "act_date":          act_date,
        "executor":          executor,
        "executor_inn":      executor_inn,
        "receiver_name":     receiver_name,
        "total_cost_with_vat": total_cost_with_vat,
    }


def _extract_waybill(text: str) -> dict:
    # document_number: use "ТРЕБОВАНИЕ-НАКЛАДНАЯ № N" pattern to skip the form-type header (М-11)
    document_number = None
    m = re.search(r'ТРЕБОВАНИЕ-НАКЛАДНАЯ\s*№\s*(\S+)', text, re.IGNORECASE)
    if m:
        document_number = m.group(1).strip(".,;")
    else:
        # fallback: skip "Форма № …" lines, take second № occurrence
        matches = list(re.finditer(r'№\s*(\S+)', text[:400]))
        if len(matches) >= 2:
            document_number = matches[1].group(1).strip(".,;")
        elif matches:
            document_number = matches[0].group(1).strip(".,;")

    document_date = None
    m = re.search(r'(\d{1,2}[.]\d{2}[.]\d{4})', text[:400])
    if m:
        document_date = _to_iso(m.group(1))

    return {
        "document_number": document_number,
        "document_date":   document_date,
    }


def _extract_order(text: str) -> dict:
    # Use pipeline field names: order_number / order_date / title

    # order_number: try "ПРИКАЗ № N" first; fall back to any № followed by a
    # multi-char token (filters out bare list-item numbers like №1, №2 and п/п headers)
    order_number = None
    m = re.search(r'ПРИКАЗ\s*(?:№|N)\s*(\S+)', text, re.IGNORECASE)
    if m:
        order_number = m.group(1).strip(".,;")
    else:
        # Academic orders put number in footer; skip п/п, single-digit, and colon-only tokens
        for m in re.finditer(r'(?:№|N)\s*(\S+)', text):
            candidate = m.group(1).strip(".,;:")
            if not candidate or candidate in ("п/п",):
                continue
            # Accept if multi-char and not just 1-2 digits (list item numbers)
            if re.fullmatch(r'\d{1,2}', candidate):
                continue
            order_number = candidate
            break

    # order_date: search full text (some orders put the date in the footer)
    order_date = None
    m = re.search(r'(\d{1,2}[.]\d{2}[.]\d{4})', text)
    if m:
        order_date = _to_iso(m.group(1))

    # title: "О…" subject clause at the start of the document; collapse newlines
    title = None
    m = re.search(r'(?:^|\n)\s*(О[б]?\s+[^\n]{5,})', text[:600], re.IGNORECASE)
    if m:
        title = " ".join(m.group(1).split())
    if not title:
        hlines = [ln.strip() for ln in text[:400].splitlines() if ln.strip()]
        for i, line in enumerate(hlines):
            if "приказ" in line.lower() and i + 1 < len(hlines):
                title = hlines[i + 1]
                break
    # Multi-line titles: join lines until we have a meaningful phrase; cap before ПРИКАЗЫВАЮ
    if title and len(title) < 10:
        more = re.search(r'О[б]?\s+(\S.{10,})', text[:600].replace('\n', ' '), re.IGNORECASE)
        if more:
            title = " ".join(more.group(0).split())
    if title:
        # Trim at ПРИКАЗЫВАЮ or first full stop to avoid including the body
        title = re.split(r'\s*(?:ПРИКАЗЫВАЮ|\.(?:\s|$))', title, maxsplit=1)[0].strip()
        title = title[:200]

    return {
        "order_number": order_number,
        "order_date":   order_date,
        "title":        title,
    }


_EXTRACTORS = {
    "ACT":     _extract_act,
    "WAYBILL": _extract_waybill,
    "ORDER":   _extract_order,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    for doc_type, (label, dirpath) in _DIRS.items():
        if not dirpath.exists():
            print(f"{label}: directory not found — {dirpath}")
            continue

        pdfs = sorted(dirpath.glob("*.pdf"))
        total = len(pdfs)
        yielded = 0

        extractor = _EXTRACTORS[doc_type]

        for pdf_path in pdfs:
            if doc_type == "ACT" and "RequirementOC" in pdf_path.name:
                print(f"  [skip ОС-1к form] {pdf_path.name}")
                total -= 1
                continue

            text = _extract_text(pdf_path)
            if text is None:
                print(f"  [scan/empty] {pdf_path.name}")
                is_scan = True
                # still record scan docs for future OCR track
                records.append({
                    "filename":      pdf_path.name,
                    "filepath":      str(pdf_path),
                    "document_type": doc_type,
                    "is_scan":       True,
                    "attributes":    {},
                })
                continue

            is_scan = False
            attrs = extractor(text)
            non_none = sum(1 for v in attrs.values() if v is not None)

            if non_none < 2:
                print(f"  [skip <2 fields] {pdf_path.name}  (got {non_none})")
                continue

            yielded += 1
            records.append({
                "filename":      pdf_path.name,
                "filepath":      str(pdf_path),
                "document_type": doc_type,
                "is_scan":       False,
                "attributes":    {k: v for k, v in attrs.items() if v is not None and str(v).strip()},
            })

        text_total = sum(1 for p in pdfs if True)  # recount after scan filter
        text_based = total - sum(
            1 for r in records
            if r["document_type"] == doc_type and r["is_scan"]
        )
        print(f"{label}: {yielded}/{total} files yielded ground truth  "
              f"({total - yielded} skipped)")

    OUT_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"\nSaved {len(records)} records → {OUT_PATH}")

    # per-category summary
    print("\nPer-category summary:")
    for doc_type in _DIRS:
        subset = [r for r in records if r["document_type"] == doc_type]
        text_docs = [r for r in subset if not r["is_scan"]]
        scan_docs = [r for r in subset if r["is_scan"]]
        print(f"  {doc_type:10s}: {len(text_docs)} text-based, {len(scan_docs)} scans")


if __name__ == "__main__":
    main()
