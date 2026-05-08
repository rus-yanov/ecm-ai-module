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
    head = text[:200]
    tail = text

    act_number = None
    m = re.search(r'№\s*(\S+)', head)
    if m:
        act_number = m.group(1).strip(".,;")

    act_date = None
    m = re.search(r'(\d{1,2}[.]\d{2}[.]\d{4})', text[:400])
    if m:
        act_date = _to_iso(m.group(1))

    contractor_name = None
    for line in text.splitlines():
        if "Исполнитель" in line and ":" in line:
            val = line.split(":", 1)[1].strip()
            if val:
                contractor_name = val
                break

    contractor_inn = None
    m = re.search(r'ИНН[:\s]+(\d{10,12})', text)
    if m:
        contractor_inn = m.group(1)

    client_name = None
    for line in text.splitlines():
        if "Заказчик" in line and ":" in line:
            val = line.split(":", 1)[1].strip()
            if val:
                client_name = val
                break

    total_amount = None
    matches = re.findall(r'([\d\s]+(?:[,.]\d+)?)\s*(?:руб|₽)', tail, re.IGNORECASE)
    if matches:
        raw = matches[-1].strip().replace(" ", "").replace(",", ".")
        try:
            total_amount = str(round(float(raw), 2))
        except ValueError:
            total_amount = raw

    return {
        "act_number":      act_number,
        "act_date":        act_date,
        "contractor_name": contractor_name,
        "contractor_inn":  contractor_inn,
        "client_name":     client_name,
        "total_amount":    total_amount,
    }


def _extract_waybill(text: str) -> dict:
    document_number = None
    m = re.search(r'№\s*(\S+)', text[:200])
    if m:
        document_number = m.group(1).strip(".,;")

    document_date = None
    m = re.search(r'(\d{1,2}[.]\d{2}[.]\d{4})', text[:400])
    if m:
        document_date = _to_iso(m.group(1))

    sender_department = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        low = stripped.lower()
        if any(kw in low for kw in ("отправитель", "сдатчик", "структурное подразделение-отправитель")):
            if ":" in stripped:
                val = stripped.split(":", 1)[1].strip()
                if val:
                    sender_department = val
                    break
            else:
                sender_department = stripped
                break

    receiver_department = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        low = stripped.lower()
        if any(kw in low for kw in ("получатель", "структурное подразделение-получатель")):
            if ":" in stripped:
                val = stripped.split(":", 1)[1].strip()
                if val:
                    receiver_department = val
                    break
            else:
                receiver_department = stripped
                break

    return {
        "document_number":     document_number,
        "document_date":       document_date,
        "sender_department":   sender_department,
        "receiver_department": receiver_department,
    }


def _extract_order(text: str) -> dict:
    head = text[:400]

    order_number = None
    m = re.search(r'ПРИКАЗ[^№]*№\s*(\S+)', text, re.IGNORECASE)
    if m:
        order_number = m.group(1).strip(".,;")
    else:
        m = re.search(r'№\s*(\S+)', head)
        if m:
            order_number = m.group(1).strip(".,;")

    order_date = None
    m = re.search(r'(\d{1,2}[.]\d{2}[.]\d{4})', head)
    if m:
        order_date = _to_iso(m.group(1))

    organization_name = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            organization_name = stripped
            break

    order_subject = None
    m = re.search(r'\bОб?\s+(.+)', head)
    if m:
        order_subject = m.group(0).strip()
    if not order_subject:
        lines = [l.strip() for l in head.splitlines() if l.strip()]
        for i, line in enumerate(lines):
            low = line.lower()
            if "приказ" in low and i + 1 < len(lines):
                order_subject = lines[i + 1]
                break

    return {
        "order_number":      order_number,
        "order_date":        order_date,
        "organization_name": organization_name,
        "order_subject":     order_subject,
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
