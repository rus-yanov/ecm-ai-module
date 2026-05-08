"""Experiment runner for ECM AI Module.

Two-track experiment:
  default (scanned)  — full OCR → LLM pipeline on corpus/scanned/
  --raw              — Track A: fitz text extraction → LLM fast-path on corpus/raw/
  --ocr-spot-check   — Track B: OCR quality report on real documents (no pipeline)

Usage:
    uv run python scripts/run_experiment.py [--limit N] [--apply-calibration]
    uv run python scripts/run_experiment.py --raw
    uv run python scripts/run_experiment.py --ocr-spot-check
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import yaml
from rapidfuzz import fuzz
from tqdm import tqdm

# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
CORPUS_DIR = ROOT / "corpus"
EXPERIMENT_DIR = ROOT / "experiment"
GROUND_TRUTH_PATH = CORPUS_DIR / "ground_truth.json"
REAL_GROUND_TRUTH_PATH = EXPERIMENT_DIR / "real_ground_truth.json"
THRESHOLDS_PATH = ROOT / "config" / "thresholds.yaml"

API_BASE = "http://localhost:8000"
ECM_BASE = "http://localhost:8001"

# Document type → schema_id for the API form field
_SCHEMA_ID: dict[str, str] = {
    "INVOICE": "invoice",
    "ACT": "act",
    "ORDER": "order",
    "WAYBILL": "waybill",
    "PAYMENT": "payment",
}

# Semantic field-type detection for comparison logic
_DATE_KEYWORDS = {"date", "deadline"}
_AMOUNT_KEYWORDS = {"amount", "sum", "total", "vat"}


def _is_date_field(name: str) -> bool:
    n = name.lower()
    return any(kw in n for kw in _DATE_KEYWORDS)


def _is_amount_field(name: str) -> bool:
    n = name.lower()
    return any(kw in n for kw in _AMOUNT_KEYWORDS)


# ===========================================================================
# Server management
# ===========================================================================

def _is_alive(url: str) -> bool:
    try:
        return httpx.get(url, timeout=2.0).status_code == 200
    except Exception:
        return False


def _wait_for(url: str, label: str, max_wait: int = 20) -> None:
    for _ in range(max_wait):
        time.sleep(1)
        if _is_alive(url):
            print(f"  {label} ready")
            return
    sys.exit(f"ERROR: {label} failed to start (timeout {max_wait}s)")


def ensure_servers() -> None:
    """Start mock_ecm and main API servers if not already responding."""
    if not _is_alive(f"{ECM_BASE}/healthz"):
        print("Starting mock ECM on :8001 ...")
        subprocess.Popen(
            ["uv", "run", "uvicorn", "mock_ecm.main:app",
             "--port", "8001", "--log-level", "warning"],
            cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        _wait_for(f"{ECM_BASE}/healthz", "mock ECM")
    else:
        print("mock ECM already running on :8001")

    if not _is_alive(f"{API_BASE}/healthz"):
        print("Starting main API on :8000 ...")
        subprocess.Popen(
            ["uv", "run", "uvicorn", "app.api.main:app",
             "--port", "8000", "--log-level", "warning"],
            cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        _wait_for(f"{API_BASE}/healthz", "main API", max_wait=25)
    else:
        print("main API already running on :8000")

    _warm_up_llm()


def _warm_up_llm() -> None:
    """Send a minimal prompt to Ollama so the model is loaded before the experiment."""
    import json as _json
    try:
        health = httpx.get(f"{API_BASE}/healthz", timeout=2.0).json()
        model = health.get("model", "")
    except Exception:
        return

    ollama_url = "http://localhost:11434/api/generate"
    print(f"Warming up LLM ({model}) ...")
    try:
        resp = httpx.post(
            ollama_url,
            json={"model": model, "prompt": "1+1=", "stream": False},
            timeout=120.0,
        )
        if resp.status_code == 200:
            duration_ms = resp.json().get("total_duration", 0) // 1_000_000
            print(f"  LLM warm  ({duration_ms} ms)")
        else:
            print(f"  LLM warm-up returned {resp.status_code}")
    except Exception as exc:
        print(f"  LLM warm-up skipped: {exc}")


# ===========================================================================
# Attribute comparison
# ===========================================================================

def _compare_value(
    attr_name: str,
    expected: str | float | int,
    extracted: str | None,
) -> bool:
    """Return True if extracted value matches expected within tolerance."""
    if extracted is None or str(extracted).strip() == "":
        return False

    exp_str = str(expected).strip()
    ext_str = str(extracted).strip()

    if _is_amount_field(attr_name):
        try:
            return abs(float(exp_str.replace(",", ".").replace(" ", ""))
                       - float(ext_str.replace(",", ".").replace(" ", ""))) < 1.0
        except ValueError:
            return False

    if _is_date_field(attr_name):
        return exp_str == ext_str  # both must be ISO-8601 after normalization

    return fuzz.ratio(exp_str.lower(), ext_str.lower()) >= 70


# ===========================================================================
# Document processing — scanned track (default)
# ===========================================================================

def process_document(
    client: httpx.Client,
    filename: str,
    expected_type: str,
    expected_attrs: dict,
) -> dict:
    """POST one scanned PDF to the API and return a structured result record."""
    scan_path = CORPUS_DIR / "scanned" / filename
    schema_id = _SCHEMA_ID.get(expected_type, expected_type.lower())

    try:
        with scan_path.open("rb") as fh:
            t0 = time.monotonic()
            response = client.post(
                f"{API_BASE}/api/v1/documents/process",
                files={"file": (filename, fh, "application/pdf")},
                data={"schema_id": schema_id},
                timeout=90.0,
            )
            elapsed = time.monotonic() - t0
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        return _error_record(filename, expected_type, expected_attrs, str(exc))

    return _parse_response(filename, expected_type, expected_attrs, data, elapsed)


# ===========================================================================
# Document processing — raw / fast-path track (--raw)
# ===========================================================================

def process_document_raw(
    client: httpx.Client,
    filename: str,
    expected_type: str,
    expected_attrs: dict,
) -> dict:
    """Track A: extract text from raw PDF with fitz, POST as .txt (OCR bypassed)."""
    import fitz  # PyMuPDF — already a project dependency

    raw_filename = filename.replace(".pdf", "_raw.pdf")
    raw_path = CORPUS_DIR / "raw" / raw_filename
    schema_id = _SCHEMA_ID.get(expected_type, expected_type.lower())
    txt_filename = filename.replace(".pdf", ".txt")

    try:
        data_bytes = raw_path.read_bytes()
        doc = fitz.open(stream=data_bytes, filetype="pdf")
        text = "\n".join(p.get_text() for p in doc)
        doc.close()

        t0 = time.monotonic()
        response = client.post(
            f"{API_BASE}/api/v1/documents/process",
            files={"file": (txt_filename, text.encode("utf-8"), "text/plain")},
            data={"schema_id": schema_id},
            timeout=150.0,
        )
        elapsed = time.monotonic() - t0
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        return _error_record(filename, expected_type, expected_attrs, str(exc))

    return _parse_response(filename, expected_type, expected_attrs, data, elapsed)


# ===========================================================================
# Document processing — real documents (--real)
# ===========================================================================

def process_real_document(
    client: httpx.Client,
    entry: dict,
) -> dict:
    """Process one real document.

    Text-based PDFs: extract with fitz, POST as .txt (OCR bypassed).
    Scanned PDFs: POST PDF bytes directly (full OCR pipeline).
    """
    filepath = Path(entry["filepath"])
    filename = entry["filename"]
    doc_type = entry["document_type"]
    expected_attrs = entry.get("attributes", {})
    schema_id = _SCHEMA_ID.get(doc_type, doc_type.lower())
    is_scan = entry.get("is_scan", False)

    try:
        if is_scan:
            data_bytes = filepath.read_bytes()
            t0 = time.monotonic()
            response = client.post(
                f"{API_BASE}/api/v1/documents/process",
                files={"file": (filename, data_bytes, "application/pdf")},
                data={"schema_id": schema_id},
                timeout=120.0,
            )
            elapsed = time.monotonic() - t0
        else:
            import fitz as _fitz
            doc = _fitz.open(str(filepath))
            text = "\n".join(p.get_text() for p in doc)
            doc.close()
            txt_name = Path(filename).stem + ".txt"
            t0 = time.monotonic()
            response = client.post(
                f"{API_BASE}/api/v1/documents/process",
                files={"file": (txt_name, text.encode("utf-8"), "text/plain")},
                data={"schema_id": schema_id},
                timeout=240.0,
            )
            elapsed = time.monotonic() - t0
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        return _error_record(filename, doc_type, expected_attrs, str(exc))

    return _parse_response(filename, doc_type, expected_attrs, data, elapsed)


# ---------------------------------------------------------------------------
# Shared response parsing helpers
# ---------------------------------------------------------------------------

def _error_record(
    filename: str, expected_type: str, expected_attrs: dict, error: str
) -> dict:
    return {
        "filename": filename,
        "expected_type": expected_type,
        "predicted_type": None,
        "type_correct": False,
        "type_confidence": 0.0,
        "requires_manual_review": False,
        "processing_time_sec": 0.0,
        "attributes": {
            k: {"expected": v, "extracted": None, "match": False, "confidence": 0.0}
            for k, v in expected_attrs.items()
        },
        "error": error,
    }


def _parse_response(
    filename: str,
    expected_type: str,
    expected_attrs: dict,
    data: dict,
    elapsed: float,
) -> dict:
    predicted_type = data.get("document_type", "UNKNOWN")
    type_confidence = float(data.get("type_confidence", 0.0))
    requires_manual_review = bool(data.get("requires_manual_review", False))
    processing_time = float(data.get("total_processing_time_sec", elapsed))

    extracted_map: dict[str, dict] = {
        a["name"]: a for a in data.get("attributes", [])
    }

    attr_results: dict[str, dict] = {}
    for attr_name, expected_val in expected_attrs.items():
        ext = extracted_map.get(attr_name)
        if ext is not None:
            ext_val = ext.get("normalized_value") or ext.get("raw_value")
            conf = float(ext.get("confidence", 0.0))
        else:
            ext_val = None
            conf = 0.0

        attr_results[attr_name] = {
            "expected": expected_val,
            "extracted": ext_val,
            "match": _compare_value(attr_name, expected_val, ext_val),
            "confidence": conf,
        }

    return {
        "filename": filename,
        "expected_type": expected_type,
        "predicted_type": predicted_type,
        "type_correct": predicted_type == expected_type,
        "type_confidence": type_confidence,
        "requires_manual_review": requires_manual_review,
        "processing_time_sec": processing_time,
        "attributes": attr_results,
        "error": data.get("error"),
    }


# ===========================================================================
# Track B — OCR spot-check on real documents
# ===========================================================================

_REAL_DOC_BASE = Path(
    "/Users/rustamakhmedzianov/Downloads/мага/вкр/ВКР/образцы документов"
)
_SPOT_CHECK_DIRS = {
    "счет-фактура": _REAL_DOC_BASE / "счет-фактура",
    "акт": _REAL_DOC_BASE / "акт",
    "приказ": _REAL_DOC_BASE / "приказ",
}


def run_ocr_spot_check() -> None:
    """Track B: OCR quality report on real documents. No pipeline, no ground truth."""
    import asyncio
    import random

    sys.path.insert(0, str(ROOT))
    from app.services.ocr.service import OcrService  # noqa: PLC0415

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    svc = OcrService()
    records: list[dict] = []

    async def _check_all() -> None:
        for label, dirpath in _SPOT_CHECK_DIRS.items():
            if not dirpath.exists():
                print(f"  [skip] {label}: directory not found ({dirpath})")
                continue

            pdfs = sorted(dirpath.glob("*.pdf"))
            if not pdfs:
                print(f"  [skip] {label}: no PDF files found")
                continue

            sample = random.sample(pdfs, min(5, len(pdfs)))
            print(f"\n  {label}/ ({len(sample)} files sampled):")

            for pdf_path in sorted(sample):
                data_bytes = pdf_path.read_bytes()
                result = await svc.recognize(data_bytes, pdf_path.name)
                text_preview = " ".join(b.text for b in result.blocks)[:120]
                print(f"    {pdf_path.name}")
                print(f"      conf={result.avg_confidence:.3f}  blocks={len(result.blocks)}")
                print(f"      text: {text_preview!r}")
                records.append({
                    "category": label,
                    "filename": pdf_path.name,
                    "avg_confidence": round(result.avg_confidence, 4),
                    "block_count": len(result.blocks),
                    "text_preview": text_preview,
                })

    asyncio.run(_check_all())

    out_path = EXPERIMENT_DIR / "ocr_spot_check.json"
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"\nOCR spot check saved → {out_path}")


# ===========================================================================
# Metrics
# ===========================================================================

DOC_TYPES = ["INVOICE", "ACT", "ORDER", "WAYBILL", "PAYMENT"]


def compute_metrics(results: list[dict]) -> dict:
    total = len(results)
    if total == 0:
        return {}

    # ------------------------------------------------------------------
    # 1. Type classification
    # ------------------------------------------------------------------
    correct = sum(1 for r in results if r["type_correct"])
    overall_accuracy = correct / total

    per_type_accuracy: dict[str, float] = {}
    for dt in DOC_TYPES:
        subset = [r for r in results if r["expected_type"] == dt]
        per_type_accuracy[dt] = (
            sum(1 for r in subset if r["type_correct"]) / len(subset)
            if subset else 0.0
        )

    all_predicted = DOC_TYPES + ["UNKNOWN"]
    confusion: dict[str, dict[str, int]] = {
        exp: {pred: 0 for pred in all_predicted} for exp in DOC_TYPES
    }
    for r in results:
        exp = r["expected_type"]
        pred = r["predicted_type"] or "UNKNOWN"
        if exp in confusion:
            if pred in confusion[exp]:
                confusion[exp][pred] += 1
            else:
                confusion[exp]["UNKNOWN"] += 1

    # ------------------------------------------------------------------
    # 2. Attribute extraction per type
    # ------------------------------------------------------------------
    attr_metrics: dict[str, dict[str, dict]] = {}
    avg_f1_per_type: dict[str, float] = {}

    for dt in DOC_TYPES:
        subset = [r for r in results if r["expected_type"] == dt]
        if not subset:
            continue

        all_attrs: set[str] = set()
        for r in subset:
            all_attrs.update(r["attributes"])

        dt_metrics: dict[str, dict] = {}
        for attr in sorted(all_attrs):
            matched = sum(
                1 for r in subset if r["attributes"].get(attr, {}).get("match", False)
            )
            extracted_count = sum(
                1 for r in subset
                if r["attributes"].get(attr, {}).get("extracted") is not None
            )
            expected_count = len(subset)

            precision = matched / extracted_count if extracted_count > 0 else 0.0
            recall = matched / expected_count if expected_count > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0.0
            )
            dt_metrics[attr] = {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "matched": matched,
                "extracted": extracted_count,
                "expected": expected_count,
            }

        attr_metrics[dt] = dt_metrics
        f1_vals = [m["f1"] for m in dt_metrics.values()]
        avg_f1_per_type[dt] = round(sum(f1_vals) / len(f1_vals), 3) if f1_vals else 0.0

    # ------------------------------------------------------------------
    # 3. Confidence calibration
    # ------------------------------------------------------------------
    correct_r = [r for r in results if r["type_correct"]]
    incorrect_r = [r for r in results if not r["type_correct"]]

    mean_conf_correct = (
        sum(r["type_confidence"] for r in correct_r) / len(correct_r)
        if correct_r else 0.0
    )
    mean_conf_incorrect = (
        sum(r["type_confidence"] for r in incorrect_r) / len(incorrect_r)
        if incorrect_r else 0.0
    )
    pct_high_conf_correct = (
        sum(1 for r in correct_r if r["type_confidence"] >= 0.8) / len(correct_r)
        if correct_r else 0.0
    )

    # ------------------------------------------------------------------
    # 4. Manual review flag
    # ------------------------------------------------------------------
    flagged = [r for r in results if r["requires_manual_review"]]
    useful_flags = sum(
        1 for r in flagged
        if any(not a["match"] for a in r["attributes"].values())
    )

    # ------------------------------------------------------------------
    # 5. Performance
    # ------------------------------------------------------------------
    times = sorted(r["processing_time_sec"] for r in results if r["processing_time_sec"] > 0)
    mean_time = sum(times) / len(times) if times else 0.0
    median_time = times[len(times) // 2] if times else 0.0
    p95_time = times[int(len(times) * 0.95)] if times else 0.0
    within_30s = sum(1 for t in times if t <= 30.0) / len(times) if times else 0.0

    return {
        "type_classification": {
            "overall_accuracy": round(overall_accuracy, 4),
            "per_type_accuracy": {k: round(v, 4) for k, v in per_type_accuracy.items()},
            "confusion_matrix": confusion,
        },
        "attribute_extraction": {
            "per_type": attr_metrics,
            "avg_f1_per_type": avg_f1_per_type,
        },
        "confidence_calibration": {
            "mean_confidence_correct": round(mean_conf_correct, 4),
            "mean_confidence_incorrect": round(mean_conf_incorrect, 4),
            "pct_correct_with_high_confidence": round(pct_high_conf_correct, 4),
        },
        "manual_review": {
            "review_rate": round(len(flagged) / total, 4),
            "useful_flag_rate": round(useful_flags / len(flagged), 4) if flagged else 0.0,
            "total_flagged": len(flagged),
            "useful_flags": useful_flags,
        },
        "performance": {
            "mean_sec": round(mean_time, 2),
            "median_sec": round(median_time, 2),
            "p95_sec": round(p95_time, 2),
            "pct_within_30s": round(within_30s, 4),
            "total_docs": total,
        },
    }


def print_metrics(metrics: dict) -> None:
    tc = metrics["type_classification"]
    ae = metrics["attribute_extraction"]
    cc = metrics["confidence_calibration"]
    mr = metrics["manual_review"]
    perf = metrics["performance"]

    sep = "=" * 62

    print(f"\n{sep}")
    print("TYPE CLASSIFICATION")
    print(sep)
    print(f"  Overall accuracy: {tc['overall_accuracy']:.1%}")
    for dt, acc in tc["per_type_accuracy"].items():
        bar = "#" * int(acc * 20)
        print(f"    {dt:12s}  {acc:.1%}  {bar}")

    print("\nConfusion matrix  (rows=expected, cols=predicted)")
    header = f"{'':11s}" + "".join(f"{t:9s}" for t in DOC_TYPES) + "  UNKNOWN"
    print("  " + header)
    for exp in DOC_TYPES:
        row = f"  {exp:10s}"
        for pred in DOC_TYPES:
            row += f"{tc['confusion_matrix'][exp].get(pred, 0):9d}"
        row += f"  {tc['confusion_matrix'][exp].get('UNKNOWN', 0):7d}"
        print(row)

    print(f"\n{sep}")
    print("ATTRIBUTE EXTRACTION")
    print(sep)
    for dt in DOC_TYPES:
        f1_avg = ae["avg_f1_per_type"].get(dt, 0.0)
        print(f"  {dt}  avg F1={f1_avg:.3f}")
        for attr, m in ae["per_type"].get(dt, {}).items():
            flags = ""
            if m["recall"] < 0.80:
                flags += "  ← low recall"
            if m["precision"] < 0.80:
                flags += "  ← low precision"
            print(
                f"    {attr:32s}  P={m['precision']:.2f}  "
                f"R={m['recall']:.2f}  F1={m['f1']:.2f}{flags}"
            )

    print(f"\n{sep}")
    print("CONFIDENCE CALIBRATION")
    print(sep)
    print(f"  Mean confidence when correct:   {cc['mean_confidence_correct']:.3f}")
    print(f"  Mean confidence when incorrect: {cc['mean_confidence_incorrect']:.3f}")
    print(f"  % correct with confidence ≥ 0.8: {cc['pct_correct_with_high_confidence']:.1%}")

    print(f"\n{sep}")
    print("MANUAL REVIEW FLAG")
    print(sep)
    print(f"  Flagged for review: {mr['total_flagged']} docs  ({mr['review_rate']:.1%})")
    print(f"  Of those, had real extraction errors: {mr['useful_flags']}  ({mr['useful_flag_rate']:.1%})")

    print(f"\n{sep}")
    print("PERFORMANCE")
    print(sep)
    print(f"  Mean   : {perf['mean_sec']:.1f}s")
    print(f"  Median : {perf['median_sec']:.1f}s")
    print(f"  P95    : {perf['p95_sec']:.1f}s")
    print(f"  Within 30s (NFR-04): {perf['pct_within_30s']:.1%}")


# ===========================================================================
# Threshold calibration
# ===========================================================================

def suggest_calibration(metrics: dict, apply: bool = False) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print("THRESHOLD CALIBRATION SUGGESTIONS")
    print(sep)

    with THRESHOLDS_PATH.open(encoding="utf-8") as fh:
        thresholds = yaml.safe_load(fh)

    doc_types_section: dict = thresholds.get("document_types", {})
    default_threshold: float = thresholds.get("default", {}).get("review_threshold", 0.7)

    # (doc_type, attr, current, suggested)
    suggestions: list[tuple[str, str, float, float]] = []

    for dt, attr_map in metrics["attribute_extraction"]["per_type"].items():
        dt_attrs: dict = doc_types_section.get(dt, {}).get("attributes", {})

        for attr, m in attr_map.items():
            current = float(dt_attrs.get(attr, default_threshold))

            if m["recall"] < 0.80 and m["extracted"] > 0:
                suggested = round(max(current - 0.05, 0.50), 2)
                if suggested < current:
                    print(
                        f"  SUGGEST: lower {dt}.{attr}  "
                        f"{current} → {suggested}  (recall={m['recall']:.2f})"
                    )
                    suggestions.append((dt, attr, current, suggested))

            elif m["precision"] < 0.80:
                suggested = round(min(current + 0.05, 0.95), 2)
                if suggested > current:
                    print(
                        f"  SUGGEST: raise {dt}.{attr}  "
                        f"{current} → {suggested}  (precision={m['precision']:.2f})"
                    )
                    suggestions.append((dt, attr, current, suggested))

    if not suggestions:
        print("  All thresholds look good — no adjustments suggested.")
        return

    if not apply:
        print(f"\n  Pass --apply-calibration to write {len(suggestions)} changes to thresholds.yaml")
        return

    for dt, attr, _, suggested in suggestions:
        if dt not in doc_types_section:
            doc_types_section[dt] = {"attributes": {}}
        if "attributes" not in doc_types_section[dt]:
            doc_types_section[dt]["attributes"] = {}
        doc_types_section[dt]["attributes"][attr] = suggested

    thresholds["document_types"] = doc_types_section
    with THRESHOLDS_PATH.open("w", encoding="utf-8") as fh:
        yaml.dump(thresholds, fh, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"\n  Applied {len(suggestions)} adjustment(s) → {THRESHOLDS_PATH}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="ECM AI experiment runner")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Process only the first N documents (quick test)")
    parser.add_argument("--apply-calibration", action="store_true",
                        help="Auto-apply threshold suggestions to thresholds.yaml")
    parser.add_argument("--no-servers", action="store_true",
                        help="Skip server startup checks (assume already running)")
    parser.add_argument("--raw", action="store_true",
                        help="Track A: fitz text extraction on raw PDFs (OCR bypassed)")
    parser.add_argument("--ocr-spot-check", action="store_true",
                        help="Track B: OCR quality report on real documents")
    parser.add_argument("--real", action="store_true",
                        help="Evaluate on real documents from experiment/real_ground_truth.json")
    parser.add_argument("--text-only", action="store_true",
                        help="(with --real) Skip scanned documents, process only text-based PDFs")
    args = parser.parse_args()

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Track B: standalone OCR spot-check
    # ------------------------------------------------------------------
    if args.ocr_spot_check:
        run_ocr_spot_check()
        return

    # ------------------------------------------------------------------
    # Real-document track (--real)
    # ------------------------------------------------------------------
    if args.real:
        if not REAL_GROUND_TRUTH_PATH.exists():
            sys.exit(f"ERROR: {REAL_GROUND_TRUTH_PATH} not found — run build_real_ground_truth.py first")

        real_gt: list[dict] = json.loads(REAL_GROUND_TRUTH_PATH.read_text())
        if args.text_only:
            real_gt = [r for r in real_gt if not r.get("is_scan", False)]
            track_label = "real text-based PDFs (fast-path)"
        else:
            track_label = "real documents (text + scan)"
        if args.limit:
            real_gt = real_gt[: args.limit]

        print(f"Loaded {len(real_gt)} real documents  [{track_label}]")

        if not args.no_servers:
            ensure_servers()

        results_path = EXPERIMENT_DIR / "results_real.json"
        metrics_path = EXPERIMENT_DIR / "metrics_real.json"

        results: list[dict] = []
        with httpx.Client() as client:
            for entry in tqdm(real_gt, desc="Processing", unit="doc", leave=True):
                result = process_real_document(client, entry)
                results.append(result)
                time.sleep(1)

                matched = sum(1 for a in result["attributes"].values() if a["match"])
                total_attrs = len(result["attributes"])
                indicator = "✓" if result["type_correct"] else "✗"
                predicted = result["predicted_type"] or "ERROR"
                scan_tag = "[scan]" if entry.get("is_scan") else "[text]"
                tqdm.write(
                    f"  {scan_tag} {result['filename'][:40]} → {predicted} {indicator} "
                    f"({result['type_confidence']:.2f}) | "
                    f"{matched}/{total_attrs} attrs | "
                    f"{result['processing_time_sec']:.1f}s"
                )

        results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"\nResults saved → {results_path}")

        metrics = compute_metrics(results)
        print_metrics(metrics)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"\nMetrics saved → {metrics_path}")
        suggest_calibration(metrics, apply=args.apply_calibration)
        print(f"\nExperiment complete. Results: {results_path} | Metrics: {metrics_path}")
        return

    # ------------------------------------------------------------------
    # Load ground truth
    # ------------------------------------------------------------------
    ground_truth: list[dict] = json.loads(GROUND_TRUTH_PATH.read_text())
    if args.limit:
        ground_truth = ground_truth[: args.limit]

    track_label = "Track A — raw PDFs / fast-path (OCR bypassed)" if args.raw else "scanned PDFs / full OCR pipeline"
    print(f"Loaded {len(ground_truth)} documents from ground_truth.json  [{track_label}]")

    if not args.no_servers:
        ensure_servers()

    # ------------------------------------------------------------------
    # Output paths — separate files per track so runs don't overwrite each other
    # ------------------------------------------------------------------
    if args.raw:
        results_path = EXPERIMENT_DIR / "results_raw.json"
        metrics_path = EXPERIMENT_DIR / "metrics_raw.json"
    else:
        results_path = EXPERIMENT_DIR / "results.json"
        metrics_path = EXPERIMENT_DIR / "metrics.json"

    # ------------------------------------------------------------------
    # Processing loop
    # ------------------------------------------------------------------
    results: list[dict] = []
    arrow = "→ (fast-path)" if args.raw else "→"

    with httpx.Client() as client:
        for entry in tqdm(ground_truth, desc="Processing", unit="doc", leave=True):
            if args.raw:
                result = process_document_raw(
                    client,
                    entry["filename"],
                    entry["document_type"],
                    entry["attributes"],
                )
            else:
                result = process_document(
                    client,
                    entry["filename"],
                    entry["document_type"],
                    entry["attributes"],
                )
            results.append(result)
            time.sleep(3)

            matched = sum(1 for a in result["attributes"].values() if a["match"])
            total_attrs = len(result["attributes"])
            indicator = "✓" if result["type_correct"] else "✗"
            predicted = result["predicted_type"] or "ERROR"
            tqdm.write(
                f"  {result['filename']} {arrow} {predicted} {indicator} "
                f"({result['type_confidence']:.2f}) | "
                f"{matched}/{total_attrs} attrs | "
                f"{result['processing_time_sec']:.1f}s"
            )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nResults saved → {results_path}")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    metrics = compute_metrics(results)
    print_metrics(metrics)

    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nMetrics saved → {metrics_path}")

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    suggest_calibration(metrics, apply=args.apply_calibration)

    print(
        f"\nExperiment complete. "
        f"Results: {results_path} | Metrics: {metrics_path}"
    )


if __name__ == "__main__":
    main()
