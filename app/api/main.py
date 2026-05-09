import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, AsyncGenerator

import structlog
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.config.settings import settings

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logging.basicConfig(level=settings.log_level)
logger = structlog.get_logger(__name__)


def _check_ocr_coherence(text: str, min_cyrillic_ratio: float = 0.35) -> bool:
    """Returns False when OCR output is garbled (too few real Cyrillic words)."""
    words = text.split()
    if not words:
        return False
    cyrillic_words = re.findall(r'[а-яёА-ЯЁ]{2,}', text)
    return len(cyrillic_words) / len(words) >= min_cyrillic_ratio


_ocr_service = None  # singleton, initialised on first scan request


def _get_ocr_service():
    global _ocr_service
    if _ocr_service is None:
        from app.services.ocr.service import OcrService
        _ocr_service = OcrService()
    return _ocr_service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("ECM AI Module starting", model=settings.ollama_model)
    yield


app = FastAPI(title="ECM AI Module", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "version": "0.1.0", "model": settings.ollama_model}


# ---------------------------------------------------------------------------
# Main pipeline endpoint
# ---------------------------------------------------------------------------


@app.post("/api/v1/documents/process")
async def process_document(
    file: Annotated[UploadFile, File()],
    schema_id: Annotated[str, Form()] = "payment",
) -> dict:
    import asyncio

    from app.adapters.ecm.adapter import EcmAdapter
    from app.domain.models import DocumentType, OcrResult, ProcessingResult
    from app.services.llm.service import LlmService
    from app.services.ocr.service import OcrService
    from app.services.router.confidence_router import ConfidenceRouter

    # document_id generated before try so it's available in error paths
    document_id = str(uuid.uuid4())
    log = structlog.get_logger().bind(document_id=document_id)
    start = time.monotonic()

    try:
        log.info("processing_started", filename=file.filename, schema_id=schema_id)

        # --- 1. Read & validate file size ---
        file_bytes = await file.read()
        max_bytes = settings.max_file_size_mb * 1024 * 1024
        if len(file_bytes) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds {settings.max_file_size_mb} MB limit",
            )

        filename = file.filename or "document"

        # --- 2. OCR (120 s hard timeout; model loads on first call ~15 s + OCR ~20 s/page) ---
        try:
            ocr_result: OcrResult = await asyncio.wait_for(
                _get_ocr_service().recognize(file_bytes, filename, document_id=document_id),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            log.warning("ocr_timeout")
            raise HTTPException(status_code=504, detail="ocr_timeout")

        # --- 3. Low-confidence OCR → skip LLM, flag for manual review ---
        if ocr_result.avg_confidence < settings.ocr_confidence_threshold and ocr_result.blocks:
            total = time.monotonic() - start
            result = ProcessingResult(
                document_id=document_id,
                document_type=DocumentType.UNKNOWN,
                type_confidence=0.0,
                attributes=[],
                ocr_result=ocr_result,
                requires_manual_review=True,
                total_processing_time_sec=total,
                error="low_ocr_confidence",
            )
            log.info(
                "processing_done",
                document_type="UNKNOWN",
                requires_manual_review=True,
                total_sec=round(total, 3),
                reason="low_ocr_confidence",
            )
            return result.model_dump(mode="json")

        # --- 4. Coherence check: garbled OCR (conf > threshold but text is noise) ---
        full_text = "\n".join(b.text for b in ocr_result.blocks) if ocr_result.blocks else filename
        if ocr_result.blocks and not _check_ocr_coherence(full_text):
            total = time.monotonic() - start
            result = ProcessingResult(
                document_id=document_id,
                document_type=DocumentType.UNKNOWN,
                type_confidence=0.0,
                attributes=[],
                ocr_result=ocr_result,
                requires_manual_review=True,
                total_processing_time_sec=total,
                error="incoherent_ocr_text",
            )
            log.info(
                "processing_done",
                document_type="UNKNOWN",
                requires_manual_review=True,
                total_sec=round(total, 3),
                reason="incoherent_ocr_text",
            )
            return result.model_dump(mode="json")

        # --- 5. ECM: schema + contractor dictionary ---
        ecm = EcmAdapter()
        schema, contractors = await _gather_ecm(ecm, schema_id)

        # --- 7. LLM classification + extraction (hard timeout = ollama_timeout_sec + 30 s) ---
        try:
            doc_type, type_conf, raw_attrs = await asyncio.wait_for(
                LlmService().classify_and_extract(full_text, schema, document_id=document_id),
                timeout=float(settings.ollama_timeout_sec) + 30.0,
            )
        except asyncio.TimeoutError:
            log.warning("llm_timeout")
            raise HTTPException(status_code=504, detail="llm_timeout")

        # --- 8. Confidence routing + normalisation ---
        router = ConfidenceRouter(dictionary=contractors)
        attributes = router.route(
            raw_attrs, doc_type, ocr_result.avg_confidence, router.thresholds
        )

        total = time.monotonic() - start
        result = ProcessingResult(
            document_id=document_id,
            document_type=doc_type,
            type_confidence=type_conf,
            attributes=attributes,
            ocr_result=ocr_result,
            requires_manual_review=any(a.requires_verification for a in attributes),
            total_processing_time_sec=total,
        )

        log.info(
            "processing_done",
            document_type=doc_type.value,
            type_confidence=round(type_conf, 4),
            num_attributes=len(attributes),
            requires_manual_review=result.requires_manual_review,
            total_sec=round(total, 3),
        )

        return result.model_dump(mode="json")

    except HTTPException:
        raise
    except Exception as exc:
        log.error("internal_error", error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail="internal_error")


async def _gather_ecm(ecm, schema_id: str) -> tuple[dict, list[str]]:
    import asyncio

    schema, contractors = await asyncio.gather(
        ecm.get_schema(schema_id),
        ecm.get_dictionary("contractors"),
    )
    return schema, contractors
