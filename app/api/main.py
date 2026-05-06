import logging
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
    schema_id: Annotated[str, Form()] = "contract",
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

        # --- 2. OCR (30 s hard timeout) ---
        try:
            ocr_result: OcrResult = await asyncio.wait_for(
                OcrService().recognize(file_bytes, filename, document_id=document_id),
                timeout=30.0,
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

        # --- 4. ECM: schema + contractor dictionary ---
        ecm = EcmAdapter()
        schema, contractors = await _gather_ecm(ecm, schema_id)

        # --- 5. LLM classification + extraction (60 s hard timeout) ---
        full_text = "\n".join(b.text for b in ocr_result.blocks) if ocr_result.blocks else filename
        try:
            doc_type, type_conf, raw_attrs = await asyncio.wait_for(
                LlmService().classify_and_extract(full_text, schema, document_id=document_id),
                timeout=60.0,
            )
        except asyncio.TimeoutError:
            log.warning("llm_timeout")
            raise HTTPException(status_code=504, detail="llm_timeout")

        # --- 6. Confidence routing + normalisation ---
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
