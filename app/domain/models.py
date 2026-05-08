from enum import Enum
from typing import Any

from pydantic import BaseModel, model_validator


class DocumentType(str, Enum):
    CONTRACT = "CONTRACT"
    INVOICE = "INVOICE"
    ACT = "ACT"
    WAYBILL = "WAYBILL"
    ORDER = "ORDER"
    UNKNOWN = "UNKNOWN"


class OcrBlock(BaseModel):
    text: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2]


class OcrResult(BaseModel):
    blocks: list[OcrBlock]
    avg_confidence: float
    page_count: int
    processing_time_sec: float


class ExtractedAttribute(BaseModel):
    name: str
    raw_value: str | None
    normalized_value: Any | None
    confidence: float
    requires_verification: bool


class ProcessingResult(BaseModel):
    document_id: str
    document_type: DocumentType
    type_confidence: float
    attributes: list[ExtractedAttribute]
    ocr_result: OcrResult
    requires_manual_review: bool
    total_processing_time_sec: float
    error: str | None = None

    @model_validator(mode="before")
    @classmethod
    def compute_requires_manual_review(cls, values: dict) -> dict:
        if "requires_manual_review" not in values or values.get("requires_manual_review") is None:
            attributes = values.get("attributes", [])
            values["requires_manual_review"] = any(
                (a.requires_verification if hasattr(a, "requires_verification") else a.get("requires_verification", False))
                for a in attributes
            )
        return values
