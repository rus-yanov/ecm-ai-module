import re
from pathlib import Path

import yaml
from rapidfuzz import fuzz, process

from app.config.settings import settings
from app.domain.models import DocumentType, ExtractedAttribute

_NAME_FIELDS = {
    "contractor", "supplier", "buyer", "customer",
    "executor", "sender", "addressee", "signatory",
}


class ConfidenceRouter:
    def __init__(self, dictionary: list[str] | None = None) -> None:
        self._dictionary: list[str] = dictionary or []

        thresholds_path = Path(settings.thresholds_config_path)
        with thresholds_path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        # Flatten into the shape used by route():
        # { "CONTRACT": {"attributes": {...}}, ..., "default": {"review_threshold": ...} }
        self._thresholds: dict = {
            **raw.get("document_types", {}),
            "default": raw.get("default", {}),
        }

    # ------------------------------------------------------------------
    # Public thresholds accessor (used by pipeline)
    # ------------------------------------------------------------------

    @property
    def thresholds(self) -> dict:
        return self._thresholds

    # ------------------------------------------------------------------
    # Weighted confidence formula
    # ------------------------------------------------------------------

    def compute_final_confidence(
        self,
        llm_conf: float,
        ocr_conf: float,
        dict_match: float,
        weights: tuple[float, float, float] = (0.6, 0.2, 0.2),
    ) -> float:
        """final_conf = w1*llm_conf + w2*ocr_conf + w3*dict_match, clipped to [0.0, 1.0]"""
        w1, w2, w3 = weights
        return max(0.0, min(1.0, w1 * llm_conf + w2 * ocr_conf + w3 * dict_match))

    # ------------------------------------------------------------------
    # Value normalisation
    # ------------------------------------------------------------------

    def normalize_value(self, name: str, raw: str | None) -> tuple[str | None, float]:
        """Returns (normalized_value, dict_match_score)."""
        if raw is None:
            return None, 0.0

        lower = name.lower()

        # Dates
        if "date" in lower or "deadline" in lower:
            return self._normalize_date(raw)

        # Amounts / sums
        if "amount" in lower or "sum" in lower:
            return self._normalize_amount(raw)

        # INN (taxpayer ID)
        if "inn" in lower:
            return self._normalize_inn(raw)

        # Entity names — fuzzy dict match
        for field in _NAME_FIELDS:
            if field in lower:
                return self._fuzzy_match(raw)

        return raw, 1.0

    _ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    @staticmethod
    def _normalize_date(raw: str) -> tuple[str | None, float]:
        from dateutil import parser as dp

        stripped = raw.strip()
        # Already ISO 8601 — return as-is to avoid dayfirst re-interpretation
        if ConfidenceRouter._ISO_DATE_RE.match(stripped):
            return stripped, 1.0
        try:
            dt = dp.parse(stripped, dayfirst=True)
            return dt.strftime("%Y-%m-%d"), 1.0
        except (ValueError, OverflowError):
            return raw, 0.5

    @staticmethod
    def _normalize_amount(raw: str) -> tuple[str | None, float]:
        # Keep digits, dots and commas; then replace comma → dot
        cleaned = re.sub(r"[^\d.,]", "", raw.replace(" ", ""))
        cleaned = cleaned.replace(",", ".")
        # If multiple dots, only keep the last as decimal separator
        parts = cleaned.split(".")
        if len(parts) > 2:
            cleaned = "".join(parts[:-1]) + "." + parts[-1]
        try:
            return str(float(cleaned)), 1.0
        except ValueError:
            return raw, 0.5

    @staticmethod
    def _normalize_inn(raw: str) -> tuple[str | None, float]:
        digits = re.sub(r"\D", "", raw)
        if len(digits) in (10, 12):
            return digits, 1.0
        return raw, 0.3

    def _fuzzy_match(self, raw: str) -> tuple[str | None, float]:
        if not self._dictionary:
            return raw, 1.0
        match = process.extractOne(raw, self._dictionary, scorer=fuzz.token_sort_ratio)
        if match is None:
            return raw, 0.4
        matched_name, score, _ = match
        if score >= 80:
            return matched_name, score / 100.0
        return raw, 0.4

    # ------------------------------------------------------------------
    # Route: normalize + compute confidence + flag for verification
    # ------------------------------------------------------------------

    def route(
        self,
        attributes: list[ExtractedAttribute],
        doc_type: DocumentType,
        ocr_avg_confidence: float,
        thresholds: dict,
    ) -> list[ExtractedAttribute]:
        doc_thresholds = thresholds.get(doc_type.value, {}).get("attributes", {})
        default_threshold = thresholds.get("default", {}).get("review_threshold", 0.7)

        result: list[ExtractedAttribute] = []
        for attr in attributes:
            normalized_value, dict_match = self.normalize_value(attr.name, attr.raw_value)
            final_conf = self.compute_final_confidence(
                attr.confidence, ocr_avg_confidence, dict_match
            )
            threshold = doc_thresholds.get(attr.name, default_threshold)
            result.append(
                ExtractedAttribute(
                    name=attr.name,
                    raw_value=attr.raw_value,
                    normalized_value=normalized_value,
                    confidence=final_conf,
                    requires_verification=final_conf < threshold,
                )
            )
        return result
