from app.domain.models import DocumentType, ExtractedAttribute


class ConfidenceRouter:
    def compute_final_confidence(
        self,
        llm_conf: float,
        ocr_conf: float,
        dict_match: float,
        weights: tuple[float, float, float] = (0.6, 0.2, 0.2),
    ) -> float:
        raise NotImplementedError

    def route(
        self, attributes: list[ExtractedAttribute], doc_type: DocumentType
    ) -> list[ExtractedAttribute]:
        raise NotImplementedError
