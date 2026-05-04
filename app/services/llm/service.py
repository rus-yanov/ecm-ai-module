from app.domain.models import DocumentType, ExtractedAttribute


class LlmService:
    async def classify_and_extract(
        self, text: str, schema: dict
    ) -> tuple[DocumentType, float, list[ExtractedAttribute]]:
        raise NotImplementedError
