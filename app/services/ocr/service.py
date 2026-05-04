from app.domain.models import OcrResult


class OcrService:
    async def recognize(self, file_bytes: bytes, filename: str) -> OcrResult:
        raise NotImplementedError
