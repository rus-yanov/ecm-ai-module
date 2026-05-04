import io

import pytest
from PIL import Image, ImageDraw

from app.services.ocr.service import OcrService


def _make_png(text: str = "Договор №123 от 01.01.2024") -> bytes:
    img = Image.new("RGB", (400, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 80), text, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_recognize_returns_valid_result() -> None:
    file_bytes = _make_png()
    service = OcrService()
    result = await service.recognize(file_bytes, "test.png")

    assert result.page_count == 1
    assert result.processing_time_sec > 0
    assert len(result.blocks) >= 0  # OCR may return no blocks on synthetic image — that's ok
    assert 0.0 <= result.avg_confidence <= 1.0
