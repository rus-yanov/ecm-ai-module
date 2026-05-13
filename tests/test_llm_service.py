import pytest

from app.domain.models import DocumentType
from app.services.llm.service import LlmService

_PAYMENT_TEXT = """\
Платёжное поручение №563 от 15 апреля 2024 года
Плательщик: ООО «Строймаш», ИНН 7701234567
Расчётный счёт плательщика: 40702810100000001234
Получатель: ООО «Сервис», ИНН 7709876543
Сумма: 125 000 рублей 00 копеек
Назначение платежа: Оплата за услуги по договору №12 от 01.03.2024
"""


@pytest.mark.integration
@pytest.mark.asyncio
async def test_classify_payment() -> None:
    service = LlmService()
    doc_type, type_confidence, attributes = await service.classify_and_extract(
        _PAYMENT_TEXT
    )

    assert doc_type == DocumentType.PAYMENT, f"Expected PAYMENT, got {doc_type}"
    assert type_confidence > 0.7, f"type_confidence too low: {type_confidence}"
    assert len(attributes) > 0, "No attributes extracted"
