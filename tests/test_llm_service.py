import pytest

from app.domain.models import DocumentType
from app.services.llm.service import LlmService

_CONTRACT_TEXT = """\
ДОГОВОР №77-Б от 10 февраля 2024 года
ООО «ТехноСтрой», ИНН 7712345678, именуемое «Исполнитель»,
и АО «ИнфраГрупп», ИНН 7798765432, именуемое «Заказчик»,
заключили договор на выполнение строительно-монтажных работ
на объекте по адресу г. Москва, ул. Ленина, д. 5.
Общая стоимость работ: 3 500 000 рублей. Срок исполнения: 30 июня 2024 года.
"""


@pytest.mark.integration
@pytest.mark.asyncio
async def test_classify_contract() -> None:
    service = LlmService()
    doc_type, type_confidence, attributes = await service.classify_and_extract(
        _CONTRACT_TEXT, {}
    )

    assert doc_type == DocumentType.CONTRACT, f"Expected CONTRACT, got {doc_type}"
    assert type_confidence > 0.7, f"type_confidence too low: {type_confidence}"
    assert len(attributes) > 0, "No attributes extracted"
