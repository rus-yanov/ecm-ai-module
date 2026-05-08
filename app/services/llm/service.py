import json
import time

import httpx
import structlog

from app.config.settings import settings
from app.domain.models import DocumentType, ExtractedAttribute

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Few-shot examples embedded in the prompt
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLES = """\
--- EXAMPLES ---

Example 1 (PAYMENT, full):
Input: "Платёжное поручение №563 от 15.04.2024 Плательщик: ООО Альфа ИНН 7701234567 Получатель: ООО Бета ИНН 7709876543 Сумма: 125000.00 Назначение платежа: Оплата по договору №12 от 01.03.2024 за услуги"
Output: {"document_type":"PAYMENT","type_confidence":0.95,"attributes":[{"name":"document_number","value":"563","confidence":0.99},{"name":"document_date","value":"2024-04-15","confidence":0.99},{"name":"payer_name","value":"ООО Альфа","confidence":0.97},{"name":"payer_inn","value":"7701234567","confidence":0.99},{"name":"receiver_name","value":"ООО Бета","confidence":0.97},{"name":"receiver_inn","value":"7709876543","confidence":0.99},{"name":"amount","value":"125000.00","confidence":0.99},{"name":"payment_purpose","value":"Оплата по договору №12 от 01.03.2024 за услуги","confidence":0.95}]}

Example 2 (PAYMENT, partial — missing receiver_inn):
Input: "Платёжное поручение №87 от 03.01.2024\\nПлательщик: ИП Иванов А.А., ИНН 500100200300\\nПолучатель: ФНС России\\nСумма: 15 000 руб.\\nНазначение: Налог на доходы физических лиц"
Output: {"document_type":"PAYMENT","type_confidence":0.93,"attributes":[{"name":"document_number","value":"87","confidence":0.99},{"name":"document_date","value":"2024-01-03","confidence":0.98},{"name":"payer_name","value":"ИП Иванов А.А.","confidence":0.97},{"name":"payer_inn","value":"500100200300","confidence":0.99},{"name":"receiver_name","value":"ФНС России","confidence":0.97},{"name":"receiver_inn","value":null,"confidence":0.0},{"name":"amount","value":"15000.00","confidence":0.96},{"name":"payment_purpose","value":"Налог на доходы физических лиц","confidence":0.95}]}

Example 3 (INVOICE, full):
Input: "СЧЁТ-ФАКТУРА №СФ-2024-0891 от 20.04.2024\\nПоставщик: ООО «Технопром», ИНН 6612345678\\nПокупатель: АО «Горизонт», ИНН 7723456789\\nСумма с НДС: 236 000,00 руб. НДС 20%: 39 333,33 руб."
Output: {"document_type":"INVOICE","type_confidence":0.99,"attributes":[{"name":"invoice_number","value":"СФ-2024-0891","confidence":0.99},{"name":"invoice_date","value":"2024-04-20","confidence":0.99},{"name":"seller_name","value":"ООО «Технопром»","confidence":0.99},{"name":"seller_inn","value":"6612345678","confidence":0.99},{"name":"buyer","value":"АО «Горизонт»","confidence":0.99},{"name":"buyer_inn","value":"7723456789","confidence":0.99},{"name":"amount_with_vat","value":"236000.00","confidence":0.99},{"name":"vat_amount","value":"39333.33","confidence":0.99}]}

Example 4 (INVOICE, partial):
Input: "Счёт-фактура №100 от 5 мая 2024\\nОт: ИП Сидоров, ИНН 771100223344\\nКому: ООО «Свет»\\nИтого: 45 000 руб. без НДС"
Output: {"document_type":"INVOICE","type_confidence":0.96,"attributes":[{"name":"invoice_number","value":"100","confidence":0.99},{"name":"invoice_date","value":"2024-05-05","confidence":0.98},{"name":"seller_name","value":"ИП Сидоров","confidence":0.97},{"name":"seller_inn","value":"771100223344","confidence":0.99},{"name":"buyer","value":"ООО «Свет»","confidence":0.97},{"name":"buyer_inn","value":null,"confidence":0.0},{"name":"amount_with_vat","value":"45000.00","confidence":0.93},{"name":"vat_amount","value":null,"confidence":0.0}]}

Example 5 (ACT):
Input: "АКТ ВЫПОЛНЕННЫХ РАБОТ №АКТ-2024-033 от 30.06.2024\\nЗаказчик: ООО «Строймонтаж», Исполнитель: ЗАО «Ремсервис»\\nПредмет: монтаж вентиляционного оборудования\\nОбщая стоимость работ: 1 200 000 рублей"
Output: {"document_type":"ACT","type_confidence":0.99,"attributes":[{"name":"act_number","value":"АКТ-2024-033","confidence":0.99},{"name":"act_date","value":"2024-06-30","confidence":0.99},{"name":"customer","value":"ООО «Строймонтаж»","confidence":0.99},{"name":"executor","value":"ЗАО «Ремсервис»","confidence":0.99},{"name":"subject","value":"монтаж вентиляционного оборудования","confidence":0.96},{"name":"amount","value":"1200000","confidence":0.99}]}

Example 6 (ACT, partial):
Input: "Акт №55 от 10 января 2024 г.\\nЗаказчик: АО «Медиагрупп»\\nВыполнены работы по техническому обслуживанию серверного оборудования."
Output: {"document_type":"ACT","type_confidence":0.94,"attributes":[{"name":"act_number","value":"55","confidence":0.99},{"name":"act_date","value":"2024-01-10","confidence":0.98},{"name":"customer","value":"АО «Медиагрупп»","confidence":0.98},{"name":"executor","value":null,"confidence":0.0},{"name":"subject","value":"техническое обслуживание серверного оборудования","confidence":0.91},{"name":"amount","value":null,"confidence":0.0}]}

Example 7 (WAYBILL, full):
Input: "Форма М-11 Требование-накладная № 71 от 28.03.2025 Отправитель: ск.776 Получатель: 099-2кл Код операции: 2042307 Наименование: Кабель ВВГ 3х2.5, 50м"
Output: {"document_type":"WAYBILL","type_confidence":0.93,"attributes":[{"name":"document_number","value":"71","confidence":0.99},{"name":"document_date","value":"2025-03-28","confidence":0.97},{"name":"sender_department","value":"ск.776","confidence":0.94},{"name":"receiver_department","value":"099-2кл","confidence":0.94},{"name":"operation_type_code","value":"2042307","confidence":0.99},{"name":"items_description","value":"Кабель ВВГ 3х2.5, 50м","confidence":0.97}]}

Example 8 (WAYBILL, partial — no operation code):
Input: "Требование-накладная № 15 от 01.04.2025\\nОтправитель: цех 101\\nПолучатель: склад 3\\nПозиции: Краска алкидная ПФ-115, 10 л; Кисть малярная, 5 шт"
Output: {"document_type":"WAYBILL","type_confidence":0.91,"attributes":[{"name":"document_number","value":"15","confidence":0.99},{"name":"document_date","value":"2025-04-01","confidence":0.98},{"name":"sender_department","value":"цех 101","confidence":0.96},{"name":"receiver_department","value":"склад 3","confidence":0.96},{"name":"operation_type_code","value":null,"confidence":0.0},{"name":"items_description","value":"Краска алкидная ПФ-115, 10 л; Кисть малярная, 5 шт","confidence":0.94}]}

Example 9 (ORDER):
Input: "ПРИКАЗ №П-2024-156 от 01.07.2024\\nО введении в действие регламента информационной безопасности\\nПодписант: Генеральный директор Смирнов А.В.\\nСрок исполнения: 01.08.2024"
Output: {"document_type":"ORDER","type_confidence":0.99,"attributes":[{"name":"order_number","value":"П-2024-156","confidence":0.99},{"name":"order_date","value":"2024-07-01","confidence":0.99},{"name":"title","value":"О введении в действие регламента информационной безопасности","confidence":0.97},{"name":"signatory","value":"Смирнов А.В., Генеральный директор","confidence":0.98},{"name":"deadline","value":"2024-08-01","confidence":0.99}]}

Example 10 (ORDER, no deadline):
Input: "Приказ №88 от 15.02.2024\\nО назначении ответственного за охрану труда\\nПодписал: Директор Козлов П.Н."
Output: {"document_type":"ORDER","type_confidence":0.97,"attributes":[{"name":"order_number","value":"88","confidence":0.99},{"name":"order_date","value":"2024-02-15","confidence":0.99},{"name":"title","value":"О назначении ответственного за охрану труда","confidence":0.96},{"name":"signatory","value":"Козлов П.Н., Директор","confidence":0.97},{"name":"deadline","value":null,"confidence":0.0}]}

--- END EXAMPLES ---"""

_FALLBACK = (DocumentType.UNKNOWN, 0.0, [])


class LlmService:
    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, text: str, schema: dict) -> str:  # noqa: ARG002
        block1 = (
            "You are a document analysis system for a Russian ECM platform.\n"
            "Extract structured information from the document text provided.\n"
            "Respond ONLY with valid JSON. Do not add any text outside JSON.\n"
            "Rules:\n"
            "- Set value to null for any attribute not found in the document\n"
            "- Set confidence to 0.0 if you are unsure about a value\n"
            "- Never invent or guess values not present in the document text\n"
            "- confidence is a float from 0.0 to 1.0"
        )

        block2 = (
            'Expected JSON response schema:\n'
            '{\n'
            '  "document_type": "one of: PAYMENT, INVOICE, ACT, WAYBILL, ORDER, UNKNOWN",\n'
            '  "type_confidence": <float 0.0-1.0>,\n'
            '  "attributes": [\n'
            '    {"name": "<attribute name>", "value": "<extracted value or null>", '
            '"confidence": <float 0.0-1.0>}\n'
            '  ]\n'
            '}'
        )

        block4 = f"Now analyze this document and respond with JSON only:\n\n{text}"

        return "\n\n".join([block1, block2, _FEW_SHOT_EXAMPLES, block4])

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw_json: str) -> tuple[DocumentType, float, list[ExtractedAttribute]]:
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            return _FALLBACK

        try:
            doc_type = DocumentType(data["document_type"])
        except (KeyError, ValueError):
            return _FALLBACK

        type_confidence = float(data.get("type_confidence", 0.0))

        attributes: list[ExtractedAttribute] = []
        for item in data.get("attributes", []):
            try:
                attributes.append(
                    ExtractedAttribute(
                        name=item["name"],
                        raw_value=item.get("value"),
                        normalized_value=item.get("value"),
                        confidence=float(item.get("confidence", 0.0)),
                        requires_verification=False,
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue

        return doc_type, type_confidence, attributes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def classify_and_extract(
        self, text: str, schema: dict, document_id: str = ""
    ) -> tuple[DocumentType, float, list[ExtractedAttribute]]:
        log = logger.bind(document_id=document_id) if document_id else logger
        prompt = self._build_prompt(text, schema)
        start = time.monotonic()

        try:
            async with httpx.AsyncClient(
                base_url=settings.ollama_base_url,
                timeout=settings.ollama_timeout_sec,
            ) as client:
                response = await client.post(
                    "/api/chat",
                    json={
                        "model": settings.ollama_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "format": "json",
                        "stream": False,
                        "options": {"temperature": 0.1, "top_p": 0.9},
                    },
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            log.error("Ollama request failed", error=str(exc))
            return _FALLBACK

        elapsed = time.monotonic() - start

        try:
            raw_json: str = response.json()["message"]["content"]
        except (KeyError, ValueError) as exc:
            log.error("Unexpected Ollama response shape", error=str(exc))
            return _FALLBACK

        doc_type, type_confidence, attributes = self._parse_response(raw_json)

        log.info(
            "LLM classify_and_extract",
            model=settings.ollama_model,
            text_length=len(text),
            document_type=doc_type.value,
            type_confidence=round(type_confidence, 4),
            num_attributes=len(attributes),
            elapsed_sec=round(elapsed, 3),
        )

        return doc_type, type_confidence, attributes
