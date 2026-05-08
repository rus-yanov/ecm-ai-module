"""Minimal mock of the ECM REST API used for local development and smoke tests."""
import uuid

from fastapi import FastAPI, Request

app = FastAPI(title="Mock ECM Server", version="0.1.0")

# ---------------------------------------------------------------------------
# Schema catalogue
# ---------------------------------------------------------------------------

_SCHEMAS: dict[str, dict] = {
    "contract": {
        "type": "CONTRACT",
        "attributes": [
            "contract_number", "contract_date", "contractor", "contractor_inn",
            "subject", "amount", "expiry_date",
        ],
    },
    "invoice": {
        "type": "INVOICE",
        "attributes": [
            "invoice_number", "invoice_date", "supplier", "supplier_inn",
            "buyer", "buyer_inn", "amount_with_vat", "vat_rate",
        ],
    },
    "act": {
        "type": "ACT",
        "attributes": [
            "act_number", "act_date", "customer", "executor", "subject", "amount",
        ],
    },
    "waybill": {
        "type": "WAYBILL",
        "attributes": [
            "document_number", "document_date", "sender_department",
            "receiver_department", "operation_type_code", "items_description",
        ],
    },
    "order": {
        "type": "ORDER",
        "attributes": [
            "order_number", "order_date", "title", "signatory", "deadline",
        ],
    },
}

# ---------------------------------------------------------------------------
# Dictionary catalogue
# ---------------------------------------------------------------------------

_DICTIONARIES: dict[str, list[str]] = {
    "contractors": [
        "ООО «Альфа Технологии»",
        "ЗАО «Бета Групп»",
        "АО «Горизонт»",
        "ООО «ТехноСтрой»",
        "АО «Медиагрупп»",
        "ООО «Ромашка»",
        "ООО «Технопром»",
        "ЗАО «Ремсервис»",
        "АО «Промснаб»",
        "ООО «ЛогистПлюс»",
        "ООО «СтройМонтаж»",
        "АО «ИнфраГрупп»",
        "ООО «ТехноПроект»",
        "ЗАО «АвтоМатика»",
        "АО «СервисГрупп»",
        "ООО «ИнфоСистемс»",
        "АО «МегаТех»",
        "ООО «ПромСервис»",
        "ЗАО «КонсалтПлюс»",
        "АО «ДатаСистемс»",
    ],
    "document_types": ["CONTRACT", "INVOICE", "ACT", "WAYBILL", "ORDER"],
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/schemas/{schema_id}")
async def get_schema(schema_id: str) -> dict:
    return _SCHEMAS.get(schema_id.lower(), {"type": "UNKNOWN", "attributes": []})


@app.get("/dictionaries/{name}")
async def get_dictionary(name: str) -> list:
    return _DICTIONARIES.get(name, [])


@app.post("/cards")
async def create_card(request: Request) -> dict:
    await request.body()  # consume body
    card_id = f"CARD-{str(uuid.uuid4())[:8].upper()}"
    return {"card_id": card_id, "status": "created"}


@app.patch("/cards/{card_id}")
async def update_card(card_id: str, request: Request) -> dict:
    await request.body()
    return {"card_id": card_id, "status": "updated"}


@app.post("/routes")
async def route_document(request: Request) -> dict:
    await request.body()
    return {"status": "routed"}
