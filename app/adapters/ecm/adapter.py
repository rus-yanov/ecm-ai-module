from app.domain.models import ProcessingResult


class EcmAdapter:
    async def get_schema(self, schema_id: str) -> dict:
        raise NotImplementedError

    async def get_dictionary(self, name: str) -> list[str]:
        raise NotImplementedError

    async def post_card(self, result: ProcessingResult) -> str:
        raise NotImplementedError
