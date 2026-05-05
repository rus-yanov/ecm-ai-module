import asyncio

import httpx
import structlog

from app.config.settings import settings
from app.domain.models import ProcessingResult

logger = structlog.get_logger(__name__)


class EcmAdapter:
    def __init__(self) -> None:
        self._schema_cache: dict[str, dict] = {}
        self._dict_cache: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Internal retry helper
    # ------------------------------------------------------------------

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        delays = settings.ecm_retry_delays
        last_exc: Exception | None = None

        for attempt in range(settings.ecm_retry_count + 1):
            try:
                async with httpx.AsyncClient(
                    base_url=settings.ecm_base_url,
                    headers={"X-API-Key": settings.ecm_api_key},
                    timeout=10.0,
                ) as client:
                    response = await client.request(method, path, **kwargs)
                    response.raise_for_status()
                    return response
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt < settings.ecm_retry_count:
                    wait = delays[attempt] if attempt < len(delays) else delays[-1]
                    logger.warning(
                        "ECM request failed, retrying",
                        path=path,
                        attempt=attempt + 1,
                        wait_sec=wait,
                        error=str(exc),
                    )
                    await asyncio.sleep(wait)

        raise RuntimeError(
            f"ECM {method} {path} failed after {settings.ecm_retry_count} retries"
        ) from last_exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_schema(self, schema_id: str) -> dict:
        if schema_id in self._schema_cache:
            return self._schema_cache[schema_id]
        response = await self._request("GET", f"/schemas/{schema_id}")
        schema = response.json()
        self._schema_cache[schema_id] = schema
        return schema

    async def get_dictionary(self, name: str) -> list[str]:
        if name in self._dict_cache:
            return self._dict_cache[name]
        response = await self._request("GET", f"/dictionaries/{name}")
        entries: list[str] = response.json()
        self._dict_cache[name] = entries
        return entries

    async def post_card(self, result: ProcessingResult) -> str:
        response = await self._request(
            "POST", "/cards", json=result.model_dump(mode="json")
        )
        return response.json()["card_id"]
