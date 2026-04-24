import os
from dataclasses import dataclass

import httpx


class MlServiceUnavailableError(RuntimeError):
    pass


@dataclass
class MlServiceClient:
    base_url: str
    timeout_seconds: float = 10.0

    @classmethod
    def from_env(cls) -> "MlServiceClient":
        return cls(
            base_url=os.getenv("ML_SERVICE_URL", "http://127.0.0.1:8000"),
            timeout_seconds=float(os.getenv("ML_SERVICE_TIMEOUT_SECONDS", "10")),
        )

    def health(self) -> dict[str, str]:
        try:
            with httpx.Client(base_url=self.base_url, timeout=self.timeout_seconds) as client:
                response = client.get("/openapi.json")
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            return {"status": "unavailable", "detail": f"ML service unavailable: {exc}"}

        if "/predict" not in (payload.get("paths") or {}):
            return {"status": "degraded", "detail": "ML service is reachable but the predict route was not advertised."}

        return {"status": "healthy", "detail": "ML service is reachable."}

    def predict_tags(self, text: str) -> list[str]:
        try:
            with httpx.Client(base_url=self.base_url, timeout=self.timeout_seconds) as client:
                response = client.post("/predict", json={"inputs": text})
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            raise MlServiceUnavailableError(f"ML service unavailable: {exc}") from exc

        tags = self._extract_tags(payload)
        return [tag for tag in tags if isinstance(tag, str)]

    @staticmethod
    def _extract_tags(payload: object) -> list[str]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, str)]

        if not isinstance(payload, dict):
            return []

        tags = payload.get("tags")
        if isinstance(tags, list):
            return [item for item in tags if isinstance(item, str)]

        predictions = payload.get("predictions")
        if isinstance(predictions, list) and predictions:
            first_item = predictions[0]
            if isinstance(first_item, list):
                return [item for item in first_item if isinstance(item, str)]
            if isinstance(first_item, str):
                return [item for item in predictions if isinstance(item, str)]

        return []
