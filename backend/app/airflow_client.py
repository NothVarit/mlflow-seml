import logging
import os
from dataclasses import dataclass

import httpx

logger = logging.getLogger("uvicorn.error")


@dataclass
class AirflowClient:
    base_url: str
    username: str
    password: str
    timeout_seconds: float = 10.0
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "AirflowClient":
        base_url = os.getenv("AIRFLOW_BASE_URL", "")
        username = os.getenv("AIRFLOW_USERNAME", "airflow")
        password = os.getenv("AIRFLOW_PASSWORD", "airflow")
        if not base_url:
            return cls(base_url="", username="", password="", enabled=False)
        return cls(base_url=base_url.rstrip("/"), username=username, password=password)

    def trigger_dag(self, dag_id: str, conf: dict | None = None) -> bool:
        if not self.enabled:
            return False
        url = f"{self.base_url}/api/v1/dags/{dag_id}/dagRuns"
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(
                    url,
                    auth=(self.username, self.password),
                    json={"conf": conf or {}},
                )
                response.raise_for_status()
            logger.info("airflow_dag_triggered dag_id=%s status=%s", dag_id, response.status_code)
            return True
        except Exception as exc:
            logger.warning("airflow_dag_trigger_failed dag_id=%s error=%r", dag_id, str(exc))
            return False
