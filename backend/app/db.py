import json
import os
import re
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS users (
        id BIGSERIAL PRIMARY KEY,
        username TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prediction_activity (
        id BIGSERIAL PRIMARY KEY,
        user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        title TEXT,
        preview TEXT NOT NULL,
        status TEXT NOT NULL,
        response_time_ms DOUBLE PRECISION,
        tags_json TEXT NOT NULL DEFAULT '[]',
        error_message TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    "ALTER TABLE prediction_activity ADD COLUMN IF NOT EXISTS title TEXT",
    "CREATE INDEX IF NOT EXISTS idx_prediction_activity_user_created_at ON prediction_activity (user_id, created_at DESC)",
)


class DuplicateUserError(RuntimeError):
    pass


def _default_database_url() -> str:
    return os.getenv("DATABASE_URL", "postgresql://admin:admin@127.0.0.1:5435/seml_app")


def _default_admin_database_url(database_url: str) -> str:
    override = os.getenv("APP_DATABASE_ADMIN_URL")
    if override:
        return override
    parts = urlsplit(database_url)
    return urlunsplit((parts.scheme, parts.netloc, "/postgres", parts.query, parts.fragment))


def _database_name(database_url: str) -> str:
    name = urlsplit(database_url).path.lstrip("/")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise RuntimeError(f"Unsupported database name: {name!r}")
    return name


@dataclass
class Database:
    database_url: str
    admin_database_url: str

    @classmethod
    def from_env(cls) -> "Database":
        database_url = _default_database_url()
        return cls(database_url=database_url, admin_database_url=_default_admin_database_url(database_url))

    def _driver(self):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError("psycopg is required for PostgreSQL persistence.") from exc
        return psycopg, dict_row

    def _connect(self, database_url: str | None = None, *, autocommit: bool = False):
        psycopg, dict_row = self._driver()
        connection = psycopg.connect(database_url or self.database_url, row_factory=dict_row)
        connection.autocommit = autocommit
        return connection

    def _ensure_database_exists(self) -> None:
        target_name = _database_name(self.database_url)
        with self._connect(self.admin_database_url, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_name,))
                if cursor.fetchone():
                    return
                cursor.execute(f'CREATE DATABASE "{target_name}"')

    def initialize(self) -> None:
        self._ensure_database_exists()
        with self._connect() as connection:
            with connection.cursor() as cursor:
                for statement in SCHEMA_STATEMENTS:
                    cursor.execute(statement)

    def health(self) -> dict[str, str]:
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1 AS ok")
                    cursor.fetchone()
            return {"status": "healthy", "detail": "Database connection is ready."}
        except Exception as exc:
            return {"status": "unavailable", "detail": f"Database unavailable: {exc}"}

    def create_user(self, username: str, password_hash: str) -> dict:
        if self.get_user_by_username(username):
            raise DuplicateUserError(username)
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO users (username, password_hash)
                    VALUES (%s, %s)
                    RETURNING id, username, password_hash, created_at
                    """,
                    (username, password_hash),
                )
                return cursor.fetchone()

    def get_user_by_username(self, username: str) -> dict | None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT id, username, password_hash, created_at FROM users WHERE username = %s",
                    (username,),
                )
                return cursor.fetchone()

    def get_user_by_id(self, user_id: int) -> dict | None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT id, username, password_hash, created_at FROM users WHERE id = %s",
                    (user_id,),
                )
                return cursor.fetchone()

    def record_prediction(
        self,
        user_id: int,
        title: str | None,
        preview: str,
        status: str,
        response_time_ms: float,
        tags: list[str],
        error_message: str | None = None,
    ) -> dict:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO prediction_activity (user_id, title, preview, status, response_time_ms, tags_json, error_message)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, title, preview, status, response_time_ms, tags_json, error_message, created_at
                    """,
                    (user_id, title, preview, status, response_time_ms, json.dumps(tags), error_message),
                )
                row = cursor.fetchone()
        return self._deserialize_record(row)

    def get_global_metrics(self) -> dict[str, float | int]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        COUNT(*) AS total_requests,
                        COALESCE(SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END), 0) AS failed_requests,
                        COALESCE(AVG(response_time_ms), 0) AS average_response_time_ms
                    FROM prediction_activity
                    """
                )
                row = cursor.fetchone() or {}
        total_requests = int(row.get("total_requests", 0) or 0)
        failed_requests = int(row.get("failed_requests", 0) or 0)
        average_response_time_ms = float(row.get("average_response_time_ms", 0.0) or 0.0)
        success_rate = 0.0 if total_requests == 0 else (total_requests - failed_requests) / total_requests
        return {
            "total_requests": total_requests,
            "failed_requests": failed_requests,
            "average_response_time_ms": average_response_time_ms,
            "success_rate": success_rate,
        }

    def get_recent_requests(self, user_id: int, *, limit: int = 10) -> list[dict]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, title, preview, status, response_time_ms, tags_json, error_message, created_at
                    FROM prediction_activity
                    WHERE user_id = %s
                    ORDER BY created_at DESC, id DESC
                    LIMIT %s
                    """,
                    (user_id, limit),
                )
                rows = cursor.fetchall()
        return [self._deserialize_record(row) for row in rows]

    def _deserialize_record(self, row: dict | None) -> dict | None:
        if row is None:
            return None
        return {
            "id": str(row["id"]),
            "title": row.get("title"),
            "preview": row["preview"],
            "status": row["status"],
            "response_time_ms": float(row["response_time_ms"] or 0.0),
            "tags": self._parse_tags(row.get("tags_json")),
            "error_message": row.get("error_message"),
            "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
        }

    @staticmethod
    def _parse_tags(value: str | None) -> list[str]:
        if not value:
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return [item for item in parsed if isinstance(item, str)]
