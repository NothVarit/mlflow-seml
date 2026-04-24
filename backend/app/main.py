import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .airflow_client import AirflowClient
from .db import Database, DuplicateUserError
from .ml_client import MlServiceClient, MlServiceUnavailableError
from .oov import OovDetector
from .schemas import Credentials, PredictRequest, PredictResponse, SessionResponse
from .security import (
    SESSION_COOKIE_NAME,
    SESSION_TTL_SECONDS,
    create_session_token,
    hash_password,
    read_session_token,
    verify_password,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
SPA_ROUTES = ("/", "/signup", "/login", "/demo", "/status")
logger = logging.getLogger("uvicorn.error")


def _set_session_cookie(response: Response, user_id: int) -> None:
    secure = os.getenv("SESSION_COOKIE_SECURE", "").strip().lower() in {"1", "true", "yes", "on"}
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=create_session_token(user_id),
        max_age=SESSION_TTL_SECONDS,
        expires=SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=secure,
        path="/",
    )


def _prediction_text(payload: PredictRequest) -> str:
    for value in (payload.text, payload.inputs, f"{payload.title}\n\n{payload.body}"):
        if value and value.strip():
            return value.strip()
    return ""


def _content_preview(body: str, fallback: str, *, limit: int = 160) -> str:
    source = body.strip() or fallback.strip()
    compact = " ".join(source.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _overall_status(database_health: dict[str, Any], ml_health: dict[str, Any]) -> str:
    database_state = str(database_health.get("status", "unknown"))
    ml_state = str(ml_health.get("status", "unknown"))
    if database_state != "healthy":
        return "unavailable"
    if ml_state != "healthy":
        return "degraded"
    return "healthy"


def _build_health_payload(database: Any, ml_client: Any) -> dict[str, Any]:
    backend = {"status": "healthy", "detail": "Application backend is running."}
    database_health = database.health()
    ml_health = ml_client.health()
    return {
        "status": _overall_status(database_health, ml_health),
        "backend": backend,
        "database": database_health,
        "ml_service": ml_health,
    }


def _register_frontend(app: FastAPI, frontend_dist: Path | None) -> None:
    if not frontend_dist or not frontend_dist.exists():
        return

    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

    index_file = frontend_dist / "index.html"
    if not index_file.exists():
        return

    def serve_index() -> FileResponse:
        return FileResponse(index_file)

    for route in SPA_ROUTES:
        app.add_api_route(route, serve_index, methods=["GET"], include_in_schema=False)


def create_app(
    db: Any | None = None,
    ml_client: Any | None = None,
    frontend_dist: Path | None = DEFAULT_FRONTEND_DIST,
) -> FastAPI:
    database = db or Database.from_env()
    inference_client = ml_client or MlServiceClient.from_env()

    oov_detector = OovDetector.from_env()
    airflow_client = AirflowClient.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.db = database
        app.state.ml_client = inference_client
        app.state.oov_detector = oov_detector
        app.state.airflow_client = airflow_client
        app.state.started_at = time.time()
        app.state.db.initialize()
        yield

    app = FastAPI(title="SEML demo backend", lifespan=lifespan)

    def current_user(request: Request) -> dict[str, Any]:
        payload = read_session_token(request.cookies.get(SESSION_COOKIE_NAME))
        if not payload:
            raise HTTPException(status_code=401, detail="Authentication required.")
        user = request.app.state.db.get_user_by_id(payload["user_id"])
        if not user:
            raise HTTPException(status_code=401, detail="Session is invalid or expired.")
        return user

    @app.get("/healthz")
    @app.get("/api/health")
    def health(request: Request) -> dict[str, Any]:
        return _build_health_payload(request.app.state.db, request.app.state.ml_client)

    @app.post("/api/auth/signup", response_model=SessionResponse)
    def signup(payload: Credentials, request: Request, response: Response) -> SessionResponse:
        username = payload.username.strip()
        if not username:
            raise HTTPException(status_code=400, detail="Username is required.")

        try:
            user = request.app.state.db.create_user(username, hash_password(payload.password))
        except DuplicateUserError as exc:
            raise HTTPException(status_code=409, detail="Username is already taken.") from exc

        _set_session_cookie(response, user["id"])
        return SessionResponse(username=user["username"], authenticated=True)

    @app.post("/api/auth/login", response_model=SessionResponse)
    def login(payload: Credentials, request: Request, response: Response) -> SessionResponse:
        username = payload.username.strip()
        user = request.app.state.db.get_user_by_username(username) if username else None
        if not user or not verify_password(payload.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid username or password.")

        _set_session_cookie(response, user["id"])
        return SessionResponse(username=user["username"], authenticated=True)

    @app.post("/api/auth/logout")
    def logout(response: Response) -> dict[str, bool]:
        response.delete_cookie(SESSION_COOKIE_NAME, path="/")
        return {"authenticated": False}

    @app.get("/api/me", response_model=SessionResponse)
    @app.get("/api/auth/me", response_model=SessionResponse)
    @app.get("/api/auth/session", response_model=SessionResponse)
    def me(user: dict[str, Any] = Depends(current_user)) -> SessionResponse:
        return SessionResponse(username=user["username"], authenticated=True)

    @app.post("/api/predict", response_model=PredictResponse)
    def predict(
        payload: PredictRequest,
        request: Request,
        user: dict[str, Any] = Depends(current_user),
    ) -> PredictResponse:
        text = _prediction_text(payload)
        if not text:
            raise HTTPException(status_code=400, detail="Provide article content before requesting tags.")

        preview = _content_preview(payload.body, text)
        title = payload.title.strip() or None
        started_at = time.perf_counter()
        try:
            tags = request.app.state.ml_client.predict_tags(text)
        except MlServiceUnavailableError as exc:
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            logger.warning(
                "prediction_failed user_id=%s title=%r text_length=%s latency_ms=%.2f error=%r",
                user["id"],
                title,
                len(text),
                elapsed_ms,
                str(exc),
            )
            request.app.state.db.record_prediction(user["id"], title, preview, "failed", elapsed_ms, [], str(exc))
            raise HTTPException(
                status_code=503,
                detail="The ML service is temporarily unavailable. Please try again shortly.",
            ) from exc

        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "prediction_succeeded user_id=%s title=%r text_length=%s tag_count=%s latency_ms=%.2f tags=%s",
            user["id"],
            title,
            len(text),
            len(tags),
            elapsed_ms,
            tags,
        )
        request.app.state.db.record_prediction(user["id"], title, preview, "success", elapsed_ms, tags, None)

        oov_rate, spike = request.app.state.oov_detector.check(text)
        if spike:
            logger.warning(
                "oov_spike_detected user_id=%s oov_rate=%.4f triggering_dag=article_tagger_retraining",
                user["id"],
                oov_rate,
            )
            request.app.state.airflow_client.trigger_dag(
                "article_tagger_retraining",
                conf={"triggered_by": "oov_spike", "oov_rate": oov_rate},
            )
        elif oov_rate > 0:
            logger.info("oov_rate user_id=%s oov_rate=%.4f spike=False", user["id"], oov_rate)

        return PredictResponse(tags=tags, message="")

    @app.get("/api/status")
    def status(request: Request, user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
        metrics = request.app.state.db.get_global_metrics()
        return {
            **metrics,
            "recent_requests": request.app.state.db.get_recent_requests(user["id"], limit=10),
            "health": _build_health_payload(request.app.state.db, request.app.state.ml_client),
        }

    _register_frontend(app, frontend_dist)
    return app


app = create_app()
