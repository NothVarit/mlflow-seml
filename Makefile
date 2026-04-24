.PHONY: postgres postgres-down backend frontend frontend-install ml app-layer app-layer-down full-stack health backend-test frontend-build bootstrap-model

postgres:
	docker compose up -d postgres

postgres-down:
	docker compose stop postgres

backend:
	cd backend && uv run uvicorn app.main:app --host 0.0.0.0 --port 8001

frontend:
	cd frontend && npm run dev

frontend-install:
	cd frontend && npm install --no-package-lock

ml:
	docker compose up --build api

app-layer:
	docker compose up --build postgres web

app-layer-down:
	docker compose stop web postgres

full-stack:
	docker compose up --build

health:
	curl -s http://127.0.0.1:8001/healthz

backend-test:
	cd backend && uv run pytest

frontend-build:
	cd frontend && npm run build

bootstrap-model:
	MLFLOW_TRACKING_URI=http://localhost:5001 uv run python bootstrap_model.py
