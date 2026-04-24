FROM node:20-bookworm-slim AS frontend-build
WORKDIR /build/frontend
COPY frontend/package.json ./package.json
COPY frontend/index.html ./index.html
COPY frontend/tsconfig.json ./tsconfig.json
COPY frontend/vite.config.ts ./vite.config.ts
COPY frontend/src ./src
RUN npm install --no-package-lock && npm run build

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.8.5 /uv /uvx /bin/
COPY backend/pyproject.toml /app/backend/pyproject.toml
COPY backend/uv.lock /app/backend/uv.lock
RUN uv sync --directory /app/backend --locked --no-dev
COPY backend/app /app/backend/app
COPY --from=frontend-build /build/frontend/dist /app/frontend/dist
WORKDIR /app/backend
CMD ["/app/backend/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
