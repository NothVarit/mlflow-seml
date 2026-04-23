FROM python:3.12-slim

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir uv && \
    uv pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
