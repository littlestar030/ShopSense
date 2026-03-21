FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_REQUIRE_HASHES=
ENV PIP_CONSTRAINT=
ENV PIP_CONFIG_FILE=/dev/null
ENV PIP_DEFAULT_TIMEOUT=1000
ENV PIP_RETRIES=10

COPY requirements.txt .
RUN grep -v '^torch$' requirements.txt > requirements.docker.txt && \
    pip install --no-cache-dir -r requirements.docker.txt && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu --retries 10 --timeout 1000 torch

COPY backend ./backend
COPY dataset ./dataset

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
