FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-bridge.txt .
RUN pip install --no-cache-dir -r requirements-bridge.txt

COPY src/       src/
COPY conf/      conf/
COPY data/json/ data/json/
COPY nifi_connector.py .

RUN mkdir -p data/resource_cache

ENV PYTHONPATH=/app/src

CMD ["python", "src/main.py", "-c", "conf/docker.json", "server", "start"]
