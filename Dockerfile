# syntax=docker/dockerfile:1.6
# ============================================================
# Stage 1 — builder: instala deps em venv isolado
# ============================================================
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Deps de sistema necessarias para build (LightGBM, lxml, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instala uv (Astral) — gerenciador rapido
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copia apenas pyproject pra cache de deps
COPY pyproject.toml ./
COPY src/ ./src/

# Cria venv e instala todas as deps em modo editavel
RUN uv venv /opt/venv --python 3.11
ENV PATH="/opt/venv/bin:${PATH}"

RUN uv pip install --python /opt/venv/bin/python \
    -e ".[ml,llm,serving,security]"

# Modelo spaCy pt_BR para Presidio
RUN python -m spacy download pt_core_news_sm


# ============================================================
# Stage 2 — runtime: imagem final enxuta
# ============================================================
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

# Apenas runtime libs (sem build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Usuario nao-root (boa pratica de seguranca)
RUN useradd --create-home --shell /bin/bash copilot

WORKDIR /app

# Copia venv pronto do builder
COPY --from=builder /opt/venv /opt/venv

# Copia codigo e artefatos minimamente necessarios
COPY --chown=copilot:copilot src/ ./src/
COPY --chown=copilot:copilot data/knowledge_base/ ./data/knowledge_base/
COPY --chown=copilot:copilot data/chroma_db/ ./data/chroma_db/
COPY --chown=copilot:copilot mlruns/ ./mlruns/
COPY --chown=copilot:copilot pyproject.toml ./

# IMPORTANTE: data/raw/ NAO entra (sao 2.6GB) — agente precisa montar via volume

USER copilot

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fs http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
