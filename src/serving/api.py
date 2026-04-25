"""Credit Underwriting Copilot - FastAPI service."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field
from starlette.responses import Response

from src.agent.react_agent import build_agent

# ---------------------- metrics (Bloco 13) ----------------------
REQ_COUNT = Counter(
    "copilot_requests_total",
    "Total de requisicoes ao /analyze",
    ["status"],
)
REQ_LATENCY = Histogram(
    "copilot_request_latency_seconds",
    "Latencia de /analyze (segundos)",
    buckets=[0.5, 1, 2, 5, 10, 20, 30, 60],
)
TOOL_CALLS = Counter(
    "copilot_tool_calls_total",
    "Total de tool calls do agente",
    ["tool"],
)


# ---------------------- schemas ----------------------
class AnalyzeRequest(BaseModel):
    sk_id_curr: int = Field(..., examples=[100002], description="ID do cliente")
    question: str | None = Field(
        default=None,
        examples=["Avalie esse cliente. Como devo proceder se a recomendacao for negar?"],
        description="Pergunta opcional. Se vazia, usa default de avaliacao completa.",
    )


class AnalyzeResponse(BaseModel):
    sk_id_curr: int
    answer: str
    tools_used: list[str]
    n_tool_calls: int
    latency_seconds: float


# ---------------------- lifespan: pre-carrega agente ----------------------
agent_executor: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_executor
    print(">>> Carregando agente (warmup)...")
    agent_executor = build_agent(verbose=False)
    print(">>> Agente pronto. API ON.")
    yield
    print(">>> API encerrando.")


# ---------------------- app ----------------------
app = FastAPI(
    title="Credit Underwriting Copilot",
    description="Agente LLM para apoio a decisao de concessao de credito (FIAP Datathon Fase 05).",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------- endpoints ----------------------
@app.get("/", tags=["meta"])
async def root():
    return {
        "service": "Credit Underwriting Copilot",
        "version": "0.1.0",
        "endpoints": ["/analyze", "/healthz", "/metrics", "/docs"],
    }


@app.get("/healthz", tags=["meta"])
async def healthz():
    if agent_executor is None:
        raise HTTPException(503, "Agent not loaded")
    return {"status": "ok"}


@app.get("/metrics", tags=["meta"])
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
async def analyze(req: AnalyzeRequest):
    if agent_executor is None:
        raise HTTPException(503, "Agent not loaded")

    question = req.question or (
        f"Analise o cliente SK_ID_CURR {req.sk_id_curr}. "
        "Como devo proceder considerando a regulacao brasileira?"
    )

    start = time.perf_counter()
    try:
        result = agent_executor.invoke({"input": question})
    except Exception as e:
        REQ_COUNT.labels(status="error").inc()
        raise HTTPException(500, f"Agent error: {e}")

    latency = time.perf_counter() - start
    REQ_LATENCY.observe(latency)
    REQ_COUNT.labels(status="ok").inc()

    tools_used = [step[0].tool for step in result.get("intermediate_steps", [])]
    for t in tools_used:
        TOOL_CALLS.labels(tool=t).inc()

    return AnalyzeResponse(
        sk_id_curr=req.sk_id_curr,
        answer=result["output"],
        tools_used=tools_used,
        n_tool_calls=len(tools_used),
        latency_seconds=round(latency, 3),
    )
