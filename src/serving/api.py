"""Credit Underwriting Copilot - FastAPI service."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from src.agent.react_agent import build_agent
from src.security.guardrails import check_input, check_output
from src.security.guardrails import summarize as gr_summarize
from src.security.pii import mask_pii

# ---------------------- metrics ----------------------
REQ_COUNT = Counter("copilot_requests_total", "Total de requisicoes ao /analyze", ["status"])
REQ_LATENCY = Histogram(
    "copilot_request_latency_seconds",
    "Latencia de /analyze (segundos)",
    buckets=[0.5, 1, 2, 5, 10, 20, 30, 60],
)
TOOL_CALLS = Counter("copilot_tool_calls_total", "Total de tool calls do agente", ["tool"])
PII_DETECTED = Counter(
    "copilot_pii_detected_total", "Itens de PII detectados em inputs", ["entity_type"]
)
GUARDRAILS_FAILED = Counter(
    "copilot_guardrails_failed_total", "Veredictos negativos de guardrails", ["stage", "severity"]
)


# ---------------------- schemas ----------------------
class AnalyzeRequest(BaseModel):
    sk_id_curr: int = Field(..., examples=[100002])
    question: str | None = Field(default=None)


class AnalyzeResponse(BaseModel):
    sk_id_curr: int
    answer: str
    tools_used: list[str]
    n_tool_calls: int
    latency_seconds: float
    pii_detected: list[str] = []
    question_masked: str | None = None
    guardrails: dict = {}


# ---------------------- lifespan ----------------------
agent_executor: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_executor
    print(">>> Carregando agente (warmup)...")
    agent_executor = build_agent(verbose=False)
    print(">>> Agente pronto. API ON.")
    yield
    print(">>> API encerrando.")


app = FastAPI(
    title="Credit Underwriting Copilot",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", tags=["meta"])
async def root():
    return {"service": "Credit Underwriting Copilot", "version": "0.1.0"}


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

    raw_question = req.question or (
        f"Analise o cliente SK_ID_CURR {req.sk_id_curr}. "
        "Como devo proceder considerando a regulacao brasileira?"
    )

    # ========= 1. Guardrails de INPUT (raw, antes de qualquer mask) =========
    input_verdicts = check_input(raw_question)
    input_summary = gr_summarize(input_verdicts)

    for v in input_verdicts:
        if not v.passed:
            GUARDRAILS_FAILED.labels(stage="input", severity=v.severity).inc()

    # Bloqueia QUALQUER falha (critical ou warning) — postura conservadora pra credito
    if input_summary["n_failed"] > 0:
        REQ_COUNT.labels(status="blocked").inc()
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Input bloqueado pelos guardrails",
                "issues": input_summary["issues"],
            },
        )

    # ========= 2. Mask PII (depois dos guardrails) =========
    question, pii_items = mask_pii(raw_question)
    pii_types = [i["entity_type"] for i in pii_items]
    for ent in pii_types:
        PII_DETECTED.labels(entity_type=ent).inc()

    # ========= 3. Injeta sk_id_curr garantido =========
    final_input = f"Analise o cliente SK_ID_CURR={req.sk_id_curr}. Contexto adicional: {question}"

    # ========= 4. Executa agente =========
    start = time.perf_counter()
    try:
        result = agent_executor.invoke({"input": final_input})
    except Exception as e:
        REQ_COUNT.labels(status="error").inc()
        raise HTTPException(500, f"Agent error: {e}") from e

    latency = time.perf_counter() - start
    REQ_LATENCY.observe(latency)

    tools_used = [step[0].tool for step in result.get("intermediate_steps", [])]
    for t in tools_used:
        TOOL_CALLS.labels(tool=t).inc()

    answer = result["output"]

    # ========= 5. Guardrails de OUTPUT =========
    output_verdicts = check_output(answer)
    output_summary = gr_summarize(output_verdicts)

    for v in output_verdicts:
        if not v.passed:
            GUARDRAILS_FAILED.labels(stage="output", severity=v.severity).inc()

    # Defesa em profundidade: se output vazou PII, mascara antes de devolver
    if any("vazou PII" in (v.reason or "") for v in output_verdicts if not v.passed):
        answer, _ = mask_pii(answer)

    REQ_COUNT.labels(status="ok").inc()

    return AnalyzeResponse(
        sk_id_curr=req.sk_id_curr,
        answer=answer,
        tools_used=tools_used,
        n_tool_calls=len(tools_used),
        latency_seconds=round(latency, 3),
        pii_detected=pii_types,
        question_masked=question if pii_types else None,
        guardrails={"input": input_summary, "output": output_summary},
    )
