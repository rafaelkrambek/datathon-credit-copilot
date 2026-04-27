"""RAGAS eval do RAG: 4 metricas (faithfulness, answer_relevancy, context_precision, context_recall)."""
from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.agent.rag import search, _embeddings

load_dotenv()

DATASET_PATH = Path("data/golden_set/rag_eval_set.json")
OUT_DIR = Path("evaluation/ragas")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Usa 8B para preservar TPD do 70B (o agente production roda no 70B)
EVAL_MODEL = "llama-3.1-8b-instant"
ANSWER_MODEL = "llama-3.1-8b-instant"


def generate_answer(llm: ChatGroq, question: str, contexts: list[str]) -> str:
    """Gera resposta baseada APENAS nos contextos retornados pelo RAG."""
    context_text = "\n\n".join(contexts)
    prompt = f"""Responda a pergunta abaixo APENAS com base no contexto fornecido.
Se o contexto nao tiver a resposta, diga "informacao nao consta nos documentos".
Cite numeros de artigos/leis quando estiverem no contexto.

CONTEXTO:
{context_text}

PERGUNTA: {question}

RESPOSTA (concisa, em portugues):"""

    resp = llm.invoke(prompt)
    return resp.content


def run_ragas():
    print(">>> Carregando dataset RAG...")
    items = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    print(f">>> {len(items)} perguntas\n")

    llm_answer = ChatGroq(
        model=ANSWER_MODEL,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    rows = []
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item['question'][:60]}...")

        # Busca contextos no Chroma
        results = search(item["question"], k=3)
        contexts = [r["snippet"] for r in results]

        # Gera resposta baseada nos contextos
        answer = generate_answer(llm_answer, item["question"], contexts)
        print(f"    answer: {answer[:80]}...")

        rows.append({
            "user_input": item["question"],
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": item["ground_truth"],
        })

    print("\n>>> Calculando metricas RAGAS...")
    dataset = Dataset.from_list(rows)

    # LLM e embeddings para o RAGAS
    eval_llm = ChatGroq(
        model=EVAL_MODEL,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    wrapped_llm = LangchainLLMWrapper(eval_llm)
    wrapped_emb = LangchainEmbeddingsWrapper(_embeddings())

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    for m in metrics:
        m.llm = wrapped_llm
        if hasattr(m, "embeddings"):
            m.embeddings = wrapped_emb

    from ragas.run_config import RunConfig
    run_config = RunConfig(
        timeout=180,         # 3 min por job
        max_retries=3,
        max_workers=1,       # sequencial (sem paralelismo, evita rate limit)
        log_tenacity=False,
    )

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=wrapped_llm,
        embeddings=wrapped_emb,
        run_config=run_config,
    )

    print("\n" + "=" * 60)
    print("RESUMO RAGAS")
    print("=" * 60)
    df = result.to_pandas()
    print(df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].describe())
    print("=" * 60)

    df.to_csv(OUT_DIR / "ragas_results.csv", index=False, encoding="utf-8")
    summary = {
        "n_samples": len(rows),
        "eval_model": EVAL_MODEL,
        "metrics_mean": {
            "faithfulness": round(float(df["faithfulness"].mean()), 4),
            "answer_relevancy": round(float(df["answer_relevancy"].mean()), 4),
            "context_precision": round(float(df["context_precision"].mean()), 4),
            "context_recall": round(float(df["context_recall"].mean()), 4),
        },
    }
    (OUT_DIR / "ragas_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n>>> Salvo em: {OUT_DIR}/")


if __name__ == "__main__":
    run_ragas()
