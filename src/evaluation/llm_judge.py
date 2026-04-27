"""LLM-as-judge para avaliar respostas do agente em 3 criterios.

Estrategia:
- Modelo de avaliacao: Llama 3.1 8B (separado do production 70B)
- Criterios:
  1. TECNICO — usa metricas/dados corretos das tools (PD, n_credits, etc.)
  2. REGULATORIO — cita norma correta para o caso (LGPD/Lei 14.181/CMN)
  3. NEGOCIO — recomendacao alinha com o tier de risco e contexto
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

JUDGE_MODEL = "llama-3.1-8b-instant"
GOLDEN_PATH = Path("data/golden_set/golden_set.json")
RESULTS_PATH = Path("evaluation/golden_set/results.json")
OUT_DIR = Path("evaluation/llm_judge")
OUT_DIR.mkdir(parents=True, exist_ok=True)


JUDGE_PROMPT = """Voce e um avaliador especializado em respostas de agentes de credito.

Avalie a RESPOSTA do agente em 3 criterios. Retorne APENAS JSON valido com 3 chaves
(tecnico, regulatorio, negocio), cada uma com:
- score: int de 1 a 5
- justificativa: string curta (1 frase)

CRITERIO 1 — TECNICO
A resposta usa numeros/dados corretos das tools chamadas?
Premio scores altos quando: PD numerica precisa, contagens (creditos, atrasos), valores R$.
Penalize quando: invencao de dados, contradicao com tools, vague ("alta", "baixa" sem numero).

CRITERIO 2 — REGULATORIO
A resposta cita norma juridica correta para o caso?
Premio scores altos quando: cita CMN 4.557/4.966, LGPD Art. 20, CDC Art. 43, Lei 14.181/21
de forma pertinente, com numero da lei/artigo.
Penalize quando: cita norma errada, ou cita generico ("a lei diz") sem fonte.

CRITERIO 3 — NEGOCIO
A recomendacao (APROVAR/REVISAO_MANUAL/NEGAR) e coerente com o risco?
Premio scores altos quando: PD baixa->APROVAR, PD media->REVISAO, PD alta->NEGAR+human-in-loop.
Penalize quando: aprova alto risco, nega baixo risco, ou ignora LGPD em negacoes.

PERGUNTA: {question}

RESPOSTA DO AGENTE:
{answer}

Retorne JSON valido (sem markdown, sem ```):
"""


def judge_one(client: Groq, question: str, answer: str, retries: int = 3) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, answer=answer)

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            return json.loads(content)
        except json.JSONDecodeError:
            if attempt == retries - 1:
                return {"error": "judge returned invalid json", "raw": content[:200]}
            time.sleep(2)
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                time.sleep(15)
                continue
            return {"error": str(e)[:200]}

    return {"error": "exhausted retries"}


def aggregate(verdicts: list[dict]) -> dict:
    """Calcula medias e distribuicao."""
    scores = {"tecnico": [], "regulatorio": [], "negocio": []}
    for v in verdicts:
        for crit in scores:
            if crit in v and isinstance(v[crit], dict) and "score" in v[crit]:
                scores[crit].append(v[crit]["score"])

    summary = {}
    for crit, vals in scores.items():
        if vals:
            summary[crit] = {
                "mean": round(sum(vals) / len(vals), 2),
                "n": len(vals),
                "distribution": {str(s): vals.count(s) for s in range(1, 6)},
            }
        else:
            summary[crit] = {"mean": None, "n": 0}

    summary["overall_mean"] = round(
        sum(s["mean"] for s in summary.values() if s.get("mean")) / 3, 2
    ) if all(summary[c].get("mean") for c in ["tecnico", "regulatorio", "negocio"]) else None

    return summary


def run_judge():
    print(">>> Carregando resultados do golden set...")
    if not RESULTS_PATH.exists():
        raise RuntimeError(f"{RESULTS_PATH} nao existe. Rode src.evaluation.golden_set_eval primeiro.")

    try:
        data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        data = json.loads(RESULTS_PATH.read_text(encoding="cp1252"))
    results = data["results"]
    valid = [r for r in results if not r.get("error")]
    print(f">>> {len(valid)} respostas validas para julgamento ({len(results) - len(valid)} com erro foram puladas)\n")

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    judged = []
    for i, item in enumerate(valid, 1):
        print(f"[{i}/{len(valid)}] Julgando {item['id']}: {item['question'][:50]}...")
        verdict = judge_one(client, item["question"], item["answer"])

        if "error" in verdict:
            print(f"    ERROR: {verdict['error']}")
        else:
            t = verdict.get("tecnico", {}).get("score", "?")
            r = verdict.get("regulatorio", {}).get("score", "?")
            n = verdict.get("negocio", {}).get("score", "?")
            print(f"    tecnico={t}/5 | regulatorio={r}/5 | negocio={n}/5")

        judged.append({"id": item["id"], "question": item["question"], "verdict": verdict})

    summary = aggregate([j["verdict"] for j in judged])

    print("\n" + "=" * 60)
    print("RESUMO LLM-AS-JUDGE")
    print("=" * 60)
    for crit in ["tecnico", "regulatorio", "negocio"]:
        s = summary[crit]
        if s.get("mean"):
            print(f"  {crit:15s}: media={s['mean']}/5  (n={s['n']})  dist={s['distribution']}")
    if summary.get("overall_mean"):
        print(f"\n  OVERALL MEAN  : {summary['overall_mean']}/5")
    print("=" * 60)

    out = OUT_DIR / "results.json"
    out.write_text(json.dumps(
        {"summary": summary, "judged": judged, "judge_model": JUDGE_MODEL},
        indent=2, ensure_ascii=False, default=str,
    ))
    print(f"\n>>> Salvo em: {out}")


if __name__ == "__main__":
    run_judge()
