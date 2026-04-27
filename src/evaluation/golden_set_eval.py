"""Avaliacao do agente em golden set de 10 perguntas."""
from __future__ import annotations

import json
import time
import unicodedata
from pathlib import Path

from src.agent.react_agent import build_agent

GOLDEN_PATH = Path("data/golden_set/golden_set.json")
OUT_DIR = Path("evaluation/golden_set")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text.lower()


def score_item(item: dict, answer: str, tools_used: list[str]) -> dict:
    answer_norm = normalize(answer)

    rec_expected = item.get("expected_recommendation", [])
    if not rec_expected:
        rec_match = None
    else:
        rec_match = any(normalize(r) in answer_norm for r in rec_expected)

    kws = item.get("expected_keywords", [])
    found = [kw for kw in kws if normalize(kw) in answer_norm]
    kw_recall = len(found) / len(kws) if kws else 1.0

    tools_expected = set(item.get("expected_tools", []))
    tools_used_set = set(tools_used)
    tool_match = tools_expected.issubset(tools_used_set) if tools_expected else True

    return {
        "recommendation_match": rec_match,
        "keyword_recall": round(kw_recall, 3),
        "keywords_found": found,
        "keywords_missed": [kw for kw in kws if normalize(kw) not in answer_norm],
        "tool_match": tool_match,
        "tools_expected": list(tools_expected),
        "tools_used": tools_used,
    }


def run_eval():
    print(">>> Carregando golden set...")
    items = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    print(f">>> {len(items)} perguntas carregadas\n")

    print(">>> Carregando agente (warmup)...")
    executor = build_agent(verbose=False)

    results = []
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item['id']} ({item['category']}): {item['question'][:60]}...")
        start = time.perf_counter()
        try:
            r = executor.invoke({"input": item["question"]})
            answer = r["output"]
            tools_used = [step[0].tool for step in r.get("intermediate_steps", [])]
            elapsed = time.perf_counter() - start

            score = score_item(item, answer, tools_used)
            score.update({
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "answer": answer,
                "latency_seconds": round(elapsed, 2),
                "error": None,
            })
            print(f"    rec={score['recommendation_match']} | "
                  f"kw_recall={score['keyword_recall']:.2f} | "
                  f"tool={score['tool_match']} | {elapsed:.1f}s")
        except Exception as e:
            score = {
                "id": item["id"], "category": item["category"],
                "question": item["question"],
                "error": str(e),
            }
            print(f"    ERROR: {e}")

        results.append(score)

    rec_scores = [r["recommendation_match"] for r in results
                  if r.get("recommendation_match") is not None]
    kw_scores = [r["keyword_recall"] for r in results if "keyword_recall" in r]
    tool_scores = [r["tool_match"] for r in results if "tool_match" in r]
    latencies = [r["latency_seconds"] for r in results if "latency_seconds" in r]

    summary = {
        "n_questions": len(results),
        "n_errors": sum(1 for r in results if r.get("error")),
        "recommendation_accuracy": round(sum(rec_scores) / len(rec_scores), 3) if rec_scores else None,
        "mean_keyword_recall": round(sum(kw_scores) / len(kw_scores), 3) if kw_scores else None,
        "tool_usage_accuracy": round(sum(tool_scores) / len(tool_scores), 3) if tool_scores else None,
        "mean_latency_seconds": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "total_latency_seconds": round(sum(latencies), 2) if latencies else None,
    }

    print("\n" + "=" * 60)
    print("RESUMO GOLDEN SET")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:30s}: {v}")
    print("=" * 60)

    out = OUT_DIR / "results.json"
    out.write_text(json.dumps(
        {"summary": summary, "results": results},
        indent=2, ensure_ascii=False, default=str,
    ))
    print(f"\n>>> Salvo em: {out}")


if __name__ == "__main__":
    run_eval()
