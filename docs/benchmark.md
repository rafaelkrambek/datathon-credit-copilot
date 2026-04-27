# Benchmark de configuracoes

## Modelos de credito (3 configs)

| Modelo | Features | valid_roc_auc | valid_gini | valid_ks |
|---|---|---|---|---|
| LogReg + WoE | application_train (82 cols) | 0.7495 | 0.4990 | 0.3655 |
| LightGBM v1 | application_train (82 cols) | ~0.76 | ~0.52 | ~0.39 |
| LightGBM v2 (enriched) | + bureau, prev, installments, pos, cc (~150 cols) | 0.7825 | 0.5650 | 0.4271 |

LogReg fica como baseline interpretavel (vai pra auditoria com WoE bins). LGBMv2
e o que vai pra producao.

Treinos rodaram com mesmo random_state (42) e split estratificado 80/20. MLflow
guarda os runs.

## LLMs no agente (2 configs)

| Modelo | Latencia media | Tokens-per-day free | Qualidade subjetiva |
|---|---|---|---|
| Llama 3.1 8B Instant | ~7s | 100k | razoavel, alucina ocasional |
| Llama 3.3 70B Versatile | ~25s | 100k | preciso, raramente alucina |

Producao usa 70B. Avaliacao (LLM-as-judge, RAGAS) usa 8B pra economizar TPD.

Em fallback (rate limit do 70B), a API pode degradar pro 8B mantendo o workflow.

## Mitigacao de fairness (2 configs)

| Modelo LGBM | Disparate Impact | EOD |
|---|---|---|
| Sem mitigacao (CODE_GENDER) | 0.818 | +0.139 |
| Com ThresholdOptimizer | 0.865 | +0.029 (-80%) |

Mitigacao reduz EOD em 80% sem perder accuracy significativa.
