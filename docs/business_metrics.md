# Mapeamento metricas de negocio -> tecnicas

Modelo de credito so faz sentido se a metrica tecnica conversa com decisao de
negocio. Aqui esta o mapeamento que usei:

| Decisao de negocio | Metrica tecnica | Threshold usado |
|---|---|---|
| "O modelo separa bom de mau pagador?" | Gini (2*AUC - 1) | >= 0.50 (aceitavel), >= 0.55 (bom) |
| "Discrimina alto risco do baixo?" | KS | >= 0.30 (regulador BR aceita) |
| "Identifica defaults reais sem inundar de FPs?" | PR-AUC | acompanhar (depende do baseline 8%) |
| "Quanto perdemos se errar?" | Expected Loss = PD * LGD * EAD | usado em simulacoes |
| "Quem aprovo / quem nego?" | PD threshold (default 0.20) | calibravel por apetite de risco |

## Tier de risco

PD < 0.10 -> BAIXO (auto-aprova)
0.10 <= PD < 0.20 -> MEDIO (revisao automatica com regras)
0.20 <= PD < 0.40 -> ALTO (revisao manual obrigatoria)
PD >= 0.40 -> MUITO_ALTO (negar com justificativa)

## Custo de erro

FP (negar bom pagador): perda de receita de juros + churn de cliente. Estimativa:
~5% do credito como receita perdida.

FN (aprovar mau pagador): perda direta proporcional a EAD. Estimativa: ~50% LGD
para credito unsecured no Brasil.

Como FN custa muito mais que FP, balanceio o threshold pra reduzir falsos negativos
mesmo aceitando mais FPs.

## Resultados que entreguei

- LightGBM enriquecido: Gini 0.565, KS 0.427 -> bate o regulator e o competitivo
- LogReg+WoE (interpretavel pra auditoria): Gini 0.499, KS 0.366
- Razao FN-cost / FP-cost considerada na escolha do threshold 0.20
