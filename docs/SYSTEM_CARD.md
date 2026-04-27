# System Card — Credit Underwriting Copilot

Versao 0.1.0 | Abril 2026 | Rafael Zampieri (FIAP Datathon Fase 05)

## O que e

Sistema agentico que ajuda analistas de credito do Home Credit a decidir
underwriting. Combina ML tradicional (LightGBM) pra calcular probabilidade de
default, agente LLM (Llama 3.3 70B via Groq) que orquestra ferramentas, e RAG
sobre regulatorios brasileiros pra fundamentar decisoes.

A saida e uma recomendacao (APROVAR / REVISAO_MANUAL / NEGAR) com
justificativa, tier de risco e fundamento regulatorio. Nao toma decisao final
sozinho — sempre marca casos de NEGAR como "REQUER ANALISE HUMANA conforme
LGPD Art. 20".

## Quem usa e pra que

- Analistas de credito que avaliam aplicacoes individuais
- Auditoria interna que precisa rastrear decisao + evidencias
- Gerente de risco que monitora performance e fairness do modelo

NAO e:
- Sistema de aprovacao automatica
- Substituto de analista humano
- Conselheiro financeiro pro tomador final

## Dados

Treinamento: Home Credit Default Risk (Kaggle), 307k aplicacoes, 7 tabelas
relacionais, default rate ~8%. Period entre 2014-2018, mercado russo
publicamente. Adaptamos a narrativa pro contexto brasileiro pra demo do
datathon, mas em producao real seria treinado com dados internos do banco.

Atributos sensiveis presentes: CODE_GENDER, NAME_EDUCATION_TYPE,
NAME_FAMILY_STATUS, idade. Auditados via Fairlearn (ver secao Fairness).

## Arquitetura

Tres camadas:

1. **Modelo de risco** (LightGBM enriquecido). 150 features, 80% treino /
   20% validacao estratificado. Servido via MLflow registry.
2. **Agente** (LangChain + Groq Llama 3.3 70B). 5 tools: profile, bureau,
   internal, score+SHAP, search_credit_policy.
3. **API + governanca** (FastAPI). Antes do agente: guardrails + Presidio.
   Depois: validacao de output. Tudo metricado em Prometheus + Langfuse.

## Performance

| Metrica | Valor | Threshold de aceite |
|---|---|---|
| Gini (LightGBM enriquecido) | 0.565 | >= 0.50 |
| KS | 0.427 | >= 0.30 (regulador BR) |
| ROC-AUC | 0.7825 | acompanhar |
| LLM-as-judge overall | 4.78/5 | (n=3, valido sob TPD) |
| RAGAS faithfulness | 0.75 | >= 0.7 |
| RAGAS context_precision | 0.75 | >= 0.7 |

## Limitacoes conhecidas

- **Modelo de avaliacao limitado**: RAGAS rodou no Llama 8B em vez do 70B por
  causa de TPD do Groq. answer_relevancy=0.26 e artefato disso. Re-execucao
  com 70B prevista.
- **Golden set de 20 itens**: amostra pequena. Em producao, expandir pra 100+.
- **Dataset estrangeiro**: Home Credit nao reflete totalmente perfil de
  credito brasileiro (renda informal, score externo, dados do BACEN).
- **Latencia alta**: ~25s por chamada do agente (5 tool calls + LLM 70B).
  Aceitavel pra analista que faz 10-20 analises por dia, nao pra autoatendimento.
- **Janela de contexto**: nao guarda historico entre chamadas. Cada analise
  e independente.
- **Sem retrain automatizado**: drift detectado dispara so alerta, retrain
  manual.

## Fairness

Auditado por CODE_GENDER e NAME_EDUCATION_TYPE.

Baseline (sem mitigacao):
- Disparate Impact (CODE_GENDER): 0.640 — abaixo do limite 4/5 do EEOC
- Equal Opportunity Difference: +0.139

Apos mitigacao com Fairlearn ThresholdOptimizer (constraint: Equal Opportunity):
- DI: 0.865 — passa rule 4/5
- EOD: +0.029 — reducao de 80%

Mitigador serializado em `evaluation/fairness/threshold_optimizer_gender.pkl`.

XNA (4 clientes sem genero declarado) excluidos por degenerancia de labels.

## Explicabilidade

Cada predicao retorna top 5 features SHAP, com direcao (aumenta/reduz risco)
e valor numerico. Vai junto na resposta do agente, citando feature por nome
(EXT_SOURCE_MEAN, etc.).

LogReg + WoE mantido como modelo "interpretavel sob demanda" — bins WoE sao
auditaveis manualmente, util pra pedidos de revisao do CDC Art. 43.

## Privacidade e LGPD

- **PII em inputs**: Presidio mascara CPF, CNPJ, email, telefone, RG, nome
  proprio antes de qualquer chamada ao LLM.
- **PII em outputs**: validador detecta vazamento e re-mascara. Defesa em
  profundidade.
- **Decisao automatizada**: nunca finaliza NEGAR sozinha. Marca "REQUER
  ANALISE HUMANA conforme LGPD Art. 20" e fica em fila pra analista.
- **Direito a revisao**: a justificativa retornada (top SHAP + fundamento
  regulatorio) atende pedido de explicacao do titular.
- **Direito de acesso ao SCR (CDC Art. 43)**: dados do bureau retornados
  via tool sao os mesmos que o cliente acessa pelo Banco Central.
- **Logs**: Langfuse retem trace por 30 dias (free tier). Em producao,
  configurar retention de acordo com politica interna.

## Seguranca

Mapeamento OWASP LLM Top 10 em `docs/owasp.md`. Resumo: 5 ameacas mapeadas
e mitigadas (LLM01 Prompt Injection, LLM02 Insecure Output, LLM06 Sensitive
Info, LLM08 Excessive Agency, LLM09 Overreliance).

Cenarios adversariais testados (5):
1. Prompt injection direta — bloqueado por guardrail (regex pattern)
2. Off-topic medico — bloqueado por guardrail (escopo)
3. Mistura de tarefa (avaliar + piada) — bloqueado por guardrail
4. Vazamento de PII em input — Presidio mascara antes do LLM
5. Vazamento de PII em output — validador re-mascara antes de devolver

## Operacao

- **Endpoint**: POST /analyze (FastAPI)
- **Healthcheck**: GET /healthz
- **Metricas**: GET /metrics (formato Prometheus, scrape a cada 15s)
- **Container**: imagem Docker multi-stage, usuario nao-root,
  ~600MB-1GB enxuta (sem torch/CUDA)
- **CI/CD**: GitHub Actions (lint, smoke imports, Docker build) em todo push

## Quando NAO usar

- Decisao final de NEGAR (sempre tem que passar por humano)
- Analise em massa sem revisao (rate limit + custo)
- Casos com renda informal predominante (modelo treinado em mercado formal)
- Casos onde nao ha historico bureau OU interno (cobertura ~85%)

## Versao e mudancas

| Versao | Data | Mudancas |
|---|---|---|
| 0.1.0 | abril/2026 | Versao inicial pra Datathon FIAP Pos-Tech Fase 05 |

## Contato

Rafael Zampieri — rafaelzampieri5@gmail.com
Repo: github.com/rafaelkrambek/datathon-credit-copilot
