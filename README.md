# Datathon Fase 05 — Credit Underwriting Copilot com Agente LLM + RAG

**FIAP Pós-Tech MLET | Solo (Rafael Zampieri) | Abril 2026**

[![CI](https://github.com/rafaelkrambek/datathon-credit-copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/rafaelkrambek/datathon-credit-copilot/actions/workflows/ci.yml)

## Problema de Negócio

Apoio à decisão de concessão de crédito ao consumidor, combinando modelo de default risk (LightGBM) com agente LLM (Llama 3.3 70B via Groq) que orquestra 5 tools — perfil do cliente, histórico externo (bureau), histórico interno, score com SHAP e RAG sobre regulatórios brasileiros (CMN 4.557, CMN 4.966, LGPD Art. 20, CDC Art. 43, Lei 14.181/21).

Métrica de negócio: Gini ≥ 0.55 e KS ≥ 0.30 (limite do regulador BR), com decisão final sempre auditável e nunca finalizada de forma automatizada nas negações (LGPD Art. 20).

Resultado atual: Gini 0.565, KS 0.427, LLM-as-judge overall 4.67/5 em 13 perguntas validadas do golden set.

O objetivo final é que a LLM sirva como um grande suporte para análise de crédito (um gargalo comum em instituições financeiras0>

## Arquitetura de Alto Nível

```
                Analista (HTTP / curl / Swagger UI)
                            |
  +-----------------------------------------------------+
  |  CAMADA 1: API FastAPI                              |
  |  - Pydantic schemas (validação de tipo)             |
  |  - lifespan warmup (sem cold start)                 |
  +-----------------------------------------------------+
                            |
  +-----------------------------------------------------+
  |  CAMADA 2: Segurança de input                       |
  |  - Guardrails (regex prompt injection, off-topic)   |
  |  - Presidio PII mask (CPF, CNPJ, email, RG, nome)   |
  +-----------------------------------------------------+
                            |
  +-----------------------------------------------------+
  |  CAMADA 3: Agente ReAct (LangChain)                 |
  |  - LLM: Llama 3.3 70B via Groq (INT8 quantized)     |
  |  - System prompt com workflow obrigatório           |
  |  - LGPD gate (negar -> revisão humana)              |
  +-----------------------------------------------------+
                            |
  +---+ +---+ +---+ +---+ +---+
  |T1 | |T2 | |T3 | |T4 | |T5 |   CAMADA 4: 5 Tools
  +---+ +---+ +---+ +---+ +---+
                            |
  +-----------------------------------------------------+
  |  CAMADA 5: Dados + modelo                           |
  |  - data/raw/*.csv (DVC tracked)                     |
  |  - data/processed/application_enriched.parquet      |
  |  - mlruns/ -> LightGBM gini 0.565                   |
  |  - data/chroma_db/ -> RAG (5 docs regulatórios)     |
  +-----------------------------------------------------+
                            |
  +-----------------------------------------------------+
  |  CAMADA 6: Output + observabilidade                 |
  |  - Guardrails de output (estrutura, sem PII)        |
  |  - Prometheus metrics (counter, histogram)          |
  |  - Langfuse trace (cada tool call no LLM)           |
  |  - Grafana dashboards (9 painéis)                   |
  +-----------------------------------------------------+
                            |
                            v
                   JSON pra o analista
```

## Início Rápido

### Pré-requisitos

- Python ≥ 3.11
- Git + Git LFS (opcional)
- Conta Kaggle com API key (para baixar o dataset Home Credit)
- Conta Groq com API key — free tier ([console.groq.com](https://console.groq.com))
- Conta Langfuse Cloud — free tier ([cloud.langfuse.com](https://cloud.langfuse.com))
- (Opcional) Docker + Docker Compose para a stack completa de observabilidade

### 1. Configurar ambiente

```bash
git clone https://github.com/rafaelkrambek/datathon-credit-copilot.git
cd datathon-credit-copilot

# uv (Astral) — package manager Python 10x mais rápido que pip
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS/Linux
# irm https://astral.sh/uv/install.ps1 | iex      # Windows PowerShell

uv venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows

uv pip install -e ".[dev,ml,llm,serving,security]"
python -m spacy download pt_core_news_sm
python -m ipykernel install --user --name credit-copilot

cp .env.example .env   # edita com GROQ_API_KEY, LANGFUSE_*
```

### 2. Baixar dataset

```bash
mkdir -p data/raw
cd data/raw
kaggle competitions download -c home-credit-default-risk
unzip home-credit-default-risk.zip && rm home-credit-default-risk.zip
cd ../..
```

### 3. Pre-builds (features + modelo + RAG index)

```bash
python -m src.features.aggregations         # gera application_enriched.parquet
python -m src.models.train_baseline --model lgbm --enriched   # treina + MLflow
python -m src.agent.rag                     # indexa knowledge_base no Chroma
```

### 4. Iniciar API

```bash
uvicorn src.serving.api:app --host 127.0.0.1 --port 8000
```

### 5. Usar endpoints

```bash
# Health check
curl http://localhost:8000/healthz

# Análise de cliente (agente roda 5 tools, ~25s no 70B)
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"sk_id_curr": 100002}'

# Análise com pergunta customizada (PII é mascarada antes do LLM)
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"sk_id_curr": 100003, "question": "Cliente apto para R$ 200 mil?"}'

# Métricas Prometheus
curl http://localhost:8000/metrics | grep copilot
```

### 6. Rodar testes

```bash
pytest tests/ -v
```

### 7. Stack completa (observabilidade)

```bash
docker compose up -d --build         # api + prometheus + grafana
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

## Estrutura do Repositório

```
datathon-credit-copilot/
├── .github/workflows/ci.yml          # GitHub Actions: lint -> test -> Docker build
├── .devcontainer/                    # config Codespaces (fallback)
├── infra/
│   ├── prometheus/prometheus.yml      # scrape config
│   └── grafana/
│       ├── provisioning/              # datasource + dashboards auto
│       └── dashboards/copilot.json    # 9 painéis prontos
├── data/
│   ├── raw/                           # 10 CSVs Kaggle (DVC tracked, gitignored)
│   ├── processed/*.parquet            # tabela enriquecida (gitignored)
│   ├── knowledge_base/*.md            # 5 docs regulatórios pro RAG
│   ├── golden_set/golden_set.json     # 20 perguntas pra avaliação
│   └── chroma_db/                     # vector store (gitignored)
├── docs/
│   ├── SYSTEM_CARD.md                 # visão completa do sistema
│   ├── owasp.md                       # OWASP LLM Top 10 mapping
│   ├── business_metrics.md            # negócio -> técnico
│   ├── quantization.md                # decisão sobre INT8 do Groq
│   └── benchmark.md                   # comparativo das 3 configs ML
├── evaluation/
│   ├── fairness/                      # Fairlearn audit + ThresholdOptimizer
│   ├── golden_set/results.json        # eval do agente
│   ├── llm_judge/results.json         # 3 critérios técnico/regulatório/negócio
│   ├── ragas/                         # 4 métricas
│   └── drift/drift.json               # KS test
├── notebooks/
│   └── 01_eda.ipynb                   # EDA com fairness preview
├── scripts/
│   └── make_eda_notebook.py           # gerador do notebook
├── src/
│   ├── agent/
│   │   ├── data_layer.py              # queries por SK_ID_CURR (lru_cache)
│   │   ├── model_layer.py             # carrega LightGBM do MLflow + SHAP
│   │   ├── tools.py                   # 5 tools com @tool decorator
│   │   ├── rag.py                     # ChromaDB + sentence-transformers
│   │   └── react_agent.py             # ChatGroq + create_tool_calling_agent
│   ├── evaluation/
│   │   ├── golden_set_eval.py         # rec accuracy, kw recall, tool usage
│   │   ├── llm_judge.py               # 3 critérios via Llama 8B
│   │   └── ragas_eval.py              # 4 métricas RAGAS
│   ├── features/
│   │   ├── preprocess.py              # drop building cols, engineered ratios
│   │   └── aggregations.py            # 7 tabelas -> 150 features
│   ├── models/
│   │   └── train_baseline.py          # LogReg+WoE e LightGBM com MLflow
│   ├── monitoring/
│   │   ├── fairness.py                # Fairlearn DI/EOD
│   │   ├── mitigation.py              # ThresholdOptimizer (Equal Opportunity)
│   │   └── drift.py                   # KS test 2-sample
│   ├── security/
│   │   ├── pii.py                     # Presidio com recognizers pt_BR
│   │   └── guardrails.py              # input + output validators
│   └── serving/
│       └── api.py                     # FastAPI com lifespan warmup
├── tests/
│   ├── test_guardrails.py             # 5 testes (injection, output, PII)
│   └── test_preprocess.py             # 3 testes (engineered features)
├── docker-compose.yml                 # api + prometheus + grafana
├── Dockerfile                         # multi-stage, non-root user
├── pyproject.toml                     # 5 grupos opcionais de deps
├── CHANGELOG.md                       # evolução por dia
└── README.md                          # você está aqui
```

## Stack Tecnológica

| Componente | Tecnologia | Justificativa |
|---|---|---|
| Modelo de risco | LightGBM 4.5 + LogReg+WoE | LGBM trata categóricas nativas e treina rápido em CPU; LogReg interpretável vai pra auditoria |
| LLM | Llama 3.3 70B Versatile (Groq) | Free tier real, INT8 quantized, latência ~10x menor que GPU |
| Quantização | INT8 do Groq (LPU custom) | Sem custo de hardware próprio; qualidade ~95% do FP16 |
| Agente | LangChain ReAct (`create_tool_calling_agent`) | Framework maduro, tool calling nativo. Pinado <1.0 (v1 quebrou API) |
| Tools | 5 tools read-only (@tool decorator) | Workflow linear, sem necessidade de Plan-and-Execute |
| RAG | ChromaDB + sentence-transformers (MiniLM-L12-v2) | Local-first, embeddings multilingue PT/EN |
| Encoding crédito | category_encoders (WOE) | Padrão clássico de credit scoring, bins auditáveis |
| Explicabilidade | SHAP TreeExplainer | Top-N feature contribution por predição |
| Tracking | MLflow 2.16 (file backend) | Tags padronizadas, lineage, model registry |
| API | FastAPI + Pydantic + Uvicorn | Async-first, OpenAPI auto, validação de tipo |
| Avaliação | RAGAS + LLM-as-judge custom | 4 métricas RAG + 3 critérios (técnico/regulatório/negócio) |
| Drift | KS test 2-sample (scipy) | Sem dependência pesada, suficiente pra demo |
| Fairness | Fairlearn (DI, EOD, ThresholdOptimizer) | Audit + mitigação Equal Opportunity |
| Segurança | Presidio (recognizers pt_BR) + guardrails regex | LGPD + OWASP LLM01/02/06/08/09 |
| Observabilidade app | Prometheus + Grafana (9 painéis) | Métricas custom: requests, latência, tool calls, PII, guardrails |
| Observabilidade LLM | Langfuse Cloud (free tier) | Trace por chamada com tokens, latência, custo |
| CI/CD | GitHub Actions | ruff lint + format + pytest + Docker build |
| Dados | DVC 3.55 | Versiona data/raw com hash, sem inflar Git |
| Container | Docker multi-stage + Compose | Imagem ~600MB-1GB (sem torch/CUDA via grupo `runtime`) |

## Design Patterns

| Padrão | Aplicação | Referência |
|---|---|---|
| Repository | `data_layer.py` centraliza acesso a `application_train.csv`, `bureau.csv`, `previous_application.csv` com `@lru_cache` | [martinfowler.com](https://martinfowler.com/eaaCatalog/repository.html) |
| Strategy | `MODEL_REGISTRY` em `train_baseline.py` permite trocar treinador (LogReg vs LightGBM) sem alterar callers | [refactoring.guru](https://refactoring.guru/design-patterns/strategy) |
| Singleton (lazy) | `_load_latest_lgbm()` e `_embeddings()` carregam modelo/embeddings uma única vez via `@lru_cache(maxsize=1)` | [refactoring.guru](https://refactoring.guru/design-patterns/singleton) |
| Decorator | `@tool` do LangChain registra funções como ferramentas do agente sem alterar a função em si | [refactoring.guru](https://refactoring.guru/design-patterns/decorator) |
| Chain of Responsibility | Pipeline de `/analyze`: input guardrail → Presidio mask → agente → output guardrail → Prometheus | [refactoring.guru](https://refactoring.guru/design-patterns/chain-of-responsibility) |
| Adapter | Cada tool wrappa um repository call em formato JSON consumível pelo LLM | [refactoring.guru](https://refactoring.guru/design-patterns/adapter) |

## Endpoints da API

| Método | Path | Tag | Descrição |
|---|---|---|---|
| GET | `/` | meta | Metadata (versão, lista de endpoints) |
| GET | `/healthz` | meta | Probe de saúde (Kubernetes-ready) |
| GET | `/metrics` | meta | Métricas em formato Prometheus |
| POST | `/analyze` | analysis | Endpoint principal: dispara agente com 5 tools |
| GET | `/docs` | meta | Swagger UI auto-gerada |

## Avaliação

| Frente | Métrica | Resultado |
|---|---|---|
| ML — discriminação | Gini (LightGBM v2) | 0.5650 |
| ML — discriminação | KS | 0.4271 |
| ML — interpretável | Gini (LogReg+WoE) | 0.4990 |
| Fairness — gênero | DI baseline → mitigado | 0.818 → 0.865 |
| Fairness — gênero | EOD baseline → mitigado | +0.139 → +0.029 (−80%) |
| Agente — recommendation accuracy | n=13 (7 com TPD esgotado) | 100% |
| Agente — keyword recall | n=13 | 68.6% |
| Agente — tool usage accuracy | n=13 | 100% |
| LLM-as-judge — técnico | 1-5 | 4.62 |
| LLM-as-judge — regulatório | 1-5 | 4.92 |
| LLM-as-judge — negócio | 1-5 | 4.46 |
| LLM-as-judge — overall | 1-5 | **4.67** |
| RAGAS — faithfulness | 0-1 | 0.75 |
| RAGAS — context_precision | 0-1 | 0.75 |
| RAGAS — context_recall | 0-1 | 0.69 |
| RAGAS — answer_relevancy | 0-1 | 0.26 (avaliado em 8B; ~0.5+ esperado em 70B) |

## Documentação

- [System Card](docs/SYSTEM_CARD.md) — visão completa, casos de uso, limitações, fairness, LGPD, segurança
- [OWASP Mapping](docs/owasp.md) — 5 ameaças LLM Top 10 com mitigação e cenários adversariais
- [Business Metrics](docs/business_metrics.md) — mapeamento métricas de negócio → técnicas
- [Quantization](docs/quantization.md) — decisão sobre INT8 do Groq, 70B em produção vs 8B em avaliação
- [Benchmark](docs/benchmark.md) — comparativo formal LogReg vs LightGBM v1 vs LightGBM v2 (3 configs)

## Licença e contato

Projeto acadêmico para FIAP Pós-Tech MLET (Datathon Fase 05). Sem licença comercial. Contato: rafaelzampieri5@gmail.com.
