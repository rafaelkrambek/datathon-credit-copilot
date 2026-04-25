# Credit Underwriting Copilot

> Assistente de análise de crédito baseado em LLM agentico, com modelo de default risk, RAG sobre regulatórios brasileiros e observabilidade end-to-end.

Projeto do **FIAP Pós-Tech Datathon — Fase 05: LLMs e Agentes**. Objetivo: demonstrar um pipeline MLOps maturity level 2 combinando machine learning tradicional, agentes com tool use e governança regulatória aplicada ao domínio de concessão de crédito.

**Dataset**: [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) — 307k aplicações, 7 tabelas relacionais, taxa de default de ~8%.

---

## O que o copiloto faz

Em produção, um analista de crédito perguntaria:

> "Cliente SK_ID_CURR 123456 — esse perfil merece aprovação? Qual o risco? Há histórico bureau preocupante? Sobre qual política regulatória devo me basear?"

O agente ReAct roteia a pergunta por **5 tools**:

1. `get_applicant_profile` — dados da aplicação atual
2. `get_bureau_history` — histórico externo de crédito
3. `get_internal_history` — histórico dentro do Home Credit
4. `score_and_explain` — PD (probabilidade de default) + SHAP por cliente
5. `search_credit_policy` — RAG sobre CMN 4.557/4.966, LGPD Art. 20, CDC Art. 43

E retorna uma **recomendação com justificativa** (aprovar / revisão manual / negar), sempre com human-in-the-loop nas negações conforme LGPD.

---

## Stack

| Camada | Tecnologias |
|---|---|
| **Dados** | Pandas, PyArrow, DVC (versionamento) |
| **ML** | scikit-learn, LightGBM, PyTorch (MLP), category_encoders (WoE), SHAP |
| **Tracking** | MLflow (experiments + model registry) |
| **Agente** | LangChain (ReAct), Groq (Llama 3.1 8B), ChromaDB, tiktoken |
| **Observabilidade** | Langfuse (LLM traces), Prometheus + Grafana (métricas), Evidently (drift) |
| **Servir** | FastAPI, Uvicorn, slowapi (rate limit) |
| **Governança** | Fairlearn (DI, Equal Opportunity), Presidio (PII pt_BR), OWASP LLM Top 10 |
| **Qualidade** | Ruff, mypy, pytest, Pandera (schemas) |

---

## Arquitetura (alvo Dia 6)
                          ┌──────────────────────┐
                          │   Analista (UI/API)  │
                          └──────────┬───────────┘
                                     │ prompt
                       ┌─────────────▼──────────────┐
                       │   FastAPI + Rate Limiter   │
                       │   Presidio (PII mask)      │
                       └─────────────┬──────────────┘
                                     │
                       ┌─────────────▼──────────────┐
                       │   Agente ReAct (Groq)      │
                       │   Prompt + Guardrails      │
                       └─────────────┬──────────────┘
                                     │
         ┌───────────┬───────────────┼────────────────┬──────────┐
         ▼           ▼               ▼                ▼          ▼
    profile     bureau          score_and_         internal    RAG
                history          explain           history  (regulatorios)
                                     │                         │
                                     ▼                         ▼
                                LightGBM                   ChromaDB
                                (MLflow)                   (CMN, LGPD)
                                     │
                                     ▼
                           Fairlearn audit + LGPD gate
                                     │
                                     ▼
                       Langfuse trace + Prometheus metrics

---

## Estrutura
datathon-credit-copilot/
├── data/
│   ├── raw/                # CSVs do Kaggle (gitignored, DVC-tracked)
│   ├── processed/          # Parquets consolidados
│   ├── knowledge_base/     # PDFs regulatorios (CMN, LGPD, CDC...)
│   └── golden_set/         # 20 pares pergunta/resposta para eval RAG
├── notebooks/
│   └── 01_eda.ipynb        # EDA inicial (TARGET, missings, fairness)
├── scripts/
│   └── make_eda_notebook.py
├── src/
│   ├── features/
│   │   ├── preprocess.py       # limpeza + feature engineering
│   │   └── aggregations.py     # agrega 6 tabelas auxiliares
│   ├── models/
│   │   └── train_baseline.py   # LogReg+WoE e LightGBM com MLflow
│   ├── agent/                  # (Dia 3) ReAct + 5 tools
│   ├── serving/                # (Dia 4) FastAPI + endpoints
│   ├── monitoring/             # (Dia 4) Prometheus + Evidently
│   └── security/               # (Dia 5) Presidio + LGPD gate
├── tests/
├── evaluation/                 # (Dia 5) golden set + Ragas + Fairlearn
├── docs/
├── configs/
├── PLANO_DATATHON.md           # plano estrategico completo
├── pyproject.toml              # deps agrupadas (dev/ml/llm/serving/security)
└── README.md

---

## Setup (replicação local, ~15 min)

### Pré-requisitos
- Python 3.11 (`python --version`)
- [uv](https://docs.astral.sh/uv/) (`irm https://astral.sh/uv/install.ps1 | iex` no Windows; `curl -LsSf https://astral.sh/uv/install.sh | sh` no macOS/Linux)
- Git
- 4 GB livres em disco

### 1. Clone e ambiente

```bash
git clone https://github.com/rafaelkrambek/datathon-credit-copilot.git
cd datathon-credit-copilot

uv venv .venv
source .venv/Scripts/activate          # Windows Git Bash
# .venv\Scripts\activate                # Windows PowerShell
# source .venv/bin/activate              # macOS/Linux

uv pip install -e ".[dev,ml,llm,serving,security]"
python -m ipykernel install --user --name credit-copilot --display-name "Python (credit-copilot)"
```

### 2. Baixar o dataset

Como o Home Credit Default Risk tem ~700 MB, o repositório não contém os CSVs. Duas formas:

**Via Kaggle API** (requer [key configurada](https://www.kaggle.com/docs/api)):

```bash
mkdir -p data/raw
cd data/raw
kaggle competitions download -c home-credit-default-risk
unzip home-credit-default-risk.zip
rm home-credit-default-risk.zip
cd ../..
```

**Manual**: baixe de https://www.kaggle.com/competitions/home-credit-default-risk/data e extraia os 10 CSVs em `data/raw/`.

### 3. Validar integridade (opcional, via DVC)

```bash
dvc status       # confirma que hashes dos CSVs batem com data/raw.dvc
```

---

## Como rodar

### EDA

```bash
# No VS Code
code notebooks/01_eda.ipynb
# Select kernel: "Python (credit-copilot)" → Run All
```

### Pipeline de features

```bash
python -m src.features.aggregations
# Gera data/processed/application_enriched.parquet (~150 colunas, 307k linhas)
```

### Treinar baselines

```bash
# LogReg + WoE (interpretavel, ~3 min)
python -m src.models.train_baseline --model logreg

# LightGBM v1 (so application_train)
python -m src.models.train_baseline --model lgbm

# LightGBM v2 (com todas as tabelas agregadas)
python -m src.models.train_baseline --model lgbm --enriched
```

### Explorar experimentos

```bash
mlflow ui --workers 1 --host 127.0.0.1 --port 5000
# Abre http://127.0.0.1:5000
```

---

## Resultados parciais (Dia 1)

| Modelo | Features | valid_roc_auc | valid_gini | valid_ks | valid_pr_auc |
|---|---|---|---|---|---|
| LogReg + WoE | application_train (82 cols) | 0.7495 | 0.4990 | 0.3655 | 0.2318 |
| LightGBM v1 | application_train (82 cols) | 0.7619 | 0.5238 | 0.3927 | 0.2495 |
| LightGBM v2 | +bureau, prev_app, installments, pos, cc (~150 cols) | 0.7825 | 0.5650 | 0.4271 | 0.2796 |

**Interpretação**: baseline LogReg já entrega KS ~0.37 (regulador brasileiro considera ≥0.30 como aceitável para crédito de consumo). O Gini de 0.50 é competitivo para um modelo lean sem as agregações das outras tabelas.

---

## Roadmap (6 dias)

- **Dia 1** — Setup, EDA, baselines, feature engineering ✅
- **Dia 2** — LightGBM enriquecido + MLP PyTorch + Pandera schemas
- **Dia 3** — Agente ReAct + 5 tools + RAG regulatorio (CMN, LGPD)
- **Dia 4** — FastAPI + Prometheus + Grafana + Langfuse
- **Dia 5** — Fairlearn audit + Presidio + golden set eval + LGPD gate
- **Dia 6** — Polimento, testes, submission, apresentação

Detalhes em [`PLANO_DATATHON.md`](PLANO_DATATHON.md).

---

## MLOps Maturity

Target: **nivel 2** da escala Microsoft, com:
- CI/CD via GitHub Actions
- Modelo versionado em MLflow Registry
- Schemas com Pandera, testes unitarios cobrindo features
- Observabilidade com metricas de negocio (nao so infra)
- Auditoria de fairness em cada treino
- Trace completo da decisao do agente (Langfuse)

---

## Governança e conformidade

O projeto explicita como atende:
- **CMN 4.557** (estrutura de gestão de risco de crédito)
- **CMN 4.966** (metodologias internas de rating)
- **LGPD Art. 20** (direito à revisão humana em decisões automatizadas)
- **CDC Art. 43** (SCR e direito de acesso)
- **Lei 14.181/21** (superendividamento — gate de affordability)

Implementacao em `src/security/` e auditoria em `evaluation/`.

---

## Autor

Rafael Zampieri — [@rafaelkrambek](https://github.com/rafaelkrambek)

FIAP Pos-Tech Datathon Fase 05, abril 2026.
