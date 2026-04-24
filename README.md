# Credit Underwriting Copilot

> **Datathon Fase 05 — FIAP Pós-Tech ML Engineering**
> Dataset: [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/competitions/home-credit-default-risk)

Copiloto para analistas de crédito baseado em:

- **Modelo de default risk** em ensemble (LogReg+WoE, LightGBM, MLP PyTorch)
- **Agente ReAct** orquestrando 5 tools sobre 7 tabelas do dataset
- **RAG** sobre documentos regulatórios brasileiros (Res. CMN 4.557/4.966, SCR, LGPD, CDC)
- **Observabilidade** end-to-end: MLflow, Prometheus+Grafana, Langfuse, Evidently
- **Governança**: Model Card, System Card, LGPD Plan, Fairness Report, OWASP mapping, Red Team

## Status

🚧 Em desenvolvimento — Datathon Fase 05

## Arquitetura

_Diagrama será adicionado em `docs/`._

## Quick start (Codespaces)

1. Clica no botão verde **Code** → aba **Codespaces** → **Create codespace on main**
2. Aguarde o devcontainer subir (≈3 min no primeiro boot)
3. `make setup` — instala dependências e baixa dataset
4. `make eda` — abre o notebook exploratório
5. `make train` — treina os 3 baselines e loga no MLflow

## Estrutura
.
├── .devcontainer/      # devcontainer + Dockerfile
├── .github/workflows/  # CI/CD (lint + test + build)
├── data/               # raw (DVC), processed, knowledge_base, golden_set
├── src/                # features, models, agent, serving, monitoring, security
├── tests/              # pytest + pandera schemas
├── evaluation/         # RAGAS + LLM-as-judge
├── docs/               # Model Card, System Card, LGPD, Fairness, OWASP, Red Team
├── notebooks/          # EDA + análise SHAP + fairness
├── infra/              # docker-compose, Prometheus, Grafana dashboards
└── configs/            # model_config, policy_config, monitoring_config

## Tech Stack

| Camada | Tecnologia |
|---|---|
| ML tabular | scikit-learn + LightGBM + PyTorch |
| Experiment tracking | MLflow |
| Data versioning | DVC |
| LLM | Groq (Llama-3.1-8B quantizado INT8) |
| Embeddings + Vector Store | text-embedding-3-small + ChromaDB |
| Agente | LangChain ReAct |
| API | FastAPI |
| Observabilidade | Langfuse + Prometheus + Grafana + Evidently |
| Governança | Fairlearn + Presidio (PII pt_BR) |
| CI/CD | GitHub Actions |

## Licença

A definir.

## Autor

Rafael Zampieri — FIAP Pós-Tech ML Engineering, Fase 05.
EOF
