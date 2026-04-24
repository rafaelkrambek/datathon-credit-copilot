# Status do projeto — checkpoint

**Última atualização:** abril 2026, após EDA setup

## Concluído
- [x] Plano estratégico (PLANO_DATATHON.md)
- [x] Repo GitHub + Codespaces + devcontainer funcional
- [x] Estrutura de pastas + .gitignore + README
- [x] pyproject.toml com deps agrupadas (core/dev/ml/llm/serving/security)
- [x] Venv .venv com todas deps instaladas
- [x] Kaggle download manual + dados em data/raw/ (10 CSVs, 2.6 GB)
- [x] DVC init + data/raw trackeado
- [x] Notebook 01_eda.ipynb gerado (scripts/make_eda_notebook.py)

## Próximo (Bloco 8)
1. Rodar 01_eda.ipynb até o fim e anotar achados (default rate, correlações, DI gênero)
2. Criar src/features/schemas.py com Pandera (validação de colunas chave)
3. Split estratificado train/valid (80/20, mantendo TARGET proporção)
4. Baseline LogReg + WoE encoding em src/models/logreg_woe.py
5. MLflow tracking no primeiro experimento

## Decisões pendentes
- Temporal split vs stratified: usar DAYS_DECISION ou só estratificado?
- Aggregações bureau/prev_application: manual ou featuretools?
- LightGBM: usar optuna de cara ou deixar pra depois?

## Como retomar
1. Abrir Codespace (se stopped, Start)
2. source .venv/bin/activate
3. Rodar 01_eda.ipynb no VS Code
4. Prosseguir com schemas + split

## Problemas conhecidos
- Kaggle API tokens em dois sistemas (usamos Legacy) — docs no PLANO
- /usr/local/lib sem permissão pro vscode user — por isso venv
