# TODO — proximas iteracoes

Coisas que ficaram fora do escopo do datathon mas merecem trabalho futuro:

- [ ] Re-rodar RAGAS com Llama 70B (estava no 8B por TPD)
- [ ] Expandir golden set pra 100+ perguntas (16 ainda nao rodaram por TPD)
- [ ] Treinar MLP PyTorch como diferencial vs LightGBM
- [ ] CI/CD com deploy pra cloud (hoje so build do Dockerfile)
- [ ] Drift monitor automatizado (hoje so script ad-hoc)
- [ ] Feature store (Feast) pra reuso entre treino e producao
- [ ] A/B testing entre 70B e 8B em producao (champion/challenger)
- [ ] Adversarial robustness com red team mais agressivo
