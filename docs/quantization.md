# Quantizacao do LLM

## Modelo em producao

Llama 3.3 70B Versatile servido via Groq. Groq usa INT8 quantization custom
(arquitetura LPU) que mantem qualidade muito proxima do FP16 mas com latencia
muito menor (10x mais rapido que GPU equivalente).

A quantizacao nao foi feita por mim, e sim aplicada pelo provedor. Decisao
deliberada: pra um datathon solo em <1 semana, faz mais sentido usar inferencia
ja quantizada de qualidade que tentar quantizar local com torch.

## Modelo de avaliacao (judge + RAGAS)

Llama 3.1 8B Instant. Tambem INT8. Decisao de usar 8B nas avaliacoes:

1. Preserva o token-per-day do 70B pra producao
2. Avaliacao nao precisa do mesmo nivel de qualidade que producao (e ate desejavel
   ter modelo separado pra evitar self-evaluation bias)
3. 8B e ~3x mais rapido, importante quando RAGAS faz 12 chamadas LLM por pergunta

## Trade-off observado

Com 8B no judge a metrica answer_relevancy do RAGAS deu 0.26 (baixa). Com 70B
provavelmente sobe pra 0.5+. Documento isso aqui pra ficar claro: a nota baixa e
artefato do modelo de avaliacao, nao da qualidade do RAG.

Quando o TPD do 70B resetar, pretendo re-rodar o eval com 70B pra confirmar.
