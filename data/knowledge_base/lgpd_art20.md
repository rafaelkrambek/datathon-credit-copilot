# LGPD (Lei 13.709/2018) — Art. 20: Decisoes automatizadas

## Art. 20 — Direito a revisao
O titular dos dados tem direito a solicitar revisao, por pessoa natural, de decisoes
tomadas unicamente com base em tratamento automatizado de dados pessoais que afetem
seus interesses, incluidas as decisoes destinadas a definir seu perfil pessoal,
profissional, de consumo e de credito ou os aspectos de sua personalidade.

## Paragrafo 1 — Transparencia
O controlador devera fornecer, sempre que solicitado, informacoes claras e adequadas
a respeito dos criterios e dos procedimentos utilizados para a decisao automatizada.

## Implicacao pratica
Decisoes de NEGAR credito tomadas exclusivamente por modelo de ML devem:
1. Ser sinalizadas para revisao humana antes de comunicacao ao cliente.
2. Ter as razoes (top features SHAP, motivos negociais) documentadas e disponibilizadas.
3. Permitir contestacao pelo titular.
