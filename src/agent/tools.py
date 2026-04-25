"""5 tools do agente Credit Underwriting Copilot."""
from __future__ import annotations

import json

from langchain_core.tools import tool

from src.agent.data_layer import (
    get_applicant,
    get_bureau_history as _bureau,
    get_internal_history as _internal,
    get_features_for_inference,
)
from src.agent.model_layer import predict_pd, shap_top_features


@tool
def get_applicant_profile(sk_id_curr: int) -> str:
    """Retorna o perfil socio-demografico e financeiro do cliente.

    Args:
        sk_id_curr: identificador unico do cliente (inteiro)

    Returns:
        JSON com gender, age, income, credit_amount, education, ext_sources etc.
    """
    data = get_applicant(sk_id_curr)
    if data is None:
        return json.dumps({"error": f"Cliente {sk_id_curr} nao encontrado."})
    return json.dumps(data, ensure_ascii=False, default=str)


@tool
def get_bureau_history(sk_id_curr: int) -> str:
    """Retorna historico de credito EXTERNO (bureau) do cliente.

    Inclui numero de creditos ativos/fechados, divida total, valores em atraso,
    tipos de credito e idade do credito mais antigo.
    """
    return json.dumps(_bureau(sk_id_curr), ensure_ascii=False, default=str)


@tool
def get_internal_history(sk_id_curr: int) -> str:
    """Retorna historico INTERNO do cliente no Home Credit.

    Inclui aplicacoes anteriores, taxa de aprovacao, contratos passados.
    """
    return json.dumps(_internal(sk_id_curr), ensure_ascii=False, default=str)


@tool
def score_and_explain(sk_id_curr: int) -> str:
    """Calcula a probabilidade de default (PD) do cliente e retorna explicacao SHAP.

    Use APOS coletar profile e historico, para fundamentar a decisao final.
    Retorna PD (0 a 1), risk_tier (BAIXO/MEDIO/ALTO/MUITO_ALTO) e top 5 features
    que mais influenciaram a predicao.
    """
    features = get_features_for_inference(sk_id_curr)
    if features is None:
        return json.dumps({"error": f"Cliente {sk_id_curr} nao tem features para predicao."})

    pred = predict_pd(features)
    explanation = shap_top_features(features, top_n=5)

    return json.dumps(
        {**pred, "top_features": explanation},
        ensure_ascii=False,
        default=str,
    )


@tool
def search_credit_policy(query: str) -> str:
    """Busca politicas e regulatorios brasileiros relevantes (CMN 4.557, CMN 4.966, LGPD Art. 20, CDC Art. 43, Lei 14.181/21).

    Use sempre que precisar fundamentar decisao em norma legal/regulatoria - especialmente
    para decisoes de NEGAR (LGPD Art. 20), avaliacao de superendividamento (Lei 14.181),
    ou metodologia de provisao (CMN 4.966).

    Args:
        query: pergunta em portugues sobre normas regulatorias

    Returns:
        JSON com top 3 trechos relevantes e suas fontes
    """
    from src.agent.rag import search
    results = search(query, k=3)
    return json.dumps({"query": query, "results": results}, ensure_ascii=False)


ALL_TOOLS = [
    get_applicant_profile,
    get_bureau_history,
    get_internal_history,
    score_and_explain,
    search_credit_policy,
]
