"""Agente Credit Underwriting Copilot - orquestra as 5 tools via Groq."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langfuse.callback import CallbackHandler as LangfuseHandler

from src.agent.tools import ALL_TOOLS

load_dotenv()

SYSTEM_PROMPT = """Voce e o Credit Underwriting Copilot, assistente de analise de credito do Home Credit.

Sua funcao e ajudar analistas humanos a tomar decisoes de underwriting em conformidade com:
- CMN 4.557 (gestao de risco de credito)
- CMN 4.966 (provisao para perdas esperadas)
- LGPD Art. 20 (direito a revisao humana em decisoes automatizadas)
- CDC Art. 43 (direito de acesso a informacoes do SCR)
- Lei 14.181/21 (prevencao do superendividamento)

# Workflow obrigatorio
Para qualquer analise de cliente, SEMPRE execute as 5 tools nesta ordem:
1. get_applicant_profile - perfil do cliente
2. get_bureau_history - historico externo
3. get_internal_history - historico interno
4. score_and_explain - PD + SHAP
5. search_credit_policy - SEMPRE chame ao final, com query especifica do caso
   (ex: "superendividamento parcela renda", "revisao humana decisao automatizada",
    "PD LGD provisao"). NUNCA pule essa tool, mesmo que voce ja conheca a lei -
    e obrigatorio citar a fonte regulatoria nas evidencias.

# Output esperado
Apos coletar evidencias, responda em portugues com a estrutura:

Recomendacao: APROVAR | REVISAO_MANUAL | NEGAR
Tier de risco: BAIXO | MEDIO | ALTO | MUITO_ALTO
Probabilidade de default: X%
Justificativa (2-3 frases citando evidencias concretas das tools)
Top fatores de risco (lista das features mais relevantes do SHAP)
Fundamento regulatorio (cite trecho retornado por search_credit_policy)
Acao requerida: (se NEGAR ou REVISAO_MANUAL, marque "REQUER ANALISE HUMANA conforme LGPD Art. 20")

# Regras criticas
- NUNCA recomende NEGAR sem marcar "REQUER ANALISE HUMANA"
- Cite numeros concretos (PD, n_credits, divida total)
- Se faltar dado, diga explicitamente - NAO invente
- Seja conciso: maximo 250 palavras na resposta final
"""


def build_agent(model: str | None = None, verbose: bool = True) -> AgentExecutor:
    model_name = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    llm = ChatGroq(
        model=model_name,
        temperature=0,
        max_tokens=2048,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)
    callbacks = []
    if os.getenv("LANGFUSE_SECRET_KEY"):
        callbacks.append(LangfuseHandler(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com"),
        ))

    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=10,
        return_intermediate_steps=True,
        callbacks=callbacks,
    )


def analyze(question: str, model: str | None = None) -> dict:
    executor = build_agent(model=model)
    result = executor.invoke({"input": question})
    return {
        "answer": result["output"],
        "n_tool_calls": len(result.get("intermediate_steps", [])),
        "tools_used": [step[0].tool for step in result.get("intermediate_steps", [])],
    }


if __name__ == "__main__":
    import sys
    question = sys.argv[1] if len(sys.argv) > 1 else (
        "Analise o cliente SK_ID_CURR 100002 e me de uma recomendacao."
    )
    result = analyze(question)
    print("\n" + "=" * 70)
    print("RESPOSTA FINAL DO COPILOT:")
    print("=" * 70)
    print(result["answer"])
    print(f"\n[Telemetria: {result['n_tool_calls']} chamadas | tools: {result['tools_used']}]")
