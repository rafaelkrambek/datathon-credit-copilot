"""Detecao e mascaramento de PII (CPF, CNPJ, email, telefone, RG) usando Presidio."""
from __future__ import annotations

import re
from functools import lru_cache

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# ---------- Recognizers customizados pt_BR ----------
CPF_PATTERN = Pattern(
    name="cpf_pattern",
    regex=r"\b\d{3}[.\s]?\d{3}[.\s]?\d{3}[-\s]?\d{2}\b",
    score=0.9,
)

CNPJ_PATTERN = Pattern(
    name="cnpj_pattern",
    regex=r"\b\d{2}[.\s]?\d{3}[.\s]?\d{3}[/\s]?\d{4}[-\s]?\d{2}\b",
    score=0.9,
)

PHONE_BR_PATTERN = Pattern(
    name="phone_br_pattern",
    regex=r"\b(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?\d{4,5}[-\s]?\d{4}\b",
    score=0.7,
)

RG_PATTERN = Pattern(
    name="rg_pattern",
    regex=r"\b\d{1,2}[.\s]?\d{3}[.\s]?\d{3}[-\s]?[\dXx]\b",
    score=0.6,
)


@lru_cache(maxsize=1)
def _analyzer() -> AnalyzerEngine:
    # NLP engine basico — usa spaCy small (rapido)
    config = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "pt", "model_name": "pt_core_news_sm"}],
    }
    nlp_engine = NlpEngineProvider(nlp_configuration=config).create_engine()

    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["pt"])

    # Adiciona recognizers BR
    for entity_type, pattern in [
        ("CPF_BR", CPF_PATTERN),
        ("CNPJ_BR", CNPJ_PATTERN),
        ("PHONE_BR", PHONE_BR_PATTERN),
        ("RG_BR", RG_PATTERN),
    ]:
        recognizer = PatternRecognizer(
            supported_entity=entity_type,
            patterns=[pattern],
            supported_language="pt",
        )
        analyzer.registry.add_recognizer(recognizer)

    return analyzer


@lru_cache(maxsize=1)
def _anonymizer() -> AnonymizerEngine:
    return AnonymizerEngine()


PII_ENTITIES = ["CPF_BR", "CNPJ_BR", "PHONE_BR", "RG_BR", "EMAIL_ADDRESS", "PERSON"]


def detect_pii(text: str) -> list[dict]:
    """Retorna lista de PIIs encontradas (sem mascarar)."""
    analyzer = _analyzer()
    results = analyzer.analyze(
        text=text,
        language="pt",
        entities=PII_ENTITIES,
    )
    return [
        {
            "entity_type": r.entity_type,
            "start": r.start,
            "end": r.end,
            "score": round(r.score, 3),
            "text": text[r.start:r.end],
        }
        for r in results
    ]


def mask_pii(text: str) -> tuple[str, list[dict]]:
    """Mascara todas PIIs encontradas. Retorna (texto_mascarado, items_detectados)."""
    analyzer = _analyzer()
    anonymizer = _anonymizer()

    detected = analyzer.analyze(
        text=text,
        language="pt",
        entities=PII_ENTITIES,
    )

    operators = {
        "CPF_BR": OperatorConfig("replace", {"new_value": "[CPF]"}),
        "CNPJ_BR": OperatorConfig("replace", {"new_value": "[CNPJ]"}),
        "PHONE_BR": OperatorConfig("replace", {"new_value": "[TELEFONE]"}),
        "RG_BR": OperatorConfig("replace", {"new_value": "[RG]"}),
        "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
        "PERSON": OperatorConfig("replace", {"new_value": "[NOME]"}),
    }

    result = anonymizer.anonymize(
        text=text,
        analyzer_results=detected,
        operators=operators,
    )

    items = [
        {
            "entity_type": r.entity_type,
            "score": round(r.score, 3),
            "text": text[r.start:r.end],
        }
        for r in detected
    ]

    return result.text, items


if __name__ == "__main__":
    samples = [
        "Cliente Joao Silva, CPF 123.456.789-01, telefone (11) 98765-4321.",
        "Email: rafael@example.com, RG 12.345.678-9. CNPJ 12.345.678/0001-90.",
        "Quero verificar o cliente SK_ID_CURR 100002 que tem renda de R$ 200 mil.",
    ]
    for s in samples:
        masked, items = mask_pii(s)
        print(f"\nORIGINAL: {s}")
        print(f"MASCARADO: {masked}")
        print(f"DETECTADOS: {[i['entity_type'] for i in items]}")
