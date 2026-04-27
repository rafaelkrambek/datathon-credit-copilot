"""Auditoria de fairness do LightGBM com Fairlearn.

Metricas:
- Demographic Parity Difference (DPD)
- Disparate Impact Ratio (DIR) — regra do 4/5 (>0.8 OK)
- Equal Opportunity Difference (EOD) — diferenca de recall entre grupos
- Selection rate por grupo
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    selection_rate,
    true_positive_rate,
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.agent.model_layer import _load_latest_lgbm
from src.features.preprocess import load_enriched

OUT_DIR = Path("evaluation/fairness")
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.20  # Threshold de classificacao (PD > 20% = nega)


def audit_attribute(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_score: pd.Series,
    sensitive: pd.Series,
    name: str,
) -> dict:
    """Roda Fairlearn MetricFrame para 1 atributo sensivel."""
    mf = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "tpr_recall": true_positive_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )

    # AUC por grupo calculado a parte (roc_auc_score precisa y_score)
    auc_by_group = {}
    for g in sensitive.unique():
        mask = sensitive == g
        if mask.sum() > 10 and y_true[mask].nunique() > 1:
            auc_by_group[str(g)] = round(float(roc_auc_score(y_true[mask], y_score[mask])), 4)

    dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)
    dir_ratio = demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive)
    eod = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive)

    report = {
        "attribute": name,
        "by_group": mf.by_group.to_dict(),
        "auc_by_group": auc_by_group,
        "demographic_parity_difference": round(float(dpd), 4),
        "disparate_impact_ratio": round(float(dir_ratio), 4),
        "equalized_odds_difference": round(float(eod), 4),
        "rule_4_5_passes": dir_ratio >= 0.80,
        "interpretation": _interpret(dir_ratio, eod),
    }
    return report


def _interpret(dir_ratio: float, eod: float) -> str:
    msgs = []
    if dir_ratio < 0.80:
        msgs.append(
            f"REGRA 4/5 VIOLADA: DIR={dir_ratio:.3f} < 0.80. "
            "Risco regulatorio (LGPD Art. 20, anti-discriminacao)."
        )
    else:
        msgs.append(f"Regra 4/5 OK (DIR={dir_ratio:.3f}).")

    if abs(eod) > 0.10:
        msgs.append(f"EOD alto ({eod:+.3f}) - desigualdade de recall entre grupos.")
    else:
        msgs.append(f"EOD aceitavel ({eod:+.3f}).")

    return " | ".join(msgs)


def run_audit():
    print(">>> Carregando dataset enriquecido + modelo...")
    df = load_enriched()
    model, run_id = _load_latest_lgbm()

    sensitive_attrs = ["CODE_GENDER", "NAME_EDUCATION_TYPE"]

    # Split estratificado igual ao do treino
    y = df["TARGET"]
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])

    X_train, X_valid, y_train, y_valid, sens_train, sens_valid = train_test_split(
        X, y, df[sensitive_attrs],
        test_size=0.2, stratify=y, random_state=42,
    )

    # Categoricas pro LightGBM
    for c in X_valid.select_dtypes(include=["object"]).columns:
        X_valid[c] = X_valid[c].astype("category")

    print(">>> Predicting valid set...")
    y_score = model.predict_proba(X_valid)[:, 1]
    y_pred = (y_score >= THRESHOLD).astype(int)

    reports = {
        "model_run_id": run_id,
        "threshold": THRESHOLD,
        "n_valid": len(y_valid),
        "default_rate": round(float(y_valid.mean()), 4),
        "audits": [],
    }

    for attr in sensitive_attrs:
        print(f">>> Auditando: {attr}")
        report = audit_attribute(
            y_true=y_valid,
            y_pred=pd.Series(y_pred, index=y_valid.index),
            y_score=pd.Series(y_score, index=y_valid.index),
            sensitive=sens_valid[attr],
            name=attr,
        )
        reports["audits"].append(report)

        print(f"    DIR: {report['disparate_impact_ratio']:.3f} | "
              f"EOD: {report['equalized_odds_difference']:+.3f} | "
              f"Rule 4/5: {'PASS' if report['rule_4_5_passes'] else 'FAIL'}")
        print(f"    {report['interpretation']}")

    out = OUT_DIR / "fairness_report.json"
    out.write_text(json.dumps(reports, indent=2, ensure_ascii=False, default=str))
    print(f"\n>>> Relatorio salvo: {out}")

    return reports


if __name__ == "__main__":
    run_audit()
