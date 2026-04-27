"""Mitigacao de viés via Fairlearn ThresholdOptimizer (Equal Opportunity)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.model_selection import train_test_split

from src.agent.model_layer import _load_latest_lgbm
from src.features.preprocess import load_enriched
from src.monitoring.fairness import audit_attribute, THRESHOLD

OUT_DIR = Path("evaluation/fairness")
SENSITIVE_ATTR = "CODE_GENDER"


def run_mitigation():
    print(">>> Carregando dados + modelo...")
    df = load_enriched()

    # Remove XNA do CODE_GENDER (poucos casos sem label utilizavel)
    n_before = len(df)
    df = df[df["CODE_GENDER"] != "XNA"].copy()
    print(f">>> Removidos {n_before - len(df)} clientes com CODE_GENDER=XNA")

    model, run_id = _load_latest_lgbm()

    y = df["TARGET"]
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])

    X_train, X_valid, y_train, y_valid, sens_train, sens_valid = train_test_split(
        X, y, df[SENSITIVE_ATTR],
        test_size=0.2, stratify=y, random_state=42,
    )

    # categoricals pra LGBM
    for c in X_train.select_dtypes(include=["object"]).columns:
        X_train[c] = X_train[c].astype("category")
        X_valid[c] = X_valid[c].astype("category")

    # ============ Baseline (sem mitigacao) ============
    print(">>> Baseline: modelo sem mitigacao")
    y_score = model.predict_proba(X_valid)[:, 1]
    y_pred_base = (y_score >= THRESHOLD).astype(int)

    base_report = audit_attribute(
        y_true=y_valid,
        y_pred=pd.Series(y_pred_base, index=y_valid.index),
        y_score=pd.Series(y_score, index=y_valid.index),
        sensitive=sens_valid,
        name=f"{SENSITIVE_ATTR}_BASELINE",
    )

    # ============ ThresholdOptimizer (Equal Opportunity) ============
    print(">>> Treinando ThresholdOptimizer (constraint: TPR parity)...")
    mitigator = ThresholdOptimizer(
        estimator=model,
        constraints="true_positive_rate_parity",
        objective="balanced_accuracy_score",
        prefit=True,
        predict_method="predict_proba",
    )
    mitigator.fit(X_train, y_train, sensitive_features=sens_train)

    print(">>> Aplicando mitigacao no valid set...")
    y_pred_mit = mitigator.predict(X_valid, sensitive_features=sens_valid)

    mit_report = audit_attribute(
        y_true=y_valid,
        y_pred=pd.Series(y_pred_mit, index=y_valid.index),
        y_score=pd.Series(y_score, index=y_valid.index),
        sensitive=sens_valid,
        name=f"{SENSITIVE_ATTR}_MITIGATED",
    )

    # ============ Comparativo ============
    comparison = {
        "model_run_id": run_id,
        "sensitive_attribute": SENSITIVE_ATTR,
        "n_valid": len(y_valid),
        "baseline": base_report,
        "mitigated": mit_report,
        "deltas": {
            "DIR_delta": round(mit_report["disparate_impact_ratio"]
                               - base_report["disparate_impact_ratio"], 4),
            "EOD_delta": round(mit_report["equalized_odds_difference"]
                               - base_report["equalized_odds_difference"], 4),
        },
    }

    print("\n" + "=" * 60)
    print(f"COMPARATIVO — {SENSITIVE_ATTR}")
    print("=" * 60)
    print(f"  DIR baseline:  {base_report['disparate_impact_ratio']:+.4f}  "
          f"(rule 4/5: {'PASS' if base_report['rule_4_5_passes'] else 'FAIL'})")
    print(f"  DIR mitigado:  {mit_report['disparate_impact_ratio']:+.4f}  "
          f"(rule 4/5: {'PASS' if mit_report['rule_4_5_passes'] else 'FAIL'})")
    print(f"  Delta DIR:     {comparison['deltas']['DIR_delta']:+.4f}")
    print()
    print(f"  EOD baseline:  {base_report['equalized_odds_difference']:+.4f}")
    print(f"  EOD mitigado:  {mit_report['equalized_odds_difference']:+.4f}")
    print(f"  Delta EOD:     {comparison['deltas']['EOD_delta']:+.4f}")
    print("=" * 60)

    out = OUT_DIR / "fairness_mitigation_report.json"
    out.write_text(json.dumps(comparison, indent=2, ensure_ascii=False, default=str))
    print(f"\n>>> Relatorio salvo: {out}")

    # Persiste o mitigator pra usar no agente (opcional)
    import joblib
    joblib.dump(mitigator, OUT_DIR / "threshold_optimizer_gender.pkl")
    print(f">>> ThresholdOptimizer salvo: {OUT_DIR / 'threshold_optimizer_gender.pkl'}")

    return comparison


if __name__ == "__main__":
    run_mitigation()
