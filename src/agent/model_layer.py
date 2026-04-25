"""Carrega o LightGBM mais recente do MLflow e expoe predict_proba + SHAP."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import mlflow
import pandas as pd

MLRUNS_PATH = Path("mlruns").resolve().as_uri()
EXPERIMENT = "credit-copilot-baseline"


@lru_cache(maxsize=1)
def _load_latest_lgbm():
    """Pega o ultimo run com run_name lgbm_baseline."""
    mlflow.set_tracking_uri(MLRUNS_PATH)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        raise RuntimeError(f"Experimento '{EXPERIMENT}' nao encontrado.")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.mlflow.runName = 'lgbm_baseline'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("Nenhum run 'lgbm_baseline' no MLflow. Treine antes.")

    run = runs[0]
    model_uri = f"runs:/{run.info.run_id}/model"
    model = mlflow.lightgbm.load_model(model_uri)
    return model, run.info.run_id


def predict_pd(features: pd.DataFrame) -> dict:
    """Retorna probabilidade de default + tier de risco."""
    model, run_id = _load_latest_lgbm()

    # Garantir mesmas categorias que o treino esperava
    X = features.copy()
    for c in X.select_dtypes(include=["object"]).columns:
        X[c] = X[c].astype("category")

    proba = float(model.predict_proba(X)[0, 1])

    if proba < 0.10:
        tier = "BAIXO"
    elif proba < 0.20:
        tier = "MEDIO"
    elif proba < 0.40:
        tier = "ALTO"
    else:
        tier = "MUITO_ALTO"

    return {
        "pd": round(proba, 4),
        "risk_tier": tier,
        "model_run_id": run_id,
    }


def shap_top_features(features: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """Top N features que mais influenciaram a predicao."""
    import shap

    model, _ = _load_latest_lgbm()
    X = features.copy()
    for c in X.select_dtypes(include=["object"]).columns:
        X[c] = X[c].astype("category")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # binary: shap_values pode ser array ou lista de arrays
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    contribs = shap_values[0]  # primeira (e unica) row

    # Top contribuintes (positivos = aumentam risco)
    pairs = sorted(
        zip(X.columns, contribs, X.iloc[0].values),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_n]

    return [
        {
            "feature": name,
            "shap_value": round(float(val), 4),
            "feature_value": str(fval) if pd.notna(fval) else "missing",
            "direction": "aumenta_risco" if val > 0 else "reduz_risco",
        }
        for name, val, fval in pairs
    ]
