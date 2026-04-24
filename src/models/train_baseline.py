"""Treino de baselines LogReg (WoE) e LightGBM com MLflow tracking.

Uso:
    python -m src.models.train_baseline --model logreg
    python -m src.models.train_baseline --model lgbm
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from category_encoders import WOEEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.preprocess import load_and_prepare, split_features_target

warnings.filterwarnings("ignore")

MLRUNS_DIR = Path("mlruns")
RANDOM_STATE = 42


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov — métrica clássica de crédito."""
    from scipy.stats import ks_2samp

    score_pos = y_score[y_true == 1]
    score_neg = y_score[y_true == 0]
    return ks_2samp(score_pos, score_neg).statistic


def gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return 2 * roc_auc_score(y_true, y_score) - 1


def evaluate(y_true, y_score, prefix=""):
    return {
        f"{prefix}roc_auc": roc_auc_score(y_true, y_score),
        f"{prefix}gini": gini(y_true, y_score),
        f"{prefix}ks": ks_statistic(y_true, y_score),
        f"{prefix}pr_auc": average_precision_score(y_true, y_score),
    }


def train_logreg(X_train, X_valid, y_train, y_valid, cat_cols, num_cols):
    """LogReg com WoE nas categóricas + Imputer+Scaler nas numéricas."""
    # Encode categóricas com WoE
    woe = WOEEncoder(cols=cat_cols, random_state=RANDOM_STATE)
    X_train_enc = woe.fit_transform(X_train, y_train)
    X_valid_enc = woe.transform(X_valid)

    # Pipeline final: impute mediana + scale + logreg balanced
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2", C=0.1, solver="lbfgs",
            class_weight="balanced", max_iter=500,
            random_state=RANDOM_STATE,
        )),
    ])

    pipe.fit(X_train_enc, y_train)

    y_train_score = pipe.predict_proba(X_train_enc)[:, 1]
    y_valid_score = pipe.predict_proba(X_valid_enc)[:, 1]

    return pipe, woe, y_train_score, y_valid_score


def train_lgbm(X_train, X_valid, y_train, y_valid, cat_cols, num_cols):
    """LightGBM — categóricas como 'category' dtype, sem encoding."""
    X_train = X_train.copy()
    X_valid = X_valid.copy()
    for c in cat_cols:
        X_train[c] = X_train[c].astype("category")
        X_valid[c] = X_valid[c].astype("category")

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 100,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": RANDOM_STATE,
        "n_estimators": 1000,
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
    }

    model = lgb.LGBMClassifier(**params, verbose=-1)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    y_train_score = model.predict_proba(X_train)[:, 1]
    y_valid_score = model.predict_proba(X_valid)[:, 1]

    return model, None, y_train_score, y_valid_score


MODEL_REGISTRY = {
    "logreg": train_logreg,
    "lgbm": train_lgbm,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="lgbm")
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--enriched", action="store_true",
                        help="Usa features agregadas de todas as tabelas")
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLRUNS_DIR.resolve().as_uri())
    mlflow.set_experiment("credit-copilot-baseline")

    print(">>> Carregando e preparando dados...")
    if args.enriched:
        from src.features.preprocess import load_enriched
        print(">>> Carregando dataset enriquecido...")
        df = load_enriched()
    else:
        df = load_and_prepare()
    X, y, num_cols, cat_cols = split_features_target(df)
    print(f"    Shape: {X.shape} | Default rate: {y.mean():.4f}")
    print(f"    Num cols: {len(num_cols)} | Cat cols: {len(cat_cols)}")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=args.valid_size,
        stratify=y, random_state=RANDOM_STATE,
    )
    print(f"    Train: {X_train.shape[0]:,} | Valid: {X_valid.shape[0]:,}")

    trainer = MODEL_REGISTRY[args.model]

    with mlflow.start_run(run_name=f"{args.model}_baseline"):
        mlflow.log_params({
            "model": args.model,
            "valid_size": args.valid_size,
            "random_state": RANDOM_STATE,
            "n_train": len(X_train),
            "n_valid": len(X_valid),
            "n_features": X.shape[1],
            "default_rate_train": y_train.mean(),
        })

        print(f">>> Treinando {args.model}...")
        model, encoder, y_train_score, y_valid_score = trainer(
            X_train, X_valid, y_train, y_valid, cat_cols, num_cols,
        )

        train_metrics = evaluate(y_train.values, y_train_score, "train_")
        valid_metrics = evaluate(y_valid.values, y_valid_score, "valid_")

        mlflow.log_metrics({**train_metrics, **valid_metrics})

        print("\n>>> Resultados:")
        for k, v in {**train_metrics, **valid_metrics}.items():
            print(f"    {k:20s}: {v:.4f}")

        # Salva o modelo
        if args.model == "lgbm":
            mlflow.lightgbm.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        print("\n>>> Run salvo no MLflow. Para abrir UI:")
        print("    mlflow ui --backend-store-uri file://mlruns")


if __name__ == "__main__":
    main()
