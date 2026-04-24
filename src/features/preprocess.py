"""Pré-processamento do application_train.csv para baselines."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/raw")

# Colunas com alto % de missing que serão DROPADAS
# (identificadas na EDA — XXX_AVG/MEDI/MODE de prédios, >45% missing)
DROP_BUILDING_COLS = True

# Features engineered simples
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona razões financeiras e flags de missingness."""
    df = df.copy()
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["CREDIT_TERM"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"].replace(0, np.nan)
    df["DAYS_EMPLOYED_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"].replace(0, np.nan)

    # Flags de missingness (EDA mostrou que são preditivas)
    for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        df[f"{c}_MISSING"] = df[c].isnull().astype(int)

    # Feature composta com média dos EXT_SOURCE
    df["EXT_SOURCE_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)

    # Fix do DAYS_EMPLOYED (há outlier 365243 = "nunca trabalhou")
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df.loc[df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan

    return df


def load_and_prepare(path: Path = DATA_DIR / "application_train.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    if DROP_BUILDING_COLS:
        # Remove colunas de características do prédio (XXX_AVG/MEDI/MODE)
        building_cols = [c for c in df.columns if any(
            s in c for s in ["_AVG", "_MEDI", "_MODE"]
        )]
        df = df.drop(columns=building_cols)

    df = add_engineered_features(df)

    return df


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Retorna X, y, lista de colunas numéricas e categóricas."""
    y = df["TARGET"]
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    return X, y, num_cols, cat_cols


if __name__ == "__main__":
    df = load_and_prepare()
    X, y, num_cols, cat_cols = split_features_target(df)
    print(f"Shape após preprocess: {df.shape}")
    print(f"Num features numéricas: {len(num_cols)}")
    print(f"Num features categóricas: {len(cat_cols)}")
    print(f"Default rate: {y.mean():.4f}")


def load_enriched(path: Path = Path("data/processed/application_enriched.parquet")) -> pd.DataFrame:
    """Carrega a versão enriquecida (application + agregações)."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não existe. Rode `python -m src.features.aggregations` primeiro."
        )
    return pd.read_parquet(path)
