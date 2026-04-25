"""Acesso aos dados do Home Credit por SK_ID_CURR. Carrega tudo na memoria 1x."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/raw")
PROCESSED = Path("data/processed")


@lru_cache(maxsize=1)
def _app() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "application_train.csv")


@lru_cache(maxsize=1)
def _bureau() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "bureau.csv")


@lru_cache(maxsize=1)
def _prev() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "previous_application.csv")


@lru_cache(maxsize=1)
def _enriched() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "application_enriched.parquet")


def get_applicant(sk_id: int) -> dict | None:
    df = _app()
    row = df[df["SK_ID_CURR"] == sk_id]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "sk_id_curr": int(sk_id),
        "gender": r["CODE_GENDER"],
        "age_years": round(-r["DAYS_BIRTH"] / 365, 1),
        "years_employed": round(-r["DAYS_EMPLOYED"] / 365, 1) if r["DAYS_EMPLOYED"] != 365243 else None,
        "income_total": float(r["AMT_INCOME_TOTAL"]),
        "credit_amount": float(r["AMT_CREDIT"]),
        "annuity": float(r["AMT_ANNUITY"]) if pd.notna(r["AMT_ANNUITY"]) else None,
        "goods_price": float(r["AMT_GOODS_PRICE"]) if pd.notna(r["AMT_GOODS_PRICE"]) else None,
        "contract_type": r["NAME_CONTRACT_TYPE"],
        "education": r["NAME_EDUCATION_TYPE"],
        "family_status": r["NAME_FAMILY_STATUS"],
        "housing_type": r["NAME_HOUSING_TYPE"],
        "ext_source_1": float(r["EXT_SOURCE_1"]) if pd.notna(r["EXT_SOURCE_1"]) else None,
        "ext_source_2": float(r["EXT_SOURCE_2"]) if pd.notna(r["EXT_SOURCE_2"]) else None,
        "ext_source_3": float(r["EXT_SOURCE_3"]) if pd.notna(r["EXT_SOURCE_3"]) else None,
    }


def get_bureau_history(sk_id: int) -> dict:
    df = _bureau()
    rows = df[df["SK_ID_CURR"] == sk_id]
    if rows.empty:
        return {"sk_id_curr": int(sk_id), "n_credits": 0, "message": "Sem historico bureau."}

    return {
        "sk_id_curr": int(sk_id),
        "n_credits": int(len(rows)),
        "n_active": int((rows["CREDIT_ACTIVE"] == "Active").sum()),
        "n_closed": int((rows["CREDIT_ACTIVE"] == "Closed").sum()),
        "total_credit_sum": float(rows["AMT_CREDIT_SUM"].sum()),
        "total_debt": float(rows["AMT_CREDIT_SUM_DEBT"].sum() if "AMT_CREDIT_SUM_DEBT" in rows else 0),
        "total_overdue": float(rows["AMT_CREDIT_SUM_OVERDUE"].sum() if "AMT_CREDIT_SUM_OVERDUE" in rows else 0),
        "credit_types": rows["CREDIT_TYPE"].value_counts().head(5).to_dict(),
        "oldest_credit_days_ago": int(-rows["DAYS_CREDIT"].min()),
        "n_prolongations": int(rows["CNT_CREDIT_PROLONG"].sum()),
    }


def get_internal_history(sk_id: int) -> dict:
    df = _prev()
    rows = df[df["SK_ID_CURR"] == sk_id]
    if rows.empty:
        return {"sk_id_curr": int(sk_id), "n_previous": 0, "message": "Cliente novo no Home Credit."}

    return {
        "sk_id_curr": int(sk_id),
        "n_previous": int(len(rows)),
        "n_approved": int((rows["NAME_CONTRACT_STATUS"] == "Approved").sum()),
        "n_refused": int((rows["NAME_CONTRACT_STATUS"] == "Refused").sum()),
        "approval_rate": round(float((rows["NAME_CONTRACT_STATUS"] == "Approved").mean()), 3),
        "total_credit_history": float(rows["AMT_CREDIT"].sum()),
        "avg_annuity": float(rows["AMT_ANNUITY"].mean() if "AMT_ANNUITY" in rows else 0),
        "contract_types": rows["NAME_CONTRACT_TYPE"].value_counts().to_dict(),
    }


def get_features_for_inference(sk_id: int) -> pd.DataFrame | None:
    """Retorna a row do dataset enriquecido pronta para o modelo."""
    df = _enriched()
    row = df[df["SK_ID_CURR"] == sk_id]
    if row.empty:
        return None
    return row.drop(columns=["TARGET", "SK_ID_CURR"])
