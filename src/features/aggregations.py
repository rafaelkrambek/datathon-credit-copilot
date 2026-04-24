"""Agregações das 6 tabelas auxiliares por SK_ID_CURR.

Estratégia:
- bureau + bureau_balance → histórico externo de crédito
- previous_application → histórico interno (Home Credit)
- installments_payments → comportamento de pagamento
- POS_CASH_balance + credit_card_balance → saldos atuais

Métricas por tabela: count, mean, sum, max, min de colunas numéricas + moda de categóricas.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/raw")
ID = "SK_ID_CURR"


def agg_bureau() -> pd.DataFrame:
    """Agrega bureau e bureau_balance por SK_ID_CURR."""
    bureau = pd.read_csv(DATA_DIR / "bureau.csv")
    bb = pd.read_csv(DATA_DIR / "bureau_balance.csv")

    # bureau_balance: status por mês → resume por SK_ID_BUREAU
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS=("MONTHS_BALANCE", "size"),
        BB_MONTHS_MIN=("MONTHS_BALANCE", "min"),
        BB_STATUS_NUNIQUE=("STATUS", "nunique"),
    ).reset_index()

    bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")

    num_cols = ["DAYS_CREDIT", "DAYS_CREDIT_ENDDATE", "AMT_CREDIT_SUM",
                "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE",
                "CNT_CREDIT_PROLONG", "BB_MONTHS", "BB_MONTHS_MIN"]

    agg = bureau.groupby(ID).agg(
        BUREAU_COUNT=(ID, "size"),
        BUREAU_ACTIVE_COUNT=("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
        BUREAU_CLOSED_COUNT=("CREDIT_ACTIVE", lambda x: (x == "Closed").sum()),
        **{f"BUREAU_{c}_MEAN": (c, "mean") for c in num_cols if c in bureau.columns},
        **{f"BUREAU_{c}_MAX": (c, "max") for c in num_cols if c in bureau.columns},
        **{f"BUREAU_{c}_SUM": (c, "sum") for c in num_cols if c in bureau.columns},
    ).reset_index()

    return agg


def agg_previous_application() -> pd.DataFrame:
    prev = pd.read_csv(DATA_DIR / "previous_application.csv")

    num_cols = ["AMT_APPLICATION", "AMT_CREDIT", "AMT_ANNUITY",
                "DAYS_DECISION", "CNT_PAYMENT"]

    agg = prev.groupby(ID).agg(
        PREV_COUNT=(ID, "size"),
        PREV_APPROVED_COUNT=("NAME_CONTRACT_STATUS", lambda x: (x == "Approved").sum()),
        PREV_REFUSED_COUNT=("NAME_CONTRACT_STATUS", lambda x: (x == "Refused").sum()),
        **{f"PREV_{c}_MEAN": (c, "mean") for c in num_cols if c in prev.columns},
        **{f"PREV_{c}_MAX": (c, "max") for c in num_cols if c in prev.columns},
    ).reset_index()

    agg["PREV_APPROVAL_RATE"] = agg["PREV_APPROVED_COUNT"] / agg["PREV_COUNT"].replace(0, np.nan)

    return agg


def agg_installments() -> pd.DataFrame:
    inst = pd.read_csv(DATA_DIR / "installments_payments.csv")

    inst["DAYS_LATE"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
    inst["AMT_SHORT"] = inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]

    agg = inst.groupby(ID).agg(
        INST_COUNT=(ID, "size"),
        INST_DAYS_LATE_MEAN=("DAYS_LATE", "mean"),
        INST_DAYS_LATE_MAX=("DAYS_LATE", "max"),
        INST_AMT_SHORT_MEAN=("AMT_SHORT", "mean"),
        INST_AMT_SHORT_SUM=("AMT_SHORT", "sum"),
        INST_AMT_PAYMENT_MEAN=("AMT_PAYMENT", "mean"),
    ).reset_index()

    return agg


def agg_pos_cash() -> pd.DataFrame:
    pos = pd.read_csv(DATA_DIR / "POS_CASH_balance.csv")

    agg = pos.groupby(ID).agg(
        POS_COUNT=(ID, "size"),
        POS_MONTHS_BALANCE_MEAN=("MONTHS_BALANCE", "mean"),
        POS_SK_DPD_MEAN=("SK_DPD", "mean"),
        POS_SK_DPD_MAX=("SK_DPD", "max"),
    ).reset_index()

    return agg


def agg_credit_card() -> pd.DataFrame:
    cc = pd.read_csv(DATA_DIR / "credit_card_balance.csv")

    agg = cc.groupby(ID).agg(
        CC_COUNT=(ID, "size"),
        CC_AMT_BALANCE_MEAN=("AMT_BALANCE", "mean"),
        CC_AMT_BALANCE_MAX=("AMT_BALANCE", "max"),
        CC_SK_DPD_MAX=("SK_DPD", "max"),
        CC_AMT_PAYMENT_MEAN=("AMT_PAYMENT_CURRENT", "mean"),
    ).reset_index()

    return agg


def build_full_feature_set() -> pd.DataFrame:
    """Carrega application_train e faz merge com todas as agregações."""
    from src.features.preprocess import load_and_prepare

    print(">>> Carregando application_train...")
    df = load_and_prepare()

    for name, fn in [
        ("bureau", agg_bureau),
        ("previous_application", agg_previous_application),
        ("installments", agg_installments),
        ("pos_cash", agg_pos_cash),
        ("credit_card", agg_credit_card),
    ]:
        print(f">>> Agregando {name}...")
        agg = fn()
        df = df.merge(agg, on=ID, how="left")
        print(f"    +{agg.shape[1] - 1} features | shape agora: {df.shape}")

    return df


if __name__ == "__main__":
    df = build_full_feature_set()
    print(f"\n>>> Shape final: {df.shape}")
    print(f">>> Colunas totais: {df.shape[1]}")
    out = Path("data/processed/application_enriched.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f">>> Salvo em: {out}")
