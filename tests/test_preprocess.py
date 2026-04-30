"""Testa que preprocess engineered features estao corretas."""

import pandas as pd

from src.features.preprocess import add_engineered_features


def _base_row():
    return pd.DataFrame(
        {
            "AMT_CREDIT": [100000.0],
            "AMT_INCOME_TOTAL": [50000.0],
            "AMT_ANNUITY": [5000.0],
            "DAYS_EMPLOYED": [-365],
            "DAYS_BIRTH": [-12000],
            "EXT_SOURCE_1": [0.5],
            "EXT_SOURCE_2": [0.6],
            "EXT_SOURCE_3": [0.7],
        }
    )


def test_credit_income_ratio():
    df = _base_row()
    out = add_engineered_features(df)
    assert out.loc[0, "CREDIT_INCOME_RATIO"] == 2.0


def test_days_employed_anomaly_flag():
    df = _base_row()
    df.loc[0, "DAYS_EMPLOYED"] = 365243
    out = add_engineered_features(df)
    assert out.loc[0, "DAYS_EMPLOYED_ANOM"] == 1


def test_ext_source_missing_flag():
    df = _base_row()
    df.loc[0, "EXT_SOURCE_1"] = None
    out = add_engineered_features(df)
    assert out.loc[0, "EXT_SOURCE_1_MISSING"] == 1
    assert out.loc[0, "EXT_SOURCE_2_MISSING"] == 0
