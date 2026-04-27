"""Drift detection minimalista via KS test."""

import json
from pathlib import Path

from scipy.stats import ks_2samp

from src.features.preprocess import load_enriched

OUT = Path("evaluation/drift")
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
]


def main():
    df = load_enriched()
    ref = df.sample(30000, random_state=42)
    cur = df.sample(30000, random_state=99).copy()

    # Drift artificial pra demo
    cur["AMT_INCOME_TOTAL"] *= 1.15
    cur["EXT_SOURCE_2"] *= 0.92

    results = {}
    for f in FEATURES:
        if f not in df.columns:
            continue
        stat, p = ks_2samp(ref[f].dropna(), cur[f].dropna())
        results[f] = {
            "ks_stat": round(float(stat), 4),
            "p_value": round(float(p), 6),
            "drift_detected": bool(p < 0.05),
        }

    drifted = [f for f, r in results.items() if r["drift_detected"]]
    summary = {
        "n_features": len(results),
        "n_drifted": len(drifted),
        "drifted_features": drifted,
        "results": results,
    }

    out = OUT / "drift.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Features monitoradas: {len(results)}")
    print(f"Com drift detected:   {len(drifted)} -> {drifted}")
    print(f"Salvo em: {out}")


if __name__ == "__main__":
    main()
