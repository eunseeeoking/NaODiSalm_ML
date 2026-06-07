"""
ARIMA(2,1,2) 백테스트 — Day 5 LSTM 비교 검증.

선행 작업:
    npm run backtest:run        # TS 측에서 reports/series/{id}_train.csv 생성

입력:
    reports/complexes.csv               단지 메타 (id, name, sigungu_code, ...)
    reports/series/{id}_train.csv       학습 시계열 (ym, price_per_m2)
    reports/series/{id}_test.csv        평가 시계열 (ym, price_per_m2)
    reports/backtest_results.csv        TS 측 MA-12 / LSTM 메트릭 (append 대상)

출력:
    reports/predictions/{id}_arima.csv  ARIMA 예측 (ym, predicted)
    reports/backtest_results.csv        ARIMA 행 추가 (기존 보존)

모델:
    ARIMA(2,1,2) — Box-Jenkins 표준 차분 1회 + AR(2) + MA(2)
    부동산 지수 시계열에 권장되는 차수.
    하이퍼파라미터 튜닝 X — 베이스라인으로 사용.

함정 방지:
    · ConvergenceWarning 다수 발생 가능 → warnings 묵음
    · 일부 단지 시계열이 잘 안 수렴 → fallback 으로 ARIMA(1,1,1) 시도
    · 둘 다 실패 시 NaN 행 출력 (skip 하지 않음 — 비교 표 행 정렬 유지)
"""
from __future__ import annotations

import csv
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Windows cp949 콘솔에서 em-dash(—)·R² 등 출력 시 UnicodeEncodeError 방지
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")  # convergence/MLE 경고 다수 → 묵음

# 실험 1 교차확인 — 차분 제거(d=0) ARIMA.
#   ARIMA_NO_DIFF=1 → order (2,0,2)/(1,0,1) 로 레벨 직접 모델링.
#   가설: 차분(d=1)이 ARIMA 우위의 진짜 이유라면, d=0 ARIMA 는 LSTM 쪽으로 무너진다.
#   기존 ARIMA 행을 덮지 않도록 model 라벨/예측 파일명을 분리(ARIMA-ND).
NO_DIFF = os.environ.get("ARIMA_NO_DIFF") == "1"
ORDERS = [(2, 0, 2), (1, 0, 1)] if NO_DIFF else [(2, 1, 2), (1, 1, 1)]
MODEL_LABEL = "ARIMA-ND" if NO_DIFF else "ARIMA"
PRED_SUFFIX = "arima_nd" if NO_DIFF else "arima"

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
SERIES_DIR = REPORTS / "series"
PRED_DIR = REPORTS / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = REPORTS / "backtest_results.csv"
COMPLEXES_CSV = REPORTS / "complexes.csv"


def calc_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """metrics.ts 와 동일한 정의."""
    mask = np.isfinite(actual) & np.isfinite(predicted)
    a = actual[mask]
    p = predicted[mask]
    n = len(a)
    if n == 0:
        return {"mape": np.nan, "rmse": np.nan, "r2": np.nan, "n": 0}
    # MAPE
    nz = np.abs(a) > 1e-9
    mape = float(np.mean(np.abs((p[nz] - a[nz]) / a[nz])) * 100) if nz.any() else np.nan
    # RMSE
    rmse = float(np.sqrt(np.mean((p - a) ** 2)))
    # R²
    if np.var(a) < 1e-12:
        r2 = np.nan
    else:
        ss_res = float(np.sum((a - p) ** 2))
        ss_tot = float(np.sum((a - np.mean(a)) ** 2))
        r2 = 1 - ss_res / ss_tot
    return {"mape": mape, "rmse": rmse, "r2": r2, "n": n}


def fit_arima(series: np.ndarray, horizon: int) -> np.ndarray | None:
    """ORDERS 순차 시도 (기본 (2,1,2)→(1,1,1), NO_DIFF 시 d=0) → 실패시 None"""
    for order in ORDERS:
        try:
            model = ARIMA(series, order=order)
            fit = model.fit(method_kwargs={"warn_convergence": False})
            forecast = fit.forecast(steps=horizon)
            return np.asarray(forecast, dtype=float)
        except Exception as e:
            print(f"    ARIMA{order} failed: {e}")
            continue
    return None


def main():
    if not COMPLEXES_CSV.exists():
        print(f"[arima] {COMPLEXES_CSV} not found. Run `npm run backtest:run` first.")
        sys.exit(1)

    complexes = pd.read_csv(COMPLEXES_CSV)
    print(f"[arima] {len(complexes)} complexes to process")

    new_rows: list[dict] = []

    for _, row in complexes.iterrows():
        cid = int(row["id"])
        name = str(row["name"])
        sigungu = str(row["sigungu_code"])
        dong = str(row["legal_dong"])

        train_csv = SERIES_DIR / f"{cid}_train.csv"
        test_csv = SERIES_DIR / f"{cid}_test.csv"
        if not train_csv.exists() or not test_csv.exists():
            print(f"  · [{cid}] {name} — missing train/test csv, skip")
            continue

        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        train_series = train_df["price_per_m2"].to_numpy(dtype=float)
        test_actual = test_df["price_per_m2"].to_numpy(dtype=float)
        horizon = len(test_actual)

        print(
            f"  · [{cid}] {name} train={len(train_series)} horizon={horizon}",
        )

        pred = fit_arima(train_series, horizon)
        if pred is None:
            print(f"    [{MODEL_LABEL}] {ORDERS[0]} and {ORDERS[1]} failed → NaN row")
            metrics = {"mape": np.nan, "rmse": np.nan, "r2": np.nan, "n": 0}
            pred_filled = [np.nan] * horizon
        else:
            metrics = calc_metrics(test_actual, pred)
            pred_filled = pred.tolist()
            print(
                f"    [{MODEL_LABEL}] MAPE={metrics['mape']:.2f}% "
                f"RMSE={metrics['rmse']:.2f} R²={metrics['r2']:.3f}",
            )

        # predictions CSV
        pred_csv = PRED_DIR / f"{cid}_{PRED_SUFFIX}.csv"
        with pred_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ym", "predicted"])
            for ym, v in zip(test_df["ym"].tolist(), pred_filled):
                w.writerow(
                    [ym, f"{v:.4f}" if np.isfinite(v) else "NaN"],
                )

        new_rows.append(
            {
                "complex_id": cid,
                "name": name,
                "sigungu_code": sigungu,
                "legal_dong": dong,
                "model": MODEL_LABEL,
                "horizon": horizon,
                "mape": f"{metrics['mape']:.4f}"
                if np.isfinite(metrics["mape"])
                else "NaN",
                "rmse": f"{metrics['rmse']:.4f}"
                if np.isfinite(metrics["rmse"])
                else "NaN",
                "r2": f"{metrics['r2']:.4f}"
                if np.isfinite(metrics["r2"])
                else "NaN",
                "n": metrics["n"],
            },
        )

    # backtest_results.csv 에 ARIMA 행 append (기존 row 보존)
    if RESULTS_CSV.exists():
        existing = pd.read_csv(RESULTS_CSV)
        # 같은 (complex_id, model) 중복 제거 — 재실행 멱등성
        existing = existing[existing["model"] != MODEL_LABEL]
        merged = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        merged = pd.DataFrame(new_rows)

    merged.to_csv(RESULTS_CSV, index=False, encoding="utf-8")

    # 요약
    print(f"\n[arima] DONE — {len(new_rows)} rows appended to {RESULTS_CSV}\n")
    for model in merged["model"].unique():
        sub = merged[merged["model"] == model]
        valid = sub[pd.to_numeric(sub["mape"], errors="coerce").notna()]
        if len(valid) > 0:
            mape_avg = pd.to_numeric(valid["mape"], errors="coerce").mean()
            rmse_avg = pd.to_numeric(valid["rmse"], errors="coerce").mean()
            r2_avg = pd.to_numeric(valid["r2"], errors="coerce").mean()
            print(
                f"  {model:<8} avg MAPE={mape_avg:.3f}% "
                f"RMSE={rmse_avg:.2f} R²={r2_avg:.3f} (n={len(valid)})",
            )


if __name__ == "__main__":
    main()
