"""
백테스트 결과 시각화 — 기획서 별첨용 PNG 생성.

선행 작업:
    npm run backtest:run       # MA-12 / LSTM
    python scripts/backtest/arima.py

입력:
    reports/complexes.csv               단지 메타
    reports/series/{id}.csv             전체 시계열 (ym, price_per_m2, phase)
    reports/predictions/{id}_ma12.csv   MA-12 예측
    reports/predictions/{id}_lstm.csv   LSTM 예측
    reports/predictions/{id}_arima.csv  ARIMA 예측
    reports/backtest_results.csv        종합 메트릭

출력:
    reports/plots/{id}_forecast.png     단지별 예측 비교 (학습+예측+실제)
    reports/plots/comparison_mape.png   3개 모델 MAPE 막대그래프
    reports/plots/comparison_rmse.png   3개 모델 RMSE 막대그래프
    reports/plots/summary.png           종합 요약 (4-패널)

스타일:
    · MA-12: 회색 점선
    · ARIMA: 청색 파선
    · LSTM:  주황 실선
    · Actual: 흑색 굵은 선
    · Train: 옅은 회색 영역
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

# Windows cp949 콘솔에서 em-dash(—)·· 등 출력 시 UnicodeEncodeError 방지
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

matplotlib.use("Agg")  # 서버/CLI 환경에서 PNG 저장만
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 한글 폰트 (Windows: Malgun Gothic, mac: AppleGothic, linux: NanumGothic)
for font in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    try:
        matplotlib.rcParams["font.family"] = font
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
SERIES_DIR = REPORTS / "series"
PRED_DIR = REPORTS / "predictions"
PLOTS_DIR = REPORTS / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = REPORTS / "backtest_results.csv"
COMPLEXES_CSV = REPORTS / "complexes.csv"

MODEL_STYLE = {
    "MA-12":    {"color": "#9CA3AF", "linestyle": "--", "linewidth": 1.8, "label": "MA-12"},
    "ARIMA":    {"color": "#3B82F6", "linestyle": "-.", "linewidth": 1.8, "label": "ARIMA(2,1,2)"},
    "ARIMA-ND": {"color": "#06B6D4", "linestyle": ":",  "linewidth": 1.8, "label": "ARIMA(2,0,2) 차분제거"},
    "LSTM":         {"color": "#F97316", "linestyle": "-",  "linewidth": 2.2, "label": "LSTM (레벨·recursive)"},
    "LSTM-RET":     {"color": "#8B5CF6", "linestyle": "-",  "linewidth": 2.4, "label": "LSTM (수익률·recursive)"},
    "LSTM-DIR":     {"color": "#EAB308", "linestyle": "-",  "linewidth": 2.2, "label": "LSTM (레벨·direct)"},
    "LSTM-RET-DIR": {"color": "#EC4899", "linestyle": "-",  "linewidth": 2.6, "label": "LSTM (수익률·direct)"},
    "LSTM-REB":     {"color": "#10B981", "linestyle": "-",  "linewidth": 2.4, "label": "LSTM + R-ONE 정규화"},
}

# 그림 표시 순서 — 결과에 존재하는 모델만 자동 필터링됨(없으면 미표시).
ALL_MODELS = [
    "MA-12", "ARIMA", "ARIMA-ND",
    "LSTM", "LSTM-RET", "LSTM-DIR", "LSTM-RET-DIR", "LSTM-REB",
]


def plot_forecast(cid: int, name: str):
    """단지별 학습+예측+실제 라인 그래프"""
    series_csv = SERIES_DIR / f"{cid}.csv"
    if not series_csv.exists():
        print(f"  · [{cid}] series csv missing, skip")
        return

    series = pd.read_csv(series_csv)
    series["t"] = np.arange(len(series))

    train = series[series["phase"] == "train"]
    test = series[series["phase"] == "test"]

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # train 구간 옅게
    ax.plot(
        train["t"],
        train["price_per_m2"],
        color="#374151",
        linewidth=1.2,
        alpha=0.5,
        label="학습 구간 (실제)",
    )
    # test 실제값
    ax.plot(
        test["t"],
        test["price_per_m2"],
        color="#111827",
        linewidth=2.5,
        label="평가 구간 (실제)",
    )

    # 모델별 예측 오버레이
    test_t = test["t"].to_numpy()
    # 파일명 매핑: "MA-12" → "ma12", "LSTM-REB" → "lstm_reb"
    file_suffix = {
        "MA-12": "ma12",
        "ARIMA": "arima",
        "ARIMA-ND": "arima_nd",
        "LSTM": "lstm",
        "LSTM-RET": "lstm_ret",
        "LSTM-DIR": "lstm_dir",
        "LSTM-RET-DIR": "lstm_ret_dir",
        "LSTM-REB": "lstm_reb",
    }
    for model_key in ALL_MODELS:
        pred_csv = PRED_DIR / f"{cid}_{file_suffix[model_key]}.csv"
        if not pred_csv.exists():
            continue
        pred = pd.read_csv(pred_csv)
        ax.plot(
            test_t,
            pred["predicted"].to_numpy(dtype=float),
            **MODEL_STYLE[model_key],
        )

    # 분리선
    if len(test) > 0:
        ax.axvline(
            test_t[0] - 0.5, color="#EF4444", linestyle=":", linewidth=1.3, alpha=0.8,
        )
        ax.text(
            test_t[0],
            ax.get_ylim()[1] * 0.97,
            " hold-out 시작",
            color="#EF4444",
            fontsize=9,
            verticalalignment="top",
        )

    ax.set_title(
        f"{name} (id={cid}) — 36개월 가격 예측 백테스트",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("월 index (학습 구간 시작 = 0)")
    ax.set_ylabel("매매가 (만원/㎡)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # x축 라벨 — 12개월마다 ym 표기
    if len(series) > 0:
        tick_idx = list(range(0, len(series), 12))
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(
            [series.iloc[i]["ym"] for i in tick_idx], rotation=45, fontsize=8,
        )

    fig.tight_layout()
    out = PLOTS_DIR / f"{cid}_forecast.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  · saved {out.name}")


def plot_comparison(results: pd.DataFrame, metric: str, title: str, unit: str):
    """3개 모델 단지별 메트릭 막대그래프"""
    pivot = results.pivot_table(
        index="complex_id",
        columns="model",
        values=metric,
        aggfunc="first",
    )
    pivot = pivot[[c for c in ALL_MODELS if c in pivot.columns]]
    names = (
        results.drop_duplicates("complex_id")
        .set_index("complex_id")["name"]
        .reindex(pivot.index)
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(pivot.index))
    n_models = len(pivot.columns)
    # 모델 수에 따라 폭과 오프셋 조정 (3개 → 0.26, 4개 → 0.20)
    width = 0.78 / max(n_models, 1)
    center_offset = (n_models - 1) / 2
    for i, model in enumerate(pivot.columns):
        vals = pd.to_numeric(pivot[model], errors="coerce").to_numpy()
        ax.bar(
            x + (i - center_offset) * width,
            vals,
            width,
            color=MODEL_STYLE[model]["color"],
            label=MODEL_STYLE[model]["label"],
            edgecolor="#1F2937",
            linewidth=0.5,
        )
        # 값 라벨
        for xi, v in zip(x + (i - center_offset) * width, vals):
            if np.isfinite(v):
                ax.text(
                    xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8,
                )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("단지")
    ax.set_ylabel(f"{metric.upper()} ({unit})")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{n}\n(id={cid})" for cid, n in names.items()],
        fontsize=8,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = PLOTS_DIR / f"comparison_{metric}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  · saved {out.name}")


def plot_summary(results: pd.DataFrame):
    """4-패널 종합 — MAPE/RMSE/R² 평균 + 단지별 best 모델 분포"""
    models = [m for m in ALL_MODELS if m in results["model"].unique()]
    if len(models) == 0:
        return

    avgs = {}
    for m in models:
        sub = results[results["model"] == m]
        avgs[m] = {
            "mape": pd.to_numeric(sub["mape"], errors="coerce").mean(),
            "rmse": pd.to_numeric(sub["rmse"], errors="coerce").mean(),
            "r2": pd.to_numeric(sub["r2"], errors="coerce").mean(),
        }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = [MODEL_STYLE[m]["color"] for m in models]

    for ax, key, ylabel, fmt in [
        (axes[0], "mape", "평균 MAPE (%)  ↓ 낮을수록 좋음", "{:.2f}%"),
        (axes[1], "rmse", "평균 RMSE (만원/㎡)  ↓ 낮을수록 좋음", "{:.1f}"),
        (axes[2], "r2", "평균 R²  ↑ 1에 가까울수록 좋음", "{:.3f}"),
    ]:
        vals = [avgs[m][key] for m in models]
        ax.bar(models, vals, color=colors, edgecolor="#1F2937", linewidth=0.6)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(
                    i, v, fmt.format(v), ha="center", va="bottom",
                    fontsize=11, fontweight="bold",
                )
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Day 5 백테스트 종합 — 36개월 가격 예측 정확도 비교 (서울 상위 단지 평균)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    out = PLOTS_DIR / "summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  · saved {out.name}")


def main():
    if not COMPLEXES_CSV.exists() or not RESULTS_CSV.exists():
        print(
            f"[visualize] inputs missing. Run `npm run backtest:run` and "
            f"`python scripts/backtest/arima.py` first.",
        )
        sys.exit(1)

    complexes = pd.read_csv(COMPLEXES_CSV)
    results = pd.read_csv(RESULTS_CSV)

    # 1) 단지별 forecast 그래프
    print("[visualize] generating per-complex forecast plots...")
    for _, row in complexes.iterrows():
        plot_forecast(int(row["id"]), str(row["name"]))

    # 2) 모델 비교 막대그래프
    print("[visualize] generating comparison bars...")
    plot_comparison(results, "mape", "MAPE 비교 — 36개월 예측 정확도", "%")
    plot_comparison(results, "rmse", "RMSE 비교 — 36개월 예측 오차", "만원/㎡")

    # 3) 종합 요약
    print("[visualize] generating summary panel...")
    plot_summary(results)

    print(f"\n[visualize] DONE — plots in {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
