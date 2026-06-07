# Day 5 백테스트 — LSTM 비교 검증

> 7day-roadmap.md §DAY 5 "구체성 결정타" 구현.
> 3개 모델 × 5단지 × 36개월 hold-out 백테스트.

## 모델

| 모델 | 위치 | 비고 |
|------|------|------|
| MA-12 | `ma12.ts` | 12개월 이동평균 + 선형 트렌드 외삽 |
| ARIMA(2,1,2) | `scripts/backtest/arima.py` | statsmodels (Python) |
| LSTM | `lstmEval.ts` | TensorFlow.js, 레벨·recursive multi-step (horizon=1 학습 후 36회 반복) |
| **LSTM-REB** ★ | `lstmEval.ts` + `rebNormalize.ts` | LSTM + R-ONE 지수 정규화 (시장 추세 노이즈 제거) — `BACKTEST_REB_NORMALIZE=1` 토글 |
| LSTM-RET | `lstmEval.ts` `predictLstmReturns` | 실험 1 — 로그수익률 표현(차분), recursive — `BACKTEST_RETURNS_SPACE=1` |
| LSTM-DIR / LSTM-RET-DIR | `lstmEval.ts` `predictLstmDirect` | 실험 3 — direct multi-horizon(recursive 제거), 레벨·수익률 양쪽 — `BACKTEST_DIRECT=1` |
| ARIMA-ND | `scripts/backtest/arima.py` | 교차검증 — 차분 제거 ARIMA(2,0,2) — `ARIMA_NO_DIFF=1` |

> 2×2(표현 × 예측방식) 실험의 결론·수치는 루트 [`README.md`](../../README.md) 의 “LSTM은 왜 졌나” 참조.

## 실행 흐름

```powershell
# 1) TS 측 — 5단지 자동 선정 + MA-12 + LSTM + CSV dump
cd C:\git\NaODiSalm_ML
npm run backtest:run

# 2) Python — ARIMA(2,1,2) 추가
pip install -r scripts/backtest/requirements.txt
npm run backtest:arima

# 3) Python — 시각화 PNG 생성
npm run backtest:visualize

# 또는 한꺼번에
npm run backtest:all
```

## 출력

```
reports/
  complexes.csv               백테스트 단지 5개 메타
  series/<id>.csv             전체 시계열 (학습+평가)
  series/<id>_train.csv       학습 구간 (Python 입력용)
  series/<id>_test.csv        평가 구간 (Python 입력용)
  predictions/<id>_ma12.csv   MA-12 예측 (ym, predicted)
  predictions/<id>_arima.csv  ARIMA 예측
  predictions/<id>_lstm.csv   LSTM 예측
  backtest_results.csv        종합 메트릭 (model × complex × MAPE/RMSE/R²)
  plots/<id>_forecast.png     단지별 라인그래프 (학습+예측+실제)
  plots/comparison_mape.png   3모델 MAPE 막대그래프
  plots/comparison_rmse.png   3모델 RMSE 막대그래프
  plots/summary.png           4-패널 종합 요약
```

## 환경변수

```
BACKTEST_TOP_N=5         단지 수 (기본 5)
BACKTEST_HORIZON=24      평가 horizon 개월 (기본 24, 커밋 산출물은 36)
BACKTEST_MIN_TRAIN=36    최소 train 개월 (기본 36)
BACKTEST_LSTM_EPOCHS=30  LSTM epochs (기본 30)
BACKTEST_REB_NORMALIZE=1 R-ONE 정규화 LSTM 추가 평가 (기본 off)
                         → LSTM-REB 행이 backtest_results.csv 에 추가
                         → 단지별 forecast.png 에 초록 라인 (#10B981) 추가
                         → comparison_*.png 에 4번째 막대 추가
                         → 예측 흐름: train series ÷ R-ONE index → z-score → LSTM
                                    → z 복원 → × (train 마지막 ym index / 100)
                         → 정규화 효과 측정: LSTM vs LSTM-REB MAPE 직접 비교
BACKTEST_RETURNS_SPACE=1 실험 1 — 로그수익률 표현 LSTM-RET 행 추가 (기본 off)
BACKTEST_DIRECT=1        실험 3 — direct multi-horizon LSTM-DIR·LSTM-RET-DIR 행 추가 (기본 off)
```

> 2×2 통제실험은 `BACKTEST_RETURNS_SPACE=1 BACKTEST_DIRECT=1` 동시 지정 시
> LSTM(레벨·rec)·LSTM-RET(수익률·rec)·LSTM-DIR(레벨·dir)·LSTM-RET-DIR(수익률·dir)
> 4행이 함께 적재되어 {표현}×{예측방식} 비교가 완성된다. 결론은 루트 README 참조.

## 평가 지표

| 지표 | 정의 | 기준 |
|------|------|------|
| MAPE | mean(\|a-p\|/\|a\|) × 100 | ↓ 낮을수록 좋음 (%) |
| RMSE | sqrt(mean((a-p)²)) | ↓ 낮을수록 좋음 (만원/㎡) |
| R² | 1 - SS_res/SS_tot | ↑ 1에 가까울수록 좋음 (음수=평균예측보다 못함) |

## 함정 (반드시 인지)

1. **LSTM recursive multi-step** — horizon=1 학습 후 36회 반복 호출. 누적 오차로 끝쪽 외삽 폭주 가능. 이게 백테스트의 "정직한" 결과.
2. **ARIMA 수렴 실패** — 일부 단지는 (2,1,2) 미수렴 → 자동 (1,1,1) fallback → 둘 다 실패 시 NaN row.
3. **재현성** — 같은 DB 상태에서 재실행 결과 동일. LSTM은 random init 영향으로 약간 변동 (±2% MAPE 정도).
