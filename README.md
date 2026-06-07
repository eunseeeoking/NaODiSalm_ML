# 나어디삶 ML — 가격 예측 백테스트 & 학습 파이프라인

**나어디삶**([데이터 기반 청년 주거 의사결정 플랫폼](https://github.com/eunseeeoking/NaODiSalm_Main))의 가격 예측 모델 **검증·학습 레포** (로컬 전용).

같은 MySQL DB(`molit_contest`)의 실거래 시계열을 사용하며, [NaODiSalm_Main](https://github.com/eunseeeoking/NaODiSalm_Main) (Vercel/Render 배포) 과 분리된 별도 레포입니다 — 클라우드 무료 호스팅의 메모리/CPU 한계를 피해 학습·백테스트를 로컬 PC 에서 수행합니다.

## 이 레포가 하는 일

| 갈래 | 위치 | 위상 |
|---|---|---|
| **① 4-모델 백테스트** (MA-12 · ARIMA(2,1,2) · LSTM · LSTM-REB) | `src/backtest/` + `scripts/backtest/*.py` | ★ **현재 핵심** — 모델 선택의 정량 증빙 |
| ② LSTM 라이브 학습 → `t_training_result` | `src/train.ts` + `models/lstm.ts` | **레거시 보조** (아래 참조) |

> ### 📌 모델 정책 — ARIMA(2,1,2) 채택, LSTM 보조
>
> 백테스트(①) 결과 **ARIMA(2,1,2)가 LSTM 대비 절반 오차**(MAPE 10.16% vs 20.41%)로 우세 → 라이브 Depth 3 가격 분석의 **메인 모델로 ARIMA 채택**.
>
> 이에 따라 LSTM 라이브 학습(②)이 적재하던 `t_training_result` z-score 방식은 **2026-05-25 프로덕션에서 폐기**(최고가 단지에서 -27%~-35% 왜곡 발생). 라이브 ARIMA 분석은 Main 서버(`server/src/routes/domains/arima.ts`)가 직접 산출합니다.
>
> LSTM 학습 코드(②)는 **백테스트 비교군·증빙 목적으로 유지**하며 라이브 서비스는 소비하지 않습니다. **LSTM 의 성능 저조는 모델 결함도 데이터 부족도 아니라 표현(representation)·구조 문제**임을 2×2 통제실험으로 규명했습니다(아래 [LSTM은 왜 졌나](#why-lstm-lost) 참조) — 가격 *레벨* 대신 *로그수익률* 을 모델링하고 recursive multi-step 을 direct 로 바꾸면 LSTM 이 ARIMA 와 같은 리그(평균 9.6% vs 7.7%)로 진입합니다. 그래도 ARIMA 가 약간 앞서고 단순·결정적이라 프로덕션은 ARIMA 유지가 맞습니다.

## 구조

```
src/
├── db.ts                            # PrismaClient 싱글톤
├── train.ts                         # [레거시] LSTM 학습 entry (npm run train)
├── data/
│   ├── fetch.ts                     # 단지 선별 + 거래 시계열 조회
│   ├── preprocess.ts                # m²당 단가, 월별 집계, IQR, train examples
│   └── rebNormalize.ts              # R-ONE 지수 정규화 (LSTM-REB 용)
├── models/
│   └── lstm.ts                      # LSTM 모델 정의/학습/예측
├── backtest/                        # ★ 현재 핵심 — 4모델 hold-out 비교
│   ├── run.ts                       # 통합 러너 (npm run backtest:run)
│   ├── selectComplexes.ts           # 거래량 상위 N단지 자동 선정
│   ├── holdout.ts                   # train/test 분할
│   ├── ma12.ts                      # MA-12 베이스라인 (12개월 이동평균 + 트렌드)
│   ├── lstmEval.ts                  # LSTM recursive multi-step (+ REB 정규화)
│   └── metrics.ts                   # MAPE / RMSE / R²
├── repository/
│   └── trainingResultRepository.ts  # t_training_result upsert
├── scripts/
│   ├── trainStats.ts                # 학습 결과 통계 (npm run train:stats)
│   └── backfillConfidence.ts        # confidence NULL 백필 (npm run train:backfill)
└── utils/
    └── confidence.ts                # MAPE 기반 confidence 산출
scripts/backtest/
├── arima.py                         # ARIMA(2,1,2) statsmodels (npm run backtest:arima)
├── visualize.py                     # 시각화 PNG 생성 (npm run backtest:visualize)
└── requirements.txt
prisma/
└── schema.prisma                    # DB 스키마 (server 와 같은 DB, TrainingResult)
reports/                             # 백테스트 산출물 — PNG 커밋, CSV 는 재실행 재생성
```

## 셋업 (1회)

```bash
# 1) 의존성 설치 — TensorFlow.js (pure JS, @tensorflow/tfjs)
npm install

# 2) .env 작성
copy .env.example .env
notepad .env
#   DATABASE_URL 을 server/.env 와 동일하게

# 3) (LSTM 학습을 돌릴 때만) t_training_result 테이블 생성
#    기존 t_apt_complex / t_apt_trade / t_apt_rent 는 server 가 owner — 영향 없음
npm run prisma:push
```

> 백테스트(①)에는 `t_training_result` 가 필요 없습니다 — read-only 평가 잡입니다.

---

## ① 백테스트 — ARIMA 채택의 정량 증빙

`reports/` 에 채점위원·외부 검증자가 재현 가능한 산출물을 패키징합니다. PNG 는 레포에 커밋(기획서 별첨용), raw CSV 는 gitignore 이며 백테스트 재실행으로 재생성됩니다.

### 종합 비교 — 3년(36개월) horizon, 서울 상위 5단지

![백테스트 종합 — MA-12·ARIMA·LSTM 비교](reports/plots/summary.png)

| 모델 | MAPE ↓ | RMSE ↓ (만원/m²) | 비고 |
|---|---|---|---|
| **ARIMA(2,1,2)** | **10.16%** | **159.7** | ✅ 메인 채택 — multi-step 누적오차 없이 LSTM 절반 |
| MA-12 (베이스라인) | 10.88% | 165.4 | 단순 12개월 이동평균, 의외로 견고 |
| LSTM (window=24) | 20.41% | 283.7 | 레벨·recursive 의 한계 — [왜 졌나](#why-lstm-lost)에서 규명 |
| LSTM-REB | (선택) | — | LSTM + R-ONE 지수 정규화, `BACKTEST_REB_NORMALIZE=1` 토글 |

> _R² 가 음수인 것은 3년(36개월) horizon multi-step 누적 평가 특성상 정상이며, 실용 지표는 MAPE/RMSE._
> _위 표·PNG 는 Day 5 시점(2025-04 데이터)의 4모델 비교. LSTM 패배의 **원인**은 아래 2×2 후속 실험에서 분리·규명했습니다._

### 재현 방법

```bash
# 0) Python 의존성 (statsmodels)
pip install -r scripts/backtest/requirements.txt

# 1) TS — 5단지 자동 선정 + MA-12 + LSTM + CSV dump
#    커밋된 reports 는 horizon=36 으로 생성 — 동일 재현 시 BACKTEST_HORIZON=36 명시
BACKTEST_HORIZON=36 npm run backtest:run

# 2) Python — ARIMA(2,1,2) 행 추가
npm run backtest:arima

# 3) Python — 시각화 PNG 재생성
npm run backtest:visualize

# 또는 1~3 한꺼번에
BACKTEST_HORIZON=36 npm run backtest:all
```

> `run.ts` 의 `BACKTEST_HORIZON` 기본값은 24 (데이터 범위 ~64개월 대응). 커밋된 PNG/표는 36 으로 생성됐으므로 동일 결과를 보려면 위처럼 36 을 지정하세요. 백테스트는 학습과 분리된 read-only 평가 — DB `t_training_result` 에 영향 없음.

### 백테스트 환경변수

| 변수 | 기본 | 설명 |
|---|---|---|
| `BACKTEST_TOP_N` | 5 | 평가 단지 수 |
| `BACKTEST_HORIZON` | 24 | 평가 horizon (개월) — 커밋 산출물은 36 |
| `BACKTEST_MIN_TRAIN` | 36 | 최소 train 개월 |
| `BACKTEST_LSTM_EPOCHS` | 30 | LSTM epochs |
| `BACKTEST_REB_NORMALIZE` | off | `=1` 시 LSTM-REB(R-ONE 정규화) 행 추가 |
| `BACKTEST_RETURNS_SPACE` | off | `=1` 시 LSTM-RET(로그수익률 표현) 행 추가 — 실험 1 |
| `BACKTEST_DIRECT` | off | `=1` 시 LSTM-DIR·LSTM-RET-DIR(direct multi-horizon) 행 추가 — 실험 3 |

> ARIMA 쪽 교차검증: `ARIMA_NO_DIFF=1 npm run backtest:arima` → 차분 제거 ARIMA(2,0,2) 를 `ARIMA-ND` 라벨로 추가(기존 ARIMA 행 보존).

자세한 모델·함정 노트는 [`src/backtest/README.md`](src/backtest/README.md) 참조.

<a id="why-lstm-lost"></a>

### 실험 — LSTM은 왜 졌나 (2×2 통제실험)

> _2026-06-06 후속 실험. 위 Day 5 표와 **데이터셋·split 이 다르므로 수치를 교차비교하지 말 것**(이 실험 내부는 전 모델 동일 holdout 이라 valid). 같은 5단지, horizon=36._

**가설:** LSTM 의 패배는 "데이터 부족"이 아니라 **표현 문제**다. ARIMA·LSTM **둘 다 외생데이터 0개**(과거 가격만)인데 ARIMA 가 2배 이겼다 → 정보가 아니라 표현의 차이다. ARIMA(2,1,**2**)의 차분(d=1)은 추세를 제거하지만, LSTM 은 가격 *레벨* 을 z-score 만 해서 비정상성에 휘둘린다.

두 축을 독립 토글로 분리해 2×2 로 측정 (MAPE 평균 / 단지간 표준편차):

| | recursive (기존) | direct multi-horizon |
|---|---|---|
| **레벨 (기존)** | 20.6% / σ3.6 | 30.0% / σ8.4 |
| **로그수익률** | 17.2% / σ**19.4** 💥 | **9.6% / σ4.6** ✅ |

_참고: 같은 split 의 ARIMA(2,1,2) = **7.7%**, 차분 제거 ARIMA(2,0,2) = 9.5%(악화)._

**규명된 3가지:**

1. **표현(차분)이 결정적** — 로그수익률이 레벨을 recursive(17.2 vs 20.6)·direct(9.6 vs 30.0) 양쪽에서 이김. ARIMA 쪽에서도 차분 제거 시 7.7%→9.5% 악화로 교차확인.
2. **recursive multi-step 이 fat-tail 의 주범** — 로그수익률·recursive 는 평균은 좋아도 한 단지(SK북한산)가 51%로 폭발해 **σ가 19.4**. direct 로 바꾸니 그 폭발이 사라지고 **σ 4.6 으로 붕괴**.
3. **두 수정은 상호작용 — 같이 가야 함** — direct 는 *수익률 공간에서만* 효과. 레벨 공간에선 direct 가 오히려 악화(20.6→30.0, 36개 절대 추세값을 한 번에 뱉어야 하므로). recursive 구조가 레벨 표현의 결함을 가리고 있었다.

**결론:** 표현(레벨→로그수익률) + 구조(recursive→direct) 를 둘 다 고치면 LSTM 은 평균 **9.6%·σ4.6** 으로 ARIMA(7.7%) 와 같은 리그에 진입한다. 다만 *이기진* 못하고 ARIMA 가 더 단순·결정적이라 **프로덕션은 ARIMA 유지**. 이 실험의 가치는 모델 교체가 아니라, "LSTM 을 왜 안 썼나"를 **"LSTM 이 왜 졌는지 정확히 안다"**로 바꾸는 데 있다.

> **재현:** `BACKTEST_HORIZON=36 BACKTEST_RETURNS_SPACE=1 BACKTEST_DIRECT=1 npm run backtest:run` → LSTM·LSTM-RET·LSTM-DIR·LSTM-RET-DIR 4행이 `backtest_results.csv` 에 함께 적재.
> **한계:** 단일 seed — direct 의 복리누적 제거는 메커니즘적으로 당연하나, SK북한산 안정화의 seed 의존성은 다회 평균으로 보강 권장.

### 단지별 forecast 곡선

<details><summary><b>표본 5단지 × forecast PNG 펼쳐 보기</b></summary>

| 단지 | 자치구·동 | 거래 건수 | id | 곡선 |
|---|---|---|---|---|
| 파크리오 | 송파 신천동 | 4,943 | 6902 | ![파크리오](reports/plots/6902_forecast.png) |
| SK북한산시티 | 강북 미아동 | 4,772 | 1313 | ![SK북한산시티](reports/plots/1313_forecast.png) |
| 중계그린1단지 | 노원 중계동 | 4,028 | 3734 | ![중계그린1단지](reports/plots/3734_forecast.png) |
| 선사현대 | 강동 암사동 | 3,556 | 739 | ![선사현대](reports/plots/739_forecast.png) |
| 신동아1 | 도봉 방학동 | 3,543 | 4082 | ![신동아1](reports/plots/4082_forecast.png) |

추가 5단지(`6901`, `6908`, `7002`, `797`, `799`)는 `reports/plots/{id}_forecast.png` 로 동일 형식 제공.

</details>

### 재생성되는 raw 데이터 (gitignore)

| 파일 | 내용 |
|---|---|
| `reports/backtest_results.csv` | complex_id, name, model, horizon, **mape, rmse, r2**, n |
| `reports/complexes.csv` | 표본 단지 메타 — trade_count, month_span, last_ym |
| `reports/series/{id}_*.csv` | 전체/train/test 시계열 (Python ARIMA 입력) |
| `reports/predictions/{id}_{model}.csv` | 월별 예측가 raw (model ∈ `arima` · `lstm` · `lstm_reb` · `ma12`) |

> `lstm_reb` = LSTM + 한국부동산원 R-ONE 지수 정규화. 외곽 지역(SK북한산시티 등)에서 MAPE 약간 개선, 시장 신호가 강한 잠실·강남 등에서는 효과 미미 — 단지·지역 특성별 정규화 정책 후속 과제.

---

## ② LSTM 라이브 학습 (레거시 보조)

> ⚠️ **이 파이프라인이 적재하는 `t_training_result`(LSTM)는 2026-05-25부터 라이브 Depth 3 에서 소비하지 않습니다.** 백테스트 비교군·증빙 목적으로 코드를 유지합니다. 라이브 가격 분석은 Main 서버의 ARIMA 가 담당합니다.

```bash
# 한 번만 순회 (테스트용)
npm run train:one

# 무한 루프 — 한 바퀴 끝나면 30분 쉬고 다시
npm run train
```

학습 환경변수 (모두 선택):

| 변수 | 기본 | 설명 |
|---|---|---|
| `MIN_SAMPLES_PER_BUCKET` | 20 | 학습 대상 단지의 최소 거래 건수 |
| `WINDOW_SIZE` | 24 | LSTM 입력 윈도우 (개월) |
| `HORIZON_MONTHS` | 36 | 예측 시점 (개월) |
| `MODEL_VERSION` | lstm-v1 | upsert 키의 일부, 모델 비교용 |
| `LIMIT` | 0 (없음) | 한 pass 에서 학습할 단지 수 제한 |

### 데이터 흐름

```
t_apt_trade (server ingest)
        │
        ▼
fetch.ts (단지 선별 + 거래 시계열)
        │
        ▼
preprocess.ts (m²당 단가, 월별 중위값, IQR, forward-fill)
        │
        ▼
LSTM.train (TensorFlow.js, MSE)
        │
        ▼
predictNext (3년 후 m²당 단가)
        │
        ▼
t_training_result (upsert)  ← 라이브 미사용, 증빙용
```

### 진단 스크립트 (read-only)

```bash
npm run train:stats        # 학습 결과 통계 (confidence NULL 분포 등)
npm run train:backfill     # t_training_result.confidence NULL → MAPE 기반 산출
npm run train:backfill:dry # 위 dry-run
```

### 운영 메모

- 학습 결과는 DB 영구 저장 — 서버 재시작과 무관
- 같은 단지/구간/`MODEL_VERSION` 은 upsert 로 최신값으로 갱신
- 한 단지 학습 평균 5~15초 → 1,000 단지면 1~4시간

---

## 라이선스

본 프로젝트는 2026 국토교통부 공공데이터 활용 공모전 출품작입니다.
</content>
</invoke>
