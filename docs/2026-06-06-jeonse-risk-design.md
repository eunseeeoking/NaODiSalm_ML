# 전세보증 위험 지수 (깡통전세 위험) — 설계 (2026-06-06)

> ML 레포 신규 하위 프로젝트. 매매가 *예측*(시계열, 불가능)을 폐기하고
> **전세보증금 회수 위험**(횡단면, 가능)으로 피벗. Main 5축 스코어링의 한 축으로 제공.

---

## 0. 목표 & 범위 (정직성 먼저)

- **만드는 것:** 전세가율 기반 **깡통전세(보증금 미회수) 위험 지수**. (단지×면적 단위)
- **안 만드는 것:** "전세사기 적발". 진짜 사기는 **근저당·선순위채권·신탁등기**(등기부등본)가 핵심인데 우리에겐 그 데이터가 **없다**. 우리는 *가격 기반 깡통 위험*까지만 정직하게 주장한다.
- **대상 유형:** APT · OFFI · VILLA (Layer 1). AVM 보강은 VILLA·OFFI 우선. SH(단독다가구)는 매매테이블·좌표·이름 없음 → 제외.

DB 현황(2026-06-06 확인): villa 매매 139,178 / 전세 423,071, offi 매매 53,399 / 전세 399,930.
매매+전세 둘 다 보유 단지 = **VILLA 51,189 · OFFI 4,890 · APT 17,757**. villa 좌표 99.5%·이름 100%.

---

## 1. 2-레이어 아키텍처

```
레이어 1 (산수, 결정적):
  매매 거래가 있는 단지×면적 → 전세가율 직접 계산 → 깡통 등급. ML 0.

레이어 2 (ML AVM, 횡단면 hedonic):
  매매 거래 없는/오래된 단지 → 주변 비교거래로 현재 매매 시세 추정
  → 전세가율 산출 가능하게. 커버리지 51K → ~147K 확장.

병합 → t_jeonse_risk (Main이 읽는 출력 계약)
```

**핵심 구분 (불가능 vs 가능):**
- ❌ 시계열 미래 예측: "이 빌라 36개월 뒤 가격" — 불가능(기존 실험에서 입증).
- ✅ 횡단면 현재 가치평가: "이 면적·연식·위치 빌라, 최근 주변 매매로 봤을 때 지금 얼마" — 표준 AVM(KB시세·Zillow), tractable.
- 레이어 2는 후자만 한다.

---

## 2. 레이어 1 — 전세가율 (결정적)

### 산출 단위
**`(complex_type, complex_id, area_bucket)`** — 전세가율은 면적에 따라 달라지므로(소형일수록 높음) 반드시 같은 면적버킷끼리 비교. `bucketArea`(기존) 재사용.

### 정의
```
최근 W개월(기본 W=18) 윈도우, IQR 이상치 컷 후:
  sale_m2   = median(매매 priceManwon / areaM2)        -- 만원/㎡
  jeonse_m2 = median(전세 depositManwon / areaM2)      -- 만원/㎡, contractType=JEONSE 만
  전세가율(jeonse_ratio) = jeonse_m2 / sale_m2
```
> 면적이 분모·분자에서 상쇄 → 전세가율 = 전세보증금/매매가. m² 정규화는 버킷 내 면적 편차 노이즈만 흡수.
> 월세(WOLSE)는 v1 제외(보증금+월세 환산 필요). JEONSE 보증금만.

### 깡통 등급 (HUG 안심전세 기준선 참고)
| 전세가율 | 등급 | 의미 |
|---|---|---|
| ≥ 1.00 | **REVERSED** | 전세 ≥ 매매, 이미 역전 — 최고위험 |
| 0.90 ~ 1.00 | **HIGH** | 고위험 |
| 0.80 ~ 0.90 | **CAUTION** | 주의 |
| < 0.80 | **SAFE** | 상대적 안전 |
(임계값 env 화)

### confidence (0~1)
`f(sale_n, jeonse_n, sale_recency)` — 표본 적거나 매매가 오래될수록↓. 최근 24개월 매매 0건 → 레이어 2로 위임(source=ESTIMATED).

---

## 3. 레이어 2 — ML AVM (횡단면 hedonic 회귀)

### 목적
매매 거래 없는/오래된 `(complex, area_bucket)`의 **현재 매매 m²단가**를 비교거래로 추정 → 레이어 1이 전세가율 산출 가능하게.

### 타깃 / 모델
- 타깃: `log(매매 m²단가)`
- 모델: **GBM (LightGBM, Python)** — tabular hedonic의 정석. (LSTM 아님 — 솔직히 여기선 GBM이 맞다.)
  - Python 사용은 기존 ARIMA 백테스트와 동일 패턴: TS가 피처 CSV dump → Python 학습/예측 → CSV 회수.

### 피처 (hedonic)
| 군 | 피처 |
|---|---|
| 구조 | 전용면적, 건축연도/연식, floor, 건물유형(VILLA/OFFI/APT) |
| 입지 | lat/lng, 행정동 FE, **k-NN 주변 매매 median**(같은유형·반경·최근), 지하철거리(subway-graph 재활용) |
| 시점/시장 | 거래 ym, 시군구 REB 지수 |

### 시점 처리 (누수·노후화 방지) — REB 정규화 재사용
각 매매가를 거래월 REB 지수로 나눠 **기준월로 detrend** → "내재가치"를 학습 → 예측 후 현재월 REB로 재인플레이트. 시장 타이밍과 내재가치를 분리(기존 `rebNormalize` 아이디어 그대로).

### 검증 (가능한 검증!)
매매 보유 단지를 hold-out → 추정가 vs 실제 매매가. AVM 표준 지표: **MAPE, ±10% 적중률, ±20% 적중률**. (시계열 예측과 달리 정답이 있어 정직하게 측정됨.)

### 출력
`(complex, area_bucket)` → 추정 매매 m²단가 + 예측구간(confidence). 레이어 1에 source=ESTIMATED로 주입.

---

## 4. 출력 계약 → Main (`t_jeonse_risk`)

ML이 upsert, Main이 5축 스코어링에서 read (t_training_result 패턴 동일).

```prisma
model JeonseRisk {
  id           Int      @id @default(autoincrement())
  complexType  String   @map("complex_type") @db.VarChar(8)  /// APT|OFFI|VILLA
  complexId    Int      @map("complex_id")
  areaBucket   AreaBucket @map("area_bucket")
  baseYm       String   @map("base_ym") @db.VarChar(7)
  saleMedianM2   Float? @map("sale_median_m2")    /// 만원/㎡ (DIRECT 또는 ESTIMATED)
  jeonseMedianM2 Float? @map("jeonse_median_m2")
  jeonseRatio    Float? @map("jeonse_ratio")
  riskLevel    String   @map("risk_level") @db.VarChar(10)    /// REVERSED|HIGH|CAUTION|SAFE
  saleSource   String   @map("sale_source") @db.VarChar(10)   /// DIRECT|ESTIMATED
  confidence   Float?                                          /// 0~1
  saleN        Int      @map("sale_n")
  jeonseN      Int      @map("jeonse_n")
  modelVersion String   @map("model_version") @db.VarChar(40)
  computedAt   DateTime @default(now()) @map("computed_at")
  @@unique([complexType, complexId, areaBucket, baseYm, modelVersion], map: "uniq_jeonse_risk")
  @@index([complexType, complexId])
  @@map("t_jeonse_risk")
}
```
> Main은 행정동 집계(위험 단지 비율)로 안전축에 합치거나 Depth 3 단지 배지로 노출.

---

## 5. 빌드 단계

| Phase | 내용 | 산출 |
|---|---|---|
| **A** | 레이어 1 — 전세가율 빌더(raw SQL, 면적버킷 매칭) + 깡통 등급 + reports CSV | 51K villa 즉시 깡통 분포 |
| **B** | 레이어 2 — AVM 피처 빌더(TS) + LightGBM(Python) + hold-out 검증 | ±10% 적중률 등 AVM 성능 |
| **C** | 병합 → `t_jeonse_risk` upsert + Main 스키마/스코어링 통합(핸드오프) | 제품 5축 반영 |

검증 우선: Phase A로 실제 깡통 분포를 눈으로 본 뒤 Phase B AVM 필요 규모(전세만 있는 단지 수) 확정.

---

## 6. 열린 결정 (디폴트 제시 — 확정 필요)

1. **임계값**: REVERSED≥1.0 / HIGH≥0.9 / CAUTION≥0.8 (HUG 기준). → 그대로?
2. **윈도우 W**: 매매·전세 median 최근 18개월. → 18 vs 12?
3. **월세 처리**: v1 JEONSE만. WOLSE 보증금 환산은 후속. → OK?
4. **AVM 모델**: Python LightGBM(권장) vs Node tfjs DNN(레포 일관성). → LightGBM?
5. **유형 범위**: 레이어1 APT+OFFI+VILLA / AVM은 VILLA·OFFI. → OK?
