/**
 * LSTM hold-out 평가.
 *
 * 전략 — Recursive Multi-Step:
 *   1) train 구간으로 horizon=1 LSTM 학습 (기존 trainLstm 재사용)
 *      입력 window=24개월 z-score → 다음 1개월 z-score 예측
 *   2) train 마지막 24개월을 시드 윈도우로 predictNext() 반복 호출
 *      예측값을 윈도우에 push → 가장 오래된 값 shift → 다음 step
 *      총 horizon 회 반복하여 1..horizon 개월 예측 시계열 생성
 *   3) z-score 복원 → 만원/㎡ 단위
 *
 * 누적 오차로 horizon 끝쪽이 외삽 폭주 가능 → MAPE/RMSE 평가에서 적나라하게 드러남.
 * 이게 백테스트의 본질: "장기 예측에서 LSTM 이 정말 단순 MA 보다 나은가?"
 *
 * R-ONE 정규화 옵션 (2026-05-24 추가):
 *   reb 옵션 전달 시 → 학습 전 시계열을 R-ONE 지수로 정규화 → 예측 후 역정규화
 *   효과: 시장 거시 추세 노이즈 제거 → 동/단지 고유 변동에만 집중
 *   백테스트 정책: train 마지막 ym 의 index 만 사용 (운영 시뮬레이션, 보수적)
 */

import {
  buildTrainExamples,
  buildReturnExamples,
  buildDirectExamples,
  type MonthlyPoint,
} from '../data/preprocess';
import {
  trainLstm,
  trainLstmMulti,
  trainLstmMV,
  predictNext,
  predictDirect,
  predictDirectMV,
} from '../models/lstm';
import {
  preloadIndex,
  normalizeSeries,
  denormalizePredictions,
  getNearestIndex,
} from '../data/rebNormalize';
import { preloadMacro, preloadUnsold } from '../data/fetchFeatures';
import { buildMultivarExamples } from '../data/featurePanel';

export interface LstmForecast {
  /** train 마지막 시점부터 1..horizon 개월 후 예측 (만원/㎡) */
  prediction: number[];
  /** 학습 검증 지표 */
  trainMape: number;
  trainMae: number;
  /** scale (z-score) 정보 */
  scale: { mean: number; std: number };
  /** R-ONE 정규화 적용 여부 + coverage (0~1) */
  rebApplied?: boolean;
  rebCoverage?: number;
  rebIndexFactor?: number;
}

const DEFAULT_WINDOW = 24;
const DEFAULT_EPOCHS = 30;

export interface LstmEvalOpts {
  window?: number;
  epochs?: number;
  /** R-ONE 정규화 활성화 시 시군구코드 필요 */
  reb?: { sigunguCode: string };
  /** 실험 1 — 로그수익률 공간 학습 (표현 문제 격리). reb 와 배타. */
  returnsSpace?: boolean;
}

export async function predictLstm(
  train: MonthlyPoint[],
  horizon: number,
  opts?: LstmEvalOpts,
): Promise<LstmForecast> {
  const window = opts?.window ?? DEFAULT_WINDOW;
  const epochs = opts?.epochs ?? DEFAULT_EPOCHS;

  if (train.length < window + 4) {
    throw new Error(
      `LSTM needs ≥${window + 4} months train, got ${train.length}`,
    );
  }

  // 실험 1 — 로그수익률 공간 (레벨 경로와 완전 분리, reb 무시)
  if (opts?.returnsSpace) {
    return predictLstmReturns(train, horizon, window, epochs);
  }

  /* ─── R-ONE 정규화 (선택) ─── */
  let workingTrain = train;
  let rebApplied = false;
  let rebCoverage = 0;
  let rebIndexFactor = 0;

  if (opts?.reb) {
    const indexMap = await preloadIndex(opts.reb.sigunguCode);
    if (indexMap.size > 0) {
      const { normalized, coverage } = normalizeSeries(train, indexMap);
      if (coverage > 0.5) {
        workingTrain = normalized;
        rebApplied = true;
        rebCoverage = coverage;
        // 백테스트 복원 정책: train 마지막 ym 의 index 사용
        const lastYm = train[train.length - 1].ym;
        rebIndexFactor = getNearestIndex(indexMap, lastYm) ?? 0;
      } else {
        console.warn(
          `[lstmEval] R-ONE coverage 낮음 (${(coverage * 100).toFixed(0)}%) — 정규화 미적용`,
        );
      }
    }
  }

  // 1) horizon=1 학습 examples 빌드
  const { examples, scale } = buildTrainExamples(workingTrain, window, 1);
  if (examples.length < 8) {
    throw new Error(
      `not enough LSTM examples: ${examples.length} (need ≥8)`,
    );
  }

  // 2) 학습 (기존 trainLstm 그대로)
  const { model, mae, mape } = await trainLstm(examples, {
    windowSize: window,
    epochs,
  });

  // 3) 시드 윈도우 = workingTrain 마지막 window 개월 z-score
  const prices = workingTrain.map((p) => p.pricePerM2);
  const seedZ = prices.slice(-window).map((v) => (v - scale.mean) / scale.std);

  // 4) Recursive multi-step
  const predictionZ: number[] = [];
  const cursor = [...seedZ];
  for (let h = 0; h < horizon; h++) {
    const nextZ = await predictNext(model, cursor);
    predictionZ.push(nextZ);
    cursor.push(nextZ);
    cursor.shift();
  }

  // 5) z-score 복원 → 정규화된 단위 (R-ONE 적용 시 "기준 100 정규화값")
  let prediction = predictionZ.map((z) => z * scale.std + scale.mean);

  // 6) R-ONE 역정규화 (예측값 × indexFactor / 100)
  if (rebApplied) {
    prediction = denormalizePredictions(prediction, rebIndexFactor);
  }

  model.dispose();

  return {
    prediction,
    trainMape: mape,
    trainMae: mae,
    scale,
    rebApplied,
    rebCoverage,
    rebIndexFactor: rebApplied ? rebIndexFactor : undefined,
  };
}

/**
 * 실험 1 — 로그수익률 공간 LSTM (표현 문제 격리).
 *
 * 레벨 경로(predictLstm 본문)와 모든 셋업(단지·horizon·epochs·recursive 구조)을
 * 동일하게 유지하고 **표현만** 가격 레벨 → 로그수익률로 바꾼다.
 *   1) buildReturnExamples 로 r_t = ln(p_t)-ln(p_{t-1}) 를 z-score 하여 학습
 *   2) train 마지막 window 개월의 z-수익률을 시드로 recursive 예측
 *   3) 예측 z-수익률 → 수익률 복원 → prevPrice × exp(ret) 로 레벨 누적 복원
 *
 * 가설: LSTM 패배가 "표현 문제"라면, 이 행(LSTM-RET)의 MAPE 가 레벨 LSTM 대비
 *       급락(→ ARIMA 수준)해야 한다. recursive 누적(범인 B)은 그대로 남으므로
 *       완전 동률이 아니면 direct multi-horizon(실험 3)을 추가 검증.
 */
async function predictLstmReturns(
  train: MonthlyPoint[],
  horizon: number,
  window: number,
  epochs: number,
): Promise<LstmForecast> {
  const { examples, scale, zReturns } = buildReturnExamples(train, window, 1);
  if (examples.length < 8) {
    throw new Error(
      `not enough LSTM(returns) examples: ${examples.length} (need ≥8)`,
    );
  }

  const { model, mae, mape } = await trainLstm(examples, {
    windowSize: window,
    epochs,
  });

  // 시드 = train 마지막 window 개월의 z-수익률 (scale 과 일관: zReturns 재사용)
  const cursor = zReturns.slice(-window);

  // recursive multi-step — 수익률 예측 → exp 누적으로 레벨 복원
  let prevPrice = train[train.length - 1].pricePerM2;
  const prediction: number[] = [];
  for (let h = 0; h < horizon; h++) {
    const nextZ = await predictNext(model, cursor);
    const ret = nextZ * scale.std + scale.mean; // z → 로그수익률
    prevPrice = prevPrice * Math.exp(ret);
    prediction.push(prevPrice);
    cursor.push(nextZ);
    cursor.shift();
  }

  model.dispose();

  return { prediction, trainMape: mape, trainMae: mae, scale };
}

/**
 * 실험 3 — Direct multi-horizon LSTM (recursive 제거).
 *
 * 모델이 1..horizon 을 **한 번에** 예측하므로 자기예측 되먹임이 없다.
 * 표현(returns)과 직교 → {레벨,수익률} × {recursive,direct} 2×2 비교 가능.
 *   · returnsSpace=false → 레벨 z 복원
 *   · returnsSpace=true  → 로그수익률 예측 후 basePrice 부터 exp 누적 복원
 *
 * 가설: 1313 처럼 recursive 에서 폭발한 단지가 direct 에서 안정화되면
 *       fat-tail 의 주범이 범인 B(복리누적)임이 확정된다.
 */
export async function predictLstmDirect(
  train: MonthlyPoint[],
  horizon: number,
  opts?: { window?: number; epochs?: number; returnsSpace?: boolean },
): Promise<LstmForecast> {
  const window = opts?.window ?? DEFAULT_WINDOW;
  const epochs = opts?.epochs ?? DEFAULT_EPOCHS;
  const returns = opts?.returnsSpace === true;

  const { examples, scale, zSeries, basePrice } = buildDirectExamples(
    train,
    window,
    horizon,
    { returns },
  );
  if (examples.length < 8) {
    throw new Error(
      `not enough direct examples: ${examples.length} (need ≥8) ` +
        `— train ${train.length}mo, window ${window}, horizon ${horizon}`,
    );
  }

  const { model, mae, mape } = await trainLstmMulti(examples, {
    windowSize: window,
    epochs,
    outputDim: horizon,
  });

  // 시드 = 마지막 window 개 z값, direct 1회 호출로 horizon 개 예측
  const seed = zSeries.slice(-window);
  const predZ = await predictDirect(model, seed); // 길이 horizon

  let prediction: number[];
  if (returns) {
    // 누적 로그수익률 → exp 복원 (예측끼리 되먹이지 않음 — 모두 실제 입력 기반)
    prediction = [];
    let cum = 0;
    for (let k = 0; k < horizon; k++) {
      cum += predZ[k] * scale.std + scale.mean;
      prediction.push(basePrice * Math.exp(cum));
    }
  } else {
    prediction = predZ.map((z) => z * scale.std + scale.mean);
  }

  model.dispose();

  return { prediction, trainMape: mape, trainMae: mae, scale };
}

/**
 * 실험 — Multivariate Direct LSTM (2026-06-06, 학습환경 보강).
 *
 * 단변량 가격 한 줄(LSTM-RET-DIR)에 거시·공급·계절 공변량을 추가해
 * "제대로 먹인 학습기가 ~8% 추세 천장을 깨는가"를 검증한다.
 *   1) 공변량 로드(macro 전국 + unsold/reb 시군구)
 *   2) buildMultivarExamples 로 [window × F] 패널 + 타깃(가격 로그수익률)
 *   3) trainLstmMV 학습 → 마지막 윈도우로 direct 예측 → exp 누적 레벨 복원
 *
 * 표현·구조는 LSTM-RET-DIR 와 동일(로그수익률·direct) — 추가된 건 오직 공변량.
 * 따라서 LSTM-RET-DIR 대비 개선분 = 공변량의 순수 기여.
 */
export async function predictLstmMultivar(
  train: MonthlyPoint[],
  horizon: number,
  sigunguCode: string,
  opts?: { window?: number; epochs?: number },
): Promise<LstmForecast> {
  const window = opts?.window ?? DEFAULT_WINDOW;
  const epochs = opts?.epochs ?? DEFAULT_EPOCHS;

  const [macro, unsold, reb] = await Promise.all([
    preloadMacro(),
    preloadUnsold(sigunguCode),
    preloadIndex(sigunguCode),
  ]);

  const panel = buildMultivarExamples(train, window, horizon, {
    macro,
    unsold,
    reb,
  });
  if (panel.examples.length < 8) {
    throw new Error(
      `not enough multivar examples: ${panel.examples.length} (need ≥8) ` +
        `— train ${train.length}mo, window ${window}, horizon ${horizon}`,
    );
  }

  const { model, mae, mape } = await trainLstmMV(panel.examples, {
    windowSize: window,
    epochs,
    outputDim: horizon,
    numFeatures: panel.numFeatures,
  });

  const predZ = await predictDirectMV(model, panel.seed); // 길이 horizon

  // 누적 로그수익률 → exp 복원 (예측끼리 되먹이지 않음 — 모두 실제 입력 기반)
  const prediction: number[] = [];
  let cum = 0;
  for (let k = 0; k < horizon; k++) {
    cum += predZ[k] * panel.scale.std + panel.scale.mean;
    prediction.push(panel.basePrice * Math.exp(cum));
  }

  model.dispose();

  return { prediction, trainMape: mape, trainMae: mae, scale: panel.scale };
}
