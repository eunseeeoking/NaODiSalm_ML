import * as tf from '@tensorflow/tfjs';
// tfjs 4.x 메인 패키지가 `import * as tf` 시 LayersModel/sequential/layers 등
// namespace 멤버를 자동 노출하지 않는 알려진 typing 이슈 → tfjs-layers 직접 import.
// 런타임은 동일 (tf.sequential / tf.layers 와 같은 인스턴스 — 4.x re-export).
import * as tfl from '@tensorflow/tfjs-layers';
import type { LayersModel } from '@tensorflow/tfjs-layers';
import type { TrainExample } from '../data/preprocess';

/**
 * 단순 LSTM 가격 예측 모델.
 *  - 입력: [windowSize] 길이의 시계열 (정규화된 m²당 단가)
 *  - 출력: horizon 후의 단일 값 (정규화된 m²당 단가)
 *  - 구조: LSTM(32) → Dropout → LSTM(16) → Dense(1)
 *  - Loss: MSE, Optimizer: Adam
 */

export interface TrainConfig {
  windowSize: number;
  epochs?: number;
  batchSize?: number;
  validationSplit?: number;
}

export interface TrainOutcome {
  model: LayersModel;
  history: { loss: number[]; valLoss?: number[] };
  mae: number;
  mape: number;
}

export async function trainLstm(
  examples: TrainExample[],
  cfg: TrainConfig,
): Promise<TrainOutcome> {
  const epochs = cfg.epochs ?? 30;
  const batchSize = cfg.batchSize ?? 16;
  const validationSplit = cfg.validationSplit ?? 0.2;

  // 텐서 빌드
  const inputs = tf.tensor3d(
    examples.map((e) => e.input.map((v) => [v])),
  ); // [samples, windowSize, 1]
  const targets = tf.tensor2d(examples.map((e) => [e.target])); // [samples, 1]

  // 모델 — tfjs-layers 직접 import (tfl), optimizer 는 core 재export 사용 (tf.train)
  const model = tfl.sequential();
  model.add(
    tfl.layers.lstm({
      units: 32,
      inputShape: [cfg.windowSize, 1],
      returnSequences: true,
    }),
  );
  model.add(tfl.layers.dropout({ rate: 0.2 }));
  model.add(tfl.layers.lstm({ units: 16, returnSequences: false }));
  model.add(tfl.layers.dense({ units: 1 }));
  model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });

  const history = await model.fit(inputs, targets, {
    epochs,
    batchSize,
    validationSplit,
    verbose: 0,
    shuffle: true,
  });

  // 평가 — validation 시계열 끝부분으로 MAE/MAPE
  const preds = model.predict(inputs) as tf.Tensor;
  const yTrue = await targets.data();
  const yPred = await preds.data();
  let absErr = 0;
  let absPct = 0;
  let valid = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const t = yTrue[i];
    const p = yPred[i];
    if (!Number.isFinite(t) || !Number.isFinite(p)) continue;
    absErr += Math.abs(p - t);
    if (Math.abs(t) > 1e-6) absPct += Math.abs((p - t) / t);
    valid += 1;
  }
  const mae = valid ? absErr / valid : 0;
  const mape = valid ? (absPct / valid) * 100 : 0;

  inputs.dispose();
  targets.dispose();
  preds.dispose();

  return {
    model,
    history: {
      loss: (history.history.loss as number[]) ?? [],
      valLoss: history.history.val_loss as number[] | undefined,
    },
    mae,
    mape,
  };
}

/** 학습된 모델로 마지막 window 입력 → 다음 horizon 후 값 예측 */
export async function predictNext(
  model: LayersModel,
  windowZ: number[],
): Promise<number> {
  const input = tf.tensor3d([windowZ.map((v) => [v])]);
  const out = model.predict(input) as tf.Tensor;
  const arr = await out.data();
  input.dispose();
  out.dispose();
  return arr[0];
}

/**
 * 실험 3 — Direct multi-horizon LSTM.
 *  - 입력: [windowSize] 시계열, 출력: Dense(outputDim) = horizon 개 값 한 번에
 *  - recursive 되먹임 없음 → 예측오차가 다음 입력으로 복리누적되지 않는다.
 *  - target 이 스칼라 대신 길이 outputDim 벡터인 점만 trainLstm 과 다름.
 */
export interface TrainExampleMulti {
  input: number[];
  target: number[]; // 길이 outputDim
}

export async function trainLstmMulti(
  examples: TrainExampleMulti[],
  cfg: TrainConfig & { outputDim: number },
): Promise<TrainOutcome> {
  const epochs = cfg.epochs ?? 30;
  const batchSize = cfg.batchSize ?? 16;
  const validationSplit = cfg.validationSplit ?? 0.2;

  const inputs = tf.tensor3d(
    examples.map((e) => e.input.map((v) => [v])),
  ); // [samples, windowSize, 1]
  const targets = tf.tensor2d(examples.map((e) => e.target)); // [samples, outputDim]

  const model = tfl.sequential();
  model.add(
    tfl.layers.lstm({
      units: 32,
      inputShape: [cfg.windowSize, 1],
      returnSequences: true,
    }),
  );
  model.add(tfl.layers.dropout({ rate: 0.2 }));
  model.add(tfl.layers.lstm({ units: 16, returnSequences: false }));
  model.add(tfl.layers.dense({ units: cfg.outputDim }));
  model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });

  const history = await model.fit(inputs, targets, {
    epochs,
    batchSize,
    validationSplit,
    verbose: 0,
    shuffle: true,
  });

  // 평가 — 전체 horizon 평탄화하여 MAE/MAPE (정규화 공간 기준 참고치)
  const preds = model.predict(inputs) as tf.Tensor;
  const yTrue = await targets.data();
  const yPred = await preds.data();
  let absErr = 0;
  let absPct = 0;
  let valid = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const t = yTrue[i];
    const p = yPred[i];
    if (!Number.isFinite(t) || !Number.isFinite(p)) continue;
    absErr += Math.abs(p - t);
    if (Math.abs(t) > 1e-6) absPct += Math.abs((p - t) / t);
    valid += 1;
  }
  const mae = valid ? absErr / valid : 0;
  const mape = valid ? (absPct / valid) * 100 : 0;

  inputs.dispose();
  targets.dispose();
  preds.dispose();

  return {
    model,
    history: {
      loss: (history.history.loss as number[]) ?? [],
      valLoss: history.history.val_loss as number[] | undefined,
    },
    mae,
    mape,
  };
}

/** Direct 모델로 window 입력 → horizon 개 예측값 한 번에 */
export async function predictDirect(
  model: LayersModel,
  windowZ: number[],
): Promise<number[]> {
  const input = tf.tensor3d([windowZ.map((v) => [v])]);
  const out = model.predict(input) as tf.Tensor;
  const arr = await out.data();
  input.dispose();
  out.dispose();
  return Array.from(arr);
}

/**
 * 실험 — Multivariate Direct LSTM (2026-06-06).
 *  - 입력: [windowSize][numFeatures] (가격수익률 + 거시·공급·계절 공변량)
 *  - 출력: Dense(outputDim) = horizon 개 가격 로그수익률 한 번에 (recursive 없음)
 *  - direct multi-horizon 이라 미래 공변량 불필요 → 누수 없음.
 */
export interface TrainExampleMV {
  input: number[][]; // [windowSize][numFeatures]
  target: number[]; // [outputDim]
}

export async function trainLstmMV(
  examples: TrainExampleMV[],
  cfg: TrainConfig & { outputDim: number; numFeatures: number },
): Promise<TrainOutcome> {
  const epochs = cfg.epochs ?? 30;
  const batchSize = cfg.batchSize ?? 16;
  const validationSplit = cfg.validationSplit ?? 0.2;

  const inputs = tf.tensor3d(
    examples.map((e) => e.input),
  ); // [samples, windowSize, numFeatures]
  const targets = tf.tensor2d(examples.map((e) => e.target)); // [samples, outputDim]

  const model = tfl.sequential();
  model.add(
    tfl.layers.lstm({
      units: 32,
      inputShape: [cfg.windowSize, cfg.numFeatures],
      returnSequences: true,
    }),
  );
  model.add(tfl.layers.dropout({ rate: 0.2 }));
  model.add(tfl.layers.lstm({ units: 16, returnSequences: false }));
  model.add(tfl.layers.dense({ units: cfg.outputDim }));
  model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });

  const history = await model.fit(inputs, targets, {
    epochs,
    batchSize,
    validationSplit,
    verbose: 0,
    shuffle: true,
  });

  const preds = model.predict(inputs) as tf.Tensor;
  const yTrue = await targets.data();
  const yPred = await preds.data();
  let absErr = 0;
  let absPct = 0;
  let valid = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const t = yTrue[i];
    const p = yPred[i];
    if (!Number.isFinite(t) || !Number.isFinite(p)) continue;
    absErr += Math.abs(p - t);
    if (Math.abs(t) > 1e-6) absPct += Math.abs((p - t) / t);
    valid += 1;
  }
  const mae = valid ? absErr / valid : 0;
  const mape = valid ? (absPct / valid) * 100 : 0;

  inputs.dispose();
  targets.dispose();
  preds.dispose();

  return {
    model,
    history: {
      loss: (history.history.loss as number[]) ?? [],
      valLoss: history.history.val_loss as number[] | undefined,
    },
    mae,
    mape,
  };
}

/** Multivariate direct 모델로 [window][F] 입력 → horizon 개 예측 한 번에 */
export async function predictDirectMV(
  model: LayersModel,
  windowRows: number[][],
): Promise<number[]> {
  const input = tf.tensor3d([windowRows]);
  const out = model.predict(input) as tf.Tensor;
  const arr = await out.data();
  input.dispose();
  out.dispose();
  return Array.from(arr);
}
