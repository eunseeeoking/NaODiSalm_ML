/**
 * 다변량 피처 패널 빌더 (2026-06-06 — 실험: multivariate-direct LSTM).
 *
 * 가격 월별 시계열에 거시·공급·계절 공변량을 같은 ym 축으로 정렬해
 * [window × F] 입력 / [horizon] 타깃(가격 로그수익률) 예제를 만든다.
 *
 * 설계 원칙 — **모든 공변량을 정상성(stationary) 형태로**:
 *   가격레벨 z-score 가 추세를 못 지운 게 LSTM 패착의 절반이었다(README "왜 졌나").
 *   거시변수도 레벨로 넣으면 같은 비정상성을 재주입하므로, 지수·물가·통화량은
 *   로그수익률, 금리는 1차차분(Δ)으로 변환한 뒤 train 구간에서 z-score 한다.
 *
 * 누수 차단: 공변량은 입력 윈도우(예측 원점 이전)에만 들어간다. direct multi-horizon
 *   이라 미래 공변량을 보지 않는다 → 발표시차로 최근월 NULL 이어도 forward-fill 안전.
 *
 * 피처 순서 (F=9):
 *   0 priceRet  가격 로그수익률 (autoregressive; 타깃과 동일 스케일)
 *   1 rebRet    R-ONE 지수 로그수익률
 *   2 dMort     주담대금리 차분 (Δ%)
 *   3 cpiRet    CPI 로그수익률 (= 물가상승률)
 *   4 m2Ret     M2 로그수익률 (= 통화증가율)
 *   5 unsoldLog 미분양 log(1+호수) 레벨
 *   6 volLog    월 거래량 log(1+건수)
 *   7 monthSin  계절성
 *   8 monthCos  계절성
 */

import type { MonthlyPoint } from './preprocess';
import type { MacroPoint } from './fetchFeatures';

export interface MultivarExample {
  input: number[][]; // [window][F]
  target: number[]; // [horizon] — z-scored 가격 로그수익률
}

export interface MultivarPanel {
  examples: MultivarExample[];
  scale: { mean: number; std: number }; // priceRet 스케일 (타깃 복원용)
  seed: number[][]; // 마지막 window 행 [window][F]
  basePrice: number; // train 마지막 실제가 (레벨 복원 기준)
  numFeatures: number;
  featureNames: string[];
}

export const FEATURE_NAMES = [
  'priceRet',
  'rebRet',
  'dMort',
  'cpiRet',
  'm2Ret',
  'unsoldLog',
  'volLog',
  'monthSin',
  'monthCos',
] as const;

const F = FEATURE_NAMES.length;

/** ym ≤ target 중 가장 큰 키의 값 (forward-fill). 없으면 null. */
function ffLookup(
  map: Map<string, number>,
  sortedYms: string[],
  ym: string,
): number | null {
  let last: number | null = null;
  for (const k of sortedYms) {
    if (k <= ym) last = map.get(k) ?? last;
    else break;
  }
  return last;
}

function logReturn(curr: number | null, prev: number | null): number {
  if (curr == null || prev == null || curr <= 0 || prev <= 0) return 0;
  return Math.log(curr) - Math.log(prev);
}

function zscoreColumn(col: number[]): { z: number[]; mean: number; std: number } {
  const mean = col.reduce((a, b) => a + b, 0) / col.length;
  const variance = col.reduce((s, v) => s + (v - mean) ** 2, 0) / col.length;
  const std = Math.sqrt(variance) || 1;
  return { z: col.map((v) => (v - mean) / std), mean, std };
}

export function buildMultivarExamples(
  train: MonthlyPoint[],
  window: number,
  horizon: number,
  cov: {
    macro: Map<string, MacroPoint>;
    unsold: Map<string, number>;
    reb: Map<string, number>;
  },
): MultivarPanel {
  const T = train.length;
  const empty: MultivarPanel = {
    examples: [],
    scale: { mean: 0, std: 1 },
    seed: [],
    basePrice: T > 0 ? train[T - 1].pricePerM2 : 0,
    numFeatures: F,
    featureNames: [...FEATURE_NAMES],
  };
  if (T < window + horizon + 1) return empty;

  // forward-fill 용 정렬 키 + macro 를 컬럼별 Map 으로 분해
  const rebYms = [...cov.reb.keys()].sort();
  const unsoldYms = [...cov.unsold.keys()].sort();
  const macroYms = [...cov.macro.keys()].sort();
  const mortMap = new Map<string, number>();
  const cpiMap = new Map<string, number>();
  const m2Map = new Map<string, number>();
  for (const [ym, m] of cov.macro) {
    if (m.mortgageRate != null) mortMap.set(ym, m.mortgageRate);
    if (m.cpi != null) cpiMap.set(ym, m.cpi);
    if (m.m2 != null) m2Map.set(ym, m.m2);
  }

  const prices = train.map((p) => p.pricePerM2);
  const yms = train.map((p) => p.ym);

  // 월별 ff 레벨
  const reb: (number | null)[] = yms.map((ym) => ffLookup(cov.reb, rebYms, ym));
  const mort: (number | null)[] = yms.map((ym) => ffLookup(mortMap, macroYms, ym));
  const cpi: (number | null)[] = yms.map((ym) => ffLookup(cpiMap, macroYms, ym));
  const m2: (number | null)[] = yms.map((ym) => ffLookup(m2Map, macroYms, ym));
  const unsold: (number | null)[] = yms.map((ym) =>
    ffLookup(cov.unsold, unsoldYms, ym),
  );

  // 원시 피처 컬럼 (변환 전)
  const rawCols: number[][] = Array.from({ length: F }, () => new Array(T).fill(0));
  for (let t = 0; t < T; t++) {
    const pPrev = t > 0 ? prices[t - 1] : null;
    rawCols[0][t] = logReturn(prices[t], pPrev); // priceRet
    rawCols[1][t] = logReturn(reb[t], t > 0 ? reb[t - 1] : null); // rebRet
    rawCols[2][t] =
      t > 0 && mort[t] != null && mort[t - 1] != null
        ? (mort[t] as number) - (mort[t - 1] as number)
        : 0; // dMort
    rawCols[3][t] = logReturn(cpi[t], t > 0 ? cpi[t - 1] : null); // cpiRet
    rawCols[4][t] = logReturn(m2[t], t > 0 ? m2[t - 1] : null); // m2Ret
    rawCols[5][t] = Math.log(1 + (unsold[t] ?? 0)); // unsoldLog
    rawCols[6][t] = Math.log(1 + train[t].sampleCount); // volLog
    const month = Number(yms[t].slice(5, 7)) || 1;
    rawCols[7][t] = Math.sin((2 * Math.PI * month) / 12); // monthSin
    rawCols[8][t] = Math.cos((2 * Math.PI * month) / 12); // monthCos
  }

  // z-score: 0~6 만 정규화, 7·8(sin/cos)은 이미 [-1,1] 이라 raw 유지
  const zCols: number[][] = [];
  let priceScale = { mean: 0, std: 1 };
  for (let f = 0; f < F; f++) {
    if (f <= 6) {
      const { z, mean, std } = zscoreColumn(rawCols[f]);
      zCols.push(z);
      if (f === 0) priceScale = { mean, std };
    } else {
      zCols.push(rawCols[f]);
    }
  }

  // 행 단위 피처 벡터로 전치
  const rows: number[][] = Array.from({ length: T }, (_, t) =>
    zCols.map((c) => c[t]),
  );

  // 슬라이딩 윈도우 예제: input=[i, i+window) 행, target=priceRet[i+window, i+window+horizon)
  const examples: MultivarExample[] = [];
  for (let i = 0; i + window + horizon <= T; i++) {
    const input = rows.slice(i, i + window);
    const target = zCols[0].slice(i + window, i + window + horizon);
    examples.push({ input, target });
  }

  return {
    examples,
    scale: priceScale,
    seed: rows.slice(T - window),
    basePrice: prices[T - 1],
    numFeatures: F,
    featureNames: [...FEATURE_NAMES],
  };
}
