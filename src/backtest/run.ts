/**
 * Day 5 백테스트 통합 러너.
 *
 * 흐름:
 *   1) selectTopComplexes() — 거래량 상위 5단지 (60개월+ 시계열, 최근 1년 거래 보장)
 *   2) 각 단지마다:
 *        a) splitHoldout(series, 36, 48) — train + test 36개월
 *        b) MA-12 예측 → CSV dump
 *        c) LSTM 예측 → CSV dump
 *        d) train 시계열 → CSV dump (Python ARIMA 입력용)
 *        e) test 실제값 → CSV dump (Python 평가 입력용)
 *   3) 두 모델 메트릭 → reports/backtest_results.csv
 *
 * 사용법:
 *   npm run backtest:run
 *     → reports/ 디렉토리에 결과 출력
 *     → 이후 npm run backtest:arima 로 ARIMA 추가
 *     → npm run backtest:visualize 로 PNG 생성
 *
 * 환경변수:
 *   BACKTEST_TOP_N=5         단지 수 (기본 5)
 *   BACKTEST_HORIZON=24      평가 horizon (기본 24, 데이터 범위 64개월 대응)
 *   BACKTEST_MIN_TRAIN=36    최소 train (기본 36)
 *   BACKTEST_LSTM_EPOCHS=30  LSTM epochs (기본 30)
 *   BACKTEST_REB_NORMALIZE=1 R-ONE 정규화 LSTM 추가 모델 평가 (기본 off)
 *                            → LSTM-REB 행을 backtest_results.csv 에 별도 추가
 *                            → comparison_*.png 에 4번째 막대로 표시
 *
 * 데이터 범위 (db-state.md):
 *   2020-01 ~ 2025-04 (~64개월) → minTrain 36 + horizon 24 = 60개월 필요 → 빠듯이 통과
 *   3년 백테스트가 필요하면 t_apt_trade 데이터 보강 후 BACKTEST_HORIZON=36 으로 재실행.
 */

import 'dotenv/config';
import fs from 'node:fs';
import path from 'node:path';
import { fetchComplexTrades } from '../data/fetch';
import { toMonthlySeries } from '../data/preprocess';
import { splitHoldout } from './holdout';
import { selectTopComplexes, type BacktestComplex } from './selectComplexes';
import { predictMa12 } from './ma12';
import { predictLstm, predictLstmDirect, predictLstmMultivar } from './lstmEval';
import { calcMetrics, type MetricResult } from './metrics';
import { disconnect } from '../db';

const TOP_N = Number(process.env.BACKTEST_TOP_N) || 5;
const HORIZON = Number(process.env.BACKTEST_HORIZON) || 24;
const MIN_TRAIN = Number(process.env.BACKTEST_MIN_TRAIN) || 36;
const LSTM_EPOCHS = Number(process.env.BACKTEST_LSTM_EPOCHS) || 30;
const REB_NORMALIZE = process.env.BACKTEST_REB_NORMALIZE === '1';
const RETURNS_SPACE = process.env.BACKTEST_RETURNS_SPACE === '1';
const DIRECT = process.env.BACKTEST_DIRECT === '1';
const MULTIVAR = process.env.BACKTEST_MULTIVAR === '1';

const REPORTS_DIR = path.resolve(process.cwd(), 'reports');
const SERIES_DIR = path.join(REPORTS_DIR, 'series');
const PRED_DIR = path.join(REPORTS_DIR, 'predictions');

interface BacktestRowResult {
  complexId: number;
  name: string;
  sigunguCode: string;
  legalDong: string;
  model:
    | 'MA-12'
    | 'LSTM'
    | 'LSTM-REB'
    | 'LSTM-RET'
    | 'LSTM-DIR'
    | 'LSTM-RET-DIR'
    | 'LSTM-MV';
  horizon: number;
  metrics: MetricResult;
}

function ensureDirs() {
  for (const d of [REPORTS_DIR, SERIES_DIR, PRED_DIR]) {
    if (!fs.existsSync(d)) fs.mkdirSync(d, { recursive: true });
  }
}

function writeCsv(filepath: string, header: string, rows: string[]) {
  fs.writeFileSync(filepath, [header, ...rows].join('\n') + '\n', 'utf-8');
}

async function runOne(c: BacktestComplex): Promise<BacktestRowResult[]> {
  console.log(
    `\n=== [${c.id}] ${c.name} (${c.sigunguCode}/${c.legalDong}) ` +
      `trades=${c.tradeCount} months=${c.monthSpan} last=${c.lastDealYm}`,
  );

  // 1) 거래 → 월별 시계열
  const trades = await fetchComplexTrades(c.id);
  const series = toMonthlySeries(trades);

  // 2) Hold-out
  const { train, test, horizon } = splitHoldout(series, HORIZON, MIN_TRAIN);
  console.log(
    `    holdout: train=${train.length}mo / test=${test.length}mo / horizon=${horizon}`,
  );

  const actual = test.map((p) => p.pricePerM2);
  const testYms = test.map((p) => p.ym);

  // 3) series CSV (학습+평가 전체)
  const seriesRows = series.map((p) => {
    const phase = p.ym >= testYms[0] ? 'test' : 'train';
    return `${p.ym},${p.pricePerM2.toFixed(4)},${phase}`;
  });
  writeCsv(
    path.join(SERIES_DIR, `${c.id}.csv`),
    'ym,price_per_m2,phase',
    seriesRows,
  );

  // 3-b) train-only CSV (Python ARIMA 입력)
  const trainRows = train.map(
    (p) => `${p.ym},${p.pricePerM2.toFixed(4)}`,
  );
  writeCsv(
    path.join(SERIES_DIR, `${c.id}_train.csv`),
    'ym,price_per_m2',
    trainRows,
  );

  // 3-c) test-only CSV (Python 평가 입력)
  const testRows = test.map(
    (p) => `${p.ym},${p.pricePerM2.toFixed(4)}`,
  );
  writeCsv(
    path.join(SERIES_DIR, `${c.id}_test.csv`),
    'ym,price_per_m2',
    testRows,
  );

  const results: BacktestRowResult[] = [];

  // 4) MA-12
  console.log(`    [MA-12] training...`);
  const ma12 = predictMa12(train, horizon);
  const ma12Metrics = calcMetrics(actual, ma12.prediction);
  console.log(
    `    [MA-12] MAPE=${ma12Metrics.mape.toFixed(2)}% ` +
      `RMSE=${ma12Metrics.rmse.toFixed(2)} R²=${ma12Metrics.r2.toFixed(3)}`,
  );
  writeCsv(
    path.join(PRED_DIR, `${c.id}_ma12.csv`),
    'ym,predicted',
    testYms.map(
      (ym, i) => `${ym},${ma12.prediction[i].toFixed(4)}`,
    ),
  );
  results.push({
    complexId: c.id,
    name: c.name,
    sigunguCode: c.sigunguCode,
    legalDong: c.legalDong,
    model: 'MA-12',
    horizon,
    metrics: ma12Metrics,
  });

  // 5) LSTM (비정규화 baseline)
  console.log(`    [LSTM ] training (epochs=${LSTM_EPOCHS})...`);
  const lstm = await predictLstm(train, horizon, { epochs: LSTM_EPOCHS });
  const lstmMetrics = calcMetrics(actual, lstm.prediction);
  console.log(
    `    [LSTM ] MAPE=${lstmMetrics.mape.toFixed(2)}% ` +
      `RMSE=${lstmMetrics.rmse.toFixed(2)} R²=${lstmMetrics.r2.toFixed(3)} ` +
      `(trainMAPE=${lstm.trainMape.toFixed(2)}%)`,
  );
  writeCsv(
    path.join(PRED_DIR, `${c.id}_lstm.csv`),
    'ym,predicted',
    testYms.map(
      (ym, i) => `${ym},${lstm.prediction[i].toFixed(4)}`,
    ),
  );
  results.push({
    complexId: c.id,
    name: c.name,
    sigunguCode: c.sigunguCode,
    legalDong: c.legalDong,
    model: 'LSTM',
    horizon,
    metrics: lstmMetrics,
  });

  // 5-b) LSTM-RET (실험 1 — 로그수익률 공간, 선택적)
  //      레벨 LSTM 과 모든 셋업 동일, 표현만 차분(로그수익률)으로 교체한 클린 A/B.
  if (RETURNS_SPACE) {
    console.log(`    [LSTM-RET] training (epochs=${LSTM_EPOCHS}, returns-space)...`);
    try {
      const lstmRet = await predictLstm(train, horizon, {
        epochs: LSTM_EPOCHS,
        returnsSpace: true,
      });
      const lstmRetMetrics = calcMetrics(actual, lstmRet.prediction);
      console.log(
        `    [LSTM-RET] MAPE=${lstmRetMetrics.mape.toFixed(2)}% ` +
          `RMSE=${lstmRetMetrics.rmse.toFixed(2)} R²=${lstmRetMetrics.r2.toFixed(3)} ` +
          `(trainMAPE=${lstmRet.trainMape.toFixed(2)}%)`,
      );
      writeCsv(
        path.join(PRED_DIR, `${c.id}_lstm_ret.csv`),
        'ym,predicted',
        testYms.map(
          (ym, i) => `${ym},${lstmRet.prediction[i].toFixed(4)}`,
        ),
      );
      results.push({
        complexId: c.id,
        name: c.name,
        sigunguCode: c.sigunguCode,
        legalDong: c.legalDong,
        model: 'LSTM-RET',
        horizon,
        metrics: lstmRetMetrics,
      });
    } catch (e) {
      console.warn(
        `    [LSTM-RET] 실패 — ${e instanceof Error ? e.message : String(e)}`,
      );
    }
  }

  // 5-c) LSTM-DIR / LSTM-RET-DIR (실험 3 — direct multi-horizon, 선택적)
  //      recursive 되먹임 제거. 레벨·수익률 두 표현 모두 평가 → 2×2 비교 완성.
  if (DIRECT) {
    const directVariants: Array<{
      model: 'LSTM-DIR' | 'LSTM-RET-DIR';
      suffix: string;
      returnsSpace: boolean;
    }> = [
      { model: 'LSTM-DIR', suffix: 'lstm_dir', returnsSpace: false },
      { model: 'LSTM-RET-DIR', suffix: 'lstm_ret_dir', returnsSpace: true },
    ];
    for (const v of directVariants) {
      console.log(`    [${v.model}] training (epochs=${LSTM_EPOCHS}, direct)...`);
      try {
        const dir = await predictLstmDirect(train, horizon, {
          epochs: LSTM_EPOCHS,
          returnsSpace: v.returnsSpace,
        });
        const dirMetrics = calcMetrics(actual, dir.prediction);
        console.log(
          `    [${v.model}] MAPE=${dirMetrics.mape.toFixed(2)}% ` +
            `RMSE=${dirMetrics.rmse.toFixed(2)} R²=${dirMetrics.r2.toFixed(3)}`,
        );
        writeCsv(
          path.join(PRED_DIR, `${c.id}_${v.suffix}.csv`),
          'ym,predicted',
          testYms.map((ym, i) => `${ym},${dir.prediction[i].toFixed(4)}`),
        );
        results.push({
          complexId: c.id,
          name: c.name,
          sigunguCode: c.sigunguCode,
          legalDong: c.legalDong,
          model: v.model,
          horizon,
          metrics: dirMetrics,
        });
      } catch (e) {
        console.warn(
          `    [${v.model}] 실패 — ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    }
  }

  // 5-d) LSTM-MV (실험 — multivariate direct, 선택적)
  //      LSTM-RET-DIR 와 표현·구조 동일(로그수익률·direct), 거시·공급·계절 공변량만 추가.
  //      LSTM-RET-DIR 대비 개선분 = 공변량의 순수 기여.
  if (MULTIVAR) {
    console.log(`    [LSTM-MV] training (epochs=${LSTM_EPOCHS}, multivar direct)...`);
    try {
      const mv = await predictLstmMultivar(train, horizon, c.sigunguCode, {
        epochs: LSTM_EPOCHS,
      });
      const mvMetrics = calcMetrics(actual, mv.prediction);
      console.log(
        `    [LSTM-MV] MAPE=${mvMetrics.mape.toFixed(2)}% ` +
          `RMSE=${mvMetrics.rmse.toFixed(2)} R²=${mvMetrics.r2.toFixed(3)}`,
      );
      writeCsv(
        path.join(PRED_DIR, `${c.id}_lstm_mv.csv`),
        'ym,predicted',
        testYms.map((ym, i) => `${ym},${mv.prediction[i].toFixed(4)}`),
      );
      results.push({
        complexId: c.id,
        name: c.name,
        sigunguCode: c.sigunguCode,
        legalDong: c.legalDong,
        model: 'LSTM-MV',
        horizon,
        metrics: mvMetrics,
      });
    } catch (e) {
      console.warn(
        `    [LSTM-MV] 실패 — ${e instanceof Error ? e.message : String(e)}`,
      );
    }
  }

  // 6) LSTM-REB (R-ONE 정규화 적용, 선택적)
  if (REB_NORMALIZE) {
    console.log(`    [LSTM-REB] training (epochs=${LSTM_EPOCHS}, sigungu=${c.sigunguCode})...`);
    try {
      const lstmReb = await predictLstm(train, horizon, {
        epochs: LSTM_EPOCHS,
        reb: { sigunguCode: c.sigunguCode },
      });
      const lstmRebMetrics = calcMetrics(actual, lstmReb.prediction);
      const appliedTag = lstmReb.rebApplied
        ? `REB-on coverage=${((lstmReb.rebCoverage ?? 0) * 100).toFixed(0)}% factor=${(lstmReb.rebIndexFactor ?? 0).toFixed(2)}`
        : 'REB-off (fallback)';
      console.log(
        `    [LSTM-REB] MAPE=${lstmRebMetrics.mape.toFixed(2)}% ` +
          `RMSE=${lstmRebMetrics.rmse.toFixed(2)} R²=${lstmRebMetrics.r2.toFixed(3)} ` +
          `(${appliedTag})`,
      );
      writeCsv(
        path.join(PRED_DIR, `${c.id}_lstm_reb.csv`),
        'ym,predicted',
        testYms.map(
          (ym, i) => `${ym},${lstmReb.prediction[i].toFixed(4)}`,
        ),
      );
      results.push({
        complexId: c.id,
        name: c.name,
        sigunguCode: c.sigunguCode,
        legalDong: c.legalDong,
        model: 'LSTM-REB',
        horizon,
        metrics: lstmRebMetrics,
      });
    } catch (e) {
      console.warn(
        `    [LSTM-REB] 실패 — ${e instanceof Error ? e.message : String(e)}`,
      );
    }
  }

  return results;
}

async function main() {
  ensureDirs();
  console.log(
    `[backtest] starting: topN=${TOP_N} horizon=${HORIZON} ` +
      `minTrain=${MIN_TRAIN} lstmEpochs=${LSTM_EPOCHS} ` +
      `reb=${REB_NORMALIZE ? 'ON' : 'OFF'} returns=${RETURNS_SPACE ? 'ON' : 'OFF'} ` +
      `direct=${DIRECT ? 'ON' : 'OFF'} multivar=${MULTIVAR ? 'ON' : 'OFF'}`,
  );

  const complexes = await selectTopComplexes({
    topN: TOP_N,
    // minMonths = MIN_TRAIN + HORIZON 로 모든 선정 단지가 동일 horizon 보장.
    // (selectComplexes 의 minMonths < splitHoldout 요구 시 horizon 자동 축소되어 비교 표 불균등)
    minMonths: MIN_TRAIN + HORIZON,
  });
  console.log(`[backtest] selected ${complexes.length} complexes:`);
  for (const c of complexes) {
    console.log(
      `  · ${c.name} (id=${c.id}) trades=${c.tradeCount} months=${c.monthSpan}`,
    );
  }

  if (complexes.length === 0) {
    console.error('[backtest] no qualified complex found.');
    await disconnect();
    process.exit(1);
  }

  // 단지 메타 CSV (Python 시각화에서 사용)
  writeCsv(
    path.join(REPORTS_DIR, 'complexes.csv'),
    'id,name,sigungu_code,legal_dong,trade_count,month_span,last_ym',
    complexes.map(
      (c) =>
        `${c.id},"${c.name.replace(/"/g, '""')}",${c.sigunguCode},${c.legalDong},${c.tradeCount},${c.monthSpan},${c.lastDealYm}`,
    ),
  );

  const allResults: BacktestRowResult[] = [];
  for (const c of complexes) {
    try {
      const r = await runOne(c);
      allResults.push(...r);
    } catch (e) {
      console.error(
        `[backtest] complex ${c.id} failed:`,
        e instanceof Error ? e.message : e,
      );
    }
  }

  // 종합 결과 CSV
  const resultsPath = path.join(REPORTS_DIR, 'backtest_results.csv');
  writeCsv(
    resultsPath,
    'complex_id,name,sigungu_code,legal_dong,model,horizon,mape,rmse,r2,n',
    allResults.map(
      (r) =>
        `${r.complexId},"${r.name.replace(/"/g, '""')}",${r.sigunguCode},${r.legalDong},${r.model},${r.horizon},` +
        `${Number.isFinite(r.metrics.mape) ? r.metrics.mape.toFixed(4) : 'NaN'},` +
        `${Number.isFinite(r.metrics.rmse) ? r.metrics.rmse.toFixed(4) : 'NaN'},` +
        `${Number.isFinite(r.metrics.r2) ? r.metrics.r2.toFixed(4) : 'NaN'},` +
        `${r.metrics.n}`,
    ),
  );

  // 요약 출력
  console.log(`\n[backtest] DONE — results saved to ${resultsPath}\n`);
  const byModel = new Map<string, MetricResult[]>();
  for (const r of allResults) {
    const arr = byModel.get(r.model) ?? [];
    arr.push(r.metrics);
    byModel.set(r.model, arr);
  }
  for (const [model, arr] of byModel) {
    const avg = (k: keyof MetricResult) => {
      const vals = arr
        .map((m) => Number(m[k]))
        .filter((v) => Number.isFinite(v));
      return vals.length
        ? (vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(3)
        : 'NaN';
    };
    console.log(
      `  ${model.padEnd(6)} avg MAPE=${avg('mape')}% RMSE=${avg('rmse')} R²=${avg('r2')} (n=${arr.length})`,
    );
  }

  console.log(
    `\n다음 단계: python scripts/backtest/arima.py && python scripts/backtest/visualize.py`,
  );

  await disconnect();
}

main().catch(async (e) => {
  console.error('[backtest] fatal', e);
  await disconnect();
  process.exit(1);
});
