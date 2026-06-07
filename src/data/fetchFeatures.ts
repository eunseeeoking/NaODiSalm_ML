/**
 * 다변량 학습용 공변량 로더 (2026-06-06 — 실험: 학습환경 보강).
 *
 * Main 레포가 적재한 거시·공급 테이블을 read-only 로 읽어온다.
 *   - t_macro_rate(ym, base_rate, mortgage_rate)   전국 단일 → ym broadcast
 *   - t_macro_econ(ym, cpi, m2)                    전국 단일 → ym broadcast
 *   - t_housing_supply(sigungu_code, ym, unsold)   서울 25구
 *   - t_reb_price_index 는 rebNormalize.preloadIndex 재사용
 *
 * 캐시 정책은 rebNormalize 와 동일 — 백테스트 한 run 내 메모리 캐시.
 */

import { prisma } from '../db';

export interface MacroPoint {
  baseRate: number | null;
  mortgageRate: number | null;
  cpi: number | null;
  m2: number | null;
}

let macroCache: Map<string, MacroPoint> | null = null;
const unsoldCache = new Map<string, Map<string, number>>();

/** 전국 거시 시계열 (금리 + CPI/M2) — ym → MacroPoint. 1회 DB 조회. */
export async function preloadMacro(): Promise<Map<string, MacroPoint>> {
  if (macroCache) return macroCache;

  const rateRows = await prisma.$queryRawUnsafe<
    Array<{ ym: string; base_rate: number | null; mortgage_rate: number | null }>
  >(`SELECT ym, base_rate, mortgage_rate FROM t_macro_rate ORDER BY ym ASC`);

  const econRows = await prisma.$queryRawUnsafe<
    Array<{ ym: string; cpi: number | null; m2: number | null }>
  >(`SELECT ym, cpi, m2 FROM t_macro_econ ORDER BY ym ASC`);

  const map = new Map<string, MacroPoint>();
  for (const r of rateRows) {
    map.set(r.ym, {
      baseRate: r.base_rate != null ? Number(r.base_rate) : null,
      mortgageRate: r.mortgage_rate != null ? Number(r.mortgage_rate) : null,
      cpi: null,
      m2: null,
    });
  }
  for (const r of econRows) {
    const prev = map.get(r.ym) ?? {
      baseRate: null,
      mortgageRate: null,
      cpi: null,
      m2: null,
    };
    prev.cpi = r.cpi != null ? Number(r.cpi) : null;
    prev.m2 = r.m2 != null ? Number(r.m2) : null;
    map.set(r.ym, prev);
  }

  macroCache = map;
  if (map.size === 0) {
    console.warn('[fetchFeatures] t_macro_rate/econ 비어있음 — 거시 피처 0');
  }
  return map;
}

/** 시군구 미분양 시계열 — ym → unsold(호수). 1회 DB 조회/캐시. */
export async function preloadUnsold(
  sigunguCode: string,
): Promise<Map<string, number>> {
  const cached = unsoldCache.get(sigunguCode);
  if (cached) return cached;

  const rows = await prisma.$queryRawUnsafe<
    Array<{ ym: string; unsold: number | null }>
  >(
    `SELECT ym, unsold FROM t_housing_supply WHERE sigungu_code = ? ORDER BY ym ASC`,
    sigunguCode,
  );

  const map = new Map<string, number>();
  for (const r of rows) {
    if (r.unsold != null) map.set(r.ym, Number(r.unsold));
  }
  unsoldCache.set(sigunguCode, map);
  return map;
}

export function clearFeatureCache() {
  macroCache = null;
  unsoldCache.clear();
}
