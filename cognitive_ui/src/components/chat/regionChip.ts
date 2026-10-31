import type { ChatRegion, MessageAttachment } from '../../types';

// The region chip: "Region · Dnipro floodplain · Kherson · 412 km² ·
// HLS L30 · 2023-06-01 → 06-09 · ≤20 % cloud". The same format as the server's
// `region.line` (server/regions.py), which replaces this once the job exists.

export const REGION_DEFAULTS = { dataset: 'HLS L30', cloud_max: 20, days: 30 } as const;

/** "412 km²"; one decimal below 10 km² (server/regions.py format_area). */
export function formatArea(km2: number): string {
  return km2 < 10 ? `${km2.toFixed(1)} km²` : `${Math.round(km2)} km²`;
}

/** "2023-06-01 → 06-09"; the year only when it changes; one date when equal. */
export function dateRange(start: string, end: string): string {
  if (start === end) return start;
  return `${start} → ${start.slice(0, 4) === end.slice(0, 4) ? end.slice(5) : end}`;
}

/** A LULC region has no dates or cloud limit: "Region · 251 km² · LULC". */
export function regionLine(region: ChatRegion): string {
  if (region.line) return region.line;
  const imagery = region.dataset === 'LULC' ? [] : [dateRange(region.start_date, region.end_date), `≤${region.cloud_max} % cloud`];
  return ['Region', region.name, region.place, formatArea(region.area_km2), region.dataset, ...imagery]
    .filter(Boolean)
    .join(' · ');
}

/** The chip's two lines: "Region · <name> · <place>" and "<area> · <dataset> · <dates> · <cloud>". */
export function regionChipLines(region: ChatRegion): { title: string; detail: string } {
  const parts = regionLine(region).split(' · ');
  const at = parts.findIndex((p) => p.endsWith('km²'));
  if (at <= 0) return { title: parts[0] ?? 'Region', detail: parts.slice(1).join(' · ') };
  return { title: parts.slice(0, at).join(' · '), detail: parts.slice(at).join(' · ') };
}

/** The first and last `metadata.date` of the sample's files, when the conversation has the sample. */
export function sampleDates(attachments: MessageAttachment[]): [string, string] | undefined {
  const dates = attachments
    .filter((a) => a.group && a.metadata?.date)
    .map((a) => a.metadata?.date as string)
    .sort();
  return dates.length > 0 ? [dates[0], dates[dates.length - 1]] : undefined;
}
