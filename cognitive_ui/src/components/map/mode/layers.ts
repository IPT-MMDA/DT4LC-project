import type { DrawnRegion, Layer, LegendClass } from '../../../store/useAppStore';
import type { MessageAttachment } from '../../../types';
import { formatArea } from '../../chat/regionChip';
import { formatKm2 } from '../../chat/result/resultModel';
import { regionAreaKm2 } from './imagery';

// Map mode's LAYERS list and legend. Pure: the
// components render these from useAppStore `mapLayers` and the selected region.

export const REGION_ROW_ID = 'region';

export interface LayerRow {
  id: string;  // a mapLayers id, or REGION_ROW_ID
  name: string;
  subtitle: string;
  /** The swatch: a ramp's stops, a preview image, or a token (the change legend tokens, the region, imagery). */
  swatch: { stops: string[] } | { image: string } | { token: 'change' | 'region' | 'imagery' };
  opacity: number;  // 0..100, as shown
  visible: boolean;
}

/** "just now", "48 s ago", "5 min ago", "2 h ago". */
export function relativeTime(iso: string | undefined, now = Date.now()): string {
  if (!iso) return '';
  const s = Math.max(0, Math.round((now - Date.parse(iso)) / 1000));
  if (s < 10) return 'just now';
  if (s < 60) return `${s} s ago`;
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  return `${Math.floor(s / 3600)} h ago`;
}

function swatchOf(layer: Layer): LayerRow['swatch'] {
  if (layer.legend?.stops) return { stops: layer.legend.stops };
  if (layer.legend?.kind === 'change') return { token: 'change' };
  if (layer.thumbnail) return { image: layer.thumbnail };
  return { token: 'imagery' };
}

function layerRow(layer: Layer, now: number): LayerRow {
  const subtitle = layer.source === 'chat'
    ? [layer.subtitle ?? 'from chat', relativeTime(layer.addedAt, now)].filter(Boolean).join(' · ')
    : layer.subtitle ?? '';
  return { id: layer.id, name: layer.name, subtitle, swatch: swatchOf(layer), opacity: Math.round(layer.opacity * 100), visible: layer.visible };
}

/**
 * The rows top to bottom, as the design lists them: results and fetched imagery
 * (newest first), the selected region, then the sample.
 */
export function layerRows(layers: Layer[], region: DrawnRegion | null, regionOpacity: number, now = Date.now()): LayerRow[] {
  const newestFirst = [...layers].reverse();
  const rows = newestFirst.filter((l) => l.source !== 'sample').map((l) => layerRow(l, now));
  if (region) {
    rows.push({
      id: REGION_ROW_ID,
      name: region.name,
      subtitle: `region · ${formatArea(regionAreaKm2(region))}`,
      swatch: { token: 'region' },
      opacity: Math.round(regionOpacity * 100),
      visible: regionOpacity > 0,
    });
  }
  rows.push(...newestFirst.filter((l) => l.source === 'sample').map((l) => layerRow(l, now)));
  return rows;
}

/** The result layer the legend and Fit to result are about: the topmost visible one from chat. */
export function activeResultLayer(layers: Layer[]): Layer | undefined {
  return [...layers].reverse().find((l) => l.visible && l.source === 'chat');
}

/** A legend tick: "−0.5", "0", "+0.5". */
export function signedTick(v: number): string {
  if (v === 0) return '0';
  return v < 0 ? `−${Math.abs(v)}` : `+${v}`;
}

export interface MapLegendModel {
  title: string;
  stops?: string[];  // the style's ramp; none: the legend-* tokens
  ticks: string[];  // "−0.5 · 0 · +0.5" for a change, "min · max" for an index
  classes: { label: string; tone: LegendClass['tone']; colour?: string; area: string | null }[];
}

/** The legend panel of a result layer: its style's ramp and range, and its class areas. */
export function legendOf(layer: Layer | undefined): MapLegendModel | undefined {
  const legend = layer?.legend;
  if (!layer || !legend) return undefined;
  if (legend.kind === 'ramp') {
    return { title: layer.name, stops: legend.stops, ticks: legend.range.map(signedTick), classes: [] };
  }
  const range = legend.range;
  return {
    title: `${layer.name} difference`,
    stops: legend.stops,
    ticks: range ? [signedTick(range[0]), '0', signedTick(range[1])] : ['lost', 'same', 'gained'],
    classes: (legend.classes ?? []).map((c) => ({
      label: c.label,
      tone: c.tone,
      colour: c.colour,
      area: c.area_km2 === undefined ? null : formatKm2(c.area_km2),
    })),
  };
}

/**
 * The sample as a true-colour layer: the later date of the conversation's sample pair
 * (attachments with a `group`), drawn by /v1/tiles in true colour by default (BR1),
 * with its footprint from `metadata.footprint_wgs84`.
 */
export function sampleLayerOf(attachments: MessageAttachment[]): Layer | undefined {
  const files = attachments.filter((a) => a.group && a.path && a.metadata?.date);
  if (files.length === 0) return undefined;
  const latest = files.reduce((a, b) => ((b.metadata?.date ?? '') > (a.metadata?.date ?? '') ? b : a));
  const date = latest.metadata?.date as string;
  const sensor = latest.metadata?.sensor;
  return {
    id: `sample-${latest.id}-${date}`,
    name: (latest.group?.name ?? latest.filename).replace(/\.tiff?$/i, ''),
    type: 'raster',
    visible: true,
    opacity: 1,
    url: latest.path,
    source: 'sample',
    subtitle: [sensor, 'true colour', date].filter(Boolean).join(' · '),
    ...(latest.metadata?.footprint_wgs84 ? { footprint: latest.metadata.footprint_wgs84 } : {}),
    ...(latest.preview_url ? { thumbnail: latest.preview_url } : {}),
  };
}

/** Layers Compare can show on either side: visible rasters with tiles. */
export function compareCandidates(layers: Layer[]): Layer[] {
  return [...layers].reverse().filter((l) => l.visible && l.url);
}

/** Compare's sides at first: the result on the left, the sample (or the next layer) on the right. */
export function defaultCompareSides(layers: Layer[]): { left: string; right: string } | null {
  const candidates = compareCandidates(layers);
  if (candidates.length < 2) return null;
  const left = activeResultLayer(layers) ?? candidates[0];
  const right = candidates.find((l) => l.source === 'sample' && l.id !== left.id) ?? candidates.find((l) => l.id !== left.id);
  return right ? { left: left.id, right: right.id } : null;
}

/** A `GET /v1/files` item, the fields the sample layer reads (docs/API.md, BR2). */
export interface LibraryFile {
  id: string;
  upload_id?: string | null;
  filename: string;
  path: string;
  kind?: 'upload' | 'sample' | 'export';
  group?: { name: string; line: string } | null;
  metadata?: MessageAttachment['metadata'] | null;
  preview_url?: string | null;
  created_at?: string;
}

/** The library's latest sample pair as attachments, for a map opened without the sample in the conversation. */
export function samplesFromFiles(files: LibraryFile[] | undefined): MessageAttachment[] {
  const samples = (files ?? []).filter((f) => f.kind === 'sample' && f.group);
  if (samples.length === 0) return [];
  const latest = samples.reduce((a, b) => ((b.created_at ?? '') > (a.created_at ?? '') ? b : a));
  return samples
    .filter((f) => f.upload_id === latest.upload_id)
    .map((f) => ({ id: f.upload_id ?? f.id, filename: f.filename, path: f.path, preview_url: f.preview_url ?? undefined, group: f.group ?? undefined, metadata: f.metadata ?? undefined }));
}
