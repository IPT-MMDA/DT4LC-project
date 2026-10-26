import type { BuiltinTier, Capabilities } from '../../api/hooks/useCapabilities';
import type { GeoTIFFFile } from '../../api/hooks/useUpload';
import type { JobRecord } from '../../api/jobs';
import type { Attachment, ChatRegion, MessageAttachment } from '../../types';
import type { PillStatus } from '../chat/analysis/StatusPill';
import { analysisReducer, type AnalysisView, type StepStatus } from '../chat/analysis/analysisReducer';
import { formatDuration } from '../chat/analysis/format';
import { asRef, downloadName, prettyFormula, type AnalysedContext, type ArtifactRef, type BandRow, type ResultCardModel, type Stat } from '../chat/result/resultModel';
import { formatRunTime } from '../library/libraryModel';

// Result detail from GET /v1/jobs/{id}: the plan
// with each step's parameters and time, the statistics table, the downloads and the
// map's meta line. Pure: pages/ResultDetail.tsx renders these and nothing else.

type Obj = Record<string, unknown>;
const obj = (v: unknown): Obj => (typeof v === 'object' && v !== null && !Array.isArray(v) ? (v as Obj) : {});
const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const str = (v: unknown): string | undefined => (typeof v === 'string' && v ? v : undefined);
const MINUS = '−';

const artifactsOf = (job: Pick<JobRecord, 'result'>): Obj => obj(obj(obj(job.result).execution).artifacts);

/** The artifact keys of an analysis, in the order resultCardOf picks the card. */
const MAIN_KEYS = ['BandMathChangeMap', 'ChangeMap', 'BandMathMap', 'NDVIMap', 'EVIMap', 'NDWIMap', 'NDSIMap', 'LULCMap', 'SnowClassification', 'FieldBoundaries', 'Statistics'];
const CHANGE_KEYS = ['BandMathChangeMap', 'ChangeMap'];

/** The job's main analysis output: its artifact key and object, or null. */
export function mainArtifact(job: Pick<JobRecord, 'result'>): { key: string; artifact: Obj } | null {
  const artifacts = artifactsOf(job);
  const key = MAIN_KEYS.find((k) => Object.keys(obj(artifacts[k])).length > 0);
  return key ? { key, artifact: obj(artifacts[key]) } : null;
}

/** "NDWI" for a change of NDWI, "NDMI" for band math; '' when unknown. */
function indexLabel(artifact: Obj): string {
  return str(artifact.name) ?? str(artifact.index_name) ?? (str(artifact.index_type)?.toUpperCase() ?? '');
}

// --- Header ---------------------------------------------------------------------

/** The page title: the stored card's, else the prompt. */
export function resultTitle(job: Pick<JobRecord, 'id' | 'prompt'> & { card?: { title?: string } | null }): string {
  return job.card?.title || job.prompt?.trim() || `Result ${job.id}`;
}

const PILL: Record<JobRecord['status'], PillStatus> = {
  completed: 'done',
  failed: 'failed',
  cancelled: 'cancelled',
  running: 'running',
  pending: 'queued',
  proposal: 'planning',
};
export const pillStatusOf = (status: JobRecord['status']): PillStatus => PILL[status] ?? 'queued';

/** The run time: the card's, else the `done` event's; '' while it runs. */
export function runTime(job: { card?: { duration_s?: number | null } | null }, view: AnalysisView): string {
  return formatRunTime(job.card?.duration_s ?? view.done?.elapsedS ?? (view.endedT !== undefined && view.startedT !== undefined ? view.endedT - view.startedT : null));
}

// --- What was analysed ------------------------------------------------------------

/** Server paths the job read: the plan's file binds and the artifacts' RasterPath*. */
export function jobInputPaths(job: Pick<JobRecord, 'result'> & { plan?: unknown }): string[] {
  const steps = Array.isArray(obj(job.plan).steps) ? (obj(job.plan).steps as unknown[]) : [];
  const fromPlan = steps.flatMap((s) => Object.values(obj(obj(s).binds))).filter((v): v is string => typeof v === 'string' && v.startsWith('/'));
  const artifacts = artifactsOf(job);
  const fromResult = ['RasterPath', 'RasterPathBefore', 'RasterPathAfter'].map((k) => artifacts[k]).filter((v): v is string => typeof v === 'string');
  return [...new Set([...fromPlan, ...fromResult])];
}

/** A GET /v1/files item as a composer attachment; the sample keeps its group (one chip). */
export function libraryFileAttachment(f: GeoTIFFFile): Attachment {
  return {
    id: f.upload_id ?? f.id,
    filename: f.filename,
    path: f.path,
    mime_type: 'image/tiff',
    size_bytes: f.size_bytes,
    ...(f.preview_url ? { preview_url: f.preview_url } : {}),
    ...(f.metadata ? { metadata: f.metadata } : {}),
    ...(f.group ? { group: f.group } : {}),
  };
}

const bboxPolygon = ([w, s, e, n]: number[]): GeoJSON.Polygon => ({
  type: 'Polygon',
  coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
});

/** A region job's echo (`job.region`) as a chip region; undefined without one. */
export function regionOfJob(job: Pick<JobRecord, 'id'> & { region?: unknown }): ChatRegion | undefined {
  const r = obj(job.region);
  const dataset = str(r.dataset);
  const box = [r.bbox, r.bounds].find((b): b is number[] => Array.isArray(b) && b.length === 4 && b.every(isNum));
  const geometry = typeof obj(r.geometry).type === 'string' ? (r.geometry as GeoJSON.Geometry) : box ? bboxPolygon(box) : undefined;
  if (!dataset || !geometry) return undefined;
  return {
    id: `job-${job.id}`,
    geometry,
    ...(str(r.name) ? { name: str(r.name) } : {}),
    ...(str(r.place) ? { place: str(r.place) } : {}),
    dataset: dataset as ChatRegion['dataset'],
    start_date: str(r.start_date) ?? '',
    end_date: str(r.end_date) ?? '',
    cloud_max: isNum(r.cloud_max) ? r.cloud_max : 20,
    ...(Array.isArray(r.bands) ? { bands: r.bands.filter((b): b is string => typeof b === 'string') } : {}),
    area_km2: isNum(r.area_km2) ? r.area_km2 : 0,
    ...(str(r.line) ? { line: str(r.line) } : {}),
  };
}

/**
 * What the job was about when this browser has no conversation holding it: the
 * library's files that the job read (by path), and the region echo of a region job.
 */
export function contextFromJob(job: Pick<JobRecord, 'id' | 'result'> & { plan?: unknown; region?: unknown }, files: GeoTIFFFile[]): AnalysedContext {
  const paths = jobInputPaths(job);
  const region = regionOfJob(job);
  return {
    attachments: files.filter((f) => paths.includes(f.path)).map(libraryFileAttachment),
    ...(region ? { region } : {}),
  };
}

// --- Plan -------------------------------------------------------------------------

interface PlanStep {
  uses: string;
  binds?: Obj;
  config?: Obj;
}

/** id → title and formula of the built-in analyses (`GET /v1/capabilities` tier 3). */
export type Registry = Record<string, { title: string; formula: string | null }>;

export function registryOf(capabilities: Capabilities | undefined): Registry {
  const builtin = capabilities?.tiers.find((t): t is BuiltinTier => t.id === 'builtin');
  return Object.fromEntries((builtin?.items ?? []).map((i) => [i.id, { title: i.title, formula: i.formula }]));
}

export interface PlanLine {
  index: number;
  name: string;
  status: StepStatus;
  /** The mono line: parameters, then the step's time, e.g. "(B5 − B6)/(B5 + B6) · after − before · threshold 0.2 · 2 s". */
  params: string;
}

export interface PlanSummary {
  /** "4 steps · 2 s". */
  heading: string;
  lines: PlanLine[];
}

/** A registry formula as the design sets it: "(Green−NIR)/(Green+NIR)" → "(Green − NIR)/(Green + NIR)". */
export const spacedFormula = (formula: string): string => formula.replace(/\s*([−+])\s*/g, ' $1 ');

/** The change threshold the analysis used: `statistics.water` / `vegetation` / `statistics.threshold`. */
function changeThreshold(job: Pick<JobRecord, 'result'>): number | undefined {
  const main = mainArtifact(job);
  if (!main || !CHANGE_KEYS.includes(main.key)) return undefined;
  const stats = obj(main.artifact.statistics);
  return [obj(stats.water).threshold, obj(stats.vegetation).threshold, stats.threshold].find(isNum);
}

const SHOWN_ELSEWHERE = new Set(['formula', 'name', 'threshold']);

function stepParams(step: PlanStep, registry: Registry, files: MessageAttachment[], threshold: number | undefined): string[] {
  const binds = obj(step.binds);
  const config = obj(step.config);
  if (step.uses.startsWith('input/')) {
    // "before · 2023-06-01": which file it is, from the bind's name and the file's date.
    return Object.entries(binds).flatMap(([key, value]) => {
      if (typeof value !== 'string') return [];
      const role = /Before$/.test(key) ? 'before' : /After$/.test(key) ? 'after' : '';
      const date = files.find((f) => f.path === value)?.metadata?.date;
      return [role, date].filter((p): p is string => Boolean(p));
    });
  }
  const change = step.uses.includes('change');
  const index = str(binds.IndexType) ?? str(config.index_type);
  const formula = str(config.formula)
    ? prettyFormula(String(config.formula))
    : (registry[change && index ? `algorithms/${index.toLowerCase()}` : step.uses]?.formula ?? null);
  const parts: string[] = [];
  if (formula) parts.push(str(config.formula) ? formula : spacedFormula(formula));
  if (change) parts.push(`after ${MINUS} before`);
  const t = isNum(config.threshold) ? config.threshold : change ? threshold : undefined;
  if (t !== undefined) parts.push(`threshold ${t}`);
  for (const [key, value] of Object.entries(config)) {
    if (SHOWN_ELSEWHERE.has(key) || !['string', 'number', 'boolean'].includes(typeof value)) continue;
    parts.push(`${key.replace(/_/g, ' ')} ${String(value)}`);
  }
  return parts;
}

function stepTime(status: StepStatus, elapsedS: number | undefined): string {
  switch (status) {
    case 'done':
      return elapsedS !== undefined ? formatDuration(elapsedS) : '';
    case 'running':
      return 'running';
    default:
      return status;
  }
}

/** The plan the job ran, or its proposal's plan. */
function planOf(job: Pick<JobRecord, 'proposal'> & { plan?: unknown }): PlanStep[] {
  const plan = obj(job.plan ?? job.proposal?.plan);
  return (Array.isArray(plan.steps) ? plan.steps : []).map(obj).filter((s): s is Obj & PlanStep => typeof s.uses === 'string');
}

/**
 * PLAN · N STEPS · total: the rows of the event log (`planning ready`, then the
 * step events) with each plan step's parameters and time. A Preparing model row has
 * no plan step. Without a row list (a proposal, an old job) the plan's steps, named
 * by the registry.
 */
export function planSummary(
  job: Pick<JobRecord, 'status' | 'result' | 'proposal' | 'events'> & { plan?: unknown; card?: { duration_s?: number | null } | null },
  registry: Registry,
  files: MessageAttachment[],
): PlanSummary {
  const view = analysisReducer(job.events ?? []);
  const steps = planOf(job);
  const threshold = changeThreshold(job);
  const rows = view.steps.length > 0
    ? view.steps
    : steps.map((s, i) => ({ index: i + 1, id: s.uses, name: registry[s.uses]?.title ?? s.uses, kind: '', status: (job.status === 'completed' ? 'done' : 'queued') as StepStatus, elapsedS: undefined }));
  let next = 0;
  const lines = rows.map((row): PlanLine => {
    const step = row.kind === 'prepare' ? undefined : (steps[next++] ?? { uses: row.id });
    const params = step ? stepParams(step, registry, files, threshold) : [];
    return { index: row.index, name: row.name, status: row.status, params: [...params, stepTime(row.status, row.elapsedS)].filter(Boolean).join(' · ') };
  });
  const n = lines.filter((_, i) => rows[i].kind !== 'prepare').length;
  const count = n === 0 ? 'no steps' : `${n} ${n === 1 ? 'step' : 'steps'}`;
  return { heading: [count, runTime(job, view)].filter(Boolean).join(' · '), lines };
}

// --- Statistics -------------------------------------------------------------------

export interface ClassRow {
  name: string;
  colour?: string;
  area: string;  // km², one decimal: "156.8", "3 570.2"
  percent: string;  // "29.5"
  mean?: string;  // signed, two decimals: "+0.81"
}

export type StatisticsTable =
  | { kind: 'classes'; meanLabel: string | null; rows: ClassRow[] }
  | { kind: 'bands'; rows: BandRow[] }
  | { kind: 'scalars'; rows: Stat[] };

/** One decimal, thousands grouped with spaces: 3570.24 → "3 570.2"; "—" when unknown. */
export function formatDecimal(value: unknown): string {
  if (!isNum(value)) return '—';
  const [int, frac] = Math.abs(value).toFixed(1).split('.');
  return `${value < 0 ? MINUS : ''}${int.replace(/\B(?=(\d{3})+(?!\d))/g, ' ')}.${frac}`;
}

/** A mean change: "+0.81", "−0.23"; "—" when unknown. */
export function formatSigned(value: unknown): string {
  if (!isNum(value)) return '—';
  const digits = Math.abs(value).toFixed(2);
  return `${value > 0 && digits !== '0.00' ? '+' : value < 0 && digits !== '0.00' ? MINUS : ''}${digits}`;
}

const isHex = (v: unknown): v is string => typeof v === 'string' && /^#[0-9a-f]{6}$/i.test(v);

/**
 * The Statistics table: per class from `class_statistics` (change with MEAN Δ<index>,
 * land cover, snow), per band for raster statistics, the scalar statistics of a
 * single index or field count; null when the result has none.
 */
export function statisticsTable(job: Pick<JobRecord, 'result'>, model: ResultCardModel | null): StatisticsTable | null {
  const main = mainArtifact(job);
  const classes = Object.values(obj(main?.artifact.class_statistics)).map(obj).filter((c) => typeof c.name === 'string');
  if (main && classes.length > 0) {
    const hasMean = classes.some((c) => isNum(c.mean_change));
    const label = indexLabel(main.artifact);
    return {
      kind: 'classes',
      meanLabel: hasMean ? `Mean Δ${label}` : null,
      rows: classes.map((c) => ({
        name: String(c.name),
        ...(isHex(c.colour) ? { colour: c.colour } : {}),
        area: formatDecimal(c.area_km2),
        percent: formatDecimal(c.percent ?? c.percentage),
        ...(hasMean ? { mean: formatSigned(c.mean_change) } : {}),
      })),
    };
  }
  if (model?.kind === 'statistics') return { kind: 'bands', rows: model.bands };
  if (model && (model.kind === 'index' || model.kind === 'fields')) return { kind: 'scalars', rows: model.stats };
  return null;
}

// --- Downloads ----------------------------------------------------------------------

export interface DownloadItem {
  name: string;  // the download's file name, e.g. "water-change_ndwi_2023-06-01_2023-06-09.tif"
  detail: string;  // "GeoTIFF · 2.0 MB"
  url: string;
}

const KIND_LABELS: Record<string, string> = { geotiff: 'GeoTIFF', gpkg: 'GeoPackage', png: 'PNG', csv: 'CSV' };
const KIND_ORDER = ['geotiff', 'gpkg', 'png'];

/** A file size: "1 KB" at least, "826 KB", "2.0 MB", "1.2 GB" (1024-based, as the upload's size_label). */
export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null || !Number.isFinite(bytes) || bytes < 0) return '';
  if (bytes < 1024 ** 2) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

function collectRefs(value: unknown, depth: number, into: ArtifactRef[]) {
  const ref = asRef(value);
  if (ref) {
    into.push(ref);
    return;
  }
  if (depth === 0) return;
  for (const v of Object.values(obj(value))) collectRefs(v, depth - 1, into);
}

/**
 * Every file the analysis wrote (artifact references `{kind, path, url, size_bytes}`,
 * the interpretation's copies left out), once each: rasters, then vectors, then images.
 */
export function downloadsOf(job: Pick<JobRecord, 'result'>): DownloadItem[] {
  const refs: ArtifactRef[] = [];
  for (const [key, value] of Object.entries(artifactsOf(job))) if (key !== 'agent_result') collectRefs(value, 3, refs);
  const seen = new Set<string>();
  const rank = (kind: string) => (KIND_ORDER.includes(kind) ? KIND_ORDER.indexOf(kind) : KIND_ORDER.length);
  return refs
    .filter((r) => !seen.has(r.path) && Boolean(seen.add(r.path)))
    .sort((a, b) => rank(a.kind) - rank(b.kind))
    .map((r) => {
      const size = (r as ArtifactRef & { size_bytes?: number }).size_bytes;
      return { name: downloadName(r), detail: [KIND_LABELS[r.kind] ?? r.kind.toUpperCase(), formatBytes(size)].filter(Boolean).join(' · '), url: r.url };
    });
}

// --- Map meta -------------------------------------------------------------------------

function analysedFiles(job: Pick<JobRecord, 'result'>, context: AnalysedContext): (MessageAttachment | undefined)[] {
  const artifacts = artifactsOf(job);
  const main = mainArtifact(job);
  const file = (p: unknown) => (typeof p === 'string' ? context.attachments.find((a) => a.path === p) : undefined);
  return main && CHANGE_KEYS.includes(main.key)
    ? [file(artifacts.RasterPathBefore), file(artifacts.RasterPathAfter)]
    : [file(artifacts.RasterPath) ?? file(main?.artifact.path)];
}

/** The analysed files' dates: [before, after] for a change, [date] for one file; '' when unknown. */
export function inputDates(job: Pick<JobRecord, 'result'>, context: AnalysedContext): string[] {
  return analysedFiles(job, context).map((f) => f?.metadata?.date ?? '');
}

/**
 * The map's mono line: "2023-06-01 → 2023-06-09 · 30 m · EPSG:32636", from the
 * analysed files' dates and resolution and the output's CRS; parts left out when unknown.
 */
export function mapMeta(job: Pick<JobRecord, 'result'>, context: AnalysedContext): string {
  const meta = obj(mainArtifact(job)?.artifact.metadata);
  const inputs = analysedFiles(job, context);
  const dates = inputDates(job, context).filter(Boolean);
  const transform = Array.isArray(meta.transform) ? meta.transform : [];
  const res = inputs.map((f) => f?.metadata?.resolution_m).find(isNum) ?? (isNum(transform[0]) ? Math.abs(transform[0]) : undefined);
  const crs = str(meta.crs) ?? inputs.map((f) => f?.metadata?.crs).find((c): c is string => typeof c === 'string');
  return [dates.join(' → '), res !== undefined ? `${Number.isInteger(res) ? res : res.toFixed(1)} m` : '', crs ?? ''].filter(Boolean).join(' · ');
}

/** The index of a change result as its maps are titled ("NDWI · before"); '' for other kinds. */
export function changeIndexLabel(job: Pick<JobRecord, 'result'>): string {
  const main = mainArtifact(job);
  return main && CHANGE_KEYS.includes(main.key) ? indexLabel(main.artifact) : '';
}

/** A legend range end: two decimals at most, a real minus, "+" when positive. */
export function formatRangeValue(value: number): string {
  const digits = String(Number(Math.abs(value).toFixed(2)));
  return `${value > 0 ? '+' : value < 0 ? MINUS : ''}${digits}`;
}
