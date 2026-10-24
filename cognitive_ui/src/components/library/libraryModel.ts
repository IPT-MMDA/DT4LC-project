import type { GeoTIFFFile } from '../../api/hooks/useUpload';
import type { JobSummary } from '../../api/library';
import { carriedContext } from '../../hooks/useSendMessage';
import { findConversationByJobId } from '../../store/useChatStore';
import type { Conversation } from '../../types';
import type { AnalysedContext } from '../chat/result/resultModel';

// The Library (design/screens/library.png): analyses (GET /v1/jobs, their stored
// `card`) and files (GET /v1/files) in one list, newest first, with filters and one
// action per row. Pure: Library.tsx renders LibraryRow and nothing else.

export type RowStatus = 'done' | 'failed' | 'cancelled' | 'running' | 'queued' | 'proposal';
/** The row's one button: Map, Chat, Retry, ↓ or CSV; null for none. */
export type RowAction = 'map' | 'chat' | 'retry' | 'download' | 'csv' | null;

interface RowBase {
  key: string;
  title: string;
  subtitle: string;  // the quoted prompt, a failure's message, or a file's line
  type: string;  // "Change detection", "File · sample"; '' when unknown
  conversation: string;  // a title, "2 conversations", or '—'
  time: number;  // ms since the epoch: the sort and the date filter
  thumbnail: string | null;  // server-relative URL
  action: RowAction;
}

export interface AnalysisRow extends RowBase {
  kind: 'analysis';
  job: JobSummary;
  status: RowStatus;
  duration: string;  // "48 s", "1 m 55 s"; '' when unknown
  conversationId: string | null;
}

export interface FileRow extends RowBase {
  kind: 'file';
  fileKind: 'upload' | 'sample' | 'export';
  files: GeoTIFFFile[];  // both dates for the sample
  deleteId: string | null;  // DELETE /v1/files/{id}; null: not deletable (an export)
}

export type LibraryRow = AnalysisRow | FileRow;

const STATUS: Record<JobSummary['status'], RowStatus> = {
  completed: 'done',
  failed: 'failed',
  cancelled: 'cancelled',
  running: 'running',
  pending: 'queued',
  proposal: 'proposal',
};

/** Card kinds whose result has a raster to put on the map. */
const RASTER_KINDS = new Set(['change', 'index', 'classification', 'fields', 'composed', 'image']);

const FILE_TYPES: Record<FileRow['fileKind'], string> = {
  sample: 'File · sample',
  upload: 'File · upload',
  export: 'File · export',
};

/** A run time as the Library shows it: "<1 s", "48 s", "1 m 55 s". */
export function formatRunTime(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return '';
  if (seconds < 1) return '<1 s';
  const total = Math.round(seconds);
  if (total < 60) return `${total} s`;
  const s = total % 60;
  return s ? `${Math.floor(total / 60)} m ${s} s` : `${Math.floor(total / 60)} m`;
}

const pad = (n: number) => String(n).padStart(2, '0');
const dayStart = (t: number) => new Date(new Date(t).toDateString()).getTime();

/** WHEN: "today 14:02", "yesterday", else "2026-10-21" (local time). */
export function whenLabel(time: number, now: number): string {
  const d = new Date(time);
  const days = Math.round((dayStart(now) - dayStart(time)) / 86_400_000);
  if (days === 0) return `today ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  if (days === 1) return 'yesterday';
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

/** WHEN of a row: the bundled sample says "bundled". */
export function rowWhen(row: LibraryRow, now: number): string {
  return row.kind === 'file' && row.fileKind === 'sample' ? 'bundled' : whenLabel(row.time, now);
}

/** The row's one action (the design's per-row button). */
export function rowAction(row: Pick<AnalysisRow, 'kind' | 'status' | 'job'> | Pick<FileRow, 'kind' | 'fileKind'>): RowAction {
  if (row.kind === 'file') return row.fileKind === 'export' ? 'download' : 'chat';
  if (row.status === 'failed') return 'retry';
  if (row.status !== 'done') return null;
  const kind = row.job.card?.kind;
  if (kind === 'statistics') return 'csv';
  return kind && RASTER_KINDS.has(kind) ? 'map' : null;
}

/** "Kakhovka reservoir — …", "2 conversations" or '—'. */
function conversationLabel(conversations: Conversation[]): string {
  if (conversations.length === 0) return '—';
  return conversations.length === 1 ? conversations[0].title : `${conversations.length} conversations`;
}

/** The conversations in this browser whose messages attach any of `paths`. */
export function conversationsWithFiles(conversations: Conversation[], paths: string[]): Conversation[] {
  const wanted = new Set(paths);
  return conversations.filter((c) => c.messages.some((m) => m.kind === 'user' && m.attachments?.some((a) => a.path && wanted.has(a.path))));
}

/** The analysis's conversation, the user message that asked for it and the files or region it was about. */
export function analysisContext(conversations: Conversation[], jobId: string): { conversation: Conversation | null; prompt: string; context: AnalysedContext } {
  const conversation = findConversationByJobId(conversations, jobId);
  const messages = conversation?.messages ?? [];
  const index = messages.findIndex((m) => m.kind === 'analysis' && m.jobId === jobId);
  const before = messages.slice(0, Math.max(index, 0));
  const asked = [...before].reverse().find((m) => m.kind === 'user');
  return { conversation, prompt: asked?.kind === 'user' ? asked.text : '', context: carriedContext(before) };
}

const parseTime = (iso: string | null | undefined, fallback = 0) => {
  const t = iso ? Date.parse(iso) : NaN;
  return Number.isFinite(t) ? t : fallback;
};

function analysisRow(job: JobSummary, conversations: Conversation[]): AnalysisRow {
  const card = job.card ?? null;
  const prompt = job.prompt?.trim() ?? '';
  const conversation = findConversationByJobId(conversations, job.id);
  const status = STATUS[job.status] ?? 'queued';
  const row: AnalysisRow = {
    kind: 'analysis',
    key: `job-${job.id}`,
    job,
    // card null (running, or a row from before BR2): the prompt is the name.
    title: card?.title || prompt || `Job ${job.id}`,
    subtitle: card?.error?.message ?? (card && prompt ? `"${prompt}"` : ''),
    type: card?.analysis?.label ?? '',
    status,
    duration: formatRunTime(card?.duration_s),
    conversation: conversation?.title ?? '—',
    conversationId: conversation?.id ?? null,
    time: parseTime(job.created_at),
    thumbnail: card?.thumbnail_url ?? null,
    action: null,
  };
  row.action = rowAction(row);
  return row;
}

/** One row per upload or export, one for both dates of a sample (they share `upload_id`). */
function fileRows(files: GeoTIFFFile[], conversations: Conversation[]): FileRow[] {
  const groups = new Map<string, GeoTIFFFile[]>();
  for (const f of files) {
    const key = f.kind === 'sample' && f.upload_id ? `sample-${f.upload_id}` : `file-${f.path}`;
    groups.set(key, [...(groups.get(key) ?? []), f]);
  }
  return [...groups.entries()].map(([key, pair]) => {
    const first = [...pair].sort((a, b) => a.filename.localeCompare(b.filename))[0];
    const fileKind = first.kind ?? (first.source === 'export' ? 'export' : 'upload');
    const meta = first.metadata;
    const line = first.group?.line ?? [meta?.line, meta?.size_label].filter(Boolean).join(' · ');
    const row: FileRow = {
      kind: 'file',
      key,
      fileKind,
      files: pair,
      // Uploads are stored as <upload_id>_<original name>.
      title: first.group?.name ?? (first.upload_id && first.filename.startsWith(`${first.upload_id}_`) ? first.filename.slice(first.upload_id.length + 1) : first.filename),
      subtitle: line,
      type: FILE_TYPES[fileKind],
      conversation: conversationLabel(conversationsWithFiles(conversations, pair.map((f) => f.path))),
      time: Math.max(...pair.map((f) => parseTime(f.created_at, f.modified * 1000))),
      thumbnail: pair.find((f) => f.preview_url)?.preview_url ?? null,
      deleteId: fileKind === 'export' ? null : (first.upload_id ?? first.id),
      action: null,
    };
    row.action = rowAction(row);
    return row;
  });
}

/** Analyses and files merged, newest first. */
export function libraryRows(jobs: JobSummary[], files: GeoTIFFFile[], conversations: Conversation[]): LibraryRow[] {
  return [...jobs.map((j) => analysisRow(j, conversations)), ...fileRows(files, conversations)].sort((a, b) => b.time - a.time);
}

// --- Filters ---------------------------------------------------------------------------

export type Show = 'all' | 'analyses' | 'files';
export type StatusFilter = 'done' | 'failed' | 'cancelled';
export type Since = 'today' | '7d' | '30d' | 'all';

export interface LibraryFilters {
  show: Show;
  statuses: StatusFilter[];  // none: any status (and files)
  type: string | null;  // null: any type
  since: Since;
}

export const DEFAULT_FILTERS: LibraryFilters = { show: 'all', statuses: [], type: null, since: '7d' };

export const SINCE_LABELS: Record<Since, string> = { today: 'Today', '7d': 'Last 7 days', '30d': 'Last 30 days', all: 'All time' };

/** The earliest time `since` keeps. */
function sinceTime(since: Since, now: number): number {
  switch (since) {
    case 'today':
      return dayStart(now);
    case '7d':
      return now - 7 * 86_400_000;
    case '30d':
      return now - 30 * 86_400_000;
    case 'all':
      return -Infinity;
  }
}

/** The rows the filters keep. A status chip keeps analyses with that status only. */
export function filterRows(rows: LibraryRow[], filters: LibraryFilters, now: number): LibraryRow[] {
  const from = sinceTime(filters.since, now);
  return rows.filter((row) => {
    if (filters.show === 'analyses' && row.kind !== 'analysis') return false;
    if (filters.show === 'files' && row.kind !== 'file') return false;
    if (filters.statuses.length > 0 && !(row.kind === 'analysis' && (filters.statuses as RowStatus[]).includes(row.status))) return false;
    if (filters.type !== null && row.type !== filters.type) return false;
    return row.time >= from;
  });
}

/** Type ▾'s choices: the analysis labels in the list, then its file types. */
export function typeOptions(rows: LibraryRow[]): string[] {
  const analyses = new Set<string>();
  const files = new Set<string>();
  for (const row of rows) if (row.type) (row.kind === 'analysis' ? analyses : files).add(row.type);
  return [...[...analyses].sort((a, b) => a.localeCompare(b)), ...Object.values(FILE_TYPES).filter((t) => files.has(t))];
}
