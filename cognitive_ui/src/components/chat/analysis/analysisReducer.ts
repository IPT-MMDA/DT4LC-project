import type { AnalysisEvent, AnalysisStatus } from '../../../types';

// One analysis message is a pure function of its job's events (DESIGN §4): the
// `{seq, type, data}` list that GET /v1/jobs/{id}/events streams and
// GET /v1/jobs/{id} returns (docs/API.md "GET /v1/jobs/{job_id}/events").

export type AnalysisPhase = 'starting' | 'planning' | 'proposal' | 'running' | 'done' | 'failed' | 'cancelled';
export type StepStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled' | 'skipped';
export type StepKind = 'input' | 'algorithm' | 'model' | 'postprocess' | 'preprocessor' | 'prepare';
export type FixKind = 'prompt' | 'retry' | 'upload' | 'open_map' | 'sample';

/** `step_progress` of a row; a Preparing model row has bytes and an ETA. */
export interface StepProgress {
  fraction?: number;
  message?: string;
  bytes?: number;
  totalBytes?: number;
  etaS?: number;
}

export interface StepRow {
  index: number;  // 1-based, as the server numbers them
  id: string;
  name: string;
  kind: StepKind | string;
  status: StepStatus;
  startedT?: number;  // server time (Unix s) of step_started
  elapsedS?: number;  // step_finished
  progress?: StepProgress;
}

export interface Fix {
  label: string;
  kind: FixKind;
  prompt?: string;
  params?: Record<string, unknown>;
}

export interface AnalysisFailure {
  code: string;
  message: string;
  fixes: Fix[];
  raw: string;
  stepIndex?: number;
}

export interface AnalysisView {
  /** Events applied so far; resume the stream with `?after=<lastSeq>`. */
  lastSeq: number;
  intent: string;
  phase: AnalysisPhase;
  /** `proposal` event; stays set after Run so the message keeps the proposal text. */
  proposal: { text: string; plan: unknown } | null;
  /** A `planning ready` came after the proposal: Run was accepted. */
  confirmed: boolean;
  /** `planning ready` `composed`: the "Composed by the assistant" badge. */
  composed: boolean;
  steps: StepRow[];
  /** The running row's index, or null. */
  current: number | null;
  /** Overall progress 0–1: finished rows plus the running row's fraction. */
  progress: number;
  /** Server time of the first and the terminal event (Unix s). */
  startedT?: number;
  endedT?: number;
  done: { elapsedS: number; summary: string; followups: string[] } | null;
  failure: AnalysisFailure | null;
  cancelled: { atStep?: number; reason?: string } | null;
}

export const TERMINAL_TYPES = ['done', 'failed', 'cancelled'] as const;

export const EMPTY_VIEW: AnalysisView = {
  lastSeq: 0,
  intent: '',
  phase: 'starting',
  proposal: null,
  confirmed: false,
  composed: false,
  steps: [],
  current: null,
  progress: 0,
  done: null,
  failure: null,
  cancelled: null,
};

const str = (v: unknown, fallback = ''): string => (typeof v === 'string' ? v : fallback);
const num = (v: unknown): number | undefined => (typeof v === 'number' && Number.isFinite(v) ? v : undefined);

/** `data` of a stored event ({seq, type, data}); a flattened event is its own data. */
export function eventData(e: AnalysisEvent): Record<string, unknown> {
  return typeof e.data === 'object' && e.data !== null ? (e.data as Record<string, unknown>) : e;
}

const seqOf = (e: AnalysisEvent): number | undefined => num(e.seq);

/**
 * Merge `incoming` into `existing` by `seq` (a resumed stream may replay events the
 * client has), sorted by `seq`. Only the latest `step_progress` of each row is kept,
 * so a long download does not grow the timeline. Events without a `seq` are kept in
 * order, after the numbered ones.
 */
export function mergeEvents(existing: AnalysisEvent[], incoming: AnalysisEvent[]): AnalysisEvent[] {
  const bySeq = new Map<number, AnalysisEvent>();
  const unnumbered: AnalysisEvent[] = [];
  for (const e of [...existing, ...incoming]) {
    const seq = seqOf(e);
    if (seq === undefined) unnumbered.push(e);
    else if (!bySeq.has(seq)) bySeq.set(seq, e);
  }
  const sorted = [...bySeq.entries()].sort(([a], [b]) => a - b).map(([, e]) => e);
  const all = [...sorted, ...unnumbered];
  const latestProgress = new Map<unknown, AnalysisEvent>();
  for (const e of all) if (e.type === 'step_progress') latestProgress.set(eventData(e).index, e);
  return all.filter((e) => e.type !== 'step_progress' || latestProgress.get(eventData(e).index) === e);
}

function toFix(v: unknown): Fix | null {
  const f = v as Partial<Fix> | null;
  if (!f || typeof f.label !== 'string' || typeof f.kind !== 'string') return null;
  return {
    label: f.label,
    kind: f.kind as FixKind,
    ...(typeof f.prompt === 'string' ? { prompt: f.prompt } : {}),
    ...(f.params && typeof f.params === 'object' ? { params: f.params } : {}),
  };
}

/** The timeline of one analysis from its events (in any order, duplicates allowed). */
export function analysisReducer(events: AnalysisEvent[]): AnalysisView {
  const ordered = mergeEvents([], events);
  const view: AnalysisView = { ...EMPTY_VIEW, steps: [] };
  const row = (index: unknown) => view.steps.find((s) => s.index === index);

  for (const e of ordered) {
    const d = eventData(e);
    const t = num(d.t);
    view.lastSeq = Math.max(view.lastSeq, seqOf(e) ?? 0);
    if (t !== undefined && view.startedT === undefined) view.startedT = t;

    switch (e.type) {
      case 'intent':
        view.intent = str(d.text);
        break;
      case 'planning':
        if (d.phase === 'ready') {
          const steps = Array.isArray(d.steps) ? (d.steps as Record<string, unknown>[]) : [];
          view.steps = steps.map((s, i) => ({
            index: num(s.index) ?? i + 1,
            id: str(s.id),
            name: str(s.name, str(s.id)),
            kind: str(s.kind),
            status: 'queued',
          }));
          view.composed = d.composed === true;
          if (view.proposal) view.confirmed = true;
        }
        view.phase = 'planning';
        break;
      case 'proposal':
        view.proposal = { text: str(d.text), plan: d.plan ?? null };
        view.phase = 'proposal';
        break;
      case 'step_started': {
        const r = row(d.index);
        if (r) {
          r.status = 'running';
          r.startedT = t;
        }
        view.current = num(d.index) ?? null;
        view.phase = 'running';
        break;
      }
      case 'step_progress': {
        const r = row(d.index);
        if (r) {
          r.progress = {
            fraction: num(d.fraction),
            message: typeof d.message === 'string' ? d.message : undefined,
            bytes: num(d.bytes),
            totalBytes: num(d.total_bytes),
            etaS: num(d.eta_s),
          };
        }
        break;
      }
      case 'step_finished': {
        const r = row(d.index);
        if (r) {
          r.status = 'done';
          r.elapsedS = num(d.elapsed_s);
        }
        if (view.current === num(d.index)) view.current = null;
        break;
      }
      case 'done':
        view.steps.forEach((s) => (s.status = 'done'));
        view.done = {
          elapsedS: num(d.elapsed_s) ?? 0,
          summary: str(d.summary),
          followups: Array.isArray(d.followups) ? d.followups.filter((f): f is string => typeof f === 'string') : [],
        };
        view.phase = 'done';
        view.current = null;
        view.endedT = t;
        break;
      case 'failed': {
        const stepIndex = num(d.step_index) ?? view.current ?? undefined;
        view.failure = {
          code: str(d.code, 'INTERNAL'),
          message: str(d.message),
          fixes: (Array.isArray(d.fixes) ? d.fixes : []).map(toFix).filter((f): f is Fix => f !== null),
          raw: str(d.raw),
          ...(stepIndex !== undefined ? { stepIndex } : {}),
        };
        view.steps.forEach((s) => {
          if (s.index === stepIndex) s.status = 'failed';
          else if (s.status === 'queued' || s.status === 'running') s.status = 'skipped';
        });
        view.phase = 'failed';
        view.current = null;
        view.endedT = t;
        break;
      }
      case 'cancelled': {
        const atStep = num(d.at_step) ?? view.current ?? undefined;
        view.cancelled = { ...(atStep !== undefined ? { atStep } : {}), ...(typeof d.reason === 'string' ? { reason: d.reason } : {}) };
        view.steps.forEach((s) => {
          if (s.status === 'running') s.status = 'cancelled';
          else if (s.status === 'queued') s.status = 'skipped';
        });
        view.phase = 'cancelled';
        view.current = null;
        view.endedT = t;
        break;
      }
      default:
        break;
    }
    // Nothing is logged after a terminal event; ignore anything that is.
    if (view.phase === 'done' || view.phase === 'failed' || view.phase === 'cancelled') break;
  }

  const n = view.steps.length;
  if (view.phase === 'done') view.progress = 1;
  else if (n > 0) {
    const finished = view.steps.filter((s) => s.status === 'done').length;
    const running = view.steps.find((s) => s.status === 'running');
    view.progress = Math.min(1, (finished + (running?.progress?.fraction ?? 0)) / n);
  }
  return view;
}

/** The store's coarse status for a view: what the conversation list and history read. */
export function statusOf(view: AnalysisView): AnalysisStatus {
  switch (view.phase) {
    case 'running':
      return 'running';
    case 'done':
    case 'failed':
    case 'cancelled':
      return view.phase;
    default:
      return 'pending';
  }
}

export const isTerminal = (view: AnalysisView) =>
  view.phase === 'done' || view.phase === 'failed' || view.phase === 'cancelled';

/** The highest `seq` among `events` (0 when none). */
export function lastSeq(events: AnalysisEvent[]): number {
  return events.reduce((max, e) => Math.max(max, seqOf(e) ?? 0), 0);
}

