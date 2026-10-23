import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { ChatError } from '../../../api/chat';
import { SESSION_QUERY_KEY } from '../../../api/hooks/useSession';
import { cancelJob, confirmJob } from '../../../api/jobs';
import { applyJobEvents, useJobEvents } from '../../../hooks/useJobEvents';
import { attachSample } from '../../../hooks/useSample';
import { useChatStore } from '../../../store/useChatStore';
import type { AnalysisMessage as AnalysisMessageData } from '../../../types';
import { cn } from '../../../utils/cn';
import { Avatar } from '../Avatar';
import { analysisReducer, isTerminal, type AnalysisView, type Fix, type StepRow } from './analysisReducer';
import { downloadLine, formatDuration } from './format';
import { changeDraft, EXPLAIN_PROPOSAL } from './proposal';
import { ResultCard } from '../result/ResultCard';
import type { AnalysedContext } from '../result/resultModel';
import { StatusGlyph, StatusPill, type PillStatus } from './StatusPill';

// One assistant message narrates one analysis (DESIGN §4): intent, proposal with
// Run · Change · Explain, the Planning skeleton, the running timeline with Cancel,
// then the collapsed Done line with the result card, interpretation and follow-ups,
// or the open Failed timeline with its fixes.

export interface AnalysisMessageProps {
  message: AnalysisMessageData;
  /** The user message that started the analysis: what `retry` and `sample` resend. */
  prompt: string;
  /** The files or region the analysis was asked about: the result card's source, dates and bounds. */
  context?: AnalysedContext;
  onSend: (text: string, options?: { params?: Record<string, unknown> }) => void;
  onOpenFilePicker: () => void;
}

const primary = 'px-3 py-[6px] rounded-control bg-accent text-on-accent font-medium disabled:opacity-50';
const outline = 'px-3 py-[6px] rounded-control border border-border bg-surface text-text hover:bg-surface2 disabled:opacity-50';

/** Now in Unix seconds, ticking while `active` (a running row's elapsed time against its server `t`). */
function useNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    if (!active) return;
    setNow(Date.now() / 1000);
    const timer = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(timer);
  }, [active]);
  return now;
}

const errorText = (err: unknown) => (err instanceof Error ? err.message : String(err));

function ComposedBadge() {
  return (
    <span data-testid="composed-badge" className="inline-flex items-center rounded-pill px-2 py-[2px] bg-accent-tint text-accent-ink font-semibold text-[11px]">
      Composed by the assistant
    </span>
  );
}

// "NDMI = (NIR − SWIR1)/(NIR + SWIR1)" in a proposal is set in mono, as in the design.
const FORMULA = /\b[A-Z][A-Z0-9]+ = \([^)]+\)\s*\/\s*\([^)]+\)/;

function ProposalText({ text }: { text: string }) {
  const m = FORMULA.exec(text);
  if (!m) return <>{text}</>;
  return (
    <>
      {text.slice(0, m.index)}
      <code className="font-mono text-[13px] bg-surface2 px-[5px] py-[1px] rounded-chip">{m[0]}</code>
      {text.slice(m.index + m[0].length)}
    </>
  );
}

/** The title of a result card of no known kind: the analysis steps, e.g. "Prithvi Reconstruction". */
function resultTitle(view: AnalysisView): string {
  const names = view.steps.filter((s) => s.kind === 'algorithm' || s.kind === 'model').map((s) => s.name);
  return names.join(' → ') || 'Result';
}

function rowGlyph(status: StepRow['status']): PillStatus {
  if (status === 'skipped') return 'queued';
  return status;
}

function rowTime(s: StepRow, view: AnalysisView, now: number): string {
  switch (s.status) {
    case 'done':
      return s.elapsedS !== undefined ? formatDuration(s.elapsedS) : '';
    case 'running':
      return s.startedT !== undefined ? formatDuration(Math.max(0, now - s.startedT)) : '';
    case 'failed':
      return s.startedT !== undefined && view.endedT !== undefined ? formatDuration(view.endedT - s.startedT) : '';
    case 'queued':
      return 'queued';
    case 'skipped':
      return 'skipped';
    default:
      return '';
  }
}

function Steps({ view, now }: { view: AnalysisView; now: number }) {
  return (
    <ol aria-label="Steps" className="flex flex-col gap-[7px] text-control">
      {view.steps.map((s) => {
        const quiet = s.status === 'queued' || s.status === 'skipped';
        const downloading = s.kind === 'prepare' && s.status === 'running' && s.progress;
        return (
          <li key={s.index} data-testid="step-row" data-status={s.status} data-kind={s.kind} className="flex flex-col gap-1">
            <div className="flex items-center gap-[9px]">
              <StatusGlyph status={rowGlyph(s.status)} size={14} />
              <span
                className={cn(
                  'flex-1 min-w-0',
                  quiet ? 'text-faint' : s.status === 'done' ? 'text-muted' : 'font-medium',
                )}
              >
                {s.name}
              </span>
              <span className={cn('font-mono text-[11.5px]', quiet ? 'text-faint' : 'text-muted')}>{rowTime(s, view, now)}</span>
            </div>
            {downloading && (
              <div className="ml-[23px] flex flex-col gap-1" data-testid="download-progress">
                <ProgressBar value={s.progress?.fraction ?? 0} label={`Downloading ${s.name}`} />
                <div className="font-mono text-[11px] text-muted">{downloadLine(s.progress ?? {})}</div>
              </div>
            )}
          </li>
        );
      })}
    </ol>
  );
}

function ProgressBar({ value, label }: { value: number; label: string }) {
  const pct = Math.round(Math.min(1, Math.max(0, value)) * 100);
  return (
    <div role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct} className="h-[3px] rounded-[2px] bg-border2 overflow-hidden">
      <div className="h-full bg-run transition-[width] duration-500" style={{ width: `${pct}%` }} />
    </div>
  );
}

/** "Step 2 of 4", "at step 1 of 2 · 12 s", "Choosing steps…". */
function phaseLabel(view: AnalysisView): string {
  const n = view.steps.length;
  const pos = (index?: number) => view.steps.findIndex((s) => s.index === index) + 1;
  const took = view.startedT !== undefined && view.endedT !== undefined ? formatDuration(view.endedT - view.startedT) : '';
  switch (view.phase) {
    case 'planning':
    case 'proposal':
      return n > 0 ? `${n} steps` : 'Choosing steps…';
    case 'running':
      return view.current !== null && pos(view.current) > 0 ? `Step ${pos(view.current)} of ${n}` : `${n} steps`;
    case 'failed': {
      const at = pos(view.failure?.stepIndex);
      return [at > 0 ? `at step ${at} of ${n}` : 'while planning', took].filter(Boolean).join(' · ');
    }
    case 'cancelled': {
      if (view.cancelled?.reason === 'expired') return 'the proposal expired';
      const at = pos(view.cancelled?.atStep);
      return at > 0 ? `at step ${at} of ${n}` : '';
    }
    default:
      return '';
  }
}

function pillOf(view: AnalysisView): PillStatus {
  switch (view.phase) {
    case 'starting':
      return 'queued';
    case 'planning':
    case 'proposal':
      return 'planning';
    default:
      return view.phase;
  }
}

function Failure({ view, onFix }: { view: AnalysisView; onFix: (fix: Fix) => void }) {
  const [showRaw, setShowRaw] = useState(false);
  const failure = view.failure;
  if (!failure) return null;
  return (
    <>
      <div data-testid="analysis-failure" className="rounded-control bg-fail-tint px-3 py-[10px] text-[13px] leading-[1.5] text-fail-ink">
        {failure.message}{' '}
        <span className="font-mono text-[11.5px] text-muted">
          <span data-testid="failure-code">{failure.code}</span>
          {failure.raw && (
            <>
              {' · '}
              <button type="button" aria-expanded={showRaw} onClick={() => setShowRaw((v) => !v)} className="underline-offset-2 hover:underline">
                {showRaw ? 'hide raw' : 'show raw'}
              </button>
            </>
          )}
        </span>
        {showRaw && (
          <pre data-testid="failure-raw" className="mt-2 whitespace-pre-wrap break-words font-mono text-[11.5px] text-text2">
            {failure.raw}
          </pre>
        )}
      </div>
      {failure.fixes.length > 0 && (
        <div aria-label="Fixes" role="group" className="flex flex-wrap gap-2 text-control">
          {failure.fixes.map((fix, i) => (
            <button key={`${fix.kind}-${fix.label}`} type="button" onClick={() => onFix(fix)} className={i === 0 ? primary : outline}>
              {fix.label}
            </button>
          ))}
        </div>
      )}
    </>
  );
}

function Timeline({
  view,
  now,
  header = true,
  onCancel,
  cancelling,
  children,
}: {
  view: AnalysisView;
  now: number;
  header?: boolean;
  onCancel?: () => void;
  cancelling?: boolean;
  children?: ReactNode;
}) {
  const label = phaseLabel(view);
  return (
    <div data-testid="analysis-timeline" className="border border-border rounded-card bg-surface px-[14px] py-3 flex flex-col gap-[10px]">
      {header && (
        <div className="flex items-center gap-[10px] flex-wrap">
          {view.composed && <ComposedBadge />}
          <StatusPill status={pillOf(view)} />
          {label && <span className="text-control text-text2">{label}</span>}
          <div className="flex-1" />
          {onCancel && (
            <button
              type="button"
              onClick={onCancel}
              disabled={cancelling}
              className="text-[12px] text-text2 px-[9px] py-1 border border-border rounded-[5px] hover:bg-surface2 disabled:opacity-50"
            >
              Cancel
            </button>
          )}
        </div>
      )}
      {view.steps.length > 0 && <Steps view={view} now={now} />}
      {view.phase === 'running' && <ProgressBar value={view.progress} label="Analysis progress" />}
      {children}
    </div>
  );
}

const NO_CONTEXT: AnalysedContext = { attachments: [] };

export function AnalysisMessage({ message, prompt, context = NO_CONTEXT, onSend, onOpenFilePicker }: AnalysisMessageProps) {
  const view = useMemo(() => analysisReducer(message.events), [message.events]);
  const terminal = isTerminal(view);
  useJobEvents(message.jobId, terminal || Boolean(message.error));
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const now = useNow(view.phase === 'running');
  const [busy, setBusy] = useState<'confirm' | 'cancel' | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [accepted, setAccepted] = useState(false);
  const [showSteps, setShowSteps] = useState(false);

  const intent = view.intent || message.text;
  const isAccepted = view.confirmed || accepted;
  const canCancel = !terminal && !message.error && (view.phase === 'planning' || view.phase === 'running' || view.phase === 'starting');

  const act = async (kind: 'confirm' | 'cancel', call: () => Promise<{ events?: unknown[] }>) => {
    setBusy(kind);
    setActionError(null);
    try {
      const job = await call();
      if (kind === 'confirm') {
        setAccepted(true);
        // A guest's analysis is counted at confirm: the badge's quota changed.
        void queryClient.invalidateQueries({ queryKey: SESSION_QUERY_KEY });
      }
      applyJobEvents(message.jobId, (job.events ?? []) as AnalysisMessageData['events']);
    } catch (err) {
      setActionError(errorText(err));
      if (err instanceof ChatError && err.code === 'GUEST_QUOTA') void queryClient.invalidateQueries({ queryKey: SESSION_QUERY_KEY });
    } finally {
      setBusy(null);
    }
  };
  const run = () => act('confirm', () => confirmJob(message.jobId));
  const cancel = () => act('cancel', () => cancelJob(message.jobId));
  const change = () => {
    useChatStore.getState().setDraft(changeDraft(prompt, view.proposal?.plan));
    document.getElementById('composer-input')?.focus();
  };

  const onFix = async (fix: Fix) => {
    setActionError(null);
    switch (fix.kind) {
      case 'prompt':
        if (fix.prompt) onSend(fix.prompt);
        break;
      case 'retry':
        onSend(prompt, fix.params ? { params: fix.params } : undefined);
        break;
      case 'upload':
        onOpenFilePicker();
        break;
      case 'open_map':
        navigate('/app/map');
        break;
      case 'sample':
        try {
          await attachSample();
          onSend(prompt);
        } catch (err) {
          setActionError(`Couldn't attach the sample: ${errorText(err)}`);
        }
        break;
    }
  };

  const alert = (actionError || message.error) && (
    <p role="alert" className="text-control text-fail-ink">
      {actionError || message.error}
    </p>
  );

  const analysis = (
    <div className="flex gap-3 min-w-0">
      <Avatar />
      <div className="flex-1 min-w-0 flex flex-col gap-3">
        <p className="text-body">{intent || 'Starting the analysis…'}</p>
        {view.phase === 'done' && view.done ? (
          <>
            <div className="flex items-center gap-[10px] text-control text-muted flex-wrap" data-testid="done-line">
              <StatusPill status="done" />
              <span>{view.done.summary}</span>
              {view.composed && <ComposedBadge />}
              {view.steps.length > 0 && (
                <button type="button" aria-expanded={showSteps} onClick={() => setShowSteps((v) => !v)} className="text-accent-ink hover:underline">
                  {showSteps ? 'Hide steps' : 'Show steps'}
                </button>
              )}
            </div>
            {showSteps && <Timeline view={view} now={now} header={false} />}
            <ResultCard jobId={message.jobId} context={context} fallbackTitle={resultTitle(view)} />
            {message.result?.summary && (
              <p className="text-body" data-testid="interpretation">
                {message.result.summary}
              </p>
            )}
            {view.done.followups.length > 0 && (
              <ul aria-label="Follow-ups" className="flex flex-wrap gap-2">
                {view.done.followups.map((f) => (
                  <li key={f}>
                    <button
                      type="button"
                      onClick={() => onSend(f)}
                      className="px-3 py-[5px] rounded-pill border border-border bg-surface text-control text-text2 hover:bg-surface2"
                    >
                      {f}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </>
        ) : view.phase !== 'starting' ? (
          <Timeline view={view} now={now} onCancel={canCancel ? cancel : undefined} cancelling={busy === 'cancel'}>
            {view.phase === 'failed' && <Failure view={view} onFix={(fix) => void onFix(fix)} />}
          </Timeline>
        ) : null}
        {!view.proposal && alert}
      </div>
    </div>
  );

  return (
    <li
      className="flex flex-col gap-5 min-w-0"
      data-testid="analysis-message"
      data-job-id={message.jobId}
      data-phase={view.phase}
    >
      {view.proposal ? (
        <>
          <div className="flex gap-3 min-w-0" data-testid="proposal">
            <Avatar />
            <div className="flex-1 min-w-0 flex flex-col gap-[10px]">
              <p className="text-body">
                <ProposalText text={view.proposal.text} />
              </p>
              {isAccepted ? (
                <p className="text-[12px] text-muted">Accepted · running below</p>
              ) : view.phase === 'proposal' ? (
                <div className="flex items-center gap-2 text-control">
                  <span className="inline-flex items-center rounded-pill px-2 py-[2px] border border-border text-muted font-semibold text-[11px]">
                    Proposal
                  </span>
                  <button type="button" onClick={() => void run()} disabled={busy !== null} className={primary}>
                    Run
                  </button>
                  <button type="button" onClick={change} className={outline}>
                    Change
                  </button>
                  <button type="button" onClick={() => onSend(EXPLAIN_PROPOSAL)} className={outline}>
                    Explain
                  </button>
                </div>
              ) : view.phase === 'cancelled' ? (
                <div className="flex items-center gap-2 text-control text-text2">
                  <StatusPill status="cancelled" />
                  {phaseLabel(view)}
                </div>
              ) : null}
              {alert}
            </div>
          </div>
          {isAccepted && analysis}
        </>
      ) : (
        analysis
      )}
    </li>
  );
}
