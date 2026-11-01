import { cn } from '../../../utils/cn';

// Status pills are always glyph + word + hue: Queued ○, Running ◌,
// Done ✓, Failed !, Cancelled –; Planning shares the running hue, Proposal the queued one.

export type PillStatus = 'queued' | 'proposal' | 'planning' | 'running' | 'done' | 'failed' | 'cancelled';

const WORD: Record<PillStatus, string> = {
  queued: 'Queued',
  proposal: 'Proposal',
  planning: 'Planning',
  running: 'Running',
  done: 'Done',
  failed: 'Failed',
  cancelled: 'Cancelled',
};

const HUE: Record<PillStatus, string> = {
  queued: 'bg-surface2 text-muted',
  proposal: 'bg-surface2 text-muted',
  planning: 'bg-run-tint text-run-ink',
  running: 'bg-run-tint text-run-ink',
  done: 'bg-done-tint text-done-ink',
  failed: 'bg-fail-tint text-fail-ink',
  cancelled: 'bg-canc-tint text-canc-ink',
};

/** The glyph alone, also used at the start of each timeline row (14 px there). */
export function StatusGlyph({ status, size = 12 }: { status: PillStatus; size?: number }) {
  const box = { width: size, height: size };
  const glyph = { ...box, color: '#fff' };  // white on the solid hue, in both themes (prototype)
  const solid = 'rounded-full flex-none flex items-center justify-center font-bold';
  switch (status) {
    case 'done':
      return <span aria-hidden style={glyph} className={cn(solid, 'bg-done text-[8px]')}>✓</span>;
    case 'failed':
      return <span aria-hidden style={glyph} className={cn(solid, 'bg-fail text-[9px]')}>!</span>;
    case 'cancelled':
      return <span aria-hidden style={glyph} className={cn(solid, 'bg-canc text-[9px]')}>–</span>;
    case 'running':
      return (
        <span
          aria-hidden
          style={box}
          className="flex-none rounded-full border-2 border-run border-t-transparent animate-spin motion-reduce:animate-none"
        />
      );
    case 'planning':
      return <span aria-hidden style={box} className="flex-none rounded-full border-[1.5px] border-dashed border-run animate-pulse motion-reduce:animate-none" />;
    case 'queued':
    case 'proposal':
      return <span aria-hidden style={box} className="flex-none rounded-full border-[1.5px] border-dashed border-faint" />;
  }
}

/** `detail` follows the word, e.g. the run time in the Library: "Done · 48 s". */
export function StatusPill({ status, detail }: { status: PillStatus; detail?: string }) {
  return (
    <span
      data-testid="status-pill"
      data-status={status}
      className={cn('inline-flex items-center gap-[5px] rounded-pill pl-[6px] pr-2 py-[2px] text-[11.5px] font-semibold', HUE[status])}
    >
      <StatusGlyph status={status} />
      {detail ? `${WORD[status]} · ${detail}` : WORD[status]}
    </span>
  );
}
