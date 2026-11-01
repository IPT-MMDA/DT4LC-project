import { useRef, useState, type ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  useCapabilities,
  type AskTier,
  type BuiltinItem,
  type BuiltinTier,
  type Capabilities as CapabilitiesDoc,
  type ComingTier,
  type ComposedTier,
  type DataTier,
  type ModelsTier,
} from '../api/hooks/useCapabilities';
import { promptHref } from '../hooks/usePromptFromQuery';
import { cn } from '../utils/cn';

// Status hue per status word (glyph + word + hue)
// screens: done for connected / installed / available, canc for the assistant offline,
// run (dashed) for a download on first use, muted for gated, not configured and coming.
type Hue = 'done' | 'canc' | 'run' | 'muted';
const HUE: Record<string, Hue> = {
  connected: 'done',
  available: 'done',
  installed: 'done',
  always: 'done',
  configured: 'done',
  offline: 'canc',
  downloads_on_first_use: 'run',
  not_available_for_guests: 'muted',
  not_installed: 'muted',
  not_configured: 'muted',
  not_bundled: 'muted',
  coming: 'muted',
};
const hueOf = (status: string): Hue => HUE[status] ?? 'muted';

/** A built-in analysis whose model this role cannot get: its prompt is shown but cannot be used. */
const isGated = (item: BuiltinItem) => item.status === 'not_available_for_guests' || item.status === 'not_installed';

const DOT: Record<Hue, string> = {
  done: 'bg-done',
  canc: 'bg-canc',
  run: 'border-[1.5px] border-dashed border-run',
  muted: 'bg-faint',
};
const PILL: Record<Hue, string> = {
  done: 'bg-done-tint text-done-ink',
  canc: 'bg-canc-tint text-canc-ink',
  run: 'bg-run-tint text-run-ink',
  muted: 'bg-surface2 text-muted',
};
const LINE: Record<Hue, string> = { done: 'text-done-ink', canc: 'text-canc-ink', run: 'text-run-ink', muted: 'text-muted' };

function Dot({ hue, size = 8 }: { hue: Hue; size?: 6 | 8 }) {
  return <span aria-hidden className={cn('rounded-full flex-none box-border', size === 6 ? 'w-[6px] h-[6px]' : 'w-2 h-2', DOT[hue])} />;
}

function StatusPill({ status, label }: { status: string; label: string }) {
  const hue = hueOf(status);
  return (
    <span
      data-testid="capability-status"
      data-status={status}
      data-hue={hue}
      className={cn('inline-flex self-start items-center gap-[6px] rounded-pill pl-[7px] pr-[10px] py-[3px] text-[11.5px] font-semibold', PILL[hue])}
    >
      <Dot hue={hue} />
      {label}
    </span>
  );
}

function StatusLine({ status, label, className }: { status: string; label: string; className?: string }) {
  const hue = hueOf(status);
  return (
    <span data-testid="capability-status" data-status={status} data-hue={hue} className={cn('flex items-center gap-[5px]', LINE[hue], className)}>
      <Dot hue={hue} size={6} />
      {label}
    </span>
  );
}

/** An example prompt: opens the Workspace with it in the composer. */
function PromptButton({ prompt, variant, disabled, title }: { prompt: string; variant: 'card' | 'inset'; disabled?: boolean; title?: string }) {
  const navigate = useNavigate();
  return (
    <button
      type="button"
      data-testid="example-prompt"
      disabled={disabled}
      title={title}
      onClick={() => navigate(promptHref(prompt))}
      className={cn(
        'text-left leading-[1.45] text-text enabled:hover:bg-accent-tint enabled:hover:text-accent-ink disabled:text-faint disabled:cursor-not-allowed',
        variant === 'card'
          ? 'px-[14px] py-3 border border-border rounded-[9px] bg-surface text-[13px]'
          : 'mt-auto px-[10px] py-2 rounded-[7px] bg-surface2 text-control',
      )}
    >
      {`"${prompt}"`}
    </button>
  );
}

const card = 'border border-border rounded-card bg-surface px-4 py-[14px] flex flex-col';
const grid4 = 'grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3';

function TierSection({
  tier,
  dimmed,
  sectionRef,
  children,
}: {
  tier: { id: string; number: number; title: string; subtitle: string | null };
  dimmed?: boolean;
  sectionRef: (el: HTMLElement | null) => void;
  children: ReactNode;
}) {
  return (
    <section
      ref={sectionRef}
      aria-labelledby={`tier-${tier.id}-title`}
      data-testid={`tier-${tier.id}`}
      data-dimmed={dimmed ? 'true' : 'false'}
      // Dimmed by colour, not opacity: its text keeps WCAG AA contrast (the status pill and badges keep their hue).
      className={cn('flex flex-col gap-3 scroll-mt-6', dimmed && '[&_:is(h2,h3,p,button,dt,dd)]:!text-muted')}
    >
      <div className="flex items-baseline gap-3 flex-wrap">
        <span className="font-mono text-[11px] font-medium text-muted">{tier.number}</span>
        <h2 id={`tier-${tier.id}-title`} className="text-[16px] font-semibold">
          {tier.title}
        </h2>
        {tier.subtitle && <span className="text-[13px] text-muted">{tier.subtitle}</span>}
      </div>
      {children}
    </section>
  );
}

function AskItems({ tier }: { tier: AskTier }) {
  return (
    <>
      {tier.status_label && (
        <div className="flex items-center gap-[10px] flex-wrap">
          <StatusPill status={tier.status} label={tier.status_label} />
        </div>
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {tier.items.map((item) => (
          <PromptButton key={item.id} prompt={item.prompt} variant="card" />
        ))}
      </div>
    </>
  );
}

function Badge({ children }: { children: ReactNode }) {
  return (
    <span className="flex-none text-[10.5px] px-[7px] py-[2px] rounded-pill bg-accent-tint text-accent-ink font-semibold">{children}</span>
  );
}

function ComposedItems({ tier }: { tier: ComposedTier }) {
  return (
    <div className={grid4}>
      {tier.items.map((item) => (
        <article key={item.id} className={cn(card, 'gap-2')}>
          <div className="flex justify-between items-baseline gap-2">
            <h3 className="text-card">{item.title}</h3>
            <Badge>{item.badge}</Badge>
          </div>
          <p className="text-[13px] text-text2 leading-[1.5]">{item.description}</p>
          <p className="font-mono text-[11.5px] text-muted leading-[1.5]">{item.recipe}</p>
          <p className="text-[11.5px] text-muted">{item.needs_label}</p>
          <PromptButton prompt={item.prompt} variant="inset" />
        </article>
      ))}
    </div>
  );
}

function BuiltinCard({ item }: { item: BuiltinItem }) {
  const meta = item.formula ?? item.model?.short ?? null;
  const gated = isGated(item);
  return (
    <article data-testid="builtin-item" data-id={item.id} className={cn(card, 'gap-[7px]')}>
      <div className="flex justify-between items-baseline gap-2 min-w-0">
        <h3 className="text-card flex-none">{item.title}</h3>
        {meta && (
          <span title={meta} className="font-mono text-[11px] text-muted truncate">
            {meta}
          </span>
        )}
      </div>
      <p className="text-[13px] text-text2 leading-[1.5]">{item.description}</p>
      <dl className="grid grid-cols-[auto_1fr] gap-x-[10px] gap-y-[2px] text-[11.5px] text-muted leading-[1.5]">
        <dt>Inputs</dt>
        <dd className="font-mono text-text2">{item.inputs.join(', ')}</dd>
        <dt>Outputs</dt>
        <dd className="text-text2">{item.outputs.join(', ')}</dd>
        <dt>Time</dt>
        <dd className="font-mono text-text2">{item.time_estimate}</dd>
      </dl>
      <StatusLine status={item.status} label={item.status_label} className="text-[11.5px]" />
      <PromptButton prompt={item.prompt} variant="inset" disabled={gated} title={gated ? item.status_label : undefined} />
    </article>
  );
}

function BuiltinItems({ tier }: { tier: BuiltinTier }) {
  const groups = tier.groups ?? [{ id: '', title: '' }];
  return (
    <>
      {groups.map((group) => {
        const items = tier.groups ? tier.items.filter((i) => i.group === group.id) : tier.items;
        if (items.length === 0) return null;
        return (
          <div key={group.id} data-testid={`builtin-group-${group.id}`} className="flex flex-col gap-[10px] mt-[6px]">
            {group.title && <h3 className="text-label uppercase text-muted">{group.title}</h3>}
            <div className={grid4}>
              {items.map((item) => (
                <BuiltinCard key={item.id} item={item} />
              ))}
            </div>
          </div>
        );
      })}
    </>
  );
}

function ModelItems({ tier }: { tier: ModelsTier }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
      {tier.items.map((model) => (
        <article key={model.id} data-testid="model-item" data-id={model.id} className={cn(card, 'gap-2')}>
          <div className="flex justify-between items-baseline gap-2">
            <h3 className="text-card">{model.title}</h3>
            <span className="font-mono text-[11px] text-muted">
              {model.size_label} ·{' '}
              <a href={model.licence_url} target="_blank" rel="noreferrer" className="hover:text-accent-ink hover:underline">
                {model.licence}
              </a>
            </span>
          </div>
          {model.description && <p className="text-[13px] text-text2 leading-[1.5]">{model.description}</p>}
          <StatusPill status={model.status} label={model.status_label} />
        </article>
      ))}
    </div>
  );
}

function DataItems({ tier }: { tier: DataTier }) {
  return (
    <div className="border border-border rounded-card bg-surface overflow-hidden text-control">
      {tier.items.map((item) => (
        <div
          key={item.id}
          data-testid="data-item"
          data-id={item.id}
          className="grid grid-cols-[1fr_1.4fr_220px] max-md:grid-cols-1 gap-[14px] max-md:gap-1 px-4 py-[10px] border-b border-border2 items-center"
        >
          <span className="font-medium">{item.title}</span>
          <span className={cn('text-text2', item.id === 'sample' && 'font-mono text-[12px]')}>{item.detail}</span>
          <StatusLine status={item.status} label={item.status_label} className="font-medium" />
        </div>
      ))}
      {tier.note && <p className="px-4 py-[10px] text-[12px] text-muted">{tier.note}</p>}
    </div>
  );
}

function ComingSection({ tier, sectionRef }: { tier: ComingTier; sectionRef: (el: HTMLElement | null) => void }) {
  const [open, setOpen] = useState(!tier.collapsed);
  return (
    <section ref={sectionRef} data-testid="tier-coming" aria-label={tier.title} className="border-t border-border pt-[14px]">
      <button
        type="button"
        aria-expanded={open}
        aria-controls="tier-coming-items"
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center gap-3 text-left"
      >
        <span className="font-mono text-[11px] font-medium text-muted">{tier.number}</span>
        <span className="text-[14px] font-semibold text-muted">{tier.title}</span>
        <span className="text-control text-muted truncate">{tier.items.map((i) => i.title).join(' · ')}</span>
        <span className="ml-auto flex-none text-[12px] text-muted">{open ? 'Hide' : 'Show'}</span>
      </button>
      {open && (
        <div id="tier-coming-items" className={cn(grid4, 'mt-3')}>
          {tier.items.map((item) => (
            <div key={item.id} className="border border-dashed border-border rounded-card px-4 py-[14px] text-muted flex flex-col gap-[6px]">
              <span className="font-semibold text-text2">{item.title}</span>
              {item.description && <span className="leading-[1.5]">{item.description}</span>}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

// The header's jump links: tier id → chip text.
const CHIPS = [
  ['ask', 'Ask'],
  ['composed', 'Composed'],
  ['builtin', 'Built-in'],
  ['models', 'Models'],
  ['data', 'Data'],
] as const;

/** The six tiers of GET /v1/capabilities. Exported for tests. */
export function CapabilitiesView({ doc }: { doc: CapabilitiesDoc }) {
  const sections = useRef<Record<string, HTMLElement | null>>({});
  const refFor = (id: string) => (el: HTMLElement | null) => {
    sections.current[id] = el;
  };
  const scrollTo = (id: string) => {
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    sections.current[id]?.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
  };
  const [ask, composed, builtin, models, data, coming] = doc.tiers;
  // Offline: tiers 1 and 2 need the assistant; greyed, never hidden.
  const dimmed = (tier: { needs_assistant: boolean }) => doc.offline && tier.needs_assistant;

  return (
    <>
      <header className="flex items-end justify-between gap-5 flex-wrap">
        <div>
          <h1 id="capabilities-title" className="text-title tracking-[-0.01em]">
            What DT4LC can do · on this server
          </h1>
          <p data-testid="capabilities-summary" className="mt-1 max-w-[720px] text-[13px] leading-[1.5] text-muted">
            Live for this installation: {doc.summary.line}.
          </p>
        </div>
        <nav aria-label="Tiers" className="flex flex-wrap gap-[6px]">
          {CHIPS.map(([id, label]) => (
            <button
              key={id}
              type="button"
              onClick={() => scrollTo(id)}
              className="px-[9px] py-[5px] border border-border rounded-control text-[12px] text-muted whitespace-nowrap hover:text-text hover:bg-surface2"
            >
              {label}
            </button>
          ))}
        </nav>
      </header>

      {doc.offline && doc.banner && (
        <p
          role="status"
          data-testid="capabilities-banner"
          className="px-[14px] py-[10px] rounded-[8px] bg-canc-tint text-canc-ink text-control leading-[1.5] flex gap-2 items-center"
        >
          <Dot hue="canc" />
          {doc.banner}
        </p>
      )}

      <div className="flex flex-col gap-3">
        <TierSection tier={ask} dimmed={dimmed(ask)} sectionRef={refFor('ask')}>
          <AskItems tier={ask} />
        </TierSection>
        <TierSection tier={composed} dimmed={dimmed(composed)} sectionRef={refFor('composed')}>
          <ComposedItems tier={composed} />
        </TierSection>
        <TierSection tier={builtin} sectionRef={refFor('builtin')}>
          <BuiltinItems tier={builtin} />
        </TierSection>
        <TierSection tier={models} sectionRef={refFor('models')}>
          <ModelItems tier={models} />
        </TierSection>
        <TierSection tier={data} sectionRef={refFor('data')}>
          <DataItems tier={data} />
        </TierSection>
        <ComingSection tier={coming} sectionRef={refFor('coming')} />
      </div>
    </>
  );
}

/** /app/capabilities ("Can do"): what this server can do, live from GET /v1/capabilities. */
export function Capabilities() {
  const { data, isPending, isError, refetch } = useCapabilities();
  return (
    <main
      data-testid="capabilities-page"
      aria-labelledby="capabilities-title"
      className="flex-1 min-w-0 h-full overflow-y-auto bg-bg px-10 max-md:px-4 md:max-lg:px-6 pt-8 max-md:pt-6 pb-12 flex flex-col gap-[30px] text-[13px]"
    >
      {data ? (
        <CapabilitiesView doc={data} />
      ) : (
        <>
          <h1 id="capabilities-title" className="text-title tracking-[-0.01em]">
            What DT4LC can do · on this server
          </h1>
          {isPending && <p className="text-muted">Checking this server…</p>}
          {isError && (
            <p className="text-fail-ink">
              Could not reach the server.{' '}
              <button type="button" onClick={() => void refetch()} className="underline">
                Try again
              </button>
            </p>
          )}
        </>
      )}
    </main>
  );
}
