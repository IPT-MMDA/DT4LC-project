import { useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from 'react';
import { Link, useParams } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { clsx as cn } from 'clsx';
import { ChatError } from '../api/chat';
import { resolveApiUrl } from '../api/client';
import { useCapabilities } from '../api/hooks/useCapabilities';
import { useSession } from '../api/hooks/useSession';
import { useListFiles } from '../api/hooks/useUpload';
import { getJob, type JobRecord } from '../api/jobs';
import { statisticsCsvUrl, type JobCard } from '../api/library';
import { analysisReducer } from '../components/chat/analysis/analysisReducer';
import { Failure } from '../components/chat/analysis/AnalysisMessage';
import { StatusGlyph, StatusPill, type PillStatus } from '../components/chat/analysis/StatusPill';
import { attachmentChips } from '../components/chat/attachmentChips';
import { FileChip, RegionChip } from '../components/chat/Chips';
import { downloadName, resultCardOf, type CardView, type ResultCardModel } from '../components/chat/result/resultModel';
import {
  changeIndexLabel,
  contextFromJob,
  downloadsOf,
  formatBytes,
  formatRangeValue,
  inputDates,
  mapMeta,
  pillStatusOf,
  planSummary,
  registryOf,
  resultTitle,
  runTime,
  statisticsTable,
  type StatisticsTable,
} from '../components/detail/resultDetailModel';
import { useResultActions } from '../components/detail/useResultActions';
import { analysisContext, whenLabel } from '../components/library/libraryModel';
import { useAttachFiles } from '../hooks/useAttachFiles';
import { interpretationOf } from '../hooks/useJobEvents';
import { useChatStore } from '../store/useChatStore';

// /app/results/:id: one analysis from
// GET /v1/jobs/{id}: the header with its actions, the maps and the statistics table,
// and beside them what was asked, the plan, the interpretation and the downloads.
// Classes join with clsx, as in the Library (utils/cn drops `text-control` before a
// text colour).

type DetailJob = JobRecord & { card?: JobCard | null; created_at?: string; plan?: unknown; region?: unknown };

const FINISHED: JobRecord['status'][] = ['completed', 'failed', 'cancelled'];
const outline = 'px-[14px] py-[8px] rounded-control border border-border bg-surface text-text text-[13px] hover:bg-surface2';
const primary = 'px-[14px] py-[8px] rounded-control bg-accent text-on-accent text-[13px] font-medium hover:opacity-90';
const small = 'px-[8px] py-[4px] rounded-[5px] border border-border text-text2 text-[12px] hover:bg-surface2';
const panel = 'border border-border rounded-card bg-surface overflow-hidden';
const sideCard = 'border border-border rounded-card bg-surface px-4 py-[14px] flex flex-col gap-[10px]';
const label = 'text-label uppercase text-muted';

/**
 * GET /v1/jobs/{id}, polled until the job finishes; a finished job is read once. An
 * unfinished record is dropped on leaving, so the result card and the Library, which
 * read `['job', id]` once, never keep it.
 */
function useDetailJob(id: string) {
  const queryClient = useQueryClient();
  useEffect(
    () => () => {
      const data = queryClient.getQueryData<DetailJob>(['job', id]);
      if (data && !FINISHED.includes(data.status)) queryClient.removeQueries({ queryKey: ['job', id], exact: true });
    },
    [queryClient, id],
  );
  return useQuery({
    queryKey: ['job', id],
    queryFn: () => getJob(id) as Promise<DetailJob>,
    staleTime: (q) => (q.state.data && FINISHED.includes(q.state.data.status) ? Infinity : 0),
    refetchInterval: (q) => (q.state.data && !FINISHED.includes(q.state.data.status) ? 2000 : false),
    retry: (n, err) => !(err instanceof ChatError && err.status === 404) && n < 2,
  });
}

/** GET /v1/jobs/{id}/statistics.csv: its size, or null when the job has none (404). */
function useCsvSize(id: string, enabled: boolean) {
  return useQuery({
    queryKey: ['job', id, 'statistics.csv'],
    enabled,
    staleTime: Infinity,
    queryFn: async (): Promise<number | null> => {
      const res = await fetch(statisticsCsvUrl(id));
      if (!res.ok) return null;
      return (await res.blob()).size;
    },
  });
}

function Unavailable() {
  const { data: session } = useSession();
  return (
    <main data-testid="result-unavailable" className="flex-1 min-w-0 overflow-y-auto bg-bg px-[40px] py-[32px]">
      <h1 className="text-title">This result isn't available</h1>
      <p className="mt-1 text-body text-text2">It may have expired (guest results are kept {session?.kept_hours ?? 24} h).</p>
      <div className="mt-4 flex gap-3 text-control font-medium">
        <Link to="/app/library" className="text-accent-ink">
          Library
        </Link>
        <Link to="/app" className="text-accent-ink">
          Chat
        </Link>
      </div>
    </main>
  );
}

function Swatch({ colour, className }: { colour?: string; className?: string }) {
  const style: CSSProperties | undefined = colour ? { background: colour } : undefined;
  return <span aria-hidden className={cn('w-[10px] h-[10px] rounded-[2px] border border-border flex-none', !colour && 'bg-surface2', className)} style={style} />;
}

/** The difference (or the single map) with its legend over the bottom-left corner. */
function MapPanel({ title, meta, view, legend, height, actions }: { title: string; meta: string; view: CardView; legend?: ReactNode; height: string; actions?: boolean }) {
  const shown = view.bare ?? view.image;
  const png = view.image ?? view.bare;
  return (
    <section data-testid="detail-map" data-view={view.key} aria-label={title} className={panel}>
      <header className="flex items-center justify-between gap-3 px-[14px] py-[10px] border-b border-border2">
        <h2 className="text-[13px] font-semibold">{title}</h2>
        <div className="flex items-center gap-2 text-[12px] text-muted min-w-0">
          {meta && (
            <span data-testid="detail-map-meta" className="font-mono truncate">
              {meta}
            </span>
          )}
          {actions && png && (
            <a href={resolveApiUrl(png.url)} download={downloadName(png)} title={downloadName(png)} className={small}>
              PNG
            </a>
          )}
          {actions && view.raster && (
            <a href={resolveApiUrl(view.raster.url)} download={downloadName(view.raster)} title={downloadName(view.raster)} className={small}>
              GeoTIFF
            </a>
          )}
        </div>
      </header>
      <div className={cn('relative bg-surface2', height)}>
        {shown ? (
          <img data-testid="detail-image" src={resolveApiUrl(shown.url)} alt={title} className="absolute inset-0 w-full h-full object-contain" />
        ) : (
          <p className="absolute inset-0 flex items-center justify-center text-control text-muted">No image for this view</p>
        )}
        {legend && (
          <div data-testid="detail-legend" className="absolute left-[10px] bottom-[10px] bg-surface/95 rounded-[6px] px-[10px] py-[7px] flex flex-wrap items-center gap-[10px] text-[11.5px] text-text2">
            {legend}
          </div>
        )}
      </div>
    </section>
  );
}

function RampBar({ stops }: { stops: string[] }) {
  return <span aria-hidden className="w-[110px] h-2 rounded-[2px]" style={{ background: `linear-gradient(90deg, ${stops.join(', ')})` }} />;
}

/** −0.5 · 0 · +0.5 ΔNDWI, then the lost and gained swatches. */
function ChangeLegend({ model, index }: { model: Extract<ResultCardModel, { kind: 'change' }>; index: string }) {
  const legend = model.views[0]?.legend;
  const ramp = legend?.kind === 'change' && legend.stops && legend.range ? { stops: legend.stops, range: legend.range } : undefined;
  return (
    <>
      {ramp && (
        <>
          <RampBar stops={ramp.stops} />
          <span className="font-mono">
            {formatRangeValue(ramp.range[0])} · 0 · {formatRangeValue(ramp.range[1])} {index && `Δ${index}`}
          </span>
        </>
      )}
      {model.legend
        .filter((l) => l.tone !== 'neutral')
        .map((l) => (
          <span key={l.label} className="flex items-center gap-[5px]">
            <Swatch colour={l.colour} />
            {l.label}
          </span>
        ))}
    </>
  );
}

function SingleLegend({ model }: { model: ResultCardModel }) {
  if (model.kind === 'index' && model.ramp) {
    return (
      <>
        <RampBar stops={model.ramp.stops} />
        <span className="font-mono">
          {formatRangeValue(model.ramp.range[0])} · {formatRangeValue(model.ramp.range[1])}
        </span>
      </>
    );
  }
  if (model.kind === 'classification') {
    return (
      <>
        {model.classes.map((c) => (
          <span key={c.name} className="flex items-center gap-[5px]">
            <Swatch colour={c.colour} />
            {c.name}
          </span>
        ))}
      </>
    );
  }
  return null;
}

const SINGLE_TITLES: Record<ResultCardModel['kind'], string> = {
  change: 'Difference map',
  index: 'Index map',
  classification: 'Class map',
  fields: 'Field boundaries',
  statistics: 'Statistics',
  image: 'Map',
};

function Maps({ job, model, meta, dates: [from, to] }: { job: DetailJob; model: ResultCardModel; meta: string; dates: string[] }) {
  if (model.kind === 'statistics') return null;
  if (model.kind === 'change') {
    const index = changeIndexLabel(job);
    const [difference, ...dates] = model.views;
    return (
      <>
        <MapPanel title="Difference map" meta={meta} view={difference} legend={<ChangeLegend model={model} index={index} />} height="h-[360px]" actions />
        <div className="grid grid-cols-2 max-md:grid-cols-1 gap-4">
          {dates.map((v) => (
            <MapPanel key={v.key} title={`${index || 'Index'} · ${v.key}`} meta={(v.key === 'before' ? from : to) ?? ''} view={v} height="h-[170px]" />
          ))}
        </div>
      </>
    );
  }
  return <MapPanel title={SINGLE_TITLES[model.kind]} meta={meta} view={model.views[0]} legend={<SingleLegend model={model} />} height="h-[360px]" actions />;
}

function StatisticsPanel({ table, csvUrl }: { table: StatisticsTable; csvUrl: string | null }) {
  const head = 'px-[14px] py-2 text-label uppercase text-muted text-right first:text-left';
  const cell = 'px-[14px] py-2 text-right font-mono';
  let columns: string[];
  let rows: ReactNode;
  switch (table.kind) {
    case 'classes':
      columns = ['Class', 'Area km²', '% of region', ...(table.meanLabel ? [table.meanLabel] : [])];
      rows = table.rows.map((r) => (
        <tr key={r.name} data-testid="statistics-row" className="border-t border-border2">
          <th scope="row" className="px-[14px] py-2 text-left font-normal">
            <span className="flex items-center gap-2">
              <Swatch colour={r.colour} />
              {r.name}
            </span>
          </th>
          <td className={cell}>{r.area}</td>
          <td className={cell}>{r.percent}</td>
          {table.meanLabel && <td className={cell}>{r.mean}</td>}
        </tr>
      ));
      break;
    case 'bands':
      columns = ['Band', 'Min', 'Max', 'Mean', 'Std', 'Valid'];
      rows = table.rows.map((b) => (
        <tr key={b.band} data-testid="statistics-row" className="border-t border-border2">
          <th scope="row" className="px-[14px] py-2 text-left font-mono font-medium">
            {b.band}
          </th>
          {[b.min, b.max, b.mean, b.std, b.valid].map((v, i) => (
            <td key={i} className={cell}>
              {v}
            </td>
          ))}
        </tr>
      ));
      break;
    case 'scalars':
      columns = ['Statistic', 'Value'];
      rows = table.rows.map((s) => (
        <tr key={s.label} data-testid="statistics-row" className="border-t border-border2">
          <th scope="row" className="px-[14px] py-2 text-left font-normal">
            {s.label}
          </th>
          <td className={cell}>{s.value}</td>
        </tr>
      ));
      break;
  }
  return (
    <section aria-label="Statistics" className={panel}>
      <header className="flex items-center justify-between px-[14px] py-[10px] border-b border-border2">
        <h2 className="text-[13px] font-semibold">Statistics</h2>
        {csvUrl && (
          <a href={csvUrl} download className={small}>
            CSV
          </a>
        )}
      </header>
      <table data-testid="statistics-table" data-kind={table.kind} className="w-full border-collapse text-control">
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c} scope="col" className={head}>
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
  );
}

const GLYPH: Partial<Record<string, PillStatus>> = { skipped: 'queued' };

export function ResultDetail() {
  const { id = '' } = useParams();
  const query = useDetailJob(id);
  const job = query.data;
  const conversations = useChatStore((s) => s.conversations);
  const fromChat = useMemo(() => analysisContext(conversations, id), [conversations, id]);
  const chatHasContext = fromChat.context.attachments.length > 0 || Boolean(fromChat.context.region);
  // This browser has no conversation with the files: the library's files the job read.
  const files = useListFiles({ enabled: Boolean(job) && !chatHasContext });
  const context = useMemo(
    () => (chatHasContext || !job ? fromChat.context : contextFromJob(job, files.data ?? [])),
    [chatHasContext, fromChat.context, job, files.data],
  );
  const capabilities = useCapabilities();
  const registry = useMemo(() => registryOf(capabilities.data), [capabilities.data]);
  const csv = useCsvSize(id, job?.status === 'completed');
  const fileInput = useRef<HTMLInputElement>(null);
  const { attachFiles } = useAttachFiles();
  const [error, setError] = useState<string | null>(null);

  const prompt = fromChat.prompt || job?.prompt?.trim() || '';
  const actions = useResultActions(
    { job: job ?? ({ id, status: 'pending' } as DetailJob), conversation: fromChat.conversation, prompt, context },
    () => fileInput.current?.click(),
  );
  const run = async (action: () => Promise<string | null> | string | null) => {
    setError(null);
    setError(await action());
  };

  if (query.isError) {
    if (query.error instanceof ChatError && query.error.status === 404) return <Unavailable />;
    return (
      <main className="flex-1 min-w-0 bg-bg px-[40px] py-[32px]">
        <p role="alert" className="text-control text-fail-ink">
          Couldn't load the result.{' '}
          <button type="button" onClick={() => void query.refetch()} className="underline-offset-2 hover:underline">
            Try again
          </button>
        </p>
      </main>
    );
  }
  if (!job) {
    return (
      <main aria-busy="true" className="flex-1 min-w-0 bg-bg px-[40px] py-[32px] text-control text-muted">
        Loading the result…
      </main>
    );
  }

  const view = analysisReducer(job.events ?? []);
  const title = resultTitle(job);
  const status = pillStatusOf(job.status);
  const model = job.status === 'completed' ? resultCardOf(job, context, title) : null;
  const table = job.status === 'completed' ? statisticsTable(job, model) : null;
  const meta = mapMeta(job, context);
  const plan = planSummary(job, registry, context.attachments);
  const interpretation = interpretationOf(job);
  const downloads = job.status === 'completed' ? downloadsOf(job) : [];
  const csvUrl = csv.data != null ? statisticsCsvUrl(id) : null;
  const created = job.created_at ? Date.parse(job.created_at) : NaN;
  const headerLine = [runTime(job, view), Number.isFinite(created) ? whenLabel(created, Date.now()) : ''].filter(Boolean).join(' · ');
  const hasLayer = Boolean(model?.views.some((v) => v.raster));
  const chips = attachmentChips(context.attachments);

  let main: ReactNode;
  if (job.status === 'completed') {
    main = (
      <>
        {model && <Maps job={job} model={model} meta={meta} dates={inputDates(job, context)} />}
        {table && <StatisticsPanel table={table} csvUrl={csvUrl} />}
      </>
    );
  } else if (job.status === 'failed') {
    main = view.failure ? (
      <section data-testid="detail-failure" aria-label="Failure" className={cn(panel, 'p-[14px] flex flex-col gap-[10px]')}>
        <Failure view={view} onFix={(f) => void run(() => actions.fix(f))} />
      </section>
    ) : (
      <section data-testid="detail-failure" className={cn(panel, 'p-[14px] text-control text-fail-ink')}>
        {job.card?.error?.message ?? (typeof job.error === 'string' ? job.error : 'The analysis failed.')}{' '}
        {job.card?.error?.code && <span className="font-mono text-[11.5px] text-muted">{job.card.error.code}</span>}
      </section>
    );
  } else if (job.status === 'proposal') {
    main = (
      <section data-testid="detail-proposal" className={cn(panel, 'p-[14px] flex flex-col gap-[10px] text-body')}>
        <p>{job.proposal?.text ?? 'The assistant proposed a plan for this request.'}</p>
        <button type="button" onClick={() => void run(actions.openInWorkspace)} className={cn(primary, 'self-start')}>
          Confirm it in the workspace
        </button>
      </section>
    );
  } else if (job.status === 'cancelled') {
    main = <section className={cn(panel, 'p-[14px] text-body text-text2')}>This analysis was cancelled.</section>;
  } else {
    main = (
      <section className={cn(panel, 'p-[14px] flex items-center gap-3 text-body text-text2')}>
        Still running.
        <button type="button" onClick={() => void run(actions.openInWorkspace)} className="text-control font-medium text-accent-ink">
          Follow it in the workspace
        </button>
      </section>
    );
  }

  return (
    <main
      data-testid="result-detail"
      data-status={job.status}
      className="flex-1 min-w-0 overflow-y-auto bg-bg px-[40px] pt-[28px] pb-[40px] max-md:px-4 max-md:pt-4 md:max-lg:px-6 flex flex-col gap-[20px]"
    >
      <nav aria-label="Breadcrumb">
        <ol data-testid="detail-breadcrumb" className="flex items-center gap-2 text-control text-muted min-w-0">
          <li>
            <Link to="/app/library" className="hover:text-text">
              Library
            </Link>
          </li>
          {fromChat.conversation && (
            <>
              <li aria-hidden>/</li>
              <li className="min-w-0 truncate">
                <Link to="/app" onClick={() => useChatStore.getState().selectConversation(fromChat.conversation!.id)} className="hover:text-text">
                  {fromChat.conversation.title}
                </Link>
              </li>
            </>
          )}
          <li aria-hidden>/</li>
          <li className="min-w-0 truncate">
            <Link to={`/app/results/${encodeURIComponent(id)}`} aria-current="page" className="text-text">
              {title}
            </Link>
          </li>
        </ol>
      </nav>

      <header className="flex items-start justify-between gap-6 flex-wrap">
        <div className="flex flex-col gap-2 min-w-0">
          <h1 data-testid="detail-title" className="text-title">
            {title}
          </h1>
          <div data-testid="detail-meta" className="flex items-center max-lg:flex-wrap gap-[10px] max-lg:gap-y-1 text-control text-muted">
            <StatusPill status={status} />
            {headerLine && <span>{headerLine}</span>}
            <span className="font-mono">{job.id}</span>
          </div>
        </div>
        <div role="group" aria-label="Result actions" className="flex max-lg:flex-wrap gap-2">
          <button type="button" onClick={() => void run(actions.openInWorkspace)} className={outline}>
            Open in workspace
          </button>
          <button type="button" onClick={() => void run(actions.rerun)} className={outline}>
            Re-run with changes
          </button>
          {hasLayer && (
            <button type="button" onClick={() => void run(actions.openOnMap)} className={primary}>
              Open on map
            </button>
          )}
        </div>
      </header>
      {error && (
        <p role="alert" className="text-control text-fail-ink">
          {error}
        </p>
      )}
      <input
        ref={fileInput}
        type="file"
        accept=".tif,.tiff"
        multiple
        hidden
        aria-label="GeoTIFF file"
        onChange={(e) => {
          const picked = [...(e.target.files ?? [])];
          e.target.value = '';
          void run(() => actions.withUploads(() => attachFiles(picked)));
        }}
      />

      {/* Below 1024 px the side column goes under the maps. */}
      <div className="grid grid-cols-[minmax(0,1fr)_360px] max-lg:grid-cols-1 gap-6 items-start">
        <div className="flex flex-col gap-4 min-w-0">{main}</div>

        <aside aria-label="About this result" className="flex flex-col gap-4">
          <section data-testid="detail-asked" className={sideCard}>
            <h2 className={label}>Asked</h2>
            {prompt && <p className="text-body">"{prompt}"</p>}
            {chips.length > 0
              ? chips.map((c) => <FileChip key={c.key} chip={c} size="md" />)
              : context.region && <RegionChip region={context.region} />}
          </section>

          <section data-testid="detail-plan" className={sideCard}>
            <h2 className={label}>Plan · {plan.heading}</h2>
            {plan.lines.length > 0 ? (
              <ol className="flex flex-col gap-[9px] text-control">
                {plan.lines.map((l) => (
                  <li key={l.index} data-testid="plan-step" data-status={l.status} className="flex gap-[9px]">
                    <span className="mt-[2px]">
                      <StatusGlyph status={GLYPH[l.status] ?? (l.status as PillStatus)} size={14} />
                    </span>
                    <div className="flex-1 min-w-0">
                      <div className="font-medium">{l.name}</div>
                      {l.params && (
                        <div data-testid="plan-step-params" className="font-mono text-[11px] text-muted">
                          {l.params}
                        </div>
                      )}
                    </div>
                  </li>
                ))}
              </ol>
            ) : (
              <p className="text-control text-muted">{job.status === 'failed' ? 'It stopped while planning.' : 'No steps yet.'}</p>
            )}
          </section>

          {interpretation && (
            <section className={sideCard}>
              <h2 className={label}>Interpretation</h2>
              <p data-testid="interpretation" className="text-[13px] leading-[1.6]">
                {interpretation}
              </p>
            </section>
          )}

          {(downloads.length > 0 || csvUrl) && (
            <section data-testid="detail-downloads" className={cn(sideCard, 'gap-2')}>
              <h2 className={label}>Downloads</h2>
              <ul className="flex flex-col">
                {downloads.map((d) => (
                  <li key={d.url} className="border-b border-border2 last:border-b-0">
                    <a data-testid="download-row" href={resolveApiUrl(d.url)} download={d.name} className="flex items-baseline justify-between gap-3 py-[6px] text-control hover:text-accent-ink">
                      {/* Long names wrap: the whole name is read and seen. */}
                      <span data-testid="download-name" className="font-mono min-w-0 [overflow-wrap:anywhere]">
                        {d.name}
                      </span>
                      <span className="text-muted flex-none">{d.detail}</span>
                    </a>
                  </li>
                ))}
                {csvUrl && (
                  <li>
                    <a data-testid="download-row" href={csvUrl} download className="flex items-baseline justify-between gap-3 py-[6px] text-control hover:text-accent-ink">
                      <span className="font-mono">statistics.csv</span>
                      <span className="text-muted flex-none">{['CSV', formatBytes(csv.data)].filter(Boolean).join(' · ')}</span>
                    </a>
                  </li>
                )}
              </ul>
            </section>
          )}
        </aside>
      </div>
    </main>
  );
}
