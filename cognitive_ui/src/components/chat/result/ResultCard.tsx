import { useEffect, useState, type CSSProperties, type ReactNode } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { resolveApiUrl } from '../../../api/client';
import { getJob } from '../../../api/jobs';
import { DOCKED_MAP_QUERY, useMediaQuery } from '../../../hooks/useMediaQuery';
import { useAppStore } from '../../../store/useAppStore';
import { findConversationByJobId, useChatStore } from '../../../store/useChatStore';
import type { AnalysisMessage } from '../../../types';
import { cn } from '../../../utils/cn';
import {
  downloadName,
  formatRangeEnd,
  resultCardOf,
  resultLayerOf,
  type AnalysedContext,
  type BandRow,
  type CardView,
  type ClassItem,
  type LegendItem,
  type ResultCardModel,
  type Stat,
  type Tone,
} from './resultModel';

// The result card of a finished analysis (design/DESIGN.md §4): one component per
// artifact kind, all on the job's artifact URLs (GET /v1/jobs/{id}), never base64.

/** react-query key of GET /v1/jobs/{id}; a finished job doesn't change, so it is fetched once. */
const jobQueryKey = (jobId: string) => ['job', jobId] as const;

const outline = 'px-[11px] py-[6px] rounded-control border border-border text-text2 hover:bg-surface2';
const SWATCH: Record<Tone, string> = { gained: 'bg-legend-gained', lost: 'bg-legend-lost', neutral: 'bg-legend-no-change' };
const TONE_TEXT: Record<Tone, string> = { gained: 'text-legend-gained', lost: 'text-lost', neutral: '' };
const COLS: Record<number, string> = { 1: 'grid-cols-1', 2: 'grid-cols-2', 3: 'grid-cols-3', 4: 'grid-cols-4' };

/** Keep the card's kind on the message, so the conversation's "N results" counts cards shown. */
function useRememberCard(jobId: string, kind: ResultCardModel['kind'] | undefined) {
  useEffect(() => {
    if (!kind) return;
    const { conversations, updateAnalysis } = useChatStore.getState();
    const message = findConversationByJobId(conversations, jobId)?.messages.find(
      (m): m is AnalysisMessage => m.kind === 'analysis' && m.jobId === jobId,
    );
    if (message && message.result?.card !== kind) updateAnalysis(jobId, { result: { ...message.result, card: kind } });
  }, [jobId, kind]);
}

function CardShell({ model, toggle, children }: { model: ResultCardModel; toggle?: ReactNode; children: ReactNode }) {
  return (
    <section
      data-testid="result-card"
      data-kind={model.kind}
      aria-label={model.title}
      className="border border-border rounded-card bg-surface overflow-hidden"
    >
      <header className="flex flex-col gap-[2px] px-3 py-[10px] border-b border-border2">
        <div className="flex items-center justify-between gap-2">
          <div className="min-w-0 flex flex-wrap items-baseline gap-x-2">
            <h3 className="text-[13px] font-semibold">{model.title}</h3>
            {model.note && <span className="font-mono text-[11px] text-muted">{model.note}</span>}
          </div>
          {toggle}
        </div>
        {model.formula && (
          <div data-testid="card-formula" className="font-mono text-[11px] text-muted truncate" title={model.formula}>
            {model.formula}
          </div>
        )}
      </header>
      {children}
    </section>
  );
}

function ViewToggle({ views, active, onChange }: { views: CardView[]; active: CardView['key']; onChange: (key: CardView['key']) => void }) {
  return (
    <div role="group" aria-label="Result view" className="flex-none flex border border-border rounded-control overflow-hidden text-[11.5px]">
      {views.map((v, i) => (
        <button
          key={v.key}
          type="button"
          aria-pressed={v.key === active}
          onClick={() => onChange(v.key)}
          className={cn(
            'px-[9px] py-1',
            i > 0 && 'border-l border-border',
            v.key === active ? 'bg-text text-surface' : 'text-text2 hover:bg-surface2',
          )}
        >
          {v.label}
        </button>
      ))}
    </div>
  );
}

/** The bare map full-bleed (transparent off the raster, over surface2); else the full figure, whole. */
function MapImage({ view, alt }: { view: CardView; alt: string }) {
  const shown = view.bare ?? view.image;
  return (
    <div className="relative h-[250px] bg-surface2 border-b border-border2">
      {shown ? (
        <img
          data-testid="result-image"
          src={resolveApiUrl(shown.url)}
          alt={alt}
          className={cn('absolute inset-0 w-full h-full', view.bare ? 'object-cover' : 'object-contain')}
        />
      ) : (
        <p className="absolute inset-0 flex items-center justify-center text-control text-faint">No image for this view</p>
      )}
      {view.overlay && (
        <span
          data-testid="image-overlay"
          className="absolute right-2 top-2 font-mono text-[10px] font-medium bg-surface/90 text-text2 px-[6px] py-[2px] rounded-[3px]"
        >
          {view.overlay}
        </span>
      )}
    </div>
  );
}

function Swatch({ className, style }: { className?: string; style?: CSSProperties }) {
  return <span aria-hidden className={cn('w-[10px] h-[10px] rounded-[2px] border border-border flex-none', className)} style={style} />;
}

/** Legend and, under it, the source line. */
function LegendBlock({ source, children }: { source: string | null; children: ReactNode }) {
  return (
    <div className="px-3 py-2 border-b border-border2 flex flex-col gap-[6px]">
      <div data-testid="legend" className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11.5px] text-text2">
        {children}
      </div>
      {source && (
        <div data-testid="source-line" className="font-mono text-[11px] text-text2">
          {source}
        </div>
      )}
    </div>
  );
}

function ChangeLegend({ items }: { items: LegendItem[] }) {
  return (
    <>
      {items.map((item) => (
        <span key={item.label} className="flex items-center gap-[5px]">
          <Swatch className={item.colour ? undefined : SWATCH[item.tone]} style={item.colour ? { background: item.colour } : undefined} />
          {item.label}
        </span>
      ))}
    </>
  );
}

function RampLegend({ ramp }: { ramp: { stops: string[]; range: [number, number] } }) {
  const [lo, hi] = ramp.range;
  return (
    <span className="flex items-center gap-2 font-mono text-[11px]">
      <span>{formatRangeEnd(lo)}</span>
      <span data-testid="legend-ramp" aria-hidden className="w-[120px] h-2 rounded-[2px]" style={{ background: `linear-gradient(90deg, ${ramp.stops.join(', ')})` }} />
      <span>{formatRangeEnd(hi)}</span>
    </span>
  );
}

function ClassLegend({ classes }: { classes: ClassItem[] }) {
  return (
    <>
      {classes.map((c) => (
        <span key={c.name} className="flex items-center gap-[5px]">
          <Swatch style={c.colour ? { background: c.colour } : undefined} />
          {c.name}
          <span className="font-mono text-[11px] text-muted">{c.value}</span>
        </span>
      ))}
    </>
  );
}

function StatsRow({ stats }: { stats: Stat[] }) {
  return (
    <dl data-testid="stats" className={cn('grid gap-[10px] p-3 border-b border-border2', COLS[Math.min(4, stats.length)])}>
      {stats.map((s) => (
        <div key={s.label}>
          <dt className="text-label uppercase text-muted">{s.label}</dt>
          <dd className={cn('font-mono text-stat mt-[2px]', s.tone && TONE_TEXT[s.tone])}>{s.value}</dd>
        </div>
      ))}
    </dl>
  );
}

function BandTable({ bands }: { bands: BandRow[] }) {
  const head = 'text-label uppercase text-muted font-semibold text-right first:text-left px-2 py-[6px]';
  const cell = 'px-2 py-1 text-right first:text-left';
  return (
    <div className="px-1 py-2 border-b border-border2 overflow-x-auto">
      <table data-testid="band-table" className="w-full font-mono text-[12px]">
        <thead>
          <tr>
            {['Band', 'Min', 'Max', 'Mean', 'Std', 'Valid'].map((h) => (
              <th key={h} scope="col" className={head}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {bands.map((b) => (
            <tr key={b.band} className="border-t border-border2">
              <th scope="row" className={cn(cell, 'font-medium')}>
                {b.band}
              </th>
              <td className={cell}>{b.min}</td>
              <td className={cell}>{b.max}</td>
              <td className={cell}>{b.mean}</td>
              <td className={cell}>{b.std}</td>
              <td className={cell}>{b.valid}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * On map · GeoTIFF · PNG · GeoPackage · Details ›, each only when its file exists.
 * PNG is the full figure (title, colour bar), or the bare map when there is none.
 */
function Actions({ jobId, model, view }: { jobId: string; model: ResultCardModel; view?: CardView }) {
  const docked = useMediaQuery(DOCKED_MAP_QUERY);
  const navigate = useNavigate();
  const addResultLayer = useAppStore((s) => s.addResultLayer);
  const raster = view?.raster;
  const image = view?.image ?? view?.bare;

  const onMap = () => {
    if (!view || !raster) return;
    addResultLayer(resultLayerOf(jobId, model, { ...view, raster }));
    // Below 1440 px there is no docked map: open the map with the layer on it.
    if (!docked) navigate('/app/map');
  };

  return (
    <div role="group" aria-label="Result actions" className="flex items-center flex-wrap gap-[6px] px-3 py-[10px] text-control">
      {raster && (
        <button type="button" onClick={onMap} className="px-[11px] py-[6px] rounded-control bg-accent-tint text-accent-ink font-medium hover:bg-accent-tint/70">
          <span aria-hidden>● </span>On map
        </button>
      )}
      {raster && (
        <a href={resolveApiUrl(raster.url)} download={downloadName(raster)} title={downloadName(raster)} className={outline}>
          GeoTIFF
        </a>
      )}
      {image && (
        <a href={resolveApiUrl(image.url)} download={downloadName(image)} title={downloadName(image)} className={outline}>
          PNG
        </a>
      )}
      {model.gpkg && (
        <a href={resolveApiUrl(model.gpkg.url)} download={downloadName(model.gpkg)} title={downloadName(model.gpkg)} className={outline}>
          GeoPackage
        </a>
      )}
      <div className="flex-1" />
      <Link to={`/app/results/${encodeURIComponent(jobId)}`} className="text-muted hover:text-text">
        Details ›
      </Link>
    </div>
  );
}

function ChangeCard({ jobId, model }: { jobId: string; model: Extract<ResultCardModel, { kind: 'change' }> }) {
  const [active, setActive] = useState<CardView['key']>('difference');
  const view = model.views.find((v) => v.key === active) ?? model.views[0];
  return (
    <CardShell model={model} toggle={<ViewToggle views={model.views} active={view.key} onChange={setActive} />}>
      <MapImage view={view} alt={`${model.title}, ${view.overlay}`} />
      <LegendBlock source={model.source}>
        <ChangeLegend items={model.legend} />
      </LegendBlock>
      <StatsRow stats={model.stats} />
      <Actions jobId={jobId} model={model} view={view} />
    </CardShell>
  );
}

function IndexCard({ jobId, model }: { jobId: string; model: Extract<ResultCardModel, { kind: 'index' }> }) {
  const [view] = model.views;
  return (
    <CardShell model={model}>
      <MapImage view={view} alt={model.title} />
      <LegendBlock source={model.source}>{model.ramp && <RampLegend ramp={model.ramp} />}</LegendBlock>
      <StatsRow stats={model.stats} />
      <Actions jobId={jobId} model={model} view={view} />
    </CardShell>
  );
}

function ClassificationCard({ jobId, model }: { jobId: string; model: Extract<ResultCardModel, { kind: 'classification' }> }) {
  const [view] = model.views;
  return (
    <CardShell model={model}>
      <MapImage view={view} alt={model.title} />
      <LegendBlock source={model.source}>
        <ClassLegend classes={model.classes} />
      </LegendBlock>
      <Actions jobId={jobId} model={model} view={view} />
    </CardShell>
  );
}

function FieldsCard({ jobId, model }: { jobId: string; model: Extract<ResultCardModel, { kind: 'fields' }> }) {
  const [view] = model.views;
  return (
    <CardShell model={model}>
      <MapImage view={view} alt={model.title} />
      {model.source && (
        <div data-testid="source-line" className="px-3 py-2 border-b border-border2 font-mono text-[11px] text-text2">
          {model.source}
        </div>
      )}
      <StatsRow stats={model.stats} />
      <Actions jobId={jobId} model={model} view={view} />
    </CardShell>
  );
}

function StatisticsCard({ jobId, model }: { jobId: string; model: Extract<ResultCardModel, { kind: 'statistics' }> }) {
  return (
    <CardShell model={model}>
      <BandTable bands={model.bands} />
      {model.source && (
        <div data-testid="source-line" className="px-3 py-2 border-b border-border2 font-mono text-[11px] text-text2">
          {model.source}
        </div>
      )}
      <Actions jobId={jobId} model={model} />
    </CardShell>
  );
}

function ImageCard({ jobId, model }: { jobId: string; model: Extract<ResultCardModel, { kind: 'image' }> }) {
  const [view] = model.views;
  return (
    <CardShell model={model}>
      <MapImage view={view} alt={model.title} />
      {model.source && (
        <div data-testid="source-line" className="px-3 py-2 border-b border-border2 font-mono text-[11px] text-text2">
          {model.source}
        </div>
      )}
      <Actions jobId={jobId} model={model} view={view} />
    </CardShell>
  );
}

export function ResultCardView({ jobId, model }: { jobId: string; model: ResultCardModel }) {
  switch (model.kind) {
    case 'change':
      return <ChangeCard jobId={jobId} model={model} />;
    case 'index':
      return <IndexCard jobId={jobId} model={model} />;
    case 'classification':
      return <ClassificationCard jobId={jobId} model={model} />;
    case 'fields':
      return <FieldsCard jobId={jobId} model={model} />;
    case 'statistics':
      return <StatisticsCard jobId={jobId} model={model} />;
    case 'image':
      return <ImageCard jobId={jobId} model={model} />;
  }
}

/**
 * The result card of finished job `jobId`, read once from GET /v1/jobs/{id}.
 * `context` is what the analysis was asked about (the source line, dates and map
 * bounds); `fallbackTitle` titles a result of no known kind.
 */
export function ResultCard({ jobId, context, fallbackTitle }: { jobId: string; context: AnalysedContext; fallbackTitle?: string }) {
  const job = useQuery({ queryKey: jobQueryKey(jobId), queryFn: () => getJob(jobId), staleTime: Infinity });
  const model = job.data ? resultCardOf(job.data, context, fallbackTitle) : null;
  useRememberCard(jobId, model?.kind);

  if (job.isPending) {
    return (
      <div data-testid="result-loading" aria-busy="true" className="h-[120px] border border-border rounded-card bg-surface2 flex items-center justify-center text-control text-faint">
        Loading the result…
      </div>
    );
  }
  if (job.isError) {
    return (
      <p role="alert" className="text-control text-fail-ink flex items-center gap-2">
        Couldn't load the result.
        <button type="button" onClick={() => void job.refetch()} className="underline-offset-2 hover:underline">
          Try again
        </button>
      </p>
    );
  }
  return model ? <ResultCardView jobId={jobId} model={model} /> : null;
}
