import { useRef, useState, type KeyboardEvent } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import { useCapabilities } from '../../../api/hooks/useCapabilities';
import { useAppStore, type DrawnRegion } from '../../../store/useAppStore';
import { useChatStore } from '../../../store/useChatStore';
import type { MessageAttachment } from '../../../types';
import { cn } from '../../../utils/cn';
import { formatArea, sampleDates } from '../../chat/regionChip';
import {
  IMAGERY_DATASETS,
  datasetOf,
  defaultImagery,
  earthEngineStatus,
  formatCentroid,
  regionAreaKm2,
  regionFromImagery,
  regionProblem,
  toggleBand,
  withDataset,
} from './imagery';
import { LayersSection } from './LayersSection';
import { GeoJSONError, presetRegion, regionFromGeoJSON, type PresetId } from './regionPresets';
import { useFetchToMap } from './useFetchToMap';
import { useMapSample } from './useMapSample';

const sectionLabel = 'text-label uppercase text-muted font-semibold';
const section = 'px-4 py-[14px] border-b border-border2 flex flex-col';

function RegionCard({ region }: { region: DrawnRegion }) {
  const updateRegion = useAppStore((s) => s.updateRegion);
  const [draft, setDraft] = useState<string | null>(null);

  const save = () => {
    const name = draft?.trim();
    if (name) updateRegion(region.id, { name });
    setDraft(null);
  };
  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') save();
    if (e.key === 'Escape') setDraft(null);
  };

  return (
    <div data-testid="region-card" className="flex items-center gap-2 px-[10px] py-2 rounded-[7px] bg-bg border border-border">
      <span aria-hidden className="w-3 h-3 flex-none rounded-[3px] bg-region" />
      <div className="flex-1 min-w-0">
        {draft === null ? (
          <div className="font-medium truncate">{region.name}</div>
        ) : (
          <input
            autoFocus
            aria-label="Region name"
            maxLength={80}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={onKeyDown}
            onBlur={save}
            className="w-full px-1 -mx-1 rounded-[4px] bg-surface border border-border font-medium"
          />
        )}
        <div data-testid="region-detail" className="font-mono text-[11px] text-muted">
          {formatArea(regionAreaKm2(region))} · {formatCentroid(region)}
        </div>
      </div>
      {draft === null && (
        <button type="button" onClick={() => setDraft(region.name)} className="text-[11.5px] text-muted hover:text-text">
          Rename
        </button>
      )}
    </div>
  );
}

const presetButton = 'px-[10px] py-[6px] rounded-[7px] border border-border bg-surface text-[12.5px] leading-[1.25] text-left text-text2 hover:bg-surface2';

/** Kakhovka dam · Brienz · Upload GeoJSON: each adds (or reselects) a region and zooms to it. */
function RegionPresets({ sample }: { sample: MessageAttachment[] }) {
  const fileInput = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);

  const pick = (region: DrawnRegion) => {
    const { drawnRegions, addRegion, updateRegion, setSelectedRegion, requestFit } = useAppStore.getState();
    if (drawnRegions.some((r) => r.id === region.id)) {
      // A preset picked again: the same region, its geometry refreshed (the sample may have changed).
      updateRegion(region.id, { geometry: region.geometry, bbox: region.bbox });
      setSelectedRegion(useAppStore.getState().drawnRegions.find((r) => r.id === region.id) ?? region);
    } else {
      addRegion(region);
    }
    requestFit(region.bbox);
  };
  const preset = (id: PresetId) => {
    setError(null);
    pick(presetRegion(id, sample));
  };
  const upload = async (file: File | undefined) => {
    if (fileInput.current) fileInput.current.value = '';
    if (!file) return;
    setError(null);
    try {
      pick(regionFromGeoJSON(await file.text(), file.name));
    } catch (err) {
      setError(err instanceof GeoJSONError ? err.message : `Couldn't read ${file.name}.`);
    }
  };

  return (
    <>
      <div role="group" aria-label="Region presets" className="flex gap-[6px]">
        <button type="button" onClick={() => preset('kakhovka')} className={presetButton}>
          Kakhovka dam
        </button>
        <button type="button" onClick={() => preset('brienz')} className={presetButton}>
          Brienz
        </button>
        <input
          ref={fileInput}
          type="file"
          accept=".geojson,.json,application/geo+json,application/json"
          hidden
          aria-label="GeoJSON file"
          onChange={(e) => void upload(e.target.files?.[0])}
        />
        <button type="button" onClick={() => fileInput.current?.click()} className={cn(presetButton, 'flex-1')}>
          Upload GeoJSON
        </button>
      </div>
      {error && (
        <p role="alert" data-testid="geojson-error" className="text-[11.5px] leading-[1.45] text-fail-ink">
          {error}
        </p>
      )}
    </>
  );
}

function RegionSection({ sample }: { sample: MessageAttachment[] }) {
  const regions = useAppStore((s) => s.drawnRegions);
  const selected = useAppStore((s) => s.selectedRegion);
  const setSelectedRegion = useAppStore((s) => s.setSelectedRegion);
  const setMapTool = useAppStore((s) => s.setMapTool);
  // Presets are picked with their own buttons.
  const others = regions.filter((r) => r.id !== selected?.id && !r.id.startsWith('preset-'));

  return (
    <div className={cn(section, 'gap-2')} aria-label="Region" role="group">
      <div className="flex justify-between items-center">
        <div className={sectionLabel}>Region</div>
        <button type="button" onClick={() => setMapTool('polygon')} className="text-[12px] text-accent-ink hover:underline">
          Draw new
        </button>
      </div>
      {selected ? (
        <RegionCard key={selected.id} region={selected} />
      ) : (
        <p className="text-[11.5px] leading-[1.45] text-muted">Draw a region on the map with Polygon or Rectangle.</p>
      )}
      <RegionPresets sample={sample} />
      {others.length > 0 && (
        <div className="flex flex-wrap gap-[6px]">
          {others.map((r) => (
            <button
              key={r.id}
              type="button"
              onClick={() => setSelectedRegion(r)}
              aria-label={`Select ${r.name}`}
              className="px-[10px] py-[6px] rounded-[7px] border border-border text-[12px] text-text2 hover:bg-surface2 truncate max-w-full"
            >
              {r.name}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * Map mode's left column: Imagery with Back to chat;
 * the region; dataset, dates, cloud and bands; Fetch to map and Send to chat; the layers.
 * `sheet`: on a phone, a sheet under the map that shows only its header until opened.
 */
export function ImageryPanel({ sheet = false }: { sheet?: boolean }) {
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();
  const region = useAppStore((s) => s.selectedRegion);
  const stored = useAppStore((s) => s.imagery);
  const setImagery = useAppStore((s) => s.setImagery);
  const setPendingRegion = useChatStore((s) => s.setPendingRegion);
  const sample = useMapSample();
  const imagery = stored ?? defaultImagery(sampleDates(sample));
  const dataset = datasetOf(imagery.dataset);
  const capabilities = useCapabilities();
  const earthEngine = earthEngineStatus(capabilities.data, { pending: capabilities.isPending, error: capabilities.isError });
  const fetcher = useFetchToMap();
  const problem = regionProblem(region, imagery);

  const fetchReason = earthEngine.reason ?? problem;
  const sendReason = problem;

  const sendToChat = () => {
    if (!region || sendReason) return;
    setPendingRegion(regionFromImagery(region, imagery));
    navigate('/app');
  };

  return (
    <aside
      aria-label="Imagery"
      className={cn(
        'flex-none bg-surface border-border flex flex-col overflow-hidden text-[13px]',
        sheet ? 'w-full max-h-[62%] border-t' : 'w-[312px] md:max-lg:w-[272px] border-r',
      )}
    >
      <div className={cn('px-4 py-[14px] border-b border-border flex items-center justify-between', sheet && 'pr-2 py-[6px] gap-3')}>
        <h1 className="text-[14px] font-semibold">Imagery</h1>
        {sheet && <div className="flex-1" />}
        <Link to="/app" className="text-[12px] text-accent-ink hover:underline">
          Back to chat
        </Link>
        {sheet && (
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            aria-label="Imagery panel"
            aria-expanded={open}
            aria-controls="imagery-sheet"
            className="w-10 h-10 flex items-center justify-center rounded-control text-muted hover:bg-surface2"
          >
            {open ? <ChevronDown aria-hidden className="w-5 h-5" /> : <ChevronUp aria-hidden className="w-5 h-5" />}
          </button>
        )}
      </div>

      <div id={sheet ? 'imagery-sheet' : undefined} className={cn('flex-1 min-h-0 overflow-y-auto flex flex-col', sheet && !open && 'hidden')}>
        <RegionSection sample={sample} />

        <div className={cn(section, 'gap-[10px]')} role="group" aria-label="Dataset">
          <div className={sectionLabel}>Dataset</div>
          <div className="flex gap-1">
            {IMAGERY_DATASETS.map((d) => (
              <button
                key={d.id}
                type="button"
                aria-pressed={d.id === imagery.dataset}
                onClick={() => setImagery(withDataset(imagery, d.id))}
                className={cn(
                  'flex-1 py-[7px] rounded-[6px] text-center',
                  d.id === imagery.dataset ? 'bg-ink text-ink-bg font-medium' : 'border border-border text-text2 hover:bg-surface2',
                )}
              >
                {d.label}
              </button>
            ))}
          </div>
          <p data-testid="dataset-description" className="text-[11.5px] leading-[1.45] text-muted">
            {dataset.description}
          </p>

          <div className="flex gap-[6px] items-center font-mono text-[12px]">
            <input
              type="date"
              aria-label="Start date"
              value={imagery.start_date}
              max={imagery.end_date}
              onChange={(e) => e.target.value && setImagery({ ...imagery, start_date: e.target.value })}
              className="flex-1 min-w-0 px-[9px] py-[6px] rounded-[6px] border border-border bg-surface"
            />
            <span aria-hidden className="text-faint">→</span>
            <input
              type="date"
              aria-label="End date"
              value={imagery.end_date}
              min={imagery.start_date}
              onChange={(e) => e.target.value && setImagery({ ...imagery, end_date: e.target.value })}
              className="flex-1 min-w-0 px-[9px] py-[6px] rounded-[6px] border border-border bg-surface"
            />
          </div>

          <label className="flex flex-col gap-[6px]">
            <span className="flex justify-between text-[11.5px] text-muted">
              <span>Max cloud cover</span>
              <span data-testid="cloud-value" className="font-mono text-text">{imagery.cloud_max} %</span>
            </span>
            <input
              type="range"
              min={0}
              max={100}
              step={5}
              value={imagery.cloud_max}
              aria-label="Max cloud cover"
              onChange={(e) => setImagery({ ...imagery, cloud_max: Number(e.target.value) })}
              className="accent-accent"
            />
          </label>

          <div className="flex flex-col gap-[6px]">
            <div className="flex justify-between text-[11.5px] text-muted">
              <span>Bands</span>
              <span>
                <button type="button" onClick={() => setImagery({ ...imagery, bands: dataset.bands.map((b) => b.code) })} className="hover:text-text">
                  Select all
                </button>
                {' · '}
                <button type="button" onClick={() => setImagery({ ...imagery, bands: [] })} className="hover:text-text">
                  Clear
                </button>
              </span>
            </div>
            <div role="group" aria-label="Bands" className="flex flex-wrap gap-1 font-mono text-[11.5px]">
              {dataset.bands.map((b) => {
                const on = imagery.bands.includes(b.code);
                return (
                  <button
                    key={b.code}
                    type="button"
                    aria-pressed={on}
                    onClick={() => setImagery(toggleBand(imagery, b.code))}
                    className={cn('px-2 py-1 rounded-[5px]', on ? 'bg-ink text-ink-bg' : 'border border-border text-text2 hover:bg-surface2')}
                  >
                    {b.label}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="flex gap-[6px]">
            <button
              type="button"
              disabled={Boolean(fetchReason) || fetcher.pending}
              aria-describedby={fetchReason ? 'fetch-reason' : undefined}
              title={fetchReason ?? undefined}
              onClick={() => region && fetcher.run(region, imagery)}
              className="flex-1 py-2 rounded-[7px] border border-border font-medium hover:bg-surface2 disabled:opacity-60 disabled:cursor-not-allowed disabled:hover:bg-transparent"
            >
              {fetcher.pending ? 'Fetching…' : 'Fetch to map'}
            </button>
            <button
              type="button"
              disabled={Boolean(sendReason)}
              title={sendReason ?? undefined}
              onClick={sendToChat}
              className="flex-1 py-2 rounded-[7px] bg-accent text-on-accent font-medium hover:bg-accent/90 disabled:opacity-60 disabled:cursor-not-allowed"
            >
              Send to chat
            </button>
          </div>
          {fetchReason && (
            <p id="fetch-reason" data-testid="fetch-reason" className="text-[11.5px] leading-[1.45] text-muted">
              {fetchReason}
            </p>
          )}
          {sendReason && sendReason !== fetchReason && (
            <p data-testid="send-reason" className="text-[11.5px] leading-[1.45] text-muted">
              {sendReason}
            </p>
          )}
          {fetcher.error && (
            <div role="alert" className="px-3 py-[10px] rounded-[8px] bg-fail-tint border border-fail/30 text-[12px] leading-[1.5] text-fail-ink">
              <span className="font-semibold">Earth Engine didn't return imagery.</span> {fetcher.error}{' '}
              <button type="button" onClick={fetcher.clearError} className="text-accent-ink hover:underline">
                Dismiss
              </button>
            </div>
          )}
        </div>

        <LayersSection />
      </div>
    </aside>
  );
}
