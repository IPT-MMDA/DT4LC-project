import { useState } from 'react';
import { Eye, EyeOff } from 'lucide-react';
import { resolveApiUrl } from '../../../api/client';
import { useAppStore } from '../../../store/useAppStore';
import { cn } from '../../../utils/cn';
import { ExportLayerDialog } from '../ExportLayerDialog';
import { layerRows, REGION_ROW_ID, type LayerRow } from './layers';

const SWATCH_TOKEN = {
  change: 'bg-gradient-to-r from-legend-lost via-legend-no-change to-legend-gained',
  region: 'bg-region',
  imagery: 'bg-legend-land border border-border',
} as const;

function Swatch({ swatch }: { swatch: LayerRow['swatch'] }) {
  if ('image' in swatch) {
    return <img alt="" src={resolveApiUrl(swatch.image)} className="w-3 h-3 flex-none rounded-[3px] object-cover bg-surface2" />;
  }
  if ('stops' in swatch) {
    return <span aria-hidden className="w-3 h-3 flex-none rounded-[3px]" style={{ background: `linear-gradient(90deg, ${swatch.stops.join(', ')})` }} />;
  }
  return <span aria-hidden className={cn('w-3 h-3 flex-none rounded-[3px]', SWATCH_TOKEN[swatch.token])} />;
}

/** Export ▾: the Earth Engine layers on the map, exported as GeoTIFF with the existing dialog. */
function ExportMenu() {
  const layers = useAppStore((s) => s.mapLayers);
  const gee = layers.filter((l) => l.type === 'gee-tiles');
  const [open, setOpen] = useState(false);
  const [target, setTarget] = useState<{ id: string; name: string } | null>(null);

  return (
    <div className="relative" onKeyDown={(e) => e.key === 'Escape' && setOpen(false)}>
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        disabled={gee.length === 0}
        title={gee.length === 0 ? 'Fetch imagery to the map to export it' : undefined}
        onClick={() => setOpen((v) => !v)}
        className="text-[12px] text-muted hover:text-text disabled:cursor-not-allowed disabled:hover:text-muted"
      >
        Export ▾
      </button>
      {open && (
        <div role="menu" aria-label="Export" className="absolute right-0 top-5 z-10 min-w-[180px] py-1 bg-surface border border-border rounded-[8px] text-[12.5px]">
          {gee.length > 1 && (
            <button type="button" role="menuitem" onClick={() => { setTarget({ id: 'all', name: 'All layers' }); setOpen(false); }} className="block w-full text-left px-3 py-[6px] hover:bg-surface2">
              All layers
            </button>
          )}
          {gee.map((l) => (
            <button key={l.id} type="button" role="menuitem" onClick={() => { setTarget({ id: l.id, name: l.name }); setOpen(false); }} className="block w-full text-left px-3 py-[6px] hover:bg-surface2 truncate">
              {l.name}
            </button>
          ))}
        </div>
      )}
      {target && <ExportLayerDialog layerId={target.id} layerName={target.name} allLayers={target.id === 'all' ? gee : undefined} onClose={() => setTarget(null)} />}
    </div>
  );
}

/** LAYERS · N: results from chat, the region and the sample, each with visibility and opacity. */
export function LayersSection() {
  const layers = useAppStore((s) => s.mapLayers);
  const region = useAppStore((s) => s.selectedRegion);
  const regionOpacity = useAppStore((s) => s.regionOpacity);
  const setLayerOpacity = useAppStore((s) => s.setLayerOpacity);
  const setRegionOpacity = useAppStore((s) => s.setRegionOpacity);
  const toggleLayerVisibility = useAppStore((s) => s.toggleLayerVisibility);
  const rows = layerRows(layers, region, regionOpacity);

  const setOpacity = (row: LayerRow, pct: number) =>
    row.id === REGION_ROW_ID ? setRegionOpacity(pct / 100) : setLayerOpacity(row.id, pct / 100);
  const toggle = (row: LayerRow) =>
    row.id === REGION_ROW_ID ? setRegionOpacity(row.visible ? 0 : 1) : toggleLayerVisibility(row.id);

  return (
    <div className="px-4 py-[14px] flex flex-col gap-2" role="region" aria-label="Map layers">
      <div className="flex justify-between items-center">
        <div className="text-label uppercase text-muted font-semibold">Layers · {rows.length}</div>
        <ExportMenu />
      </div>
      {rows.length === 0 ? (
        <p className="text-[11.5px] leading-[1.45] text-muted">Results from the chat appear here as layers.</p>
      ) : (
        <ul className="flex flex-col gap-[6px]">
          {rows.map((row, i) => (
            <li key={row.id} data-testid="map-layer-row" className={cn('group flex items-center gap-2 px-2 py-[7px] rounded-[6px]', i === 0 && 'bg-bg')}>
              <Swatch swatch={row.swatch} />
              <div className={cn('flex-1 min-w-0', !row.visible && 'opacity-50')}>
                <div className="font-medium truncate">{row.name}</div>
                <div className="text-[11px] text-muted truncate">{row.subtitle}</div>
              </div>
              <button
                type="button"
                onClick={() => toggle(row)}
                aria-label={`${row.visible ? 'Hide' : 'Show'} ${row.name}`}
                aria-pressed={!row.visible}
                className={cn('text-muted hover:text-text', row.visible && 'opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 focus:opacity-100 max-md:opacity-100 [@media(hover:none)]:opacity-100')}
              >
                {row.visible ? <Eye aria-hidden className="w-[14px] h-[14px]" /> : <EyeOff aria-hidden className="w-[14px] h-[14px]" />}
              </button>
              <input
                type="range"
                min={0}
                max={100}
                step={5}
                value={row.opacity}
                aria-label={`Opacity of ${row.name}`}
                onChange={(e) => setOpacity(row, Number(e.target.value))}
                // The top row shows its slider; the others on hover or
                // focus, and always on a phone or a touch screen, which have no hover.
                className={cn('w-[50px] accent-accent', i > 0 && 'hidden group-hover:block group-focus-within:block max-md:block [@media(hover:none)]:block')}
              />
              <span className="w-[22px] text-right font-mono text-[11px] text-muted">{row.opacity}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
