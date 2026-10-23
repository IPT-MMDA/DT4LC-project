import { Link } from 'react-router-dom';
import { useAppStore, type LayerLegend } from '../../store/useAppStore';
import { cn } from '../../utils/cn';
import { formatRangeEnd } from '../chat/result/resultModel';
import { MapContainer } from '../map/MapContainer';

/** lost · same · gained in the map legend tokens, for a change layer without its style's ramp. */
const CHANGE_GRADIENT = 'bg-gradient-to-r from-legend-lost via-legend-land to-legend-gained';

/** The layer's ramp as the server draws its tiles (the raster's style). */
const rampStyle = (legend?: LayerLegend) =>
  legend?.stops ? { background: `linear-gradient(90deg, ${legend.stops.join(', ')})` } : undefined;

const tokenGradient = (legend?: LayerLegend) => legend?.kind === 'change' && !legend.stops && CHANGE_GRADIENT;

function LegendRow({ legend }: { legend: LayerLegend }) {
  return (
    <div data-testid="layers-legend" className="flex items-center gap-[6px] pt-[6px] border-t border-border2 text-[11px] text-text2">
      <div data-testid="layers-legend-ramp" aria-hidden className={cn('w-[84px] h-[7px] rounded-[2px]', tokenGradient(legend))} style={rampStyle(legend)} />
      <span className="font-mono">{legend.kind === 'change' ? 'lost · same · gained' : `${formatRangeEnd(legend.range[0])} · ${formatRangeEnd(legend.range[1])}`}</span>
    </div>
  );
}

/**
 * The map docked beside the chat at ≥1440 px (design/DESIGN.md §1): the map, the
 * Fit to result / Compare controls, "Open full map" and the Layers panel with the
 * result layers added from the chat's cards (useAppStore `mapLayers`).
 */
export function DockedMap() {
  const control = 'px-[10px] py-[5px] rounded-[5px] text-text2 hover:bg-surface2 disabled:hover:bg-transparent disabled:cursor-not-allowed disabled:opacity-60';
  const layers = useAppStore((s) => s.mapLayers);
  const setLayerOpacity = useAppStore((s) => s.setLayerOpacity);
  const requestFit = useAppStore((s) => s.requestFit);
  const latest = [...layers].reverse();
  const fitTarget = latest.find((l) => l.bounds)?.bounds;
  const legend = latest.find((l) => l.legend)?.legend;

  return (
    <section aria-label="Map" className="flex-1 min-w-0 relative">
      <MapContainer docked />

      <div className="absolute top-3 left-3 right-3 flex justify-between items-start gap-2 pointer-events-none">
        <div className="pointer-events-auto flex gap-1 p-1 bg-surface border border-border rounded-[8px] text-[12px]">
          <button
            type="button"
            disabled={!fitTarget}
            title={fitTarget ? undefined : 'No result on the map yet'}
            onClick={() => fitTarget && requestFit(fitTarget)}
            className={control}
          >
            Fit to result
          </button>
          {/* Compare (a before/after swipe) is map mode, Phase 4. */}
          <button type="button" disabled title="Arrives with map mode" className={control}>
            Compare
          </button>
        </div>
        <Link
          to="/app/map"
          className="pointer-events-auto px-3 py-[7px] rounded-[8px] bg-surface border border-border text-control font-medium"
        >
          Open full map ↗
        </Link>
      </div>

      <div
        aria-label="Layers"
        role="region"
        className="absolute left-3 bottom-3 w-[264px] bg-surface border border-border rounded-card px-3 py-[10px] flex flex-col gap-2 text-control"
      >
        <div className="flex justify-between text-label uppercase text-muted">
          <span>Layers</span>
          <span className="normal-case tracking-normal font-normal font-mono">opacity</span>
        </div>
        {layers.length === 0 ? (
          <p className="text-[11.5px] leading-[1.45] text-faint">Results from the chat appear here as layers.</p>
        ) : (
          <ul className="flex flex-col gap-[6px]">
            {layers.map((layer) => {
              const pct = Math.round(layer.opacity * 100);
              return (
                <li key={layer.id} data-testid="layer-row" className="flex items-center gap-2">
                  <span
                    aria-hidden
                    className={cn('w-3 h-3 rounded-[3px] flex-none bg-surface2', tokenGradient(layer.legend))}
                    style={rampStyle(layer.legend)}
                  />
                  <span className="flex-1 min-w-0 truncate font-medium">{layer.name}</span>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    step={5}
                    value={pct}
                    aria-label={`Opacity of ${layer.name}`}
                    onChange={(e) => setLayerOpacity(layer.id, Number(e.target.value) / 100)}
                    className="w-[44px] accent-accent"
                  />
                  <span className="w-[22px] text-right font-mono text-[11px] text-muted">{pct}</span>
                </li>
              );
            })}
          </ul>
        )}
        {legend && <LegendRow legend={legend} />}
      </div>
    </section>
  );
}
