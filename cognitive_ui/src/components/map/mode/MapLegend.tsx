import { useAppStore } from '../../../store/useAppStore';
import { cn } from '../../../utils/cn';
import { activeResultLayer, legendOf } from './layers';

const TOKEN_GRADIENT = 'bg-gradient-to-r from-legend-lost via-legend-no-change to-legend-gained';
const TONE_TEXT = { gained: 'text-legend-gained', lost: 'text-lost', neutral: '' } as const;

/** The active result layer's legend, bottom left: title, ramp with its range, class areas. */
export function MapLegend() {
  const layers = useAppStore((s) => s.mapLayers);
  const legend = legendOf(activeResultLayer(layers));
  if (!legend) return null;

  return (
    <div
      role="region"
      aria-label="Legend"
      data-testid="map-legend"
      className="absolute left-3 bottom-3 max-w-[calc(100%-96px)] bg-surface border border-border rounded-[8px] px-3 py-2 flex flex-col gap-[6px] text-[11.5px] text-text2"
    >
      <div data-testid="map-legend-title" className="font-semibold text-[12px] text-text">{legend.title}</div>
      <div className="flex items-center gap-2">
        <div
          data-testid="map-legend-ramp"
          aria-hidden
          className={cn('w-[120px] h-2 rounded-[2px] flex-none', !legend.stops && TOKEN_GRADIENT)}
          style={legend.stops ? { background: `linear-gradient(90deg, ${legend.stops.join(', ')})` } : undefined}
        />
        <span data-testid="map-legend-ticks" className="font-mono">{legend.ticks.join(' · ')}</span>
      </div>
      {legend.classes.length > 0 && (
        <ul data-testid="map-legend-classes" className="flex flex-wrap gap-x-[10px] gap-y-1">
          {legend.classes.map((c) => (
            <li key={c.label} className="flex items-center gap-1">
              <span
                aria-hidden
                className={cn('w-[9px] h-[9px] rounded-[2px]', c.tone === 'neutral' && 'border border-border2')}
                style={c.colour ? { background: c.colour } : undefined}
              />
              <span className={TONE_TEXT[c.tone]}>{[c.label, c.area].filter(Boolean).join(' ')}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
