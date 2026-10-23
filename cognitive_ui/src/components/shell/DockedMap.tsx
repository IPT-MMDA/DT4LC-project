import { Link } from 'react-router-dom';
import { MapContainer } from '../map/MapContainer';

/**
 * The map docked beside the chat at ≥1440 px (design/DESIGN.md §1): the map, the
 * Fit to result / Compare controls, "Open full map" and the Layers panel shell.
 * Result layers, the legend and the two controls arrive with the result cards (F4).
 */
export function DockedMap() {
  const control = 'px-[10px] py-[5px] rounded-[5px] text-text2 disabled:cursor-not-allowed';
  return (
    <section aria-label="Map" className="flex-1 min-w-0 relative">
      <MapContainer docked />

      <div className="absolute top-3 left-3 right-3 flex justify-between items-start gap-2 pointer-events-none">
        <div className="pointer-events-auto flex gap-1 p-1 bg-surface border border-border rounded-[8px] text-[12px]">
          <button type="button" disabled title="No result on the map yet" className={control}>
            Fit to result
          </button>
          <button type="button" disabled title="No result on the map yet" className={control}>
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
        {/* F4: one row per result layer (swatch, name, opacity) and the legend below them. */}
        <p className="text-[11.5px] leading-[1.45] text-faint">Results from the chat appear here as layers.</p>
      </div>
    </section>
  );
}
