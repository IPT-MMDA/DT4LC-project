import { useEffect, useRef, useState } from 'react';
import type { MapRef } from 'react-map-gl/maplibre';
import { useAppStore } from '../../../store/useAppStore';
import { cn } from '../../../utils/cn';
import { BASEMAPS } from '../basemaps';
import { activeResultLayer, defaultCompareSides } from './layers';

const box = 'px-[10px] py-[6px] rounded-[7px] bg-surface border border-border text-text2';
const action = cn(box, 'hover:bg-surface2 disabled:opacity-60 disabled:cursor-not-allowed disabled:hover:bg-surface');

/** "46.812, 33.402": the pointer's latitude and longitude, else the map's centre. */
function Coordinates({ map }: { map: MapRef | null }) {
  const [at, setAt] = useState<[number, number] | null>(null);

  useEffect(() => {
    if (!map) return;
    const m = map.getMap();
    const centre = () => {
      const c = m.getCenter();
      setAt([c.lat, c.lng]);
    };
    const pointer = (e: { lngLat: { lat: number; lng: number } }) => setAt([e.lngLat.lat, e.lngLat.lng]);
    centre();
    m.on('mousemove', pointer);
    m.on('moveend', centre);
    return () => {
      m.off('mousemove', pointer);
      m.off('moveend', centre);
    };
  }, [map]);

  return (
    <div data-testid="map-coordinates" aria-label="Coordinates" className={cn(box, 'font-mono max-md:hidden')}>
      {at ? `${at[0].toFixed(3)}, ${at[1].toFixed(3)}` : '—'}
    </div>
  );
}

function BasemapPicker() {
  const current = useAppStore((s) => s.currentBasemap);
  const setBasemap = useAppStore((s) => s.setBasemap);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent | KeyboardEvent) => {
      if (e instanceof KeyboardEvent ? e.key === 'Escape' : !ref.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', close);
    document.addEventListener('keydown', close);
    return () => {
      document.removeEventListener('mousedown', close);
      document.removeEventListener('keydown', close);
    };
  }, [open]);

  return (
    <div ref={ref} className="relative">
      <button type="button" aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen((v) => !v)} className={action}>
        Basemap ▾
      </button>
      {open && (
        <div role="menu" aria-label="Basemap" className="absolute right-0 top-[calc(100%+4px)] min-w-[140px] py-1 bg-surface border border-border rounded-[8px]">
          {BASEMAPS.map((b) => (
            <button
              key={b.id}
              type="button"
              role="menuitemradio"
              aria-checked={b.id === current}
              onClick={() => {
                setBasemap(b.id);
                setOpen(false);
              }}
              className={cn('block w-full text-left px-3 py-[6px] hover:bg-surface2', b.id === current && 'font-medium text-text')}
            >
              {b.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/** Top right of the map (MapContainer places it): coordinates, Fit to result, Compare, Basemap ▾. A phone has no pointer to follow, so no coordinates. */
export function MapControls({ map }: { map: MapRef | null }) {
  const layers = useAppStore((s) => s.mapLayers);
  const requestFit = useAppStore((s) => s.requestFit);
  const compare = useAppStore((s) => s.compare);
  const setCompare = useAppStore((s) => s.setCompare);
  const fitTarget = activeResultLayer(layers)?.bounds;
  const sides = defaultCompareSides(layers);

  return (
    <div data-testid="map-controls" className="pointer-events-auto flex gap-[6px] text-[12px]">
      <Coordinates map={map} />
      <button
        type="button"
        disabled={!fitTarget}
        title={fitTarget ? undefined : 'No result on the map yet'}
        onClick={() => fitTarget && requestFit(fitTarget)}
        className={action}
      >
        Fit to result
      </button>
      <button
        type="button"
        aria-pressed={compare !== null}
        disabled={!compare && !sides}
        title={compare || sides ? undefined : 'Compare needs two layers on the map'}
        onClick={() => setCompare(compare ? null : sides && { ...sides, split: 0.5 })}
        className={cn(action, compare && 'bg-ink text-ink-bg border-ink hover:bg-ink')}
      >
        Compare
      </button>
      <BasemapPicker />
    </div>
  );
}
