import { useRef, type KeyboardEvent, type PointerEvent } from 'react';
import Map from 'react-map-gl/maplibre';
import type { StyleSpecification } from 'maplibre-gl';
import { useAppStore, type CompareState } from '../../../store/useAppStore';
import { JobLayersManager, RasterAnchor } from '../JobLayersManager';
import { clipRightOf, splitFromKey, splitFromPointer } from './compare';
import { compareCandidates } from './layers';

interface CompareSwipeProps {
  viewState: { longitude: number; latitude: number; zoom: number };
  mapStyle: string | StyleSpecification;
}

const select = 'pointer-events-auto max-w-[200px] px-2 py-1 rounded-[6px] bg-surface border border-border text-[11.5px] font-mono text-text2';

/**
 * Compare: a second map, in step with the first, shows the right-hand layer and is
 * clipped to the right of the divider; the first map shows the left-hand layer. The
 * handle drags, and moves with the arrow, Page, Home and End keys.
 */
export function CompareSwipe({ viewState, mapStyle }: CompareSwipeProps) {
  const compare = useAppStore((s) => s.compare);
  const setCompare = useAppStore((s) => s.setCompare);
  const layers = useAppStore((s) => s.mapLayers);
  const boxRef = useRef<HTMLDivElement>(null);
  if (!compare) return null;

  const candidates = compareCandidates(layers);
  const nameOf = (id: string) => candidates.find((l) => l.id === id)?.name ?? id;
  const update = (next: Partial<CompareState>) => setCompare({ ...compare, ...next });
  const pct = `${(compare.split * 100).toFixed(2)}%`;

  const drag = (e: PointerEvent<HTMLDivElement>) => {
    const rect = boxRef.current?.getBoundingClientRect();
    if (rect && e.currentTarget.hasPointerCapture(e.pointerId)) update({ split: splitFromPointer(e.clientX, rect) });
  };
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const split = splitFromKey(compare.split, e.key, e.shiftKey);
    if (split === null) return;
    e.preventDefault();
    update({ split });
  };

  return (
    <div ref={boxRef} data-testid="compare" className="absolute inset-0 pointer-events-none">
      <div data-testid="compare-right" className="absolute inset-0" style={{ clipPath: clipRightOf(compare.split) }}>
        <Map {...viewState} interactive={false} attributionControl={false} mapStyle={mapStyle} style={{ width: '100%', height: '100%' }}>
          <RasterAnchor />
          <JobLayersManager only={compare.right} footprints={false} />
        </Map>
      </div>

      <div
        data-testid="compare-divider"
        className="absolute top-0 bottom-0 w-3 -ml-[6px] pointer-events-auto cursor-ew-resize touch-none flex justify-center"
        style={{ left: pct }}
        onPointerDown={(e) => e.currentTarget.setPointerCapture(e.pointerId)}
        onPointerMove={drag}
        onPointerUp={(e) => e.currentTarget.releasePointerCapture(e.pointerId)}
      >
        <div aria-hidden className="w-[2px] h-full bg-surface" />
        <div
          role="slider"
          tabIndex={0}
          aria-label="Compare divider"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(compare.split * 100)}
          aria-valuetext={`${nameOf(compare.left)} on the left, ${nameOf(compare.right)} on the right`}
          onKeyDown={onKeyDown}
          className="absolute top-1/2 -translate-y-1/2 w-7 h-7 rounded-full bg-surface border border-border text-text2 text-[12px] flex items-center justify-center focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
        >
          ⇆
        </div>
      </div>

      <div className="absolute top-[60px] -translate-x-full pr-2" style={{ left: pct }}>
        <select aria-label="Left of the divider" value={compare.left} onChange={(e) => update({ left: e.target.value })} className={select}>
          {candidates.filter((l) => l.id !== compare.right).map((l) => <option key={l.id} value={l.id}>{l.name}</option>)}
        </select>
      </div>
      <div className="absolute top-[60px] pl-2" style={{ left: pct }}>
        <select aria-label="Right of the divider" value={compare.right} onChange={(e) => update({ right: e.target.value })} className={select}>
          {candidates.filter((l) => l.id !== compare.left).map((l) => <option key={l.id} value={l.id}>{l.name}</option>)}
        </select>
      </div>
    </div>
  );
}
