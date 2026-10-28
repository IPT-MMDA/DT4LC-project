import { useAppStore, type MapTool } from '../../../store/useAppStore';
import { cn } from '../../../utils/cn';

const TOOLS: { tool: MapTool; label: string; hint: string }[] = [
  { tool: 'pan', label: 'Pan', hint: '' },
  { tool: 'polygon', label: 'Polygon', hint: 'Click to add points, double-click to finish · Esc cancels' },
  { tool: 'rectangle', label: 'Rectangle', hint: 'Drag to draw a rectangle · Esc cancels' },
];

/** Pan / Polygon / Rectangle / Clear, top left of the map (MapContainer places it). Clear deletes the selected region. */
export function MapToolbar() {
  const tool = useAppStore((s) => s.mapTool);
  const setMapTool = useAppStore((s) => s.setMapTool);
  const selected = useAppStore((s) => s.selectedRegion);
  const deleteRegion = useAppStore((s) => s.deleteRegion);
  const hint = TOOLS.find((t) => t.tool === tool)?.hint;

  return (
    <div className="flex flex-col items-start gap-[6px] pointer-events-none">
      <div role="toolbar" aria-label="Draw" data-testid="map-toolbar" className="pointer-events-auto flex gap-1 p-1 bg-surface border border-border rounded-[8px] text-[12px]">
        {TOOLS.map((t) => (
          <button
            key={t.tool}
            type="button"
            aria-pressed={tool === t.tool}
            onClick={() => setMapTool(t.tool)}
            className={cn('px-[10px] py-[5px] rounded-[5px]', tool === t.tool ? 'bg-ink text-ink-bg font-medium' : 'text-text2 hover:bg-surface2')}
          >
            {t.label}
          </button>
        ))}
        <button
          type="button"
          disabled={!selected}
          onClick={() => selected && deleteRegion(selected.id)}
          aria-label="Clear selected region"
          title={selected ? `Delete ${selected.name}` : 'Select a region to delete it'}
          className="px-[10px] py-[5px] rounded-[5px] text-text2 hover:bg-surface2 disabled:opacity-60 disabled:cursor-not-allowed disabled:hover:bg-transparent"
        >
          Clear
        </button>
      </div>
      {hint && (
        <p role="status" className="px-[10px] py-[5px] bg-surface border border-border rounded-[7px] text-[11.5px] text-text2">
          {hint}
        </p>
      )}
    </div>
  );
}
