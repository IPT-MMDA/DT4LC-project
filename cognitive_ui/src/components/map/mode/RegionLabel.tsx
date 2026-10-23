import { Marker } from 'react-map-gl/maplibre';
import { useAppStore } from '../../../store/useAppStore';
import { formatArea } from '../../chat/regionChip';
import { regionAreaKm2 } from './imagery';

/** "Dnipro floodplain · Kherson · 412 km²" in mono above the selected region's north-west corner. */
export function RegionLabel() {
  const region = useAppStore((s) => s.selectedRegion);
  const opacity = useAppStore((s) => s.regionOpacity);
  if (!region || opacity === 0) return null;
  const [west, , , north] = region.bbox;

  return (
    <Marker longitude={west} latitude={north} anchor="bottom-left" offset={[0, -6]} style={{ pointerEvents: 'none' }}>
      <div data-testid="region-label" className="font-mono font-medium text-[10px] bg-surface text-text2 px-[6px] py-[2px] rounded-[3px] whitespace-nowrap">
        {region.name} · {formatArea(regionAreaKm2(region))}
      </div>
    </Marker>
  );
}
