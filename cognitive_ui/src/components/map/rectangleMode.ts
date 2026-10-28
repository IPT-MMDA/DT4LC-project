import type MapboxDraw from '@mapbox/mapbox-gl-draw';

// A MapboxDraw mode for the toolbar's Rectangle: press, drag, release. Hand-built,
// as the brief allows no new dependency; it fires draw.create like draw_polygon.

export const RECTANGLE_MODE = 'draw_rectangle';

type LngLat = [number, number];

/** The open ring of the rectangle spanned by two corners: NW, NE, SE, SW. */
export function rectangleRing(a: LngLat, b: LngLat): LngLat[] {
  const [west, east] = a[0] <= b[0] ? [a[0], b[0]] : [b[0], a[0]];
  const [south, north] = a[1] <= b[1] ? [a[1], b[1]] : [b[1], a[1]];
  return [[west, north], [east, north], [east, south], [west, south]];
}

interface RectangleState {
  rect: MapboxDraw.DrawPolygon;
  start: LngLat | null;
}

const lngLat = (e: MapboxDraw.MapMouseEvent): LngLat => [e.lngLat.lng, e.lngLat.lat];

function stretch(state: RectangleState, e: MapboxDraw.MapMouseEvent) {
  if (state.start) state.rect.setCoordinates([rectangleRing(state.start, lngLat(e))]);
}

export const rectangleMode: MapboxDraw.DrawCustomMode<RectangleState> = {
  onSetup() {
    const rect = this.newFeature({
      type: 'Feature',
      properties: {},
      geometry: { type: 'Polygon', coordinates: [[]] },
    }) as MapboxDraw.DrawPolygon;
    this.addFeature(rect);
    this.clearSelectedFeatures();
    this.updateUIClasses({ mouse: 'add' });
    this.map.dragPan.disable();
    return { rect, start: null };
  },

  onMouseDown(state, e) {
    state.start = lngLat(e);
  },

  onDrag: stretch,
  onMouseMove: stretch,

  onMouseUp(state, e) {
    if (!state.start) return;
    const end = lngLat(e);
    // A click without a drag draws nothing.
    if (end[0] === state.start[0] || end[1] === state.start[1]) {
      state.start = null;
      return;
    }
    state.rect.setCoordinates([rectangleRing(state.start, end)]);
    state.start = null;
    this.map.fire('draw.create', { features: [state.rect.toGeoJSON()] });
    this.changeMode('simple_select');
  },

  onKeyUp(_state, e) {
    if (e.key === 'Escape') this.changeMode('simple_select');
  },

  onStop(state) {
    this.map.dragPan.enable();
    this.updateUIClasses({ mouse: 'none' });
    const id = String(state.rect.id);
    if (this.getFeature(id) && !state.rect.isValid()) this.deleteFeature(id, { silent: true });
  },

  toDisplayFeatures(_state, geojson, display) {
    display(geojson);
  },
};
