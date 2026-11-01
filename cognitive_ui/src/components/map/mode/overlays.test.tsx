import type { ReactNode } from 'react';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useAppStore, type Layer } from '../../../store/useAppStore';
import { CompareSwipe } from './CompareSwipe';
import { MapLegend } from './MapLegend';

// jsdom has no WebGL: Compare's second map is a plain box.
vi.mock('react-map-gl/maplibre', () => ({ default: ({ children }: { children?: ReactNode }) => <div data-testid="second-map">{children}</div> }));
vi.mock('../JobLayersManager', () => ({
  RasterAnchor: () => null,
  JobLayersManager: ({ only }: { only?: string }) => <span data-testid="second-map-layer">{only}</span>,
}));

const result: Layer = {
  id: 'result-1-difference', name: 'Water change · NDWI', type: 'raster', visible: true, opacity: 0.8, url: '/o/change.tif', source: 'chat',
  legend: {
    kind: 'change',
    stops: ['#8c4a2f', '#c4cad6', '#2d5f9e'],
    range: [-0.5, 0.5],
    classes: [
      { label: 'lost', tone: 'lost', colour: '#8c4a2f', area_km2: 0.603 },
      { label: 'no change', tone: 'neutral', colour: '#c4cad6', area_km2: 373.476 },
      { label: 'gained', tone: 'gained', colour: '#2d5f9e', area_km2: 156.761 },
    ],
  },
};
const sample: Layer = { id: 'sample-1-2023-06-09', name: 'kakhovka_hls_l30', type: 'raster', visible: true, opacity: 1, url: '/u/0609.tif', source: 'sample' };

beforeEach(() => {
  useAppStore.setState({ mapLayers: [sample, result], compare: null });
});
afterEach(cleanup);

describe('the map legend', () => {
  it("shows the active result's title, ramp over its real range and class areas", () => {
    render(<MapLegend />);
    expect(screen.getByTestId('map-legend-title').textContent).toBe('Water change · NDWI difference');
    expect(screen.getByTestId('map-legend-ticks').textContent).toBe('−0.5 · 0 · +0.5');
    expect(screen.getByTestId('map-legend-ramp').style.background).toBe('linear-gradient(90deg, #8c4a2f, #c4cad6, #2d5f9e)');
    expect(screen.getByTestId('map-legend-classes').textContent).toBe('lost 0.6 km²no change 373 km²gained 157 km²');
  });

  it('says what every colour means in text: the ramp and swatches are decoration, the ticks and classes are words', () => {
    render(<MapLegend />);
    const legend = screen.getByRole('region', { name: 'Legend' });
    for (const el of legend.querySelectorAll<HTMLElement>('[style*="background"]')) expect(el.getAttribute('aria-hidden')).toBe('true');
    expect(legend.textContent).toBe('Water change · NDWI difference−0.5 · 0 · +0.5lost 0.6 km²no change 373 km²gained 157 km²');
    // Gained and lost in the theme's text tokens, which have a dark variant, not in the swatch colours.
    const words = [...screen.getByTestId('map-legend-classes').querySelectorAll('li > span:last-child')];
    expect(words.map((el) => el.className)).toEqual(['text-lost', '', 'text-gained']);
  });

  it('is gone without a visible result layer', () => {
    useAppStore.setState({ mapLayers: [sample, { ...result, visible: false }] });
    render(<MapLegend />);
    expect(screen.queryByTestId('map-legend')).toBeNull();
  });
});

describe('Compare', () => {
  const view = { longitude: 32.54, latitude: 46.56, zoom: 11 };

  it('shows the right-hand layer on a second map clipped at the divider, which the keyboard moves', () => {
    const after: Layer = { ...result, id: 'result-1-after', name: 'NDWI · 2023-06-09', legend: undefined };
    useAppStore.setState({ mapLayers: [sample, after, result], compare: { left: result.id, right: sample.id, split: 0.5 } });
    render(<CompareSwipe viewState={view} mapStyle="style.json" />);
    expect(screen.getByTestId('second-map-layer').textContent).toBe(sample.id);
    expect(screen.getByTestId('compare-right').style.clipPath).toBe('inset(0 0 0 50.00%)');

    const handle = screen.getByRole('slider', { name: 'Compare divider' });
    expect(handle.getAttribute('aria-valuenow')).toBe('50');
    expect(handle.getAttribute('aria-valuetext')).toBe('Water change · NDWI on the left, kakhovka_hls_l30 on the right');
    fireEvent.keyDown(handle, { key: 'ArrowLeft', shiftKey: true });
    expect(useAppStore.getState().compare?.split).toBeCloseTo(0.4);
    fireEvent.keyDown(handle, { key: 'End' });
    expect(useAppStore.getState().compare?.split).toBe(1);
    expect(screen.getByTestId('compare-right').style.clipPath).toBe('inset(0 0 0 100.00%)');

    // Another layer on the left (before vs after); a side never lists the other side's layer.
    const left = screen.getByRole('combobox', { name: 'Left of the divider' });
    expect([...left.querySelectorAll('option')].map((o) => o.textContent)).toEqual(['Water change · NDWI', 'NDWI · 2023-06-09']);
    fireEvent.change(left, { target: { value: 'result-1-after' } });
    expect(useAppStore.getState().compare).toMatchObject({ left: 'result-1-after', right: sample.id });
  });

  it('renders nothing when off', () => {
    render(<CompareSwipe viewState={view} mapStyle="style.json" />);
    expect(screen.queryByTestId('compare')).toBeNull();
  });
});
