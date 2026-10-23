import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useAppStore, type Layer } from '../../store/useAppStore';
import { DockedMap } from './DockedMap';

// jsdom has no WebGL: the map itself is a stand-in.
vi.mock('../map/MapContainer', () => ({ MapContainer: () => <div data-testid="maplibre-map" /> }));

const BOUNDS: Layer['bounds'] = [32.388779, 46.455331, 32.691118, 46.663879];
const change: Layer = {
  id: 'result-job-1-difference', name: 'Water change · NDWI', type: 'raster', visible: true, opacity: 0.8,
  url: '/app/resources/.cache/outputs/job-1/2_ChangeMap_change.tif', bounds: BOUNDS,
  legend: { kind: 'change', stops: ['#8c4a2f', '#c4cad6', '#2d5f9e'] },
};

const renderMap = () => render(<MemoryRouter><DockedMap /></MemoryRouter>);
const panel = () => within(screen.getByRole('region', { name: 'Layers' }));
const fit = () => screen.getByRole('button', { name: 'Fit to result' });

beforeEach(() => {
  useAppStore.setState({ mapLayers: [], fitRequest: null });
});
afterEach(cleanup);

describe('docked map Layers panel', () => {
  it('without result layers: the hint, Fit to result and Compare disabled', () => {
    renderMap();
    expect(panel().getByText('Results from the chat appear here as layers.')).toBeTruthy();
    expect(fit().hasAttribute('disabled')).toBe(true);
    expect(screen.getByRole('button', { name: 'Compare' }).hasAttribute('disabled')).toBe(true);
    expect(screen.queryByTestId('layers-legend')).toBeNull();
  });

  it('lists a result layer with its opacity, the lost · same · gained legend, and Fit to result zooms to it', () => {
    renderMap();
    act(() => {
      useAppStore.getState().addResultLayer(change);
      useAppStore.getState().clearFitRequest();
    });
    const row = panel().getByTestId('layer-row');
    expect(row.textContent).toBe('Water change · NDWI80');
    expect(screen.getByTestId('layers-legend').textContent).toBe('lost · same · gained');
    // The change style's ramp, as the tiles are drawn: brown → grey → blue.
    expect(screen.getByTestId('layers-legend-ramp').style.background).toBe('linear-gradient(90deg, #8c4a2f, #c4cad6, #2d5f9e)');

    fireEvent.change(panel().getByRole('slider', { name: 'Opacity of Water change · NDWI' }), { target: { value: '45' } });
    expect(useAppStore.getState().mapLayers[0].opacity).toBe(0.45);
    expect(panel().getByTestId('layer-row').textContent).toBe('Water change · NDWI45');

    expect(fit().hasAttribute('disabled')).toBe(false);
    fireEvent.click(fit());
    expect(useAppStore.getState().fitRequest).toEqual(BOUNDS);
  });

  it('a change layer without its style\'s ramp (an older job) falls back to the legend tokens', () => {
    useAppStore.getState().addResultLayer({ ...change, legend: { kind: 'change' } });
    renderMap();
    const ramp = screen.getByTestId('layers-legend-ramp');
    expect(ramp.className).toContain('from-legend-lost');
    expect(ramp.style.background).toBe('');
  });

  it('shows the newest layer\'s index ramp with its range', () => {
    useAppStore.getState().addResultLayer(change);
    useAppStore.getState().addResultLayer({ ...change, id: 'result-job-2-map', name: 'NDVI · 9 June', legend: { kind: 'ramp', stops: ['#994c1a', '#006600'], range: [-0.2, 0.8] } });
    renderMap();
    expect(panel().getAllByTestId('layer-row').map((r) => r.textContent)).toEqual(['Water change · NDWI80', 'NDVI · 9 June80']);
    expect(screen.getByTestId('layers-legend').textContent).toBe('−0.2 · 0.8');
  });
});
