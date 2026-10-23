import { forwardRef, useEffect, useImperativeHandle, type ReactNode } from 'react';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useAppStore } from '../../store/useAppStore';
import { MapContainer } from './MapContainer';

// jsdom has no WebGL: render the map as plain DOM so the toolbar and region sources can be inspected.
// It "loads" on mount; fitBounds is recorded.
const fitBounds = vi.fn();
vi.mock('react-map-gl/maplibre', () => ({
  default: forwardRef(({ children, onLoad }: { children?: ReactNode; onLoad?: () => void }, ref) => {
    useImperativeHandle(ref, () => ({ getMap: () => null, fitBounds, getContainer: () => document.createElement('div') }));
    // A map loads once.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    useEffect(() => onLoad?.(), []);
    return <div>{children}</div>;
  }),
  NavigationControl: () => null,
  ScaleControl: () => null,
  Source: ({ id, children }: { id: string; children?: ReactNode }) => (
    <div data-testid={id}>{children}</div>
  ),
  Layer: () => null,
}));

// Minimal stand-in for the MapboxDraw wrapper: it holds features until deleteAll(),
// as the real control does. MapContainer calls deleteAll() after every draw.create.
let drawFeatures: unknown[] = [];
let fireCreate: (event: { features: unknown[] }) => void = () => {};
vi.mock('./DrawControl', () => ({
  DrawControl: forwardRef(
    ({ onCreate }: { onCreate: (event: { features: unknown[] }) => void }, ref) => {
      fireCreate = (event) => {
        drawFeatures = [...drawFeatures, ...event.features];
        onCreate(event);
      };
      useImperativeHandle(ref, () => ({
        changeMode: () => {},
        deleteAll: () => {
          drawFeatures = [];
        },
        getSelected: () => ({ type: 'FeatureCollection', features: drawFeatures }),
      }));
      return null;
    }
  ),
}));

vi.mock('./JobLayersManager', () => ({ JobLayersManager: () => null }));
vi.mock('./OblastBoundaries', () => ({ OblastBoundaries: () => null }));
vi.mock('./LocationQuickAccess', () => ({ LocationQuickAccess: () => null }));

const polygon = {
  type: 'Feature',
  properties: {},
  geometry: {
    type: 'Polygon',
    coordinates: [[[33, 46], [34, 46], [34, 47], [33, 47], [33, 46]]],
  },
};

beforeEach(() => {
  localStorage.clear();
  useAppStore.setState({ drawnRegions: [], selectedRegion: null, fitRequest: null });
  fitBounds.mockClear();
});

afterEach(() => {
  cleanup();
});

describe('MapContainer delete tool (#67 bug 5)', () => {
  it('removes the region just drawn from the store and the map', () => {
    render(<MapContainer />);

    act(() => fireCreate({ features: [polygon] }));
    const drawn = useAppStore.getState().selectedRegion;
    expect(drawn).not.toBeNull();
    expect(screen.getByTestId(`region-${drawn!.id}`)).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: 'Delete selected region' }));

    expect(useAppStore.getState().drawnRegions).toEqual([]);
    expect(useAppStore.getState().selectedRegion).toBeNull();
    expect(screen.queryByTestId(`region-${drawn!.id}`)).toBeNull();
  });
});

describe('MapContainer fit requests (On map, Fit to result)', () => {
  it('zooms to a request made before the map loaded, then clears it', () => {
    useAppStore.getState().requestFit([32.38, 46.45, 32.69, 46.66]);
    render(<MapContainer docked />);
    expect(fitBounds).toHaveBeenCalledWith([[32.38, 46.45], [32.69, 46.66]], expect.objectContaining({ padding: 48 }));
    expect(useAppStore.getState().fitRequest).toBeNull();
  });

  it('zooms to a request made while it is shown', () => {
    render(<MapContainer docked />);
    expect(fitBounds).not.toHaveBeenCalled();
    act(() => useAppStore.getState().requestFit([1, 2, 3, 4]));
    expect(fitBounds).toHaveBeenCalledWith([[1, 2], [3, 4]], expect.anything());
    expect(useAppStore.getState().fitRequest).toBeNull();
  });
});
