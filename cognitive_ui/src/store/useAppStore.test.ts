import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { DrawnRegion } from './useAppStore';

const STORAGE_KEY = 'dt4lc-app-storage';

const region: DrawnRegion = {
  id: 'region-1',
  name: 'Region 1',
  type: 'polygon',
  geometry: {
    type: 'Feature',
    properties: {},
    geometry: {
      type: 'Polygon',
      coordinates: [[[33, 46], [34, 46], [34, 47], [33, 47], [33, 46]]],
    },
  },
  bbox: [33, 46, 34, 47],
  createdAt: '2026-01-01T00:00:00.000Z',
};

// A fresh module instance creates a fresh store that rehydrates from localStorage,
// which is what a page reload does.
async function loadFreshStore() {
  vi.resetModules();
  return (await import('./useAppStore')).useAppStore;
}

beforeEach(() => {
  localStorage.clear();
});

describe('selected region persistence (#67 bug 6)', () => {
  it('restores the selected region after a reload', async () => {
    const other: DrawnRegion = { ...region, id: 'region-2', name: 'Region 2' };
    const store = await loadFreshStore();
    store.getState().addRegion(region);
    store.getState().addRegion(other);
    store.getState().setSelectedRegion(region);

    const persisted = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}');
    expect(persisted.state.selectedRegion).toEqual(region);

    const reloaded = await loadFreshStore();
    expect(reloaded.getState().drawnRegions).toEqual([region, other]);
    expect(reloaded.getState().selectedRegion).toEqual(region);
  });
});

describe('deleting the selected region (#67 bug 5)', () => {
  it('removes it from the store, clears the selection and stays gone after a reload', async () => {
    const other: DrawnRegion = { ...region, id: 'region-2', name: 'Region 2' };
    const store = await loadFreshStore();
    store.getState().addRegion(other);
    store.getState().addRegion(region);

    const selected = store.getState().selectedRegion;
    expect(selected?.id).toBe(region.id);
    store.getState().deleteRegion(selected!.id);

    expect(store.getState().drawnRegions).toEqual([other]);
    expect(store.getState().selectedRegion).toBeNull();

    const reloaded = await loadFreshStore();
    expect(reloaded.getState().drawnRegions).toEqual([other]);
    expect(reloaded.getState().selectedRegion).toBeNull();
  });
});
