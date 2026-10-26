import type { ReactNode } from 'react';
import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { OblastBoundaries } from './OblastBoundaries';

// jsdom has no WebGL: the Source renders its props as DOM, and there is no map to reorder layers on.
vi.mock('react-map-gl/maplibre', () => ({
  useMap: () => ({ current: undefined }),
  Source: ({ id, data, attribution, children }: { id: string; data: string; attribution?: string; children?: ReactNode }) => (
    <div data-testid={id} data-data={data} data-attribution={attribution}>
      {children}
    </div>
  ),
  Layer: ({ id }: { id: string }) => <span data-testid={id} />,
}));

afterEach(cleanup);

describe('OblastBoundaries', () => {
  it('credits OCHA / HDX and the licence on the boundaries source', () => {
    render(<OblastBoundaries />);
    const source = screen.getByTestId('oblast-boundaries');
    expect(source.getAttribute('data-data')).toBe('/ukraine_oblasts.json');
    expect(source.getAttribute('data-attribution')).toBe('Oblast boundaries: OCHA / HDX, CC BY-IGO 3.0');
    expect(screen.getByTestId('oblast-boundaries-line')).toBeTruthy();
  });
});
