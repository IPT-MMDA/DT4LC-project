import { act, renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { readMapColours, toMapColour, useMapColours } from './mapColours';

describe('toMapColour', () => {
  it('converts oklch() to rgb(), which MapLibre can parse', () => {
    expect(toMapColour('oklch(1 0 0)')).toBe('rgb(255, 255, 255)');
    expect(toMapColour('oklch(0 0 0)')).toBe('rgb(0, 0, 0)');
    // CSS Color 4 reference value for sRGB red.
    expect(toMapColour('oklch(0.62796 0.25768 29.2339)')).toBe('rgb(255, 0, 0)');
    expect(toMapColour('oklch(62.796% 0.25768 29.2339deg)')).toBe('rgb(255, 0, 0)');
  });

  it('converts the light accent token', () => {
    // --accent: oklch(0.5 0.16 265), a mid indigo.
    const [r, g, b] = toMapColour(' oklch(0.5 0.16 265) ').match(/\d+/g)!.map(Number);
    expect(b).toBeGreaterThan(r);
    expect(b).toBeGreaterThan(g);
  });

  it('keeps alpha', () => {
    expect(toMapColour('oklch(1 0 0 / 0.5)')).toBe('rgba(255, 255, 255, 0.5)');
    expect(toMapColour('oklch(1 0 0 / 50%)')).toBe('rgba(255, 255, 255, 0.5)');
  });

  it('passes other colours through, and never returns an empty colour', () => {
    expect(toMapColour('#2d5f9e')).toBe('#2d5f9e');
    expect(toMapColour('rgb(1, 2, 3)')).toBe('rgb(1, 2, 3)');
    expect(toMapColour('')).toBe('rgb(128, 128, 128)');
  });
});

describe('map colours follow the theme tokens (QA D2)', () => {
  const sheet = document.createElement('style');
  sheet.textContent = `
    :root, [data-theme="light"] { --accent: #0000ff; --region: oklch(0 0 0); --canc: #ffaa00; --surface: #ffffff; }
    [data-theme="dark"] { --region: oklch(1 0 0); --canc: #aa5500; --surface: #111111; }
  `;

  afterEach(() => {
    sheet.remove();
    document.documentElement.removeAttribute('data-theme');
  });

  it('reads each role from its token', () => {
    document.head.append(sheet);
    document.documentElement.dataset.theme = 'light';
    expect(readMapColours()).toEqual({
      regionSelected: 'rgb(0, 0, 0)',
      region: 'rgb(0, 0, 0)',
      drawing: '#ffaa00',
      vertexFill: '#ffffff',
      boundaries: '#ffaa00',
      footprint: '#ffaa00',
    });
  });

  it('paints regions amber (--region), not with the accent', () => {
    document.head.append(sheet);
    document.documentElement.dataset.theme = 'light';
    const colours = readMapColours();
    expect([colours.region, colours.regionSelected]).toEqual(['rgb(0, 0, 0)', 'rgb(0, 0, 0)']);
    expect(Object.values(colours)).not.toContain('#0000ff');
  });

  it('re-reads when data-theme changes', async () => {
    document.head.append(sheet);
    document.documentElement.dataset.theme = 'light';
    const { result } = renderHook(() => useMapColours());
    expect(result.current.regionSelected).toBe('rgb(0, 0, 0)');

    await act(async () => {
      document.documentElement.dataset.theme = 'dark';
      await Promise.resolve(); // MutationObserver callbacks run as a microtask
    });
    expect(result.current.regionSelected).toBe('rgb(255, 255, 255)');
    expect(result.current.region).toBe('rgb(255, 255, 255)');
  });
});
