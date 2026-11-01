import { describe, expect, it } from 'vitest';
import { clampSplit, clipRightOf, splitFromKey, splitFromPointer } from './compare';

describe('the compare divider', () => {
  it('follows the pointer across the map, clamped to its edges', () => {
    const box = { left: 374, width: 1066 };
    expect(splitFromPointer(374 + 533, box)).toBe(0.5);
    expect(splitFromPointer(374 + 1066 * 0.25, box)).toBe(0.25);
    expect(splitFromPointer(100, box)).toBe(0);
    expect(splitFromPointer(2000, box)).toBe(1);
    expect(splitFromPointer(500, { left: 0, width: 0 })).toBe(0.5);
  });

  it('moves with the keyboard: arrows by 2 %, Shift or Page keys by 10 %, Home and End to the edges', () => {
    expect(splitFromKey(0.5, 'ArrowLeft')).toBeCloseTo(0.48);
    expect(splitFromKey(0.5, 'ArrowRight')).toBeCloseTo(0.52);
    expect(splitFromKey(0.5, 'ArrowRight', true)).toBeCloseTo(0.6);
    expect(splitFromKey(0.5, 'PageDown')).toBeCloseTo(0.4);
    expect(splitFromKey(0.99, 'ArrowRight')).toBe(1);
    expect(splitFromKey(0.01, 'ArrowLeft')).toBe(0);
    expect(splitFromKey(0.5, 'Home')).toBe(0);
    expect(splitFromKey(0.5, 'End')).toBe(1);
    expect(splitFromKey(0.5, 'Enter')).toBeNull();
  });

  it('clips the right-hand map to the right of the divider', () => {
    expect(clipRightOf(0.5)).toBe('inset(0 0 0 50.00%)');
    expect(clipRightOf(0.125)).toBe('inset(0 0 0 12.50%)');
    expect(clipRightOf(1.5)).toBe('inset(0 0 0 100.00%)');
    expect(clampSplit(-1)).toBe(0);
  });
});
