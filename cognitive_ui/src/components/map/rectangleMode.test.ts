import { describe, expect, it } from 'vitest';
import { rectangleRing } from './rectangleMode';

describe('rectangleRing', () => {
  it('spans two corners as NW, NE, SE, SW, whichever way the drag went', () => {
    const ring = [[32.4, 46.66], [32.69, 46.66], [32.69, 46.46], [32.4, 46.46]];
    expect(rectangleRing([32.4, 46.66], [32.69, 46.46])).toEqual(ring);
    expect(rectangleRing([32.69, 46.46], [32.4, 46.66])).toEqual(ring);
    expect(rectangleRing([32.4, 46.46], [32.69, 46.66])).toEqual(ring);
  });
});
