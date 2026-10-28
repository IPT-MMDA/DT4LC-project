// Compare's swipe divider (hand-built): a second map shows the right-hand layer and is
// clipped to the right of the divider. `split` is the divider's place across the map, 0..1.

export const clampSplit = (split: number): number => Math.min(1, Math.max(0, split));

/** The divider under the pointer: `clientX` within the map's box. */
export function splitFromPointer(clientX: number, box: { left: number; width: number }): number {
  return box.width > 0 ? clampSplit((clientX - box.left) / box.width) : 0.5;
}

const STEP = 0.02;
const BIG_STEP = 0.1;

/** The divider after a key on its handle, or null for a key it ignores. */
export function splitFromKey(split: number, key: string, shift = false): number | null {
  const step = shift ? BIG_STEP : STEP;
  switch (key) {
    case 'ArrowLeft':
    case 'ArrowDown':
      return clampSplit(split - step);
    case 'ArrowRight':
    case 'ArrowUp':
      return clampSplit(split + step);
    case 'PageDown':
      return clampSplit(split - BIG_STEP);
    case 'PageUp':
      return clampSplit(split + BIG_STEP);
    case 'Home':
      return 0;
    case 'End':
      return 1;
    default:
      return null;
  }
}

/** The right-hand map's CSS clip: everything left of the divider is cut away. */
export const clipRightOf = (split: number): string => `inset(0 0 0 ${(clampSplit(split) * 100).toFixed(2)}%)`;
