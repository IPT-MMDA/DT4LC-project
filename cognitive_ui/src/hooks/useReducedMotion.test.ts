import { act, renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { REDUCED_MOTION_QUERY, useReducedMotion } from './useReducedMotion';

// A matchMedia whose reduced-motion answer can change, as when the OS setting flips.
function stubMatchMedia(initial: boolean) {
  let matches = initial;
  const listeners = new Set<() => void>();
  vi.stubGlobal('matchMedia', (query: string) => ({
    get matches() {
      return query === REDUCED_MOTION_QUERY && matches;
    },
    media: query,
    addEventListener: (_: string, fn: () => void) => listeners.add(fn),
    removeEventListener: (_: string, fn: () => void) => listeners.delete(fn),
  }));
  return {
    set(value: boolean) {
      matches = value;
      listeners.forEach((fn) => fn());
    },
    listeners,
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('useReducedMotion', () => {
  it('reads prefers-reduced-motion: reduce', () => {
    stubMatchMedia(true);
    expect(renderHook(() => useReducedMotion()).result.current).toBe(true);
  });

  it('is false when the user has no preference', () => {
    stubMatchMedia(false);
    expect(renderHook(() => useReducedMotion()).result.current).toBe(false);
  });

  it('follows the setting when it changes, and stops listening on unmount', () => {
    const media = stubMatchMedia(false);
    const { result, unmount } = renderHook(() => useReducedMotion());
    act(() => media.set(true));
    expect(result.current).toBe(true);
    act(() => media.set(false));
    expect(result.current).toBe(false);
    unmount();
    expect(media.listeners.size).toBe(0);
  });

  it('is false without matchMedia', () => {
    vi.stubGlobal('matchMedia', undefined);
    expect(renderHook(() => useReducedMotion()).result.current).toBe(false);
  });
});
