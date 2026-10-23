import { useSyncExternalStore } from 'react';

/** The docked map sits beside the chat from this width (design/DESIGN.md section 1). */
export const DOCKED_MAP_QUERY = '(min-width: 1440px)';

/** Whether a CSS media query matches, updated when it changes. False without matchMedia. */
export function useMediaQuery(query: string): boolean {
  return useSyncExternalStore(
    (onChange) => {
      if (typeof window.matchMedia !== 'function') return () => {};
      const mql = window.matchMedia(query);
      mql.addEventListener('change', onChange);
      return () => mql.removeEventListener('change', onChange);
    },
    () => (typeof window.matchMedia === 'function' ? window.matchMedia(query).matches : false),
  );
}
