import { useSyncExternalStore } from 'react';

/** The docked map sits beside the chat from this width. */
export const DOCKED_MAP_QUERY = '(min-width: 1440px)';

/**
 * Phone and tablet widths, on Tailwind's `md` (768) and `lg` (1024) breakpoints, so the
 * `max-md:` and `md:max-lg:` classes switch at the same widths. From 1024 px up the
 * desktop layout applies unchanged.
 */
export const PHONE_QUERY = '(max-width: 767px)';
export const TABLET_QUERY = '(min-width: 768px) and (max-width: 1023px)';

export type Layout = 'phone' | 'tablet' | 'desktop';

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

/** Phone (< 768 px), tablet (768–1023 px) or desktop; desktop without matchMedia. */
export function useLayout(): Layout {
  const phone = useMediaQuery(PHONE_QUERY);
  const tablet = useMediaQuery(TABLET_QUERY);
  return phone ? 'phone' : tablet ? 'tablet' : 'desktop';
}
