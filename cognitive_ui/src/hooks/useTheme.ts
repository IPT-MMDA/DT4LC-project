import { useEffect } from 'react';
import { useAppStore } from '../store/useAppStore';
import { applyTheme, DARK_QUERY, resolveTheme } from '../utils/theme';
import { useMediaQuery } from './useMediaQuery';

/** The theme that renders: the stored one, or the system's for `auto`. */
export function useResolvedTheme(): 'light' | 'dark' {
  const theme = useAppStore((state) => state.theme);
  const prefersDark = useMediaQuery(DARK_QUERY);
  return resolveTheme(theme, prefersDark);
}

/**
 * Applies the theme to the document for every route (the landing too), and again when
 * the stored theme or, for `auto`, the system's preference changes.
 */
export function useApplyTheme() {
  const theme = useResolvedTheme();
  useEffect(() => {
    applyTheme(theme);
  }, [theme]);
}
