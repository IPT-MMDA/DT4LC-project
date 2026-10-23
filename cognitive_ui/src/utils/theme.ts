export type ThemeSetting = 'light' | 'dark' | 'auto';

/** The system's dark preference, which `auto` follows. */
export const DARK_QUERY = '(prefers-color-scheme: dark)';

/** The theme that renders: `auto` follows the system (light when it says nothing). */
export function resolveTheme(theme: ThemeSetting, prefersDark: boolean): 'light' | 'dark' {
  if (theme === 'auto') return prefersDark ? 'dark' : 'light';
  return theme;
}

function systemPrefersDark(): boolean {
  return typeof window.matchMedia === 'function' && window.matchMedia(DARK_QUERY).matches;
}

/** Sets the `data-theme` attribute the design tokens and Tailwind `dark:` key on. */
export function applyTheme(
  theme: ThemeSetting,
  root: HTMLElement = document.documentElement,
  prefersDark: boolean = systemPrefersDark()
) {
  root.dataset.theme = resolveTheme(theme, prefersDark);
}
