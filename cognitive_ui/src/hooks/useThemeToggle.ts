import { useAppStore } from '../store/useAppStore';

/** Light/dark toggle for the rail and the account menu; the Shell applies it with applyTheme(). */
export function useThemeToggle() {
  const theme = useAppStore((state) => state.theme);
  const setTheme = useAppStore((state) => state.setTheme);
  const isDark = theme === 'dark';
  return {
    isDark,
    label: isDark ? 'Switch to light theme' : 'Switch to dark theme',
    toggle: () => setTheme(isDark ? 'light' : 'dark'),
  };
}
