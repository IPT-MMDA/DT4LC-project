import { useAppStore } from '../store/useAppStore';
import { useResolvedTheme } from './useTheme';

/** Light/dark toggle for the rail and the account menu; from `auto`, it flips what the system shows. */
export function useThemeToggle() {
  const setTheme = useAppStore((state) => state.setTheme);
  const isDark = useResolvedTheme() === 'dark';
  return {
    isDark,
    label: isDark ? 'Switch to light theme' : 'Switch to dark theme',
    toggle: () => setTheme(isDark ? 'light' : 'dark'),
  };
}
