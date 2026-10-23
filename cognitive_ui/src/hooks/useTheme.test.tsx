import { act, cleanup, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useAppStore } from '../store/useAppStore';
import { DARK_QUERY } from '../utils/theme';
import { useThemeToggle } from './useThemeToggle';
import { useApplyTheme } from './useTheme';

// A system dark preference that can flip, as when the OS switches at night.
let systemDark = false;
const listeners = new Set<() => void>();
function flipSystem(dark: boolean) {
  systemDark = dark;
  act(() => listeners.forEach((fn) => fn()));
}

beforeEach(() => {
  systemDark = false;
  listeners.clear();
  vi.stubGlobal('matchMedia', (query: string) => ({
    get matches() {
      return query === DARK_QUERY && systemDark;
    },
    media: query,
    addEventListener: (_: string, fn: () => void) => listeners.add(fn),
    removeEventListener: (_: string, fn: () => void) => listeners.delete(fn),
  }));
  useAppStore.setState({ theme: 'light' });
  delete document.documentElement.dataset.theme;
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const theme = () => document.documentElement.dataset.theme;

describe('useApplyTheme', () => {
  it('applies the stored theme and follows changes to it', () => {
    renderHook(() => useApplyTheme());
    expect(theme()).toBe('light');
    act(() => useAppStore.getState().setTheme('dark'));
    expect(theme()).toBe('dark');
  });

  it('auto follows the system, also when it changes', () => {
    useAppStore.setState({ theme: 'auto' });
    renderHook(() => useApplyTheme());
    expect(theme()).toBe('light');
    flipSystem(true);
    expect(theme()).toBe('dark');
    flipSystem(false);
    expect(theme()).toBe('light');
  });

  it('a stored light or dark wins over the system', () => {
    systemDark = true;
    renderHook(() => useApplyTheme());
    expect(theme()).toBe('light');
  });
});

describe('useThemeToggle', () => {
  it('from auto flips what is shown, and stores it', () => {
    useAppStore.setState({ theme: 'auto' });
    systemDark = true;
    const { result } = renderHook(() => useThemeToggle());
    expect(result.current.isDark).toBe(true);
    expect(result.current.label).toBe('Switch to light theme');
    act(() => result.current.toggle());
    expect(useAppStore.getState().theme).toBe('light');
    expect(result.current.isDark).toBe(false);
  });
});
