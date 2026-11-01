import { afterEach, describe, expect, it, vi } from 'vitest';
import { applyTheme, DARK_QUERY, resolveTheme } from './theme';

afterEach(() => {
  vi.unstubAllGlobals();
});

function stubSystem(dark: boolean) {
  vi.stubGlobal('matchMedia', (query: string) => ({
    matches: query === DARK_QUERY ? dark : false,
    media: query,
    addEventListener: () => {},
    removeEventListener: () => {},
  }));
}

describe('resolveTheme', () => {
  it('keeps light and dark, and turns auto into what the system prefers', () => {
    expect(resolveTheme('light', true)).toBe('light');
    expect(resolveTheme('dark', false)).toBe('dark');
    expect(resolveTheme('auto', true)).toBe('dark');
    expect(resolveTheme('auto', false)).toBe('light');
  });
});

describe('applyTheme', () => {
  it('sets data-theme on the root element', () => {
    const root = document.createElement('html');
    applyTheme('dark', root);
    expect(root.dataset.theme).toBe('dark');
    applyTheme('light', root);
    expect(root.dataset.theme).toBe('light');
  });

  it('follows prefers-color-scheme for auto', () => {
    const root = document.createElement('html');
    stubSystem(true);
    applyTheme('auto', root);
    expect(root.dataset.theme).toBe('dark');
    stubSystem(false);
    applyTheme('auto', root);
    expect(root.dataset.theme).toBe('light');
  });

  it('treats auto as light without matchMedia', () => {
    vi.stubGlobal('matchMedia', undefined);
    const root = document.createElement('html');
    applyTheme('auto', root);
    expect(root.dataset.theme).toBe('light');
  });

  it('defaults to document.documentElement', () => {
    applyTheme('dark');
    expect(document.documentElement.dataset.theme).toBe('dark');
  });
});
