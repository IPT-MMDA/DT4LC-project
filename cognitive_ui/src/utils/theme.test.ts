import { describe, expect, it } from 'vitest';
import { applyTheme } from './theme';

describe('applyTheme', () => {
  it('sets data-theme on the root element', () => {
    const root = document.createElement('html');
    applyTheme('dark', root);
    expect(root.dataset.theme).toBe('dark');
    applyTheme('light', root);
    expect(root.dataset.theme).toBe('light');
  });

  it('treats auto as light, as before', () => {
    const root = document.createElement('html');
    applyTheme('auto', root);
    expect(root.dataset.theme).toBe('light');
  });

  it('defaults to document.documentElement', () => {
    applyTheme('dark');
    expect(document.documentElement.dataset.theme).toBe('dark');
  });
});
