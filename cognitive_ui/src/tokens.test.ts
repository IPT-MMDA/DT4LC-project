// @vitest-environment node
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

// Every CSS custom property in design/DESIGN.md section 2.
const DESIGN_TOKENS = [
  'bg', 'surface', 'surface2', 'bubble', 'border', 'border2',
  'text', 'text2', 'muted', 'faint',
  'accent', 'on-accent', 'accent-tint', 'accent-ink',
  'done', 'done-tint', 'done-ink',
  'run', 'run-tint', 'run-ink',
  'fail', 'fail-tint', 'fail-ink',
  'canc', 'canc-tint', 'canc-ink',
  'lost', 'ink', 'ink-bg',
];

const css = readFileSync(fileURLToPath(new URL('./index.css', import.meta.url)), 'utf8');

function declarations(selector: string): Map<string, string> {
  const start = css.indexOf(`${selector} {`);
  expect(start, `no "${selector} {" block in index.css`).toBeGreaterThanOrEqual(0);
  const body = css.slice(start, css.indexOf('}', start));
  return new Map(
    [...body.matchAll(/--([\w-]+)\s*:\s*([^;]+);/g)].map((m) => [m[1], m[2].trim()])
  );
}

describe('design tokens in index.css', () => {
  const light = declarations(':root,\n[data-theme="light"]');
  const dark = declarations('[data-theme="dark"]');

  it.each(DESIGN_TOKENS)('--%s is defined in the light and the dark theme', (name) => {
    expect(light.get(name)).toBeTruthy();
    expect(dark.get(name)).toBeTruthy();
  });

  it('both themes define the same set of tokens', () => {
    expect([...dark.keys()].sort()).toEqual([...light.keys()].sort());
  });

  it('defines the region amber in both themes, apart from the cancelled hue', () => {
    expect(light.get('region')).toBe('oklch(0.85 0.12 85)');
    expect(dark.get('region')).toBeTruthy();
    expect(light.get('region')).not.toBe(light.get('canc'));
  });

  it('uses the DESIGN values for accent and the inverted monogram ink', () => {
    expect(light.get('accent')).toBe('oklch(0.5 0.16 265)');
    expect(dark.get('accent')).toBe('oklch(0.72 0.12 265)');
    expect(light.get('ink')).toBe(light.get('text'));
    expect(light.get('ink-bg')).toBe(light.get('surface'));
    expect(dark.get('ink')).toBe(dark.get('text'));
    expect(dark.get('ink-bg')).toBe(dark.get('bg'));
  });
});
