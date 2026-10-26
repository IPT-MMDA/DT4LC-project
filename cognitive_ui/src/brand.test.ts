// @vitest-environment node
import { existsSync, readFileSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const brand = (file: string) => fileURLToPath(new URL(`../public/brand/${file}`, import.meta.url));

describe('public/brand', () => {
  it.each(['dt-monogram.svg', 'logo_mmda.png', 'logo_umd.png', 'logo_unige.png', 'logo_unige_grey.png', 'README.md'])(
    '%s exists and is not empty',
    (file) => {
      expect(existsSync(brand(file))).toBe(true);
      expect(statSync(brand(file)).size).toBeGreaterThan(0);
    }
  );

  it('the logos are PNG files', () => {
    for (const file of ['logo_mmda.png', 'logo_umd.png', 'logo_unige.png', 'logo_unige_grey.png']) {
      expect(readFileSync(brand(file)).subarray(1, 4).toString('latin1'), file).toBe('PNG');
    }
  });

  it('the monogram draws with the --ink / --ink-bg tokens', () => {
    const svg = readFileSync(brand('dt-monogram.svg'), 'utf8');
    expect(svg).toContain('var(--ink,');
    expect(svg).toContain('var(--ink-bg,');
    expect(svg).toMatch(/>DT</);
  });

  it('the README "Missing" section names SNSF and no longer names UMD', () => {
    const readme = readFileSync(brand('README.md'), 'utf8');
    const missing = readme.split(/^## Missing\b.*$/m)[1]?.split(/^## /m)[0];
    expect(missing, 'the README has a "## Missing" section').toBeDefined();
    expect(missing).toContain('logo_snsf.png');
    expect(missing).not.toContain('logo_umd.png');
  });
});
