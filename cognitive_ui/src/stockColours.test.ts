// @vitest-environment node
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import tailwindColors from 'tailwindcss/colors';

// Files allowed to keep stock colours. Empty since v0.6.0 deleted the old
// pages and job cards; everything must use the design tokens.
const LEGACY = new Set<string>([]);

const HUES =
  'slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose|primary|geo|white';
// A Tailwind colour utility on a stock hue, e.g. `bg-blue-500`, `dark:hover:text-gray-300`,
// `text-white`. `bg-black/50` modal scrims are allowed.
const STOCK = new RegExp(`(?<![\\w-])(?:[a-z-]+:)*[a-z]+(?:-[a-z]+)*-(?:${HUES})(?:-\\d+)?(?:/\\d+)?(?![\\w-])`, 'g');

// Every hex value of the stock Tailwind palette (`#3b82f6`, `#2563eb`, ...), e.g. a
// MapLibre paint colour copied from blue-500. White and black are not stock hues.
// The renamed palettes (lightBlue, ...) are getters that only log a warning.
const RENAMED = new Set(['lightBlue', 'warmGray', 'trueGray', 'coolGray', 'blueGray']);
const STOCK_HEX = new Set(
  Object.keys(tailwindColors)
    .filter((name) => !RENAMED.has(name))
    .flatMap((name) => {
      const shades = (tailwindColors as unknown as Record<string, unknown>)[name];
      return typeof shades === 'object' && shades ? Object.values(shades as Record<string, string>) : [];
    })
    .map((hex) => hex.toLowerCase()),
);
const HEX = /#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b/g;
const stockHex = (text: string) => (text.match(HEX) ?? []).filter((hex) => STOCK_HEX.has(hex.toLowerCase()));

const srcDir = fileURLToPath(new URL('.', import.meta.url));
const surviving = (readdirSync(srcDir, { recursive: true }) as string[])
  .map((f) => f.split('\\').join('/'))
  .filter((f) => /\.(tsx?|jsx?)$/.test(f) && !/\.test\.tsx?$/.test(f) && !LEGACY.has(f));

describe('surviving components use design tokens, not the stock Tailwind palette', () => {
  it('finds the files it checks', () => {
    expect(surviving).toContain('components/shell/Rail.tsx');
    expect(surviving).toContain('components/map/mode/MapToolbar.tsx');
  });

  it.each(surviving)('%s has no stock palette classes', (file) => {
    const text = readFileSync(srcDir + file, 'utf8');
    expect(text.match(STOCK) ?? []).toEqual([]);
  });

  it.each(surviving)('%s has no stock palette hex values', (file) => {
    const text = readFileSync(srcDir + file, 'utf8');
    expect(stockHex(text)).toEqual([]);
  });

  it('the hex check catches stock blue but not the design tokens', () => {
    expect(STOCK_HEX.has('#3b82f6')).toBe(true);
    expect(stockHex("'fill-color': isSelected ? '#3b82f6' : '#10B981', '#2563eb'")).toEqual([
      '#3b82f6',
      '#10B981',
      '#2563eb',
    ]);
    // Legend colours and the monogram ink from design/DESIGN.md section 2.
    expect(stockHex("'#2d5f9e' '#8c4a2f' '#c4cad6' '#f1efe8' '#1a1d23' '#fff' '#000'")).toEqual([]);
  });

  it('the pattern catches stock blue', () => {
    expect('a bg-blue-500 dark:hover:text-blue-300 b'.match(STOCK)).toEqual([
      'bg-blue-500',
      'dark:hover:text-blue-300',
    ]);
    expect('bg-accent text-on-accent bg-black/50 border-accent/30'.match(STOCK)).toBeNull();
  });
});
