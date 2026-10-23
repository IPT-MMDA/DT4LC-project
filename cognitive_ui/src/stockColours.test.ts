// @vitest-environment node
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

// Old pages and job cards that Phase 2 of v0.6.0 replaces; they keep their stock
// colours until they are deleted. Everything else must use the design tokens.
const LEGACY = new Set([
  'pages/ChatPage.tsx',
  'pages/Dashboard.tsx',
  'pages/DataPage.tsx',
  'pages/JobDetailsPage.tsx',
  'pages/JobsPage.tsx',
  'pages/ModelsPage.tsx',
  'components/chat/ChatHistory.tsx',
  'components/chat/FileUploadDropzone.tsx',
  'components/chat/JobResultCard.tsx',
  'components/jobs/JobResult.tsx',
]);

const HUES =
  'slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose|primary|geo|white';
// A Tailwind colour utility on a stock hue, e.g. `bg-blue-500`, `dark:hover:text-gray-300`,
// `text-white`. `bg-black/50` modal scrims are allowed.
const STOCK = new RegExp(`(?<![\\w-])(?:[a-z-]+:)*[a-z]+(?:-[a-z]+)*-(?:${HUES})(?:-\\d+)?(?:/\\d+)?(?![\\w-])`, 'g');

const srcDir = fileURLToPath(new URL('.', import.meta.url));
const surviving = (readdirSync(srcDir, { recursive: true }) as string[])
  .map((f) => f.split('\\').join('/'))
  .filter((f) => /\.(tsx?|jsx?)$/.test(f) && !/\.test\.tsx?$/.test(f) && !LEGACY.has(f));

describe('surviving components use design tokens, not the stock Tailwind palette', () => {
  it('finds the files it checks', () => {
    expect(surviving).toContain('components/layout/Sidebar.tsx');
    expect(surviving).toContain('components/map/DrawToolbar.tsx');
  });

  it.each(surviving)('%s has no stock palette classes', (file) => {
    const text = readFileSync(srcDir + file, 'utf8');
    expect(text.match(STOCK) ?? []).toEqual([]);
  });

  it('the pattern catches stock blue', () => {
    expect('a bg-blue-500 dark:hover:text-blue-300 b'.match(STOCK)).toEqual([
      'bg-blue-500',
      'dark:hover:text-blue-300',
    ]);
    expect('bg-accent text-on-accent bg-black/50 border-accent/30'.match(STOCK)).toBeNull();
  });
});
