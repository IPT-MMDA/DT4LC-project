import { describe, expect, it } from 'vitest';
import { attachmentChips } from './attachmentChips';

describe('attachmentChips', () => {
  it('shows a group as one chip and other files with their metadata line and size', () => {
    const group = { name: 'kakhovka_hls_l30', line: 'HLS L30 · B2–B6 · 30 m · 4.0 MB · 2023-06-01 → 06-09' };
    const chips = attachmentChips([
      { id: 's', filename: 'hlsl_20230601.tif', path: '/u/a.tif', group },
      { id: 'f', filename: 'forest_2024.tif', path: '/u/f.tif', metadata: { line: 'Sentinel-2 · B2 B3 B4 B8', size_label: '7.2 MB' } },
      { id: 's', filename: 'hlsl_20230609.tif', path: '/u/b.tif', group },
      { id: 'x', filename: 'scan.tif', path: '/u/x.tif', metadata: { line: '' } },
    ]);
    expect(chips.map(({ key, name, line, attachments }) => [key, name, line, attachments.length])).toEqual([
      ['group:kakhovka_hls_l30', 'kakhovka_hls_l30', group.line, 2],
      ['file:/u/f.tif', 'forest_2024.tif', 'Sentinel-2 · B2 B3 B4 B8 · 7.2 MB', 1],
      ['file:/u/x.tif', 'scan.tif', '', 1],
    ]);
  });
});
