import { describe, expect, it } from 'vitest';
import { downloadLine, formatDuration } from './format';
import { changeDraft } from './proposal';
import { NDMI_PLAN } from '../../../test/jobEvents';

describe('formatDuration', () => {
  it.each([
    [0, '<1 s'],
    [0.54, '<1 s'],
    [9.4, '9 s'],
    [59.4, '59 s'],
    [72, '1.2 min'],
    [-1, ''],
  ])('%s s → %s', (s, text) => expect(formatDuration(s)).toBe(text));
});

describe('downloadLine', () => {
  const message = 'Downloading Prithvi EO v1 (100M) · 450 MB';
  it('message, bytes done and time left', () => {
    expect(downloadLine({ message, bytes: 52428801, totalBytes: 471859200, etaS: 56 })).toBe(`${message} · 50 MB done · 56 s left`);
  });
  it('no ETA before bytes arrive, nor an absurd one', () => {
    expect(downloadLine({ message, bytes: 0, totalBytes: 471859200, etaS: 75 })).toBe(message);
    expect(downloadLine({ message, bytes: 1, totalBytes: 471859200, etaS: 472641388 })).toBe(`${message} · 0 MB done`);
  });
});

describe('changeDraft', () => {
  it('the prompt with the proposed formula and threshold, to edit', () => {
    expect(changeDraft('Map moisture change with NDMI between the two dates', NDMI_PLAN)).toBe(
      'Map moisture change with NDMI between the two dates, using NDMI = (B5 - B6) / (B5 + B6) and a change threshold of 0.2',
    );
  });
  it('the prompt alone when the plan has no formula', () => {
    expect(changeDraft('Calculate NDVI', { steps: [{ uses: 'algorithms/ndvi', config: {} }] })).toBe('Calculate NDVI');
  });
});
