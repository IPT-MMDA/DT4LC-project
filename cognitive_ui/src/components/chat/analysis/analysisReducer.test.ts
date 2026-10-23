import { describe, expect, it } from 'vitest';
import {
  EE_FAILED,
  EE_FIXES,
  NDMI_CONFIRMED,
  NDMI_PLAN,
  NDMI_PROPOSAL,
  NDMI_PROPOSAL_TEXT,
  NDVI_CANCELLED,
  NDVI_DONE,
  NDVI_FAILED_AT_2,
  NDVI_FOLLOWUPS,
  PRITHVI_PREPARING,
} from '../../../test/jobEvents';
import type { JobEvent } from '../../../types';
import { analysisReducer, isTerminal, lastSeq, mergeEvents, statusOf, type AnalysisView } from './analysisReducer';

const statuses = (v: AnalysisView) => v.steps.map((s) => s.status);

describe('analysisReducer over the server sequences', () => {
  it.each<[string, JobEvent[], Partial<AnalysisView>, string[]]>([
    [
      'intent only',
      NDVI_DONE.slice(0, 1),
      { phase: 'starting', intent: "I'll compute NDVI for 1 June from the sample.", lastSeq: 1 },
      [],
    ],
    ['planning started: no skeleton yet', NDVI_DONE.slice(0, 2), { phase: 'planning' }, []],
    ['planning ready: the skeleton, all queued', NDVI_DONE.slice(0, 3), { phase: 'planning', composed: false }, ['queued', 'queued', 'queued']],
    ['step 2 running', NDVI_DONE.slice(0, 6), { phase: 'running', current: 2 }, ['done', 'running', 'queued']],
    ['NDVI done', NDVI_DONE, { phase: 'done', current: null, progress: 1, lastSeq: 10 }, ['done', 'done', 'done']],
    ['failed in step 2: later rows skipped', NDVI_FAILED_AT_2, { phase: 'failed' }, ['done', 'failed', 'skipped']],
    ['cancelled at step 2', NDVI_CANCELLED, { phase: 'cancelled', cancelled: { atStep: 2 } }, ['done', 'cancelled', 'skipped']],
    ['refused at planning: no rows', EE_FAILED, { phase: 'failed', steps: [] }, []],
    ['proposal waits', NDMI_PROPOSAL, { phase: 'proposal', confirmed: false }, []],
    ['proposal confirmed and done', NDMI_CONFIRMED, { phase: 'done', confirmed: true, composed: true }, ['done', 'done', 'done', 'done']],
  ])('%s', (_name, events, expected, rows) => {
    const view = analysisReducer(events);
    expect(view).toMatchObject(expected);
    expect(statuses(view)).toEqual(rows);
  });

  it('NDVI done: names, elapsed times, summary and follow-ups', () => {
    const view = analysisReducer(NDVI_DONE);
    expect(view.steps.map((s) => [s.index, s.name, s.kind, s.elapsedS])).toEqual([
      [1, 'Read hlsl_20230601.tif', 'input', 0],
      [2, 'NDVI', 'algorithm', 0.54],
      [3, 'Interpretation', 'postprocess', 0],
    ]);
    expect(view.done).toEqual({ elapsedS: 0.7, summary: '3 steps · <1 s · NDVI', followups: NDVI_FOLLOWUPS });
    expect(statusOf(view)).toBe('done');
    expect(isTerminal(view)).toBe(true);
  });

  it('the proposal keeps its text and plan after Run', () => {
    expect(analysisReducer(NDMI_PROPOSAL).proposal).toEqual({ text: NDMI_PROPOSAL_TEXT, plan: NDMI_PLAN });
    const done = analysisReducer(NDMI_CONFIRMED);
    expect(done.proposal?.text).toBe(NDMI_PROPOSAL_TEXT);
    expect(done.done?.summary).toBe('4 steps · 2 s · NDMI');
    expect(statusOf(analysisReducer(NDMI_PROPOSAL))).toBe('pending');
  });

  it('failed: code, plain message, fixes and raw', () => {
    expect(analysisReducer(EE_FAILED).failure).toEqual({
      code: 'EE_NOT_CONFIGURED',
      message: "Earth Engine isn't configured on this server, so I can't fetch imagery for this region.",
      fixes: EE_FIXES,
      raw: "Earth Engine isn't configured on this server, so I can't fetch imagery for this region.",
    });
    const midRun = analysisReducer(NDVI_FAILED_AT_2).failure;
    expect(midRun?.stepIndex).toBe(2);
    expect(midRun?.fixes.map((f) => f.kind)).toEqual(['retry', 'prompt', 'open_map']);
    expect(midRun?.fixes[0].params).toEqual({ mode: 'template' });
  });

  it('the Preparing model row carries bytes, total and time left', () => {
    const view = analysisReducer(PRITHVI_PREPARING);
    const prepare = view.steps[1];
    expect(prepare).toMatchObject({ kind: 'prepare', status: 'running', name: 'Preparing model · Prithvi EO v1 (100M)' });
    expect(prepare.progress).toEqual({
      fraction: 0.111,
      message: 'Downloading Prithvi EO v1 (100M) · 450 MB',
      bytes: 52428801,
      totalBytes: 471859200,
      etaS: 56,
    });
    // One of four rows finished plus 11.1 % of the running one.
    expect(view.progress).toBeCloseTo((1 + 0.111) / 4, 5);
    expect(view.current).toBe(2);
  });

  it('ignores order, duplicates and anything after a terminal event', () => {
    const shuffled = [...NDVI_DONE].reverse().concat(NDVI_DONE.slice(2, 5));
    expect(analysisReducer(shuffled)).toEqual(analysisReducer(NDVI_DONE));
    const late = { seq: 11, type: 'step_started', data: { index: 3, t: 9 } };
    expect(analysisReducer([...NDVI_CANCELLED, late]).steps[2].status).toBe('skipped');
  });
});

describe('mergeEvents: resuming with ?after=<lastSeq>', () => {
  it('a reconnect that replays from an older seq gives the same timeline as one stream', () => {
    const before = NDVI_DONE.slice(0, 8);
    expect(lastSeq(before)).toBe(8);
    // The stream after a reload replays 7..10 (the persisted log lost its tail).
    const merged = mergeEvents(before, NDVI_DONE.slice(6));
    expect(merged.map((e) => (e as JobEvent).seq)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    expect(analysisReducer(merged)).toEqual(analysisReducer(NDVI_DONE));
  });

  it('keeps only the latest step_progress of each row', () => {
    const merged = mergeEvents(PRITHVI_PREPARING.slice(0, 7), PRITHVI_PREPARING.slice(7));
    expect(merged.filter((e) => e.type === 'step_progress').map((e) => (e as JobEvent).seq)).toEqual([8]);
    expect(lastSeq(merged)).toBe(8);
  });

  it('lastSeq of nothing is 0, so the first stream replays the whole log', () => {
    expect(lastSeq([])).toBe(0);
  });
});
