// @vitest-environment node
import { describe, expect, it } from 'vitest';
import { demoReducer, initialDemoState, type DemoEvent, type DemoState } from './demoMachine';

const run = (state: DemoState, ...events: DemoEvent[]) => events.reduce(demoReducer, state);

describe('the How-it-works demo state machine', () => {
  it('starts paused at 0 s, or on the poster frame (step 3) with reduced motion', () => {
    expect(initialDemoState(false)).toEqual({ mode: 'idle', step: 0, next: null, reduced: false });
    expect(initialDemoState(true)).toEqual({ mode: 'poster', step: 3, next: null, reduced: true });
  });

  it('play resets for a frame, then plays from step 0 and lights the steps as it goes', () => {
    const reset = run(initialDemoState(false), { type: 'play' });
    expect(reset).toMatchObject({ mode: 'off', step: 0, next: 'play' });
    const playing = run(reset, { type: 'restarted' });
    expect(playing).toMatchObject({ mode: 'play', step: 0, next: null });
    expect(run(playing, { type: 'tick', step: 1 }, { type: 'tick', step: 2 }).step).toBe(2);
  });

  it.each([1, 2, 3] as const)('a click on step %i jumps the demo to that step and holds it', (step) => {
    const playing = run(initialDemoState(false), { type: 'play' }, { type: 'restarted' }, { type: 'tick', step: 1 });
    const scrubbed = run(playing, { type: 'scrub', step }, { type: 'restarted' });
    expect(scrubbed).toMatchObject({ mode: 'scrub', step, next: null });
    // A tick left over from the play does not move a scrubbed demo.
    expect(run(scrubbed, { type: 'tick', step: 3 }).step).toBe(step);
  });

  it('a scrub from the idle demo works without playing first', () => {
    expect(run(initialDemoState(false), { type: 'scrub', step: 2 }, { type: 'restarted' })).toMatchObject({ mode: 'scrub', step: 2 });
  });

  it('restarted outside the reset frame changes nothing', () => {
    const idle = initialDemoState(false);
    expect(run(idle, { type: 'restarted' })).toBe(idle);
  });

  describe('reduced motion', () => {
    it('never plays: neither on scroll-in nor on Play', () => {
      const poster = initialDemoState(true);
      expect(run(poster, { type: 'play' })).toBe(poster);
    });

    it('the step buttons still move it between still frames', () => {
      const scrubbed = run(initialDemoState(true), { type: 'scrub', step: 1 }, { type: 'restarted' });
      expect(scrubbed).toMatchObject({ mode: 'scrub', step: 1, reduced: true });
      expect(run(scrubbed, { type: 'play' })).toBe(scrubbed);
    });

    it('turned on mid-play stops on the poster; turned off holds the end state until the next play', () => {
      const playing = run(initialDemoState(false), { type: 'play' }, { type: 'restarted' });
      const poster = run(playing, { type: 'reducedMotion', reduced: true });
      expect(poster).toEqual(initialDemoState(true));
      const off = run(poster, { type: 'reducedMotion', reduced: false });
      expect(off).toEqual({ mode: 'off', step: 3, next: null, reduced: false });
      expect(run(off, { type: 'play' }, { type: 'restarted' }).mode).toBe('play');
    });

    it('the same setting again changes nothing', () => {
      const idle = initialDemoState(false);
      expect(run(idle, { type: 'reducedMotion', reduced: false })).toBe(idle);
    });
  });
});
