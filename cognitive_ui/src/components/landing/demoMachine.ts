// The "How it works" demo (design/prototype/Landing v2.dc.html): one 8-s CSS timeline
// that plays once when the demo scrolls into view. The three step buttons scrub it to
// a fixed frame; reduced motion shows a poster frame and only the buttons move it.
//
// Modes, as `data-mode` on the demo (landing.css):
//   idle   the timeline paused at 0 s, before the demo is seen
//   off    no animation: every element at its end state. Also the one-frame reset
//          that restarts the CSS animations before `play` or `scrub`
//   play   the timeline running, from 0 s
//   scrub  the timeline paused at SCRUB_AT[step]
//   poster reduced motion: no animation, the end state (step 3)

export type DemoStep = 0 | 1 | 2 | 3;
export type DemoMode = 'idle' | 'off' | 'play' | 'scrub' | 'poster';

export interface DemoState {
  mode: DemoMode;
  /** The highlighted step button; 0 for none. */
  step: DemoStep;
  /** The mode to enter after the `off` reset frame. */
  next: 'play' | 'scrub' | null;
  reduced: boolean;
}

export type DemoEvent =
  | { type: 'play' }
  | { type: 'scrub'; step: Exclude<DemoStep, 0> }
  | { type: 'restarted' }
  | { type: 'tick'; step: Exclude<DemoStep, 0> }
  | { type: 'reducedMotion'; reduced: boolean };

/** The timeline's length; landing.css runs every demo animation over it. */
export const DEMO_DURATION_S = 8;

/** Where a step button parks the timeline (a negative animation-delay). */
export const SCRUB_AT: Record<Exclude<DemoStep, 0>, string> = { 1: '-2.2s', 2: '-4.3s', 3: '-7.6s' };

/** While playing, the step that lights up at each time (ms from the start). */
export const PLAY_TICKS: ReadonlyArray<readonly [number, Exclude<DemoStep, 0>]> = [
  [300, 1],
  [2700, 2],
  [4600, 3],
];

export function initialDemoState(reduced: boolean): DemoState {
  return reduced ? { mode: 'poster', step: 3, next: null, reduced } : { mode: 'idle', step: 0, next: null, reduced };
}

export function demoReducer(state: DemoState, event: DemoEvent): DemoState {
  switch (event.type) {
    case 'play':
      // Reduced motion never plays the timeline.
      return state.reduced ? state : { ...state, mode: 'off', step: 0, next: 'play' };
    case 'scrub':
      return { ...state, mode: 'off', step: event.step, next: 'scrub' };
    case 'restarted':
      return state.mode === 'off' && state.next ? { ...state, mode: state.next, next: null } : state;
    case 'tick':
      return state.mode === 'play' ? { ...state, step: event.step } : state;
    case 'reducedMotion':
      if (event.reduced === state.reduced) return state;
      // Turned on mid-play: stop on the poster. Turned off: stay on the end state
      // until the next play (scrolling into view for the first time, or Play).
      return event.reduced ? initialDemoState(true) : { mode: 'off', step: 3, next: null, reduced: false };
  }
}
