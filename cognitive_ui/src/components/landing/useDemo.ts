import { useEffect, useReducer, useRef } from 'react';
import { useReducedMotion } from '../../hooks/useReducedMotion';
import { PLAY_TICKS, demoReducer, initialDemoState, type DemoStep } from './demoMachine';

/**
 * Drives the How-it-works demo: plays it once when a third of it is in view, lights
 * the steps while it plays, and lets the step buttons scrub it. With reduced motion
 * it never plays; the step buttons still move it between still frames.
 */
export function useDemo() {
  const reduced = useReducedMotion();
  const [state, dispatch] = useReducer(demoReducer, reduced, initialDemoState);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    dispatch({ type: 'reducedMotion', reduced });
  }, [reduced]);

  // Self-running: once, the first time the demo is seen.
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === 'undefined') return;
    const io = new IntersectionObserver(
      (entries) => {
        if (!entries.some((e) => e.isIntersecting)) return;
        io.disconnect();
        dispatch({ type: 'play' });
      },
      { threshold: 0.35 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  // `off` for two frames restarts the CSS animations, so play starts at 0 s and a
  // scrub lands on its own frame.
  const { mode, next } = state;
  useEffect(() => {
    if (mode !== 'off' || !next) return;
    let second = 0;
    const first = requestAnimationFrame(() => {
      second = requestAnimationFrame(() => dispatch({ type: 'restarted' }));
    });
    return () => {
      cancelAnimationFrame(first);
      cancelAnimationFrame(second);
    };
  }, [mode, next]);

  useEffect(() => {
    if (mode !== 'play') return;
    const timers = PLAY_TICKS.map(([ms, step]) => setTimeout(() => dispatch({ type: 'tick', step }), ms));
    return () => timers.forEach(clearTimeout);
  }, [mode]);

  return {
    ref,
    state,
    play: () => dispatch({ type: 'play' }),
    scrub: (step: Exclude<DemoStep, 0>) => dispatch({ type: 'scrub', step }),
  };
}
