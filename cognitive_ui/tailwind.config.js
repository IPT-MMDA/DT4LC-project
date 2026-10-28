/** @type {import('tailwindcss').Config} */

// Design tokens live in src/index.css. color-mix keeps
// opacity modifiers such as `bg-accent/20` working on top of the CSS variables.
const token = (name) =>
  `color-mix(in oklch, var(--${name}) calc(<alpha-value> * 100%), transparent)`;

module.exports = {
  darkMode: ['selector', '[data-theme="dark"]'],
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: token('bg'),
        surface: token('surface'),
        surface2: token('surface2'),
        bubble: token('bubble'),
        border: token('border'),
        border2: token('border2'),
        text: token('text'),
        text2: token('text2'),
        muted: token('muted'),
        faint: token('faint'),
        accent: token('accent'),
        'on-accent': token('on-accent'),
        'accent-tint': token('accent-tint'),
        'accent-ink': token('accent-ink'),
        done: token('done'),
        'done-tint': token('done-tint'),
        'done-ink': token('done-ink'),
        run: token('run'),
        'run-tint': token('run-tint'),
        'run-ink': token('run-ink'),
        fail: token('fail'),
        'fail-tint': token('fail-tint'),
        'fail-ink': token('fail-ink'),
        canc: token('canc'),
        'canc-tint': token('canc-tint'),
        'canc-ink': token('canc-ink'),
        region: token('region'),
        lost: token('lost'),
        gained: token('gained'),
        ink: token('ink'),
        'ink-bg': token('ink-bg'),
        // Legacy palette, unused.
        geo: {
          vegetation: '#22c55e',
          water: '#0ea5e9',
          urban: '#ef4444',
          terrain: '#a16207',
        },
        // Map legend (same in both themes).
        legend: {
          gained: '#2d5f9e',
          lost: '#8c4a2f',
          'no-change': '#c4cad6',
          land: '#f1efe8',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'monospace'],
      },
      fontSize: {
        title: ['20px', { lineHeight: '1.3', fontWeight: '600' }],
        card: ['14px', { lineHeight: '1.4', fontWeight: '600' }],
        body: ['14px', { lineHeight: '1.55', fontWeight: '400' }],
        control: ['12.5px', { lineHeight: '1.4' }],
        // Pair with `uppercase`.
        label: ['10.5px', { lineHeight: '1.4', letterSpacing: '.06em', fontWeight: '600' }],
        // Pair with `font-mono`.
        stat: ['16px', { lineHeight: '1.3', fontWeight: '500' }],
        // Landing hero: 84/1/-0.03em and 20/1.4, smaller on short
        // viewports as in the prototype.
        hero: ['min(84px, 9.3vh)', { lineHeight: '1', letterSpacing: '-0.03em', fontWeight: '600' }],
        'hero-sub': ['min(20px, 2.3vh)', { lineHeight: '1.4' }],
        // The hero on a phone: four lines at 390 px instead of seven.
        'hero-phone': ['min(44px, 11vw)', { lineHeight: '1.05', letterSpacing: '-0.03em', fontWeight: '600' }],
      },
      spacing: {
        1: '4px',
        2: '8px',
        3: '12px',
        4: '16px',
        6: '24px',
        8: '32px',
      },
      borderRadius: {
        chip: '4px',
        control: '6px',
        card: '10px',
        composer: '12px',
        pill: '999px',
      },
      boxShadow: {
        composer: '0 1px 2px rgba(0, 0, 0, .04)',
      },
    },
  },
  plugins: [],
};
