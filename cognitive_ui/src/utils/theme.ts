/** Sets the `data-theme` attribute the design tokens and Tailwind `dark:` key on. */
export function applyTheme(
  theme: 'light' | 'dark' | 'auto',
  root: HTMLElement = document.documentElement
) {
  root.dataset.theme = theme === 'dark' ? 'dark' : 'light';
}
