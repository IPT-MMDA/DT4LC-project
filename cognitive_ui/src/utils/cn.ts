import { clsx, type ClassValue } from 'clsx';
import { extendTailwindMerge } from 'tailwind-merge';

// The type scale in tailwind.config.js `fontSize`. Without it, tailwind-merge reads
// `text-control` as a colour and drops it next to `text-text2`.
const twMerge = extendTailwindMerge({
  extend: {
    classGroups: {
      'font-size': [{ text: ['title', 'card', 'body', 'control', 'label', 'stat', 'hero', 'hero-sub', 'hero-phone'] }],
    },
  },
});

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
