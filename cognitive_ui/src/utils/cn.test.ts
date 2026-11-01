import { describe, expect, it } from 'vitest';
import { cn } from './cn';

describe('cn', () => {
  it('keeps a type-scale size next to a text colour', () => {
    expect(cn('text-control text-text2')).toBe('text-control text-text2');
    expect(cn('font-mono text-stat', 'text-gain-ink')).toBe('font-mono text-stat text-gain-ink');
  });

  it('keeps every type-scale key from tailwind.config.js', () => {
    for (const size of ['title', 'card', 'body', 'control', 'label', 'stat', 'hero', 'hero-sub', 'hero-phone']) {
      expect(cn(`text-${size}`, 'text-muted')).toBe(`text-${size} text-muted`);
    }
  });

  it('lets a later size win over an earlier one', () => {
    expect(cn('text-body', 'text-control')).toBe('text-control');
    expect(cn('text-body', 'text-[11px]')).toBe('text-[11px]');
  });
});
