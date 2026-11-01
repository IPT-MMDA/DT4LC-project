import { cleanup, render } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { Markdown } from './Markdown';

afterEach(cleanup);

describe('Markdown', () => {
  it('renders lists, bold, inline code and GFM tables', () => {
    const { container } = render(
      <Markdown
        text={[
          'I can **map water** with `NDWI`:',
          '',
          '- Compare two dates',
          '- Classify land cover',
          '',
          '| index | bands |',
          '| --- | --- |',
          '| NDMI | B5, B6 |',
        ].join('\n')}
      />,
    );
    expect(container.querySelector('strong')?.textContent).toBe('map water');
    expect(container.querySelector('code')?.textContent).toBe('NDWI');
    expect([...container.querySelectorAll('ul > li')].map((li) => li.textContent)).toEqual([
      'Compare two dates',
      'Classify land cover',
    ]);
    expect([...container.querySelectorAll('td')].map((td) => td.textContent)).toEqual(['NDMI', 'B5, B6']);
  });

  it('escapes HTML in a reply instead of rendering it', () => {
    const { container } = render(
      <Markdown text={'Hello <b>bold</b> <img src=x onerror="alert(1)"> [x](javascript:alert(1))\n\n<script>alert(1)</script>'} />,
    );
    expect(container.querySelector('b')).toBeNull();
    expect(container.querySelector('img')).toBeNull();
    expect(container.querySelector('script')).toBeNull();
    expect(container.querySelector('[onerror]')).toBeNull();
    expect(container.textContent).toContain('Hello <b>bold</b>');
    expect(container.textContent).toContain('<script>alert(1)</script>');
    expect(container.querySelector('a')?.getAttribute('href') ?? '').not.toContain('javascript');
  });
});
