// @vitest-environment node
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { SAMPLE_SUGGESTIONS, SUGGESTIONS } from './suggestions';

// Every chip prompt the Workspace sends is routed by the backend's table
// (CONTRACTS "Routing fixes"): a new suggestion must be added there too.
const ROUTING_TEST = fileURLToPath(new URL('../../../../tests/test_ui_prompt_routing.py', import.meta.url));

describe('empty-state suggestions', () => {
  it('are the screenshots\' prompts: three without the sample, five with it', () => {
    expect(SUGGESTIONS.map((s) => `${s.prompt} ${s.time}`)).toEqual([
      'Compare water before and after the dam breach ~50 s',
      'Detect field boundaries east of the river ~2 min',
      'Classify land cover on the flooded fields ~90 s',
    ]);
    expect(SAMPLE_SUGGESTIONS.slice(3).map((s) => [s.prompt, s.badge, s.time])).toEqual([
      ['Map moisture change with NDMI between the two dates', 'Composed', '~60 s'],
      ['Fill the cloud gaps in the 9 June image', undefined, 'Prithvi · ~3 min'],
    ]);
  });

  it('are all in tests/test_ui_prompt_routing.py EMPTY_STATE', () => {
    const source = readFileSync(ROUTING_TEST, 'utf8');
    const block = source.slice(source.indexOf('EMPTY_STATE = ('), source.indexOf(')\n', source.indexOf('EMPTY_STATE = (')));
    for (const { prompt } of SAMPLE_SUGGESTIONS) expect(block).toContain(`"${prompt}"`);
  });
});
