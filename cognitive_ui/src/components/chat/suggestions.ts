// The empty state's suggested prompts (design/screens/workspace-empty-*.png and the
// prototype's askWater/askFields/askClassify/askBurn/askPrithvi). Every prompt must be
// in tests/test_ui_prompt_routing.py's table; suggestions.test.ts checks it.

export interface Suggestion {
  prompt: string;
  time: string;
  badge?: string;
}

export const SUGGESTIONS: Suggestion[] = [
  { prompt: 'Compare water before and after the dam breach', time: '~50 s' },
  { prompt: 'Detect field boundaries east of the river', time: '~2 min' },
  { prompt: 'Classify land cover on the flooded fields', time: '~90 s' },
];

/** With the sample attached: also the composed NDMI example (never NBR: the sample has no B7) and Prithvi. */
export const SAMPLE_SUGGESTIONS: Suggestion[] = [
  ...SUGGESTIONS,
  { prompt: 'Map moisture change with NDMI between the two dates', time: '~60 s', badge: 'Composed' },
  { prompt: 'Fill the cloud gaps in the 9 June image', time: 'Prithvi · ~3 min' },
];
