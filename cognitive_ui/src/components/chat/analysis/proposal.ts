// The proposal's Change and Explain buttons.

/** What Explain sends: a conversation reply, even with files attached (routing fix). */
export const EXPLAIN_PROPOSAL = 'Explain this proposal';

/** "Change": the prompt with the proposed formula, for the user to edit and send. */
export function changeDraft(prompt: string, plan: unknown): string {
  const steps = ((plan as { steps?: { config?: Record<string, unknown> }[] } | null)?.steps ?? []);
  const config = steps.map((s) => s.config ?? {}).find((c) => typeof c.formula === 'string');
  if (!config) return prompt;
  const name = typeof config.name === 'string' && config.name !== config.formula ? `${config.name} = ` : '';
  const threshold = typeof config.threshold === 'number' ? ` and a change threshold of ${config.threshold}` : '';
  return `${prompt}, using ${name}${config.formula as string}${threshold}`;
}
