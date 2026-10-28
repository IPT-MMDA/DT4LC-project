/** What `/` shows: the landing, or a redirect to the Workspace. */
export type LandingDecision = 'landing' | 'app';

/**
 * The landing is for every visitor at `/`: a guest, a returning guest, a member or an
 * operator, so the DT mark in the app leads there. Only a Workspace deep link
 * (`?sample=` or `?prompt=`) opens `/app`, with its query. The answer needs neither
 * the session nor the stored conversations, so it is given at once.
 */
export function landingDecision(search: string): LandingDecision {
  const params = new URLSearchParams(search);
  return params.has('sample') || params.has('prompt') ? 'app' : 'landing';
}
