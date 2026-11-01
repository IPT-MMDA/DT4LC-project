// Times and sizes in the analysis timeline: "9 s", "1.2 min", "<1 s"; "52 MB".

/** A step's elapsed time, or a time left. */
export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '';
  if (seconds < 1) return '<1 s';
  if (seconds < 60) return `${Math.round(seconds)} s`;
  return `${(seconds / 60).toFixed(1)} min`;
}

const MB = 1024 * 1024;

export const formatMB = (bytes: number) => `${Math.round(bytes / MB)} MB`;

/**
 * The line under a downloading Preparing model row:
 * "Downloading Prithvi EO v1 (100M) · 450 MB · 52 MB done · 56 s left". The ETA is
 * left out until bytes arrive, and when it is absurd (the first poll can say 15 years).
 */
export function downloadLine(p: { message?: string; bytes?: number; totalBytes?: number; etaS?: number }): string {
  const parts: string[] = [];
  if (p.message) parts.push(p.message);
  else if (p.totalBytes) parts.push(formatMB(p.totalBytes));
  if (p.bytes !== undefined && p.bytes > 0) parts.push(`${formatMB(p.bytes)} done`);
  if (p.bytes && p.etaS !== undefined && p.etaS >= 0 && p.etaS < 24 * 3600) parts.push(`${formatDuration(p.etaS)} left`);
  return parts.join(' · ');
}
