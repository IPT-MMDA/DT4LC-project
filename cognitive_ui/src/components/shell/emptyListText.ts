/** The empty list's line: a guest's conversations stay for `keptHours`, a member's until deleted. */
export function emptyListText(keptHours: number | null): string {
  const kept = keptHours === null ? '' : ` and stay for ${keptHours} h`;
  return `No conversations yet. They appear here as you work${kept}.`;
}
