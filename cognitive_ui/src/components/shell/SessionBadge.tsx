import { useHealth } from '../../api/hooks/useHealth';
import { sessionBadge, useSession } from '../../api/hooks/useSession';

/** "Guest. 4 of 5 analyses left this hour · kept 24 h · no account." from GET /v1/session. */
export function SessionBadge() {
  const { data: session } = useSession();
  const { data: health } = useHealth();
  if (!session) return null;
  const { label, line } = sessionBadge(session, health?.assistant?.offline ?? false);

  return (
    <div
      data-testid="session-badge"
      className="m-[10px] px-3 py-[10px] rounded-control bg-bg border border-border text-[11.5px] leading-[1.45] text-muted"
    >
      <span className="font-semibold text-text2">{label}</span> {line}
    </div>
  );
}
