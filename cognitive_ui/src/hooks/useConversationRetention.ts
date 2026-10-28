import { useEffect } from 'react';
import { useSession } from '../api/hooks/useSession';
import { GUEST_KEPT_HOURS, useChatStore } from '../store/useChatStore';

/**
 * Keep this browser's conversations as the session says: a guest's for
 * `kept_hours`, a member's or operator's without a time limit. Runs on every
 * session fetch, so a long-open tab drops what has expired too.
 */
export function useConversationRetention(): void {
  const { data: session } = useSession();
  const setKeptHours = useChatStore((s) => s.setKeptHours);
  useEffect(() => {
    if (!session) return;
    setKeptHours(session.guest ? (session.kept_hours ?? GUEST_KEPT_HOURS) : null);
  }, [session, setKeptHours]);
}
