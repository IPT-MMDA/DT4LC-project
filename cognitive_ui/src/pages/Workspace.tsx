import { ConversationList } from '../components/shell/ConversationList';
import { ChatColumn } from '../components/shell/ChatColumn';
import { DockedMap } from '../components/shell/DockedMap';
import { DOCKED_MAP_QUERY, useMediaQuery } from '../hooks/useMediaQuery';
import { useSampleFromQuery } from '../hooks/useSample';

/** /app: conversation list, chat column and, at ≥1440 px, the docked map. */
export function Workspace() {
  const docked = useMediaQuery(DOCKED_MAP_QUERY);
  useSampleFromQuery();

  return (
    <>
      <ConversationList />
      <ChatColumn className={docked ? 'w-[560px] flex-none' : 'flex-1'} />
      {docked && <DockedMap />}
    </>
  );
}
