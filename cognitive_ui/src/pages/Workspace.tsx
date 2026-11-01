import { useState } from 'react';
import { PanelLeftOpen } from 'lucide-react';
import { ConversationList } from '../components/shell/ConversationList';
import { ChatColumn } from '../components/shell/ChatColumn';
import { DockedMap } from '../components/shell/DockedMap';
import { useConversationRetention } from '../hooks/useConversationRetention';
import { DOCKED_MAP_QUERY, useLayout, useMediaQuery } from '../hooks/useMediaQuery';
import { usePromptFromQuery } from '../hooks/usePromptFromQuery';
import { useSampleFromQuery } from '../hooks/useSample';

/**
 * /app: conversation list, chat column and, at ≥1440 px, the docked map. On a phone the
 * list is in the menu (Shell) and the chat spans the screen; on a tablet the list hides.
 */
export function Workspace() {
  const docked = useMediaQuery(DOCKED_MAP_QUERY);
  const layout = useLayout();
  const [listOpen, setListOpen] = useState(true);
  useSampleFromQuery();
  usePromptFromQuery();
  useConversationRetention();

  let list = null;
  if (layout === 'desktop') list = <ConversationList />;
  else if (layout === 'tablet' && listOpen) list = <ConversationList onCollapse={() => setListOpen(false)} />;
  else if (layout === 'tablet') {
    list = (
      <div className="w-11 flex-none bg-surface border-r border-border flex flex-col items-center pt-3">
        <button
          type="button"
          onClick={() => setListOpen(true)}
          aria-label="Show conversations"
          aria-expanded={false}
          title="Show conversations"
          className="w-8 h-8 flex items-center justify-center rounded-control text-muted hover:bg-surface2"
        >
          <PanelLeftOpen aria-hidden className="w-4 h-4" strokeWidth={1.8} />
        </button>
      </div>
    );
  }

  return (
    <>
      {list}
      <ChatColumn className={docked ? 'w-[560px] flex-none' : 'flex-1'} />
      {docked && <DockedMap />}
    </>
  );
}
