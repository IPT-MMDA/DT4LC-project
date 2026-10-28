import { useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useChatStore } from '../store/useChatStore';

/** The Workspace link that opens with `prompt` in the composer (an example prompt on Capabilities). */
export function promptHref(prompt: string): string {
  return `/app?${new URLSearchParams({ prompt })}`;
}

/**
 * `/app?prompt=<text>`: put the text in the composer without sending it, then drop
 * the parameter so a reload does not repeat it. The active conversation and the
 * files already in the composer stay as they are; nothing is attached.
 */
export function usePromptFromQuery() {
  const [params, setParams] = useSearchParams();
  const prompt = params.get('prompt');

  useEffect(() => {
    if (prompt === null) return;
    useChatStore.getState().setDraft(prompt);
    setParams(
      (current) => {
        const next = new URLSearchParams(current);
        next.delete('prompt');
        return next;
      },
      { replace: true },
    );
  }, [prompt, setParams]);
}
