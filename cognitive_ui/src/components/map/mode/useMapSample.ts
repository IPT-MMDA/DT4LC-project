import { useMemo } from 'react';
import { useListFiles } from '../../../api/hooks/useUpload';
import { carriedContext } from '../../../hooks/useSendMessage';
import { selectActiveMessages, useChatStore } from '../../../store/useChatStore';
import type { MessageAttachment } from '../../../types';
import { samplesFromFiles } from './layers';

/**
 * The sample pair for map mode: the conversation's (composer or last sent message),
 * else the library's latest (`GET /v1/files`, kind `sample`).
 */
export function useMapSample(): MessageAttachment[] {
  const messages = useChatStore(selectActiveMessages);
  const pending = useChatStore((s) => s.pendingAttachments);
  const fromChat = useMemo(
    () => [...pending, ...carriedContext(messages).attachments].filter((a) => a.group),
    [pending, messages],
  );
  // The Library's query (one shape under one key): it is invalidated when a file is deleted.
  const files = useListFiles({ enabled: fromChat.length === 0 });
  const fromLibrary = useMemo(() => samplesFromFiles(files.data), [files.data]);
  return fromChat.length > 0 ? fromChat : fromLibrary;
}
