import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import apiClient from '../../../api/client';
import { carriedContext } from '../../../hooks/useSendMessage';
import { selectActiveMessages, useChatStore } from '../../../store/useChatStore';
import type { MessageAttachment } from '../../../types';
import { samplesFromFiles, type LibraryFile } from './layers';

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
  const files = useQuery({
    queryKey: ['files'],
    queryFn: () => apiClient.get<{ ok: boolean; files: LibraryFile[] }>('/v1/files'),
    enabled: fromChat.length === 0,
    staleTime: 60_000,
  });
  const fromLibrary = useMemo(() => samplesFromFiles(files.data?.files), [files.data]);
  return fromChat.length > 0 ? fromChat : fromLibrary;
}
