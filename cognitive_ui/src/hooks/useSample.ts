import { useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { FILES_QUERY_KEY } from '../api/hooks/useUpload';
import { queryClient } from '../api/queryClient';
import { postSample, sampleAttachments } from '../api/sample';
import { useChatStore } from '../store/useChatStore';
import type { Attachment } from '../types';
import { apiLogger as logger } from '../utils/logger';

/**
 * Attach the Kakhovka sample to the composer: POST /v1/sample, then both files go
 * into `pendingAttachments` as one group (one chip). Safe to call twice: the server
 * returns the same files and the store keys attachments by path. Also what the
 * `sample` fix kind of a failed analysis needs before it resends. The sample is now
 * among the caller's files, so the file list (From library, map mode) is fetched again.
 */
export async function attachSample(): Promise<Attachment[]> {
  const attachments = sampleAttachments(await postSample());
  void queryClient.invalidateQueries({ queryKey: FILES_QUERY_KEY });
  const { addAttachment } = useChatStore.getState();
  attachments.forEach(addAttachment);
  return attachments;
}

/**
 * `/app?sample=1` (the landing's "try the sample"): start a new conversation with
 * the sample in the composer, then drop the parameter so a reload does not repeat it.
 * `?sample=0` or no parameter leaves the composer alone.
 */
export function useSampleFromQuery() {
  const [params, setParams] = useSearchParams();
  const wantsSample = params.get('sample') === '1';

  useEffect(() => {
    if (!wantsSample) return;
    let cancelled = false;
    useChatStore.getState().newConversation();
    attachSample()
      .catch((err: unknown) => {
        // 404 SAMPLE_NOT_BUNDLED: the empty state simply shows no sample.
        if (!cancelled) logger.warn('Could not attach the sample:', err);
      })
      .finally(() => {
        if (cancelled) return;
        setParams(
          (current) => {
            const next = new URLSearchParams(current);
            next.delete('sample');
            return next;
          },
          { replace: true },
        );
      });
    return () => {
      cancelled = true;
    };
  }, [wantsSample, setParams]);
}
