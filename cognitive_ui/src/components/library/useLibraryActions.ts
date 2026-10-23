import { useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { ChatError } from '../../api/chat';
import { FILES_QUERY_KEY, type GeoTIFFFile } from '../../api/hooks/useUpload';
import { getJob } from '../../api/jobs';
import { deleteFile, deleteJob, JOBS_QUERY_KEY } from '../../api/library';
import { attachSample } from '../../hooks/useSample';
import { useSendMessage } from '../../hooks/useSendMessage';
import { useAppStore } from '../../store/useAppStore';
import { useChatStore } from '../../store/useChatStore';
import type { Attachment } from '../../types';
import { resultCardOf, resultLayerOf } from '../chat/result/resultModel';
import { analysisContext, type AnalysisRow, type FileRow, type LibraryRow } from './libraryModel';

const errorText = (err: unknown) => (err instanceof Error ? err.message : String(err));

/** A Library file as a composer attachment, metadata kept for chat and jobs. */
export function fileAttachment(f: GeoTIFFFile): Attachment {
  return {
    id: f.upload_id ?? f.id,
    filename: f.filename,
    path: f.path,
    mime_type: 'image/tiff',
    size_bytes: f.size_bytes,
    preview_url: f.preview_url ?? undefined,
    metadata: f.metadata ?? undefined,
  };
}

/**
 * What a Library row does: open it (Result detail, or the Workspace with the file),
 * its one action (Map, Chat, Retry; ↓ and CSV are links) and Delete. Each returns an
 * error sentence for the page to show, or null.
 */
export function useLibraryActions() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { send } = useSendMessage();

  /** Chat: a new conversation with the file in the composer (the sample as its pair). */
  const openInChat = useCallback(
    async (row: FileRow): Promise<string | null> => {
      useChatStore.getState().newConversation();
      try {
        if (row.fileKind === 'sample') await attachSample();
        else row.files.forEach((f) => useChatStore.getState().addAttachment(fileAttachment(f)));
      } catch (err) {
        return `Couldn't attach ${row.title}: ${errorText(err)}`;
      }
      navigate('/app');
      return null;
    },
    [navigate],
  );

  /** Map: the result's first raster as a layer, then map mode; Result detail when it has none. */
  const openOnMap = useCallback(
    async (row: AnalysisRow): Promise<string | null> => {
      const { id } = row.job;
      try {
        const job = await queryClient.fetchQuery({ queryKey: ['job', id], queryFn: () => getJob(id), staleTime: Infinity });
        const { context } = analysisContext(useChatStore.getState().conversations, id);
        const model = resultCardOf(job, context, row.title);
        const view = model?.views.find((v) => v.raster);
        if (!model || !view?.raster) {
          navigate(`/app/results/${encodeURIComponent(id)}`);
          return null;
        }
        useAppStore.getState().addResultLayer(resultLayerOf(id, model, { ...view, raster: view.raster }));
        navigate('/app/map');
        return null;
      } catch (err) {
        return `Couldn't load ${row.title}: ${errorText(err)}`;
      }
    },
    [navigate, queryClient],
  );

  /** Retry: the prompt again in the analysis's conversation (a new one if this browser has none). */
  const retry = useCallback(
    (row: AnalysisRow): string | null => {
      const { conversations, selectConversation, newConversation } = useChatStore.getState();
      const { conversation, prompt } = analysisContext(conversations, row.job.id);
      const text = row.job.prompt?.trim() || prompt;
      if (!text) return `Nothing to retry: ${row.title} has no prompt.`;
      if (conversation) selectConversation(conversation.id);
      else newConversation();
      navigate('/app');
      void send(text);
      return null;
    },
    [navigate, send],
  );

  /** A row click: Result detail for an analysis, the Workspace with the file for a file. */
  const open = useCallback(
    async (row: LibraryRow): Promise<string | null> => {
      if (row.kind === 'file') return openInChat(row);
      navigate(`/app/results/${encodeURIComponent(row.job.id)}`);
      return null;
    },
    [navigate, openInChat],
  );

  /** Delete (after the confirm); refreshes the jobs and files lists. 409: a running job. */
  const remove = useCallback(
    async (row: LibraryRow): Promise<string | null> => {
      try {
        if (row.kind === 'analysis') await deleteJob(row.job.id);
        else if (row.deleteId) await deleteFile(row.deleteId);
        else return `${row.title} can't be deleted here.`;
      } catch (err) {
        if (err instanceof ChatError && err.status === 409) return `${row.title} is still running. Cancel it first.`;
        // 404: gone already (expired, or deleted in another tab); refresh the list anyway.
        if (!(err instanceof ChatError && err.status === 404)) return `Couldn't delete ${row.title}: ${errorText(err)}`;
      }
      if (row.kind === 'analysis') queryClient.removeQueries({ queryKey: ['job', row.job.id] });
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: JOBS_QUERY_KEY }),
        queryClient.invalidateQueries({ queryKey: FILES_QUERY_KEY }),
      ]);
      return null;
    },
    [queryClient],
  );

  return { open, openInChat, openOnMap, retry, remove };
}
