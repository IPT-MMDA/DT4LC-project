import { useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import type { JobRecord } from '../../api/jobs';
import { attachSample } from '../../hooks/useSample';
import { useSendMessage } from '../../hooks/useSendMessage';
import { useAppStore } from '../../store/useAppStore';
import { useChatStore } from '../../store/useChatStore';
import type { Attachment, Conversation } from '../../types';
import type { Fix } from '../chat/analysis/analysisReducer';
import { changeDraft } from '../chat/analysis/proposal';
import { resultCardOf, resultLayerOf, type AnalysedContext } from '../chat/result/resultModel';
import { sampleLayerOf } from '../map/mode/layers';

const errorText = (err: unknown) => (err instanceof Error ? err.message : String(err));

export interface ResultSource {
  job: JobRecord;
  /** The conversation in this browser that holds the job, or null. */
  conversation: Conversation | null;
  /** What was asked: the user message, else the job's prompt. */
  prompt: string;
  /** The files or region it was about. */
  context: AnalysedContext;
}

/** The context's files and region into the composer. */
function attachContext(context: AnalysedContext) {
  const { addAttachment, setPendingRegion } = useChatStore.getState();
  context.attachments.filter((a): a is Attachment => Boolean(a.path)).forEach(addAttachment);
  if (context.attachments.length === 0 && context.region) setPendingRegion(context.region);
}

/**
 * Result detail's actions: Open in workspace,
 * Re-run with changes, Open on map, and a failure's fixes as the Workspace runs them
 * (AnalysisMessage's onFix). Each returns an error sentence for the page, or null.
 * `openFilePicker` is the page's file input, for the `upload` fix.
 */
export function useResultActions({ job, conversation, prompt, context }: ResultSource, openFilePicker: () => void) {
  const navigate = useNavigate();
  const { send } = useSendMessage();

  /**
   * The job's conversation with an empty composer, or a new one. `attach`: the files
   * (or region) in the composer: always, or only in a new conversation (an existing
   * one carries them from its messages).
   */
  const toConversation = useCallback(
    (attach: 'always' | 'if-new' | 'never') => {
      const store = useChatStore.getState();
      if (conversation) store.selectConversation(conversation.id);
      else store.newConversation();
      store.clearAttachments();
      store.setPendingRegion(null);
      if (attach === 'always' || (attach === 'if-new' && !conversation)) attachContext(context);
    },
    [conversation, context],
  );

  /** The conversation holding the job; a new one with the files when this browser has none. */
  const openInWorkspace = useCallback((): string | null => {
    if (conversation) useChatStore.getState().selectConversation(conversation.id);
    else toConversation('always');
    navigate('/app');
    return null;
  }, [conversation, navigate, toConversation]);

  /** The prompt (with a composed plan's formula and threshold) in the composer with the files, not sent. */
  const rerun = useCallback((): string | null => {
    toConversation('always');
    useChatStore.getState().setDraft(changeDraft(prompt, job.plan ?? job.proposal?.plan));
    navigate('/app');
    return null;
  }, [job, prompt, navigate, toConversation]);

  /**
   * Every raster of the result as a layer (the difference on top and visible, before
   * and after hidden under it), the sample in true colour under them, then map mode.
   */
  const openOnMap = useCallback((): string | null => {
    const model = resultCardOf(job, context);
    const views = (model?.views ?? []).filter((v) => v.raster);
    if (!model || views.length === 0) return 'This result has no map layer.';
    const shown = (key: string) => key === 'difference' || key === 'map';
    const { addResultLayer, addBaseLayer } = useAppStore.getState();
    for (const view of [...views.filter((v) => !shown(v.key)), ...views.filter((v) => shown(v.key))]) {
      if (!view.raster) continue;
      addResultLayer({ ...resultLayerOf(job.id, model, { ...view, raster: view.raster }), visible: shown(view.key) });
    }
    const sample = sampleLayerOf(context.attachments);
    if (sample) addBaseLayer(sample);
    navigate('/app/map');
    return null;
  }, [job, context, navigate]);

  /** A failure's fix, as the Workspace's analysis message runs it, in the job's conversation. */
  const fix = useCallback(
    async (f: Fix): Promise<string | null> => {
      switch (f.kind) {
        case 'prompt':
          if (!f.prompt) return null;
          toConversation('if-new');
          navigate('/app');
          void send(f.prompt);
          return null;
        case 'retry':
          toConversation('if-new');
          navigate('/app');
          void send(prompt, f.params ? { params: f.params } : undefined);
          return null;
        case 'sample':
          toConversation('never');
          try {
            await attachSample();
          } catch (err) {
            return `Couldn't attach the sample: ${errorText(err)}`;
          }
          navigate('/app');
          void send(prompt);
          return null;
        case 'upload':
          openFilePicker();
          return null;
        case 'open_map':
          navigate('/app/map');
          return null;
      }
      return null;
    },
    [navigate, openFilePicker, prompt, send, toConversation],
  );

  /** After the `upload` fix's picker: the job's conversation with the new files and the prompt in the composer. */
  const withUploads = useCallback(
    async (attachFiles: () => Promise<void>): Promise<string | null> => {
      toConversation('never');
      await attachFiles();
      useChatStore.getState().setDraft(prompt);
      navigate('/app');
      return null;
    },
    [navigate, prompt, toConversation],
  );

  return { openInWorkspace, rerun, openOnMap, fix, withUploads };
}
