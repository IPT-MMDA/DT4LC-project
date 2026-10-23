import { useState, useEffect, useRef } from 'react';
import { Send, Loader2, Paperclip, Info, PanelLeftOpen, PanelLeftClose, XCircle, ExternalLink, X, Plus, StopCircle } from 'lucide-react';
import {
  MAX_CONTEXT_TOKENS,
  findConversationByJobId,
  getContextForBackend,
  selectActiveMessages,
  selectContextTokens,
  useChatStore,
} from '../store/useChatStore';
import { useSubmitJob, useJob, useCancelJob } from '../api/hooks/useJobs';
import { Link, useSearchParams } from 'react-router-dom';
import { JobResultCard, parseJobResult } from '../components/chat/JobResultCard';
import { ChatHistory } from '../components/chat/ChatHistory';
import { FileUploadDropzone } from '../components/chat/FileUploadDropzone';
import { ImagePreviewModal } from '../components/chat/ImagePreviewModal';
import { useJobSync } from '../hooks/useJobSync';
import { resolveApiUrl } from '../api/client';
import { chatLogger as logger } from '../utils/logger';
import type { ChatMessage, MessageAttachment } from '../types';

export function ChatPage() {
  const [showHistory, setShowHistory] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  const [previewImage, setPreviewImage] = useState<{ src: string; filename: string } | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [searchParams, setSearchParams] = useSearchParams();

  // One chat store: conversations, messages and the composer
  const messages = useChatStore(selectActiveMessages);
  const contextTokenCount = useChatStore(selectContextTokens);
  const currentSessionId = useChatStore((state) => state.activeConversationId);
  const addMessage = useChatStore((state) => state.addMessage);
  const updateAnalysis = useChatStore((state) => state.updateAnalysis);
  const loadSession = useChatStore((state) => state.selectConversation);
  const createNewSession = useChatStore((state) => state.newConversation);
  const input = useChatStore((state) => state.draft);
  const setInput = useChatStore((state) => state.setDraft);
  const uploadedAttachments = useChatStore((state) => state.pendingAttachments);
  const clearAttachments = useChatStore((state) => state.clearAttachments);
  const removeAttachment = useChatStore((state) => state.removeAttachment);

  const submitJob = useSubmitJob();
  const cancelJob = useCancelJob();

  // Sync pending jobs on mount and periodically
  useJobSync();

  // Handle loading session from URL query param (e.g., from Jobs page)
  useEffect(() => {
    const sessionId = searchParams.get('session');
    const jobId = searchParams.get('job');

    if (sessionId && sessionId !== currentSessionId) {
      loadSession(sessionId);
      setSearchParams({});
    } else if (jobId) {
      // Find session containing this job
      const session = findConversationByJobId(useChatStore.getState().conversations, jobId);
      if (session && session.id !== currentSessionId) {
        loadSession(session.id);
      }
      setSearchParams({});
    }
  }, [searchParams]);

  // Track the most recent job for polling
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const { data: currentJob } = useJob(currentJobId || undefined, !!currentJobId);

  // Reset currentJobId when session changes (e.g., New Chat clicked)
  useEffect(() => {
    setCurrentJobId(null);
  }, [currentSessionId]);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, currentJob]);

  // Handle job completion - settle the job's analysis message
  useEffect(() => {
    if (currentJob && currentJob.status === 'completed') {
      updateAnalysis(currentJob.id, { status: 'done', result: parseJobResult(currentJob) });
      setCurrentJobId(null);
    } else if (currentJob && currentJob.status === 'failed') {
      updateAnalysis(currentJob.id, { status: 'failed', error: currentJob.error || 'Unknown error' });
      setCurrentJobId(null);
    } else if (currentJob && currentJob.status === 'cancelled') {
      updateAnalysis(currentJob.id, { status: 'cancelled' });
      setCurrentJobId(null);
    } else if (currentJob && currentJob.status === 'running') {
      updateAnalysis(currentJob.id, { status: 'running' });
    }
  }, [currentJob?.status]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || submitJob.isPending) return;

    // Include attachment previews AND paths in the message for display and context
    const messageAttachments: MessageAttachment[] | undefined = uploadedAttachments.length > 0
      ? uploadedAttachments.map(att => ({
          id: att.id,
          filename: att.filename,
          preview_url: att.preview_url,
          path: att.path,  // Include path for follow-up context
        }))
      : undefined;

    const { conversationId } = addMessage({
      kind: 'user',
      text: input,
      attachments: messageAttachments,
    });
    setInput('');

    try {
      // Build context for backend (includes previous attachments from this conversation)
      const conversation = useChatStore.getState().conversations.find((c) => c.id === conversationId);
      const backendContext = getContextForBackend(conversation?.messages ?? []);

      logger.debug('Submitting job with:', {
        prompt: input,
        attachmentsCount: uploadedAttachments.length,
        attachments: uploadedAttachments,
        context: backendContext,
      });

      const job = await submitJob.mutateAsync({
        prompt: input,
        mode: 'hybrid',
        attachments: uploadedAttachments.length > 0 ? uploadedAttachments : undefined,
        context: backendContext,  // Send context with previous attachments
      });

      // One analysis message narrates the job; polling settles it in place
      addMessage({ kind: 'analysis', jobId: job.id, status: 'pending', text: '', events: [] }, conversationId);
      setCurrentJobId(job.id);

      // Clear attachments after successful submission
      clearAttachments();
    } catch (error) {
      addMessage({ kind: 'assistant', text: `Error: ${(error as Error).message}`, error: true }, conversationId);
    }
  };

  const quickPrompts = [
    'Calculate NDVI for vegetation analysis',
    'Run Prithvi reconstruction',
    'Classify land cover',
    'Detect snow and ice coverage',
  ];

  // Render a message based on its type
  const renderMessage = (message: ChatMessage) => {
    const isUser = message.kind === 'user';
    const isError = message.kind === 'assistant' && message.error;

    // A done analysis shows its result card
    if (message.kind === 'analysis' && message.status === 'done') {
      const resultData = message.result;

      return (
        <div key={message.id} className="flex justify-start">
          <div className="max-w-[90%]">
            {resultData ? (
              <JobResultCard
                job={{ id: message.jobId, status: 'completed', progress: 1 }}
                resultData={resultData}
                compact={false}
              />
            ) : (
              <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 p-4">
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Job {message.jobId} completed.
                </p>
                <Link
                  to={`/jobs/${message.jobId}`}
                  className="mt-2 inline-flex items-center gap-1 text-sm text-primary-500 hover:text-primary-600"
                >
                  View job details
                </Link>
              </div>
            )}
          </div>
        </div>
      );
    }

    // A pending, running, failed or cancelled analysis
    if (message.kind === 'analysis') {
      const isActiveJob = message.jobId === currentJobId;
      const job = isActiveJob && currentJob ? currentJob : null;
      const isCancelled = message.status === 'cancelled';

      // If job failed or cancelled, show appropriate state
      if (message.status === 'failed' || isCancelled) {
        return (
          <div key={message.id} className="flex justify-start">
            <div className="max-w-[90%]">
              <div className={`rounded-lg border p-3 ${
                isCancelled
                  ? 'bg-orange-50 dark:bg-orange-950 border-orange-200 dark:border-orange-800'
                  : 'bg-red-50 dark:bg-red-950 border-red-200 dark:border-red-800'
              }`}>
                <div className="flex items-center gap-2">
                  {isCancelled ? (
                    <StopCircle className="w-4 h-4 text-orange-600 dark:text-orange-400" />
                  ) : (
                    <XCircle className="w-4 h-4 text-red-600 dark:text-red-400" />
                  )}
                  <span className={`text-sm ${
                    isCancelled
                      ? 'text-orange-700 dark:text-orange-300'
                      : 'text-red-700 dark:text-red-300'
                  }`}>
                    {isCancelled ? 'Job cancelled' : `Job failed${message.error ? `: ${message.error}` : ''}`}
                  </span>
                  <Link
                    to={`/jobs/${message.jobId}`}
                    className={`ml-auto flex items-center gap-1 text-xs hover:underline ${
                      isCancelled
                        ? 'text-orange-600 dark:text-orange-400'
                        : 'text-red-600 dark:text-red-400'
                    }`}
                  >
                    <ExternalLink className="w-3 h-3" />
                    Details
                  </Link>
                </div>
              </div>
            </div>
          </div>
        );
      }

      const handleCancelJob = () => {
        if (!cancelJob.isPending) {
          cancelJob.mutate(message.jobId);
        }
      };

      return (
        <div key={message.id} className="flex justify-start">
          <div className="max-w-[90%]">
            {job ? (
              <JobResultCard job={job} compact={true} />
            ) : (
              <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin text-primary-500" />
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      Processing your request...
                    </span>
                  </div>
                  <button
                    onClick={handleCancelJob}
                    disabled={cancelJob.isPending}
                    className="flex items-center gap-1 px-2 py-1 text-xs text-red-600 dark:text-red-400
                             bg-red-100 dark:bg-red-900 rounded hover:bg-red-200 dark:hover:bg-red-800
                             disabled:opacity-50 transition-colors"
                    title="Cancel job"
                  >
                    <StopCircle className="w-3 h-3" />
                    {cancelJob.isPending ? 'Cancelling...' : 'Cancel'}
                  </button>
                </div>
                <Link
                  to={`/jobs/${message.jobId}`}
                  className="mt-2 inline-flex items-center gap-1 text-xs text-primary-500 hover:text-primary-600"
                >
                  View in Jobs page
                </Link>
              </div>
            )}
          </div>
        </div>
      );
    }

    // Regular text messages
    return (
      <div
        key={message.id}
        className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}
      >
        <div
          className={`max-w-[80%] rounded-lg p-4 ${
            isUser
              ? 'bg-primary-500 text-white'
              : isError
              ? 'bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800'
              : 'bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800'
          }`}
        >
          {/* Attachment previews for user messages */}
          {message.kind === 'user' && message.attachments && message.attachments.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-3">
              {message.attachments.map((att) => (
                <button
                  key={att.id}
                  onClick={() => att.preview_url && setPreviewImage({
                    src: resolveApiUrl(att.preview_url),
                    filename: att.filename,
                  })}
                  className="group relative rounded-lg overflow-hidden border-2 border-white/30 hover:border-white/60 transition-colors cursor-pointer"
                  title={`Click to view: ${att.filename}`}
                >
                  {att.preview_url ? (
                    <img
                      src={resolveApiUrl(att.preview_url)}
                      alt={att.filename}
                      className="w-20 h-20 object-cover"
                    />
                  ) : (
                    <div className="w-20 h-20 bg-white/10 flex items-center justify-center">
                      <Paperclip className="w-6 h-6 text-white/60" />
                    </div>
                  )}
                  <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors flex items-center justify-center">
                    <span className="text-white text-xs opacity-0 group-hover:opacity-100 transition-opacity">
                      View
                    </span>
                  </div>
                </button>
              ))}
            </div>
          )}
          <p
            className={`text-sm ${
              isUser
                ? 'text-white'
                : isError
                ? 'text-red-700 dark:text-red-300'
                : 'text-gray-900 dark:text-white'
            }`}
          >
            {message.text}
          </p>
        </div>
      </div>
    );
  };

  return (
    <div className="h-full flex">
      {/* Chat History Sidebar */}
      {showHistory && (
        <div className="w-72 border-r border-gray-200 dark:border-gray-800 flex-shrink-0">
          <ChatHistory onClose={() => setShowHistory(false)} />
        </div>
      )}

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Chat Header */}
        <div className="border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <button
                onClick={() => setShowHistory(!showHistory)}
                className={`p-2 rounded-lg transition-colors ${
                  showHistory
                    ? 'bg-primary-100 dark:bg-primary-900 text-primary-600 dark:text-primary-400'
                    : 'hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-600 dark:text-gray-400'
                }`}
                title={showHistory ? 'Hide chat history' : 'Show chat history'}
              >
                {showHistory ? (
                  <PanelLeftClose className="w-5 h-5" />
                ) : (
                  <PanelLeftOpen className="w-5 h-5" />
                )}
              </button>
              <div>
                <h1 className="text-xl font-semibold text-gray-900 dark:text-white">
                  AI Chat
                </h1>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Ask questions about your geospatial data
                </p>
              </div>
            </div>
            {/* Actions */}
            <div className="flex items-center gap-3">
              {/* Context indicator */}
              <div className="flex items-center gap-2 text-xs text-gray-500">
                <Info className="w-3 h-3" />
                <span>
                  Context: {contextTokenCount} / {MAX_CONTEXT_TOKENS} tokens
                </span>
              </div>
              {/* New Chat button */}
              <button
                onClick={() => {
                  createNewSession();
                  clearAttachments();
                  setCurrentJobId(null);
                }}
                className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
                title="Start new chat"
              >
                <Plus className="w-4 h-4" />
                New Chat
              </button>
            </div>
          </div>
        </div>

      {/* Messages */}
      <div className="flex-1 overflow-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                Welcome to DT4LC Chat
              </h2>
              <p className="text-gray-600 dark:text-gray-400 max-w-md">
                I can help you analyze geospatial data, calculate NDVI, detect
                changes, extract field boundaries, and more.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-2xl">
              {quickPrompts.map((prompt) => (
                <button
                  key={prompt}
                  onClick={() => setInput(prompt)}
                  className="text-left p-4 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg hover:border-primary-500 dark:hover:border-primary-500 transition-colors"
                >
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    {prompt}
                  </p>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((message) => renderMessage(message))}
            <div ref={messagesEndRef} />
          </>
        )}

        {submitJob.isPending && (
          <div className="flex justify-start">
            <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg p-4">
              <Loader2 className="w-5 h-5 animate-spin text-gray-400" />
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="border-t border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-4">
        {/* Upload Dropdown */}
        {showUpload && (
          <div className="mb-3">
            <FileUploadDropzone
              onClose={() => setShowUpload(false)}
              compact={true}
            />
          </div>
        )}

        {/* Attached Files */}
        {uploadedAttachments.length > 0 && !showUpload && (
          <div className="mb-3 flex flex-wrap gap-2">
            {uploadedAttachments.map((attachment) => (
              <div
                key={attachment.id}
                className="flex items-center gap-2 px-3 py-1 bg-primary-100 dark:bg-primary-900 text-primary-700 dark:text-primary-300 rounded-full text-sm"
              >
                <Paperclip className="w-3 h-3" />
                <span className="truncate max-w-[200px]">{attachment.filename}</span>
                <button
                  onClick={() => removeAttachment(attachment.id)}
                  className="hover:text-primary-900 dark:hover:text-primary-100"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ))}
            <button
              onClick={() => setShowUpload(true)}
              className="flex items-center gap-1 px-3 py-1 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-full text-sm hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
            >
              <Paperclip className="w-3 h-3" />
              Add more
            </button>
          </div>
        )}

        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={
              uploadedAttachments.length > 0
                ? `Ask about your ${uploadedAttachments.length} file(s)...`
                : 'Type your message...'
            }
            className="flex-1 px-4 py-2 bg-gray-50 dark:bg-gray-950 border border-gray-200 dark:border-gray-800 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 text-gray-900 dark:text-white"
            disabled={submitJob.isPending}
          />
          <button
            type="button"
            onClick={() => setShowUpload(!showUpload)}
            className={`px-4 py-2 rounded-lg transition-colors flex items-center gap-2 ${
              showUpload || uploadedAttachments.length > 0
                ? 'bg-primary-100 dark:bg-primary-900 text-primary-600 dark:text-primary-400'
                : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
            }`}
            title="Attach file"
          >
            <Paperclip className="w-5 h-5" />
            {uploadedAttachments.length > 0 && (
              <span className="text-xs font-medium">{uploadedAttachments.length}</span>
            )}
          </button>
          <button
            type="submit"
            disabled={!input.trim() || submitJob.isPending}
            className="px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {submitJob.isPending ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </form>
      </div>
      </div>{/* End Main Chat Area */}

      {/* Image Preview Modal */}
      {previewImage && (
        <ImagePreviewModal
          imageSrc={previewImage.src}
          filename={previewImage.filename}
          onClose={() => setPreviewImage(null)}
        />
      )}
    </div>
  );
}
