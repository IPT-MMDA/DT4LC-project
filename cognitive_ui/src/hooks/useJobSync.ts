import { useEffect, useCallback } from 'react';
import { useChatStore } from '../store/useChatStore';
import apiClient from '../api/client';
import { parseJobResult } from '../components/chat/JobResultCard';
import { jobSyncLogger as logger } from '../utils/logger';
import type { Job } from '../types';

/**
 * Hook to sync pending jobs with their completion state.
 * This handles the case where a page reload happens while jobs are running.
 */
export function useJobSync() {
  const updateAnalysis = useChatStore((state) => state.updateAnalysis);

  // Sync a single job
  const syncJob = useCallback(async (jobId: string) => {
    try {
      const job = await apiClient.get<Job>(`/v1/jobs/${jobId}`);

      logger.debug(`Job ${jobId.slice(0, 8)} status: ${job.status}`);

      if (job.status === 'completed') {
        updateAnalysis(jobId, { status: 'done', result: parseJobResult(job) });
        logger.debug(`Job ${jobId.slice(0, 8)} completed, result added to chat`);
        return true;
      } else if (job.status === 'failed') {
        updateAnalysis(jobId, { status: 'failed', error: job.error || 'Unknown error' });
        logger.debug(`Job ${jobId.slice(0, 8)} failed: ${job.error}`);
        return true;
      } else if (job.status === 'cancelled') {
        updateAnalysis(jobId, { status: 'cancelled' });
        logger.debug(`Job ${jobId.slice(0, 8)} cancelled`);
        return true;
      }

      // Job still in progress
      return false;
    } catch (error: unknown) {
      // Handle 404 - job not found (server restarted or job expired)
      const isNotFound = error instanceof Error &&
        (error.message.includes('404') || error.message.includes('not found'));

      if (isNotFound) {
        logger.warn(`Job ${jobId.slice(0, 8)} not found on server (may have expired)`);
        updateAnalysis(jobId, {
          status: 'failed',
          error: 'Job not found. The server may have been restarted. Please try your request again.',
        });
        return true; // Consider it "handled"
      }

      logger.error(`Error syncing job ${jobId}:`, error);
      return false;
    }
  }, [updateAnalysis]);

  // On mount, check every unfinished analysis in every kept conversation
  useEffect(() => {
    const checkPendingJobs = async () => {
      const pending = useChatStore.getState().conversations.flatMap((c) =>
        c.messages.filter(
          (m) => m.kind === 'analysis' && (m.status === 'pending' || m.status === 'running'),
        ),
      );
      for (const message of pending) {
        if (message.kind !== 'analysis') continue;
        logger.debug(`Checking pending job: ${message.jobId.slice(0, 8)}`);
        await syncJob(message.jobId);
      }
    };

    checkPendingJobs();
  }, []); // Only run on mount

  // Active jobs are polled by ChatPage via useJob hook.
  // This hook only syncs on mount for jobs that completed while the page was closed.

  return { syncJob };
}
