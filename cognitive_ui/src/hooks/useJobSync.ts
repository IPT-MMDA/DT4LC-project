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
  const {
    messages,
    addJobResultMessage,
    updateMessageByJobId,
    untrackJob,
  } = useChatStore();

  // Find all job_submitted messages that might need syncing
  const pendingJobMessages = messages.filter(
    (m) => m.type === 'job_submitted' && m.jobId
  );

  // Sync a single job
  const syncJob = useCallback(async (jobId: string) => {
    try {
      const job = await apiClient.get<Job>(`/v1/jobs/${jobId}`);

      logger.debug(`Job ${jobId.slice(0, 8)} status: ${job.status}`);

      if (job.status === 'completed') {
        // Job completed - add result message
        const resultData = parseJobResult(job);
        addJobResultMessage(job, resultData);

        // Update the original "Processing..." message
        updateMessageByJobId(jobId, {
          content: `Job completed successfully`,
          type: 'job_result',
        });

        logger.debug(`Job ${jobId.slice(0, 8)} completed, result added to chat`);
        return true;
      } else if (job.status === 'failed') {
        // Job failed - update message
        updateMessageByJobId(jobId, {
          content: `Job failed: ${job.error || 'Unknown error'}`,
          type: 'error',
        });
        untrackJob(jobId);
        logger.debug(`Job ${jobId.slice(0, 8)} failed: ${job.error}`);
        return true;
      } else if (job.status === 'cancelled') {
        updateMessageByJobId(jobId, {
          content: `Job was cancelled`,
          type: 'error',
        });
        untrackJob(jobId);
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
        updateMessageByJobId(jobId, {
          content: `Job not found. The server may have been restarted. Please try your request again.`,
          type: 'error',
        });
        untrackJob(jobId);
        return true; // Consider it "handled"
      }

      logger.error(`Error syncing job ${jobId}:`, error);
      return false;
    }
  }, [addJobResultMessage, updateMessageByJobId, untrackJob]);

  // On mount, check all pending job messages
  useEffect(() => {
    const checkPendingJobs = async () => {
      for (const message of pendingJobMessages) {
        if (!message.jobId) continue;

        // Check if we already have a result message for this job
        const hasResult = messages.some(
          (m) => m.type === 'job_result' && m.jobId === message.jobId
        );

        if (!hasResult) {
          logger.debug(`Checking pending job: ${message.jobId.slice(0, 8)}`);
          await syncJob(message.jobId);
        }
      }
    };

    checkPendingJobs();
  }, []); // Only run on mount

  // Active jobs are polled by ChatPage via useJob hook.
  // This hook only syncs on mount for jobs that completed while the page was closed.

  return { syncJob };
}
