import { useEffect, useCallback } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { useChatStore } from '../store/useChatStore';
import apiClient from '../api/client';
import { parseJobResult } from '../components/chat/JobResultCard';
import type { Job } from '../types';

/**
 * Hook to sync pending jobs with their completion state.
 * This handles the case where a page reload happens while jobs are running.
 */
export function useJobSync() {
  const queryClient = useQueryClient();
  const {
    messages,
    addJobResultMessage,
    updateMessageByJobId,
    activeJobIds,
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

      console.log(`[JobSync] Job ${jobId.slice(0, 8)} state: ${job.state}`);

      if (job.state === 'completed' || job.state === 'succeeded') {
        // Job completed - add result message
        const resultData = parseJobResult(job);
        addJobResultMessage(job, resultData);

        // Update the original "Processing..." message
        updateMessageByJobId(jobId, {
          content: `Job completed successfully`,
          type: 'job_result',
        });

        console.log(`[JobSync] Job ${jobId.slice(0, 8)} completed, result added to chat`);
        return true;
      } else if (job.state === 'failed') {
        // Job failed - update message
        updateMessageByJobId(jobId, {
          content: `Job failed: ${job.error || 'Unknown error'}`,
          type: 'error',
        });
        untrackJob(jobId);
        console.log(`[JobSync] Job ${jobId.slice(0, 8)} failed: ${job.error}`);
        return true;
      } else if (job.state === 'cancelled') {
        updateMessageByJobId(jobId, {
          content: `Job was cancelled`,
          type: 'error',
        });
        untrackJob(jobId);
        console.log(`[JobSync] Job ${jobId.slice(0, 8)} cancelled`);
        return true;
      }

      // Job still in progress
      return false;
    } catch (error) {
      console.error(`[JobSync] Error syncing job ${jobId}:`, error);
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
          console.log(`[JobSync] Checking pending job: ${message.jobId.slice(0, 8)}`);
          await syncJob(message.jobId);
        }
      }
    };

    checkPendingJobs();
  }, []); // Only run on mount

  // Also periodically check active jobs
  useEffect(() => {
    if (activeJobIds.length === 0) return;

    const interval = setInterval(async () => {
      for (const jobId of activeJobIds) {
        const completed = await syncJob(jobId);
        if (completed) {
          // Invalidate queries to refresh UI
          queryClient.invalidateQueries({ queryKey: ['jobs'] });
        }
      }
    }, 3000); // Check every 3 seconds

    return () => clearInterval(interval);
  }, [activeJobIds, syncJob, queryClient]);

  return { syncJob };
}
