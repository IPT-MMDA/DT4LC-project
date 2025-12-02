import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import apiClient from '../client';
import type { Job, JobSubmitRequest, JobsListResponse, JobState } from '../../types';

interface JobFilters {
  status?: string;
  limit?: number;
  offset?: number;
}

// Transform backend job response to frontend Job type
// Backend uses "status", frontend uses "state"
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function transformJob(raw: any): Job {
  return {
    id: raw.id,
    state: (raw.status || raw.state) as JobState,
    progress: raw.progress || 0,
    message: raw.message,
    result: raw.result,
    plan: raw.plan,
    error: raw.error,
    created_at: raw.created_at,
    updated_at: raw.updated_at,
  };
}

export function useSubmitJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (data: JobSubmitRequest): Promise<Job> => {
      // Debug logging for job submission
      console.log('[useSubmitJob] Sending to API:', JSON.stringify(data, null, 2));
      const raw = await apiClient.post<JobSubmitRequest, unknown>('/v1/jobs', data);
      console.log('[useSubmitJob] API response:', raw);
      return transformJob(raw);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
    },
  });
}

export function useJob(jobId: string | undefined, enabled = true) {
  return useQuery({
    queryKey: ['job', jobId],
    queryFn: async (): Promise<Job> => {
      const raw = await apiClient.get<unknown>(`/v1/jobs/${jobId}`);
      return transformJob(raw);
    },
    enabled: enabled && !!jobId,
    refetchInterval: (query) => {
      const data = query.state.data as Job | undefined;
      const status = data?.state;
      // Poll every 2 seconds for pending/running jobs
      return status && ['pending', 'queued', 'running'].includes(status) ? 2000 : false;
    },
  });
}

export function useJobs(filters?: JobFilters) {
  return useQuery({
    queryKey: ['jobs', filters],
    queryFn: async (): Promise<JobsListResponse> => {
      const raw = await apiClient.get<{ jobs: unknown[]; total: number; limit: number; offset: number }>('/v1/jobs', { params: filters });
      return {
        jobs: raw.jobs.map(transformJob),
        total: raw.total,
        limit: raw.limit,
        offset: raw.offset,
      };
    },
  });
}

export function useCancelJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (jobId: string) =>
      apiClient.post(`/v1/jobs/${jobId}/cancel`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
    },
  });
}
