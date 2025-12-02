import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import apiClient from '../client';
import { apiLogger as logger } from '../../utils/logger';
import type { Job, JobSubmitRequest, JobsListResponse } from '../../types';

interface JobFilters {
  status?: string;
  limit?: number;
  offset?: number;
}

export function useSubmitJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (data: JobSubmitRequest): Promise<Job> => {
      logger.debug('Sending to API:', JSON.stringify(data, null, 2));
      const job = await apiClient.post<JobSubmitRequest, Job>('/v1/jobs', data);
      logger.debug('API response:', job);
      return job;
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
      return apiClient.get<Job>(`/v1/jobs/${jobId}`);
    },
    enabled: enabled && !!jobId,
    refetchInterval: (query) => {
      const data = query.state.data as Job | undefined;
      const jobStatus = data?.status;
      // Poll every 2 seconds for pending/running jobs
      return jobStatus && ['pending', 'running'].includes(jobStatus) ? 2000 : false;
    },
  });
}

export function useJobs(filters?: JobFilters) {
  return useQuery({
    queryKey: ['jobs', filters],
    queryFn: async (): Promise<JobsListResponse> => {
      return apiClient.get<JobsListResponse>('/v1/jobs', { params: filters });
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
