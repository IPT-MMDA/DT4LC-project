import { useQuery } from '@tanstack/react-query';
import apiClient from '../client';
import type { HealthResponse } from '../../types';

export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: () => apiClient.get<HealthResponse>('/v1/health'),
    refetchInterval: 30000, // Check health every 30 seconds
  });
}
