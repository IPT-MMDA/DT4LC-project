import { QueryClient } from '@tanstack/react-query';

/** The app's one react-query client, also for code outside components (attachSample). */
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});
