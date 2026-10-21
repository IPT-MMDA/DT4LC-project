import axios, { AxiosError, AxiosRequestConfig } from 'axios';

export const API_BASE_URL: string = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

/** Resolve a server-relative URL (e.g. an artifact `url`) against the API base URL. */
export function resolveApiUrl(url: string): string {
  return `${API_BASE_URL}${url}`;
}

/**
 * Server-relative `url` of an artifact reference ({kind, path, url, stats}, see
 * docs/ARCHITECTURE.md "Job artifacts"), or undefined for anything else, such as
 * the base64 strings of results produced before v0.5.0.
 */
export function artifactUrl(ref: unknown): string | undefined {
  const url = (ref as { url?: unknown } | null | undefined)?.url;
  return typeof url === 'string' && url ? url : undefined;
}

const axiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
axiosInstance.interceptors.request.use(
  (config) => {
    // Add auth token if exists
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor - unwraps response.data
axiosInstance.interceptors.response.use(
  (response) => response.data,
  (error: AxiosError) => {
    // Handle errors globally
    if (error.response?.status === 401) {
      // Redirect to login or clear auth
      localStorage.removeItem('token');
    }

    const data = error.response?.data as any;
    let errorMessage = 'An error occurred';

    if (data) {
      if (data.error && data.error.message) {
        // New standardized format: {"ok": false, "error": {"message": "..."}}
        errorMessage = data.error.message;
      } else if (data.detail) {
        // FastAPI default or legacy format: {"detail": "..."}
        errorMessage = Array.isArray(data.detail) 
          ? data.detail[0]?.msg || JSON.stringify(data.detail)
          : data.detail;
      } else if (data.error && typeof data.error === 'string') {
        // Simple legacy format: {"error": "..."}
        errorMessage = data.error;
      }
    } else {
      errorMessage = error.message;
    }

    return Promise.reject(new Error(errorMessage));
  }
);

// Typed wrapper that correctly returns T instead of AxiosResponse<T>
const apiClient = {
  get: <T>(url: string, config?: AxiosRequestConfig): Promise<T> =>
    axiosInstance.get(url, config) as Promise<T>,

  post: <TData, TResponse>(url: string, data?: TData, config?: AxiosRequestConfig): Promise<TResponse> =>
    axiosInstance.post(url, data, config) as Promise<TResponse>,

  put: <TData, TResponse>(url: string, data?: TData, config?: AxiosRequestConfig): Promise<TResponse> =>
    axiosInstance.put(url, data, config) as Promise<TResponse>,

  patch: <TData, TResponse>(url: string, data?: TData, config?: AxiosRequestConfig): Promise<TResponse> =>
    axiosInstance.patch(url, data, config) as Promise<TResponse>,

  delete: <T>(url: string, config?: AxiosRequestConfig): Promise<T> =>
    axiosInstance.delete(url, config) as Promise<T>,
};

export default apiClient;
