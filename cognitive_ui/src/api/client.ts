import axios, { AxiosError, AxiosRequestConfig } from 'axios';

const axiosInstance = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
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

    const errorMessage = error.response?.data
      ? (error.response.data as any).detail || 'An error occurred'
      : error.message;

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
