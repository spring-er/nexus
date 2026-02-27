import axios, { AxiosError, InternalAxiosRequestConfig } from "axios";
import type { AuthTokens } from "@/lib/types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 30000,
});

// Request interceptor — attach access token
api.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    if (typeof window !== "undefined") {
      const tokens = localStorage.getItem("auth_tokens");
      if (tokens) {
        const { access_token } = JSON.parse(tokens) as AuthTokens;
        config.headers.Authorization = `Bearer ${access_token}`;
      }
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor — handle 401 and token refresh
api.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config as InternalAxiosRequestConfig & {
      _retry?: boolean;
    };

    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;

      try {
        const tokens = localStorage.getItem("auth_tokens");
        if (!tokens) throw new Error("No tokens");

        const { refresh_token } = JSON.parse(tokens) as AuthTokens;

        const { data } = await axios.post<AuthTokens>(
          `${API_BASE_URL}/auth/refresh`,
          { refresh_token }
        );

        localStorage.setItem("auth_tokens", JSON.stringify(data));
        originalRequest.headers.Authorization = `Bearer ${data.access_token}`;

        return api(originalRequest);
      } catch {
        localStorage.removeItem("auth_tokens");
        localStorage.removeItem("user");
        if (typeof window !== "undefined") {
          window.location.href = "/login";
        }
        return Promise.reject(error);
      }
    }

    return Promise.reject(error);
  }
);

export default api;
