import { create } from "zustand";
import api from "@/lib/api";
import type {
  User,
  AuthTokens,
  AuthResponse,
  LoginRequest,
  RegisterRequest,
} from "@/lib/types";

interface AuthState {
  user: User | null;
  tokens: AuthTokens | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;

  login: (credentials: LoginRequest) => Promise<void>;
  register: (data: RegisterRequest) => Promise<void>;
  logout: () => void;
  loadFromStorage: () => void;
  clearError: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  tokens: null,
  isAuthenticated: false,
  isLoading: false,
  error: null,

  login: async (credentials: LoginRequest) => {
    set({ isLoading: true, error: null });
    try {
      const { data } = await api.post<AuthResponse>("/auth/login", credentials);
      localStorage.setItem("auth_tokens", JSON.stringify(data.tokens));
      localStorage.setItem("user", JSON.stringify(data.user));
      set({
        user: data.user,
        tokens: data.tokens,
        isAuthenticated: true,
        isLoading: false,
      });
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : "Login failed";
      set({ error: message, isLoading: false });
    }
  },

  register: async (data: RegisterRequest) => {
    set({ isLoading: true, error: null });
    try {
      const { data: response } = await api.post<AuthResponse>(
        "/auth/register",
        data
      );
      localStorage.setItem("auth_tokens", JSON.stringify(response.tokens));
      localStorage.setItem("user", JSON.stringify(response.user));
      set({
        user: response.user,
        tokens: response.tokens,
        isAuthenticated: true,
        isLoading: false,
      });
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : "Registration failed";
      set({ error: message, isLoading: false });
    }
  },

  logout: () => {
    localStorage.removeItem("auth_tokens");
    localStorage.removeItem("user");
    set({
      user: null,
      tokens: null,
      isAuthenticated: false,
      error: null,
    });
  },

  loadFromStorage: () => {
    if (typeof window === "undefined") return;
    const tokens = localStorage.getItem("auth_tokens");
    const user = localStorage.getItem("user");
    if (tokens && user) {
      set({
        tokens: JSON.parse(tokens),
        user: JSON.parse(user),
        isAuthenticated: true,
      });
    }
  },

  clearError: () => set({ error: null }),
}));
