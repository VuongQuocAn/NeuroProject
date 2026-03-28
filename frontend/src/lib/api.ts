import axios from "axios";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const api = axios.create({
  baseURL: API_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// Request interceptor to add JWT token
api.interceptors.request.use(
  (config) => {
    if (typeof window !== "undefined") {
      const token = localStorage.getItem("token");
      if (token && config.headers) {
        config.headers.Authorization = `Bearer ${token}`;
      }
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for handling 401s
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401 && typeof window !== "undefined") {
      // Clear token and redirect to login
      localStorage.removeItem("token");
      window.location.href = "/login";
    }
    return Promise.reject(error);
  }
);

// API Service Wrapper (Handles Mocks)
export const apiService = {
  auth: {
    login: async (credentials: any) => {
      const formData = new URLSearchParams();
      formData.append('username', credentials.username);
      formData.append('password', credentials.password);
      
      return api.post("/auth/login", formData, {
        headers: { "Content-Type": "application/x-www-form-urlencoded" }
      });
    },
  },
  patients: {
    getAll: async () => {
      return api.get("/records/patients/");
    },
    getHistory: async () => {
      return api.get("/records/patients/"); 
    },
    getById: async (id: string) => {
      return api.get(`/records/patients/${id}`);
    },
    create: async (data: { name: string; external_id?: string; age?: number; gender?: string }) => {
      return api.post("/records/patients/", data);
    }
  },
  analysis: {
    getResult: async (patientId: string) => {
      return api.get(`/records/analysis/${patientId}`);
    },
    getSurvivalCurve: async (patientId: string) => {
      return api.get(`/analytics/survival/${patientId}`);
    },
    getXaiOverlay: async (imageId: string) => {
      return api.get(`/records/analysis/${imageId}/xai-overlay`);
    }
  },
  upload: {
    mri: async (patientId: string, file: File) => {
      const formData = new FormData();
      formData.append("file", file);
      return api.post(`/upload/mri/?patient_id=${patientId}`, formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });
    },
    rna: async (file: File) => {
      const formData = new FormData();
      formData.append("file", file);
      return api.post(`/upload/rna/`, formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });
    },
    clinical: async (patientId: string, data: any) => {
      return api.patch(`/records/patients/${patientId}/clinical`, data);
    }
  }
};
