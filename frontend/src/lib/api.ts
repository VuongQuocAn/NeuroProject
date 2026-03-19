import axios from "axios";
import { mockPatients, mockHistory, mockAnalysisResult, mockLogin, mockSurvivalData, mockXaiOverlay } from "./mock-data";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const USE_MOCK = process.env.NEXT_PUBLIC_USE_MOCK_DATA === "true";

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
      if (USE_MOCK) return { data: mockLogin(credentials) };
      
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
      if (USE_MOCK) return { data: mockPatients };
      return api.get("/records/patients/");
    },
    getHistory: async () => {
      if (USE_MOCK) return { data: mockHistory };
      return api.get("/records/patients/"); 
    },
    getById: async (id: string) => {
      if (USE_MOCK) {
        const patient = mockPatients.find(p => p.id.includes(id) || id.includes(p.id.replace('#', '')));
        return { data: { patient: patient || mockPatients[0], images: [] } };
      }
      return api.get(`/records/patients/${id}`);
    }
  },
  analysis: {
    getResult: async (patientId: string) => {
      if (USE_MOCK) return { data: mockAnalysisResult };
      return api.get(`/records/analysis/${patientId}`);
    },
    getSurvivalCurve: async (patientId: string) => {
      if (USE_MOCK) return { data: mockSurvivalData };
      return api.get(`/analytics/survival/${patientId}`);
    },
    getXaiOverlay: async (imageId: string) => {
      if (USE_MOCK) return { data: mockXaiOverlay };
      return api.get(`/records/analysis/${imageId}/xai-overlay`);
    }
  },
  upload: {
    mri: async (patientId: string, file: File) => {
      if (USE_MOCK) return { data: { message: "Mock upload successful" } };
      const formData = new FormData();
      formData.append("file", file);
      return api.post(`/upload/mri/?patient_id=${patientId}`, formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });
    },
    rna: async (file: File) => {
      if (USE_MOCK) return { data: { message: "Mock RNA upload successful" } };
      const formData = new FormData();
      formData.append("file", file);
      return api.post(`/upload/rna/`, formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });
    },
    clinical: async (patientId: string, data: any) => {
      if (USE_MOCK) return { data: { message: "Mock clinical update successful" } };
      return api.patch(`/records/patients/${patientId}/clinical`, data);
    }
  }
};
