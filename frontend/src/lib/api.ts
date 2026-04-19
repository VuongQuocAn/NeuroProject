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
    },
    deleteImage: async (imageId: string | number) => {
      return api.delete(`/records/images/${imageId}`);
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
    },
    getImageResult: async (imageId: string | number) => {
      return api.get(`/records/analysis/image/${imageId}`);
    },
    downloadReport: async (imageId: string | number) => {
      return api.get(`/records/analysis/image/${imageId}/report`, {
        responseType: "blob",
      });
    }
  },
  inference: {
    runMri: async (imageId: string | number) => {
      return api.post(`/inference/mri/${imageId}`);
    },
    runPrognosis: async (patientId: string | number) => {
      return api.post(`/inference/prognosis/${patientId}`);
    },
    getTask: async (taskId: string | number) => {
      return api.get(`/inference/tasks/${taskId}`);
    },
    waitForTask: async (taskId: string | number, intervalMs = 2000, timeoutMs = 180000) => {
      const startedAt = Date.now();

      while (Date.now() - startedAt < timeoutMs) {
        const response = await api.get(`/inference/tasks/${taskId}`);
        const task = response.data;

        if (task.status === "done" || task.status === "completed") {
          return task;
        }

        if (task.status === "failed") {
          throw new Error(task.error_message || "AI task failed");
        }

        await new Promise((resolve) => setTimeout(resolve, intervalMs));
      }

      throw new Error("Inference timeout");
    },
  },
  upload: {
    mri: async (patientId: string, file: File) => {
      const formData = new FormData();
      formData.append("file", file);
      return api.post(`/upload/mri/?patient_id=${patientId}`, formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });
    },
    rna: async (patientId: string, file: File) => {
      const formData = new FormData();
      formData.append("file", file);
      return api.post(`/upload/rna/?patient_id=${encodeURIComponent(patientId)}`, formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });
    },
    clinical: async (patientId: string, data: any) => {
      return api.patch(`/records/patients/${patientId}/clinical`, data);
    }
  }
};
