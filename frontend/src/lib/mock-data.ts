export const mockLogin = (credentials: any) => {
  const role = credentials.username === "admin" ? "researcher" : "doctor";
  // Create a base64 encoded payload so jwt-decode doesn't throw an error
  const payload = {
    sub: credentials.username,
    role: role,
    exp: 9999999999 // Non-expiring for mock
  };
  const base64Payload = typeof window !== 'undefined' 
    ? btoa(JSON.stringify(payload)) 
    : Buffer.from(JSON.stringify(payload)).toString('base64');
  
  return {
    access_token: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.${base64Payload}.mock_signature`,
    token_type: "bearer",
    role: role,
  };
};

export const mockPatients = [
  { id: "#ND-8821", name: "Johnathan Doe", age: 45, gender: "M", lastVisit: "2023-10-24", diagnosis: "GBM (Rủi ro cao)", riskScore: 92 },
  { id: "#ND-7712", name: "Jane Smith", age: 32, gender: "F", lastVisit: "2023-10-23", diagnosis: "Meningioma", riskScore: 45 },
  { id: "#ND-6605", name: "Robert Chen", age: 67, gender: "M", lastVisit: "2023-10-22", diagnosis: "Multiple Sclerosis", riskScore: 15 },
  { id: "#ND-5590", name: "Sarah Miller", age: 28, gender: "F", lastVisit: "2023-10-21", diagnosis: "Kết quả bình thường", riskScore: 2 },
  { id: "#ND-4481", name: "William Ross", age: 54, gender: "M", lastVisit: "2023-10-20", diagnosis: "Glioma", riskScore: 78 },
  { id: "#ND-3372", name: "Elena Vance", age: 41, gender: "F", lastVisit: "2023-10-19", diagnosis: "Aneurysm", riskScore: 62 },
];

export const mockHistory = [
  { id: 1, date: "Oct 24, 2023", time: "14:30 GMT+7", patientName: "John Doe", patientId: "P-8821", status: "AI: GBM (High Confidence)", doctorReview: "Doctor: GBM (Đã xác nhận)", bgClass: "border-teal-500/30" },
  { id: 2, date: "Oct 22, 2023", time: "10:15 GMT+7", patientName: "Mary Jane Watson", patientId: "P-7412", status: "AI: Multiple Sclerosis (Low Confidence)", doctorReview: "Doctor: Chờ duyệt", bgClass: "border-slate-700" },
  { id: 3, date: "Oct 21, 2023", time: "09:45 GMT+7", patientName: "Robert Downey", patientId: "P-9921", status: "AI: Normal (High Confidence)", doctorReview: "Doctor: Normal (Đã xác nhận)", bgClass: "border-slate-700" },
  { id: 4, date: "Oct 19, 2023", time: "16:20 GMT+7", patientName: "Scarlett Johansson", patientId: "P-3334", status: "AI: Metastasis (High Confidence)", doctorReview: "Doctor: Metastasis (Đã xác nhận)", bgClass: "border-slate-700" },
];

export const mockAnalysisResult = {
  patient_id: "ND-8829-X",
  tumor_label: "Glioblastoma (GBM)",
  classification_confidence: 0.78,
  dice_score: 0.85,
  iou_score: 0.82,
  accuracy: 0.94,
  c_index: 0.75,
  risk_score: 0.72,
  risk_group: "High Risk",
  gradcam_path: "/mock/gradcam.png",
  mask_path: "/mock/mask.png",
  explanations: [
    { label: "Enhancing tumor volume", value: "24.1 cm³" },
    { label: "Peritumoral Edema", value: "Grade III" },
    { label: "KI-67 Prediction", value: "28% ± 5%" }
  ],
  summary: "Imaging reveals a large, peripherally enhancing necrotic mass in the left frontal lobe. Significant mass effect and midline shift (4mm) observed. Radiomic features strongly suggest High-Grade Glioma (WHO Grade IV). Recommended follow-up: Histopathological confirmation and MR Spectroscopy.",
  other_classifications: [
    { label: "Meningioma", value: 0.12 },
    { label: "LGG", value: 0.10 }
  ]
};

export const mockSurvivalData = [
  { month: 0, highRisk: 100, lowRisk: 100 },
  { month: 12, highRisk: 65, lowRisk: 95 },
  { month: 24, highRisk: 40, lowRisk: 88 },
  { month: 36, highRisk: 25, lowRisk: 82 },
  { month: 48, highRisk: 15, lowRisk: 75 },
  { month: 60, highRisk: 5, lowRisk: 70 },
];

export const mockXaiOverlay = {
  gradcam_url: "https://via.placeholder.com/512/0f172a/0d9488?text=Grad-CAM+Overlay",
  mask_url: "https://via.placeholder.com/512/0f172a/ef4444?text=Segmentation+Mask",
  expires_in: 3600
};
