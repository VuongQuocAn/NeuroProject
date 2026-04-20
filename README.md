# NeuroDiagnosis AI Platform

A comprehensive, multi-modal AI-assisted oncology diagnostic platform designed for neuro-imaging analysis. This repository contains both the FastAPI backend pipeline for AI model inferences and the Next.js frontend dashboard.

## 🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You need to have the following installed on your system:
- [Node.js](https://nodejs.org/) (v18.x or newer) and npm
- [Python](https://www.python.org/) 3.10+
- [Docker](https://www.docker.com/) & Docker Compose (for running backend infrastructure like Redis and Celery easily)

### 1. Running the Frontend (Web Dashboard)

The frontend is built with Next.js 14, React, and Tailwind CSS. It comes pre-configured to run with mock data out-of-the-box, meaning you can test the UI without needing the backend or AI models running.

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install JavaScript dependencies:**
   ```bash
   npm install
   ```

3. **Environment Setup:**
   Ensure there is a `.env.local` file in the `frontend` folder with the following configuration:
   ```env
   NEXT_PUBLIC_API_URL=http://localhost:8000
   NEXT_PUBLIC_USE_MOCK_DATA=true
   ```
   *(Set `NEXT_PUBLIC_USE_MOCK_DATA` to `false` when the real backend is operational).*

4. **Start the development server:**
   ```bash
   npm run dev
   ```

5. **Access the Application:**
   Open your browser and navigate to [http://localhost:3000](http://localhost:3000).
   - **Default Login:** Use username `admin` and any password (e.g. `123456`) to access the dashboard with researcher privileges.

### 2. Running the Backend (API & AI Inference Engine)

The backend is built with FastAPI, integrating Celery for asynchronous AI processing and Redis as the message broker.

1. **Start infrastructure via Docker Compose:**
   Navigate to the root directory of the project and run:
   ```bash
   docker-compose up --build
   ```
   *This command builds the Python backend container, starts Redis, and initializes Celery workers.*

2. **Access the API Documentation:**
   Once running successfully, the interactive Swagger UI documentation for all endpoints is available at [http://localhost:8000/docs](http://localhost:8000/docs).

## 🛠 Tech Stack

- **Frontend:** Next.js 14 (App Router), React, TypeScript, Tailwind CSS v4, Recharts, Lucide Icons.
- **Backend:** Python 3.10, FastAPI, Celery, Redis, Pydantic v2.
- **AI/ML Integration Prepared For:** PyTorch, YOLOv5, U-Net, DenseNet. Features endpoints for Multi-modal inputs (MRI + RNA-seq + Clinical Data like KI-67 index).

## 📄 Architecture Overview

- **`/frontend`**: The unified dashboard for Patient Management, Uploading multi-modal diagnostic data, and viewing detailed AI Clinical Reports with explainable AI (Grad-CAM overlays).
- **`/backend`**: High-performance REST APIs grouped into:
  - `auth`: JWT-based Access Control.
  - `multimodal`: Endpoints to handle and validate DICOM, WSI, and RNA expressions.
  - `inference`: Asynchronous task delegation to Celery for heavy image processing tasks.
  - `analysis`: Synthesis of tumor classification score and survival index (C-index based).





  @'
from ai_core.pipeline import TumorAnalysisPipeline
import os

weights_dir = os.path.join(os.getcwd(), "ai_core", "weights")
pipeline = TumorAnalysisPipeline(weights_dir=weights_dir, device="cpu")

result = pipeline.run_inference(
    image_source=os.path.join(os.getcwd(), "test_mri.dcm"),
    output_dir=os.path.join(os.getcwd(), "test_output_manual")
)
print(result)
'@ | python -
