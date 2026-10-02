<div align="center">

# NeuroDiagnosis AI

### Multimodal Deep Learning and Explainable AI for Brain Tumor Diagnosis and Survival Prognosis

An end-to-end research prototype that combines MRI tumor detection, segmentation and classification; multimodal CoxPH prognosis from MRI, WSI, RNA-seq and clinical variables; task-specific XAI; and a clinical web decision-support system.

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Clinical%20Metadata-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Celery](https://img.shields.io/badge/Celery-Asynchronous%20Inference-37814A?logo=celery&logoColor=white)](https://docs.celeryq.dev/)
[![Status](https://img.shields.io/badge/status-research%20prototype-F59E0B)](#clinical-and-research-use)

**Research title:** *Development of a Multimodal Deep Learning Model with XAI for Supporting Brain Tumor Diagnosis and Prognosis*

**Web demo:** [neurodiagnosisai.vercel.app/login](https://neurodiagnosisai.vercel.app/login)

**Lưu ý:** Kinh phí duy trì backend trên AWS đã hết nên backend hiện không còn chạy trên AWS. Để sử dụng đầy đủ dự án, hãy [clone repo và chạy local bằng Docker](#getting-started). Frontend Vercel chỉ hoạt động đầy đủ khi backend local được kết nối qua tunnel theo [hướng dẫn deploy](#website-deployment).

</div>

> [!IMPORTANT]
> This repository is a research and clinical decision-support prototype. Its outputs are not a medical diagnosis and must not be used without review by qualified clinicians, institutional validation, and appropriate regulatory approval.

## Table of Contents

- [Overview](#overview)
- [Research Contributions](#research-contributions)
- [System Architecture](#system-architecture)
- [Module 1: MRI Diagnosis](#module-1-mri-diagnosis)
- [Module 2: Multimodal Prognosis](#module-2-multimodal-prognosis)
- [Explainable AI](#explainable-ai)
- [Clinical Chatbox Agent](#clinical-chatbox-agent)
- [Web CDSS and Asynchronous Processing](#web-cdss-and-asynchronous-processing)
- [Experimental Design](#experimental-design)
- [Experimental Results](#experimental-results)
- [Repository Structure](#repository-structure)
- [API Overview](#api-overview)
- [Security and Data Governance](#security-and-data-governance)
- [Limitations](#limitations)
- [Citation](#citation)
- [Authors](#authors)
- [Acknowledgment](#acknowledgment)
- [Clone, cài Docker và chạy local](#getting-started)
  - [Cấu hình sẵn Chatbox, RAG và LLM](#llm-setup)
- [Frontend Vercel + backend local qua tunnel](#website-deployment)
- [Hướng dẫn sử dụng toàn bộ web](#user-guide)
  - [Danh sách và tạo bệnh nhân](#guide-patients)
  - [Upload MRI, WSI, RNA và chỉ số lâm sàng](#guide-upload)
  - [Chi tiết bệnh nhân](#guide-patient-detail)
  - [Đọc kết quả chẩn đoán và tiên lượng](#guide-results)
  - [Chatbox Agentic](#guide-agent)
  - [Lịch sử và báo cáo chẩn đoán](#guide-history)
  - [NeuroBoard / TumorBoard và hội chẩn ROI](#guide-neuroboard)
  - [Kết quả không phát hiện khối u](#guide-no-tumor)
  - [File dữ liệu và PDF để thử nghiệm](#guide-test-files)

## Overview

NeuroDiagnosis AI is a web-based clinical decision-support system designed to connect two complementary AI workflows:

1. **Automated MRI diagnosis** detects a suspected tumor, segments its pixel-level boundary, classifies the tumor type, and generates an explanation for every prediction stage.
2. **Multimodal survival prognosis** fuses MRI, whole-slide histopathology, RNA-seq and clinical variables to estimate a continuous CoxPH risk score, risk group, survival curve and modality-specific explanations.

The platform separates the user interface, business API, asynchronous GPU inference and data storage layers. FastAPI handles short-lived HTTP work; Redis and Celery move compute-intensive AI tasks to GPU workers; PostgreSQL stores structured records and audit data; MinIO or Cloudflare R2 stores medical images and generated artifacts.

### Core capabilities

| Domain | Capability | Primary output |
|---|---|---|
| MRI detection | YOLOv11 lesion localization | Bounding box, detection confidence, ODAM |
| MRI segmentation | DynUNet pixel-level delineation | Tumor mask, Dice/IoU metadata, Seg-Eigen-CAM |
| MRI classification | DenseNet169 ROI classification | Glioma, Meningioma or Pituitary label; confidence; Finer-CAM |
| Series inference | Consensus majority voting | Series-level tumor decision and representative slice |
| Multimodal prognosis | Masked Gated Attention Fusion with CoxPH | Risk score, risk group, survival curve, fusion attention |
| Visual XAI | Gradient- and activation-based explanations | ODAM, Seg-Eigen-CAM, Finer-CAM, Grad-CAM family |
| Genomic XAI | Input x Gradient attribution | Top genes associated with higher-risk or protective contribution |
| Human review | Expert confirmation and relabeling | Auditable final label and review status |
| Clinical Agent | Context-grounded conversational assistance | Patient QA, history analysis, quick MRI workflow and notifications |

## Research Contributions

The work addresses several practical gaps in AI-assisted neuro-oncology:

- **End-to-end MRI reasoning:** detection, segmentation and classification are integrated into one conditional pipeline rather than evaluated as isolated models.
- **Task-specific explainability:** each inference task uses an explanation mechanism aligned with its output structure: instance-aware ODAM for detection, Seg-Eigen-CAM for masks, and class-discriminative Finer-CAM for classification.
- **Missing-modality-tolerant prognosis:** zero-vector padding and binary modality masks prevent absent MRI, WSI or RNA branches from contaminating the fused representation.
- **Dual visual and genomic XAI:** image evidence is explained with CAM-based methods, while RNA-seq contribution is summarized with Input x Gradient attribution.
- **Human-in-the-loop review:** clinicians can confirm or correct AI labels regardless of confidence; low confidence changes the warning state, not the clinician's authority to review.
- **Production-oriented orchestration:** long-running inference is executed outside the HTTP request path through Redis and Celery.
- **Grounded language assistance:** RAG and Gemini convert structured AI/XAI evidence into readable explanations while retaining explicit clinical-safety constraints.

## System Architecture

### Research pipeline architecture

The MRI subsystem is organized as an inference layer and an XAI/storage layer. Tumor-positive cases continue through ROI segmentation and classification; tumor-negative cases stop before prognosis so that an invalid risk score is not produced.

![Two-layer MRI and XAI architecture](assets/readme/eureka/mri_xai_architecture.png)

### End-to-end MRI, XAI and RAG flow

![MRI diagnosis pipeline with task-specific XAI, RAG and LLM reporting](assets/readme/eureka/mri_diagnosis_pipeline.png)

The operational sequence is:

```text
MRI upload
  -> YOLOv11 detection
     -> no tumor: stop downstream tumor prognosis and return a review notice
     -> tumor detected:
        -> crop ROI
        -> DynUNet segmentation
        -> DenseNet169 classification
        -> ODAM + Seg-Eigen-CAM + Finer-CAM
        -> persist structured results and visual artifacts
        -> retrieve supporting knowledge
        -> generate a grounded clinical explanation
```

## Module 1: MRI Diagnosis

### Inputs

- single MRI images in supported raster or DICOM form;
- DICOM series with representative-slice selection;
- NIfTI data where supported by the preprocessing workflow.

### Pipeline stages

| Stage | Model or method | Purpose | Stored evidence |
|---|---|---|---|
| Preprocessing | intensity normalization, resize/resample, orientation and format handling | Standardize heterogeneous MRI input | Prepared image or series metadata |
| Detection | YOLOv11 | Decide whether a suspicious tumor region exists and localize it | Bounding box, confidence, ODAM |
| ROI extraction | bbox-guided crop | Restrict downstream reasoning to the detected lesion | ROI and masked ROI |
| Segmentation | MONAI DynUNet | Delineate the tumor at pixel level | Binary mask, contour, Dice/IoU fields, Seg-Eigen-CAM |
| Classification | DenseNet169 | Classify tumor-positive ROI into three primary classes | Class probabilities, AI label, confidence, Finer-CAM |
| Series aggregation | consensus majority voting | Improve consistency across a DICOM sequence | Series label, key slice and per-slice evidence |
| Interpretation | local RAG + Gemini | Explain the classification and XAI evidence | Natural-language explanation and retrieval metadata |

### Tumor-negative safety rule

If detection concludes `no_tumor_detected = true`, the system returns **No tumor detected**, leaves tumor classification fields empty, and does not calculate or present a multimodal risk score for that MRI event. This rule is enforced in the backend pipeline rather than being a frontend-only display condition.

### Representative MRI outputs

| Detection | Segmentation | Classification XAI |
|---|---|---|
| ![MRI detection result](assets/readme/examples/mri_detection_example.png) | ![MRI segmentation result](assets/readme/examples/mri_segmentation_example.png) | ![MRI classification Finer-CAM](assets/readme/examples/mri_classification_xai_example.png) |

## Module 2: Multimodal Prognosis

The prognosis model estimates relative survival risk from up to four modalities. Every available branch is projected to a 512-dimensional representation before masked gated-attention fusion and a CoxPH prediction head.

![Multimodal survival model architecture](assets/readme/eureka/survival_model_architecture.png)

### Modality-specific processing

| Modality | Preprocessing | Encoder | Representation |
|---|---|---|---|
| MRI | orientation normalization, smart slicing, Min-Max scaling, 256 x 256 resize | Frozen DenseNet121 image encoder + slice attention | 512-D |
| WSI | OpenSlide access, tissue-aware grid tiling, HSV saturation filtering, 256 x 256 tiles | Frozen DenseNet121 image encoder + tile attention | 512-D |
| RNA-seq | STAR-count parsing, Ensembl ID normalization, duplicate aggregation, `log2(TPM + 1)` | MLP with LayerNorm and dropout | 512-D |
| Clinical | survival-event construction, time normalization, Min-Max and one-hot encoding | Two-layer clinical MLP | 512-D |

![Parallel preprocessing of MRI, WSI, RNA-seq and clinical data](assets/readme/eureka/multimodal_preprocessing.png)

### Missing-modality handling

For each patient, the model builds a mask:

```text
[has_mri, has_wsi, has_rna, has_clinical]
```

An absent modality receives a zero-valued 512-D vector. Before softmax, its attention logit is assigned a large negative value, forcing its normalized fusion weight toward zero. This preserves tensor shape while preventing missing branches from influencing the prediction.

![Masked attention computation](assets/readme/eureka/masked_attention_flow.png)

### Prognosis outputs

- continuous CoxPH risk score;
- categorical risk group;
- modality-level fusion-attention weights;
- Kaplan-Meier-compatible survival-curve data;
- MRI/WSI risk heatmaps using Grad-CAM, Grad-CAM++ and LayerCAM;
- top RNA-seq features with high-risk or protective attribution;
- generated clinical interpretation.

![Visual and genomic XAI flow for multimodal prognosis](assets/readme/eureka/multimodal_xai_flow.png)

## Explainable AI

The system does not apply one generic heatmap to all model branches. Explanation targets are selected according to the semantics of each task.

### Detection: ODAM

ODAM explains one YOLO detection instance using gradients of both the class score and decoded bounding-box coordinates. The maps are combined and projected back to the original image space.

![ODAM architecture](assets/readme/xai_odam_diagram.png)

![ODAM example with original MRI, ground-truth box, predicted box and heatmap](assets/readme/eureka/odam_example.png)

### Segmentation: Seg-Eigen-CAM

Seg-Eigen-CAM backpropagates from tumor-region logits, combines absolute gradients with activations, extracts the dominant component through SVD, corrects its sign, normalizes it and overlays the result on the ROI.

![Seg-Eigen-CAM architecture](assets/readme/xai_seg_eigen_cam_diagram.png)

![Seg-Eigen-CAM example](assets/readme/eureka/seg_eigen_cam_example.png)

### Classification: Finer-CAM

Finer-CAM explains the difference between the predicted class logit and a semantically close reference class. This contrastive objective emphasizes class-discriminative evidence rather than generic salient anatomy.

![Finer-CAM architecture](assets/readme/xai_finer_cam_diagram.png)

![Finer-CAM example](assets/readme/eureka/finer_cam_example.png)

### Prognosis: visual and genomic attribution

For risk prediction, CAM variants explain image branches and Input x Gradient explains RNA-seq contribution. Positive gene attribution is presented as risk-increasing evidence and negative attribution as protective evidence; neither is interpreted as causal biology.

![Top genomic features from Input x Gradient attribution](assets/readme/eureka/genomic_xai.png)

> [!NOTE]
> XAI visualizes evidence used by the model. A heatmap is not histopathological proof, and a gene attribution score is not a causal biomarker claim.

## Clinical Chatbox Agent

The floating NeuroDiagnosis Agent is an application layer over the existing clinical APIs and AI workflows. It is available across dashboard pages and receives the user's message together with page context such as `current_page`, `patient_id`, `image_id` and an optional selected image region.

### Agent execution graph

```mermaid
flowchart LR
    UI[Chat widget] --> P[Gemini planner]
    P --> V[Tool and argument validator]
    V --> T[Approved clinical tools]
    T --> F[Final grounded Gemini response]
    F --> SSE[SSE token stream]
    SSE --> UI
    P -. checkpoint .-> CP[(PostgresSaver)]
    F -. conversation .-> DB[(AgentConversation and AgentMessage)]
    F -. long-term memory .-> LS[(PostgresStore)]
    T -. audit .-> AL[(AgentAuditLog)]
```

### Implemented Agent capabilities

- LLM-based intent planning rather than page-only or keyword-only routing;
- patient-profile, diagnosis-history, image-analysis and notification tools;
- validation of allowed tools and required `patient_id` or `image_id` arguments;
- server-sent event streaming, with the complete assistant message persisted only after generation completes;
- conversation history with reload, soft delete, trim and LLM summary operations;
- LangGraph checkpointing with `PostgresSaver`;
- long-term user memory with `PostgresStore`;
- patient- and image-aware audit logging;
- quick MRI upload from the chatbox;
- human-in-the-loop interruption when quick MRI diagnosis has no patient context;
- automatic continuation after patient selection, result summary in chat and navigation to the full result page;
- Markdown rendering, copy actions, timestamps and zoomable MRI result images in the chat UI.

### Grounding and RAG

Classification explanations use a local child-parent retrieval store with precomputed BGE-M3 child embeddings. Query embeddings and optional reranking are obtained through Hugging Face inference endpoints. The top grounded contexts, structured model outputs and XAI metadata are passed to Gemini; generated text is validated and falls back to a deterministic explanation if generation is unavailable or unsafe.

### Human review semantics

Every tumor classification can be reviewed, including high-confidence predictions. Confidence at or below 0.95 triggers a stronger warning. A review records the original AI label, AI confidence, expert label, comment and whether the expert **confirmed** or **corrected** the prediction.

## Web CDSS and Asynchronous Processing

### Four-tier application architecture

![Four-tier web CDSS architecture](assets/readme/eureka/web_layered_architecture.png)

| Tier | Main components | Responsibility |
|---|---|---|
| Presentation | Next.js, React, TypeScript, Cornerstone | Patient workflows, uploads, result visualization, XAI and chat |
| Application | FastAPI, Pydantic, SQLAlchemy, JWT/RBAC | Validation, authorization, records, task creation and reporting |
| Async inference | Redis, Celery, GPU worker | MRI and prognosis pipelines outside the HTTP request lifecycle |
| Data | PostgreSQL, MinIO/R2, Redis result backend | Structured records, object artifacts, task state and memory |

### Why Redis and Celery are required

GPU inference is too long-running and resource-intensive for a normal HTTP request. FastAPI therefore validates the request, creates an `InferenceTask`, sends a Celery signature to Redis, and immediately returns a task identifier. A Celery worker consumes the queued task, runs the AI pipeline, stores the result and updates its status. The frontend polls task state and renders the completed output without holding an HTTP connection open for the entire inference run.

![Asynchronous inference sequence](assets/readme/eureka/async_celery_flow.png)

### Data model

The relational model separates patient identity, uploaded modalities, task lifecycle, AI outputs, review records, history reports, accounts and audit logs.

![PostgreSQL entity-relationship diagram](assets/readme/eureka/postgresql_erd.png)

The current implementation also includes Agent-specific tables:

- `agent_conversations` for thread-level context and summaries;
- `agent_messages` for user and assistant messages;
- `agent_audit_logs` for tool and chat activity;
- LangGraph checkpoint/store tables managed by `PostgresSaver` and `PostgresStore`.

### Container deployment

![Docker Compose deployment topology](assets/readme/eureka/docker_deployment.png)

Two deployment modes are maintained:

- **Local:** Next.js + FastAPI + Celery + Redis + PostgreSQL + MinIO.
- **Tunnel deployment:** Vercel frontend + Cloudflare Quick Tunnel + local FastAPI/Celery/GPU + PostgreSQL/Redis + Cloudflare R2.

## Experimental Design

### MRI diagnosis dataset

The research uses the Bangladesh Brain Cancer MRI dataset for model development and a standardized held-out evaluation set of 424 images:

| Class | Evaluation samples | Share |
|---|---:|---:|
| Glioma | 116 | 27.36% |
| Meningioma | 102 | 24.06% |
| Pituitary | 108 | 25.47% |
| Normal | 98 | 23.11% |
| **Total** | **424** | **100%** |

Detection is evaluated on all 424 images. Segmentation and three-class ROI classification are evaluated on the 326 tumor-positive images.

### Multimodal prognosis cohort

The merged cohort contains 899 patients from five neuro-oncology sources:

| Cohort | Patients |
|---|---:|
| UPENN-GBM | 585 |
| TCGA-GBM | 133 |
| TCGA-LGG | 122 |
| IvyGAP | 31 |
| CPTAC-GBM | 28 |
| **Total** | **899** |

The report uses 719 patients for training and 180 for independent validation, with event-stratified splitting for survival evaluation.

## Experimental Results

All values in this section are transcribed from the supplied 2026 research report. They describe the documented experimental setting, not external prospective clinical validation.

### Summary

| Task | End-to-end or broad result | Tumor-ROI or task-specific result |
|---|---:|---:|
| YOLOv11 detection | Precision 95.25%, Recall 86.20%, F1 90.50% | mAP@50 85.39%, mean IoU on matched tumor boxes 73.34% |
| DynUNet segmentation | Mean Dice 87.42% | Dice 92.77%, IoU 87.03% on tumor-positive ROIs |
| DenseNet169 classification | Accuracy 93.86%, AUC > 0.92 | Accuracy 97.55%, Macro-F1 97.54% on detected tumor ROIs |
| Multimodal CoxPH prognosis | Best validation C-index 0.6936 | C-index 0.654-0.668 under random WSI/RNA missingness |

The distinction between broad pipeline performance and ROI-only performance is essential: ROI metrics evaluate downstream models after a tumor region is available and therefore should not be presented as end-to-end diagnostic performance.

### Detection

![Detection metric summary](assets/readme/eureka/detection_summary.png)

| Metric | Value |
|---|---:|
| Precision@0.5 | 95.25% |
| Recall@0.5 | 86.20% |
| F1@0.5 | 90.50% |
| mAP@50 over the 424-image evaluation set | 85.39% |
| Mean IoU on images with matched ground-truth boxes | 73.34% |
| Mean detection inference time | 0.0345 s/image |

![Detection precision-recall and confidence-threshold curves](assets/readme/eureka/detection_pr_curves.png)

The report identifies Glioma as the most difficult detection group because of heterogeneous shape, diffuse boundaries and surrounding edema. This is reflected in lower group AP and a higher false-negative burden than for Pituitary tumors.

### Segmentation

| Metric on 326 tumor-positive ROIs | Value |
|---|---:|
| Dice | 92.77% |
| IoU | 87.03% |
| Sensitivity | 92.77% |
| Specificity | 96.83% |
| HD95 | 5.52 px |

![DynUNet segmentation result summary](assets/readme/eureka/segmentation_summary.png)

Meningioma achieved the strongest class-wise segmentation result in the report (Dice 95.41%, IoU 91.41%), while Glioma remained more difficult because of irregular and infiltrative boundaries.

### Classification

| Metric on 326 tumor ROIs | Value |
|---|---:|
| Accuracy | 97.55% |
| Macro-F1 | 97.54% |
| Weighted-F1 | 97.55% |
| Top-2 accuracy | 100.00% |
| Total classification errors | 8/326 |

![DenseNet169 confusion matrix](assets/readme/eureka/classification_confusion_matrix.png)

![DenseNet169 one-vs-rest ROC and precision-recall curves](assets/readme/eureka/classification_roc_pr.png)

At a 0.95 confidence threshold, the documented ROI subset reaches 99.00% accuracy and 99.00% Macro-F1 at 92.33% coverage. The application uses this threshold as a review-warning policy, not as an automatic prohibition on expert review.

![Classification performance and coverage by confidence threshold](assets/readme/eureka/classification_threshold.png)

### XAI evaluation

| XAI method | Evaluation population | Key reported evidence |
|---|---|---|
| ODAM | 293 valid matched detections | Pointing game 98.98%; confidence drop 59.36%, 70.28% and 82.55% when masking the top 5%, 10% and 20% regions |
| Seg-Eigen-CAM | 326 tumor ROIs | Pointing game on ground-truth mask 99.08%; energy inside GT mask 78.08%; energy inside predicted mask 79.02% |
| Finer-CAM | 326 classified ROIs | Mean confidence 98.17%; compactness@0.5 72.00%; Drop@20 23.77%; Relative Drop@20 41.50% |

![ODAM confidence-drop faithfulness test](assets/readme/eureka/odam_faithfulness.png)

These metrics test localization and faithfulness of model explanations. They do not establish clinical validity by themselves.

### Multimodal survival prognosis

The best validation C-index is **0.6936 at epoch 11**. Under random WSI or RNA-seq omission, masked gated-attention fusion retains a C-index in the **0.654-0.668** range without pipeline failure, supporting the intended missing-modality tolerance.

![CoxPH training and validation loss](assets/readme/eureka/survival_training_loss.png)

![Validation C-index over training epochs](assets/readme/eureka/survival_c_index.png)

### Web system evaluation

| System measure | Reported value |
|---|---:|
| API error rate under the documented test load | 0.04% |
| 30-day health-check availability | 99.92% |
| Average end-to-end MRI response | 4.2 s/case |
| Average end-to-end multimodal response | 12.8 s/case |
| Lighthouse Performance | 84/100 |
| Lighthouse Accessibility | 92/100 |
| Lighthouse Best Practices | 95/100 |
| Lighthouse SEO | 90/100 |
| Specialist trust/acceptance rating | 4.6/5 |

## Repository Structure

```text
.
|-- backend/
|   |-- agent/                 LangGraph planner, validator, tools and memory
|   |-- ai_core/               MRI, prognosis and XAI model implementations
|   |-- rag/xai/               Local child-parent RAG artifacts and embeddings
|   |-- routers/               FastAPI route modules
|   |-- services/              RAG and Gemini explanation services
|   |-- migrations/            Database migrations
|   |-- celery_app.py          Celery configuration
|   |-- task.py                Asynchronous AI tasks
|   `-- main.py                FastAPI application entry point
|-- frontend/
|   |-- src/app/               Next.js routes and clinical screens
|   |-- src/components/        Layout, result, visualization and Agent UI
|   |-- src/contexts/          Authentication, theme and Agent context
|   |-- src/hooks/             Streaming and UI hooks
|   `-- src/lib/               API clients and shared utilities
|-- assets/readme/             Architecture and experimental figures
|-- docker-compose.local.yml   Standalone local stack, CPU by default
|-- docker-compose.gpu.yml     Optional NVIDIA GPU overlay
|-- docker-compose.yml         Original development stack
|-- docker-compose.tunnel.yml  Local backend stack for Vercel/tunnel deployment
|-- docker/minio/Dockerfile    Build pinned MinIO and mc from official sources
|-- SETUP_GUIDE.md             Complete setup instructions
`-- TUNNEL_DEPLOY_GUIDE.md     Tunnel deployment workflow
```

## API Overview

The complete, executable API specification is available from FastAPI Swagger at `/docs`. Major endpoint groups include:

| Group | Representative endpoints |
|---|---|
| Authentication | `POST /auth/login` |
| Patients and history | `GET/POST /records/patients`, `GET /records/patients/{patient_id}`, `GET /records/patients/{patient_id}/history-report` |
| Upload | `POST /upload/mri/`, `POST /upload/mri/series`, `POST /upload/wsi/series`, `POST /upload/rna/` |
| Inference | `POST /inference/mri/{image_id}`, `POST /inference/prognosis/{patient_id}`, `GET /inference/tasks/{task_id}` |
| Results and XAI | `GET /records/analysis/image/{image_id}`, `GET /records/analysis/image/{image_id}/report`, `POST /records/analysis/image/{image_id}/validate` |
| Classification review | `POST /records/analysis/image/{image_id}/classification-review` |
| Agent | `POST /agent/chat`, `POST /agent/chat/stream`, `GET /agent/conversations`, `POST /agent/quick-mri` |
| NeuroBoard | `GET /neuroboard/feed`, `POST /neuroboard/posts`, `GET /neuroboard/me` |
| Administration | `GET /admin/logs` |

## Security and Data Governance

- JWT bearer authentication protects authenticated API operations.
- RBAC separates doctor and researcher permissions.
- DICOM upload preprocessing removes identifying tags before object storage where configured.
- Presigned object URLs avoid exposing storage credentials to the browser.
- Access logs and Agent audit logs support traceability.
- Patient, inference, review and Agent records are stored separately in PostgreSQL.
- Backend credentials are loaded from environment variables. JWT, database and storage credentials remain private. The owner has explicitly authorized sharing the Gemini/Hugging Face demo keys in the setup instructions below; shared demo access is subject to the owner's quota.

Before any real-world use, replace all development credentials, enable managed TLS and secret rotation, define retention and backup policies, review DICOM de-identification against local regulation, and complete institutional security and clinical validation.

## Limitations

- The reported results are retrospective and dataset-specific; they are not evidence of prospective clinical effectiveness.
- The MRI diagnosis dataset is public and does not represent every scanner, protocol, institution or population.
- ROI-only segmentation and classification metrics are conditional on tumor localization and must not be interpreted as whole-pipeline metrics.
- The survival cohort remains small relative to the 60,664-dimensional RNA input, creating overfitting and generalization risk.
- Missing-modality robustness does not eliminate bias from systematically unavailable modalities.
- XAI localization and perturbation metrics measure model behavior, not biological or clinical causality.
- The Quick Tunnel deployment is appropriate for demonstration, not a production hospital network.
- The system requires external validation, calibration, monitoring, governance and regulatory assessment before clinical deployment.

## Clinical and Research Use

NeuroDiagnosis AI is intended to support research, education and clinician review. It must not autonomously determine treatment, replace radiological or pathological assessment, or be used as the sole basis for patient care.

## Citation

If this repository supports your research, cite the project as:

```bibtex
@misc{neurodiagnosisai2026,
  title        = {Development of a Multimodal Deep Learning Model with XAI for Supporting Brain Tumor Diagnosis and Prognosis},
  author       = {Pham Huynh Quoc Dat and Nguyen Hai Dang and Vuong Quoc An},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/Dang123-ui/brain-tumor-multimodal-xai},
  note         = {Research prototype submitted to the 28th Eureka Student Research Award}
}
```

## Authors

- **Phạm Huỳnh Quốc Đạt**
- **Nguyễn Hải Đăng**
- **Vương Quốc An**

Academic advisor: **Dr. Trịnh Hùng Cường**

## Acknowledgment

Architecture diagrams, evaluation plots and reported measurements in this README were adapted from the supplied 2026 full research report. The implementation additionally documents the current LangGraph clinical Agent and human-in-the-loop workflows present in this repository.

## Getting Started

Hướng dẫn dưới đây dành cho người clone repo lần đầu, chạy **toàn bộ dự án bằng Docker** trên máy cá nhân. Cách cài phần mềm, GPU, chạy frontend riêng và xử lý lỗi nằm trong [SETUP_GUIDE.md](SETUP_GUIDE.md).

Bộ local dùng cùng source frontend/backend với bộ `neuroproject-tunnel`, có đầy đủ **Chatbox Agent, NeuroBoard, hồ sơ bệnh nhân, MRI và phân tích đa mô thức**. Web chạy tại `localhost:3000`, API tại `localhost:8001`; PostgreSQL và MinIO lưu dữ liệu riêng trên máy người chạy. Không cần Vercel hay tunnel để dùng web local.

### 1. Cài Git, Git LFS và Docker

- Cài [Git](https://git-scm.com/downloads) và [Git LFS](https://git-lfs.com/).
- Windows: cài [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/), bật WSL 2 và dùng Linux containers. Mở Docker Desktop, đợi engine chạy trước khi dùng lệnh Docker.
- Linux: cài [Docker Engine](https://docs.docker.com/engine/install/) và [Compose plugin](https://docs.docker.com/compose/install/linux/).
- Khuyến nghị máy x86_64 có ít nhất 16 GB RAM và khoảng 25 GB ổ trống cho source, model, image và cache. Đây là mức chuẩn bị tham khảo; dữ liệu WSI và inference đa mô thức có thể cần thêm tài nguyên.
- Local mặc định chạy CPU. NVIDIA GPU là tùy chọn tăng tốc; không cần cài Python, Node.js hay CUDA Toolkit trên host khi chạy mọi thành phần trong Docker.

Kiểm tra trong PowerShell:

```powershell
git --version
git lfs version
docker version
docker compose version
```

### 2. Clone repository và tải model

```powershell
git lfs install
git clone https://github.com/VuongQuocAn/NeuroProject.git
cd NeuroProject
git lfs pull origin
git lfs fsck
```

HTTPS giúp clone public repo mà không cần tạo SSH key. Hai file `.pth` dùng Git LFS; file pointer vài trăm byte chưa phải model. Các file cần có:

| File trong `backend/ai_core/weights/` | Chức năng | Kích thước tham khảo |
|---|---|---|
| `yolo_weights.pt` | Phát hiện u | 40.5 MB |
| `unet_weights.pt` | Phân đoạn u | 46.5 MB |
| `densenet169_weights.pth` | Phân loại u | 51 MB |
| `best_multimodal_model.pth` | Tiên lượng đa mô thức | 313 MB |

Giữ nguyên `backend/rag/xai/`: `embedding_config.json`, `child_embeddings.npy`, `child_chunks_metadata.jsonl`, `parent_chunks_lookup.json` đã có trong repo. Database local tự nạp hai bệnh nhân demo `UCSF-003` và `UCSF-001`, cùng file dữ liệu và kết quả được đóng gói trong repo.

### 3. Tạo cấu hình local

Chạy tại thư mục gốc chứa `docker-compose.local.yml`:

```powershell
Copy-Item .env.example .env
$localSecret = [guid]::NewGuid().ToString('N') + [guid]::NewGuid().ToString('N')
(Get-Content .env) -replace '^SECRET_KEY=.*$', "SECRET_KEY=$localSecret" | Set-Content .env -Encoding ascii
notepad .env
```

Chỉ copy template lần đầu, tránh ghi đè cấu hình đã có. Linux dùng `cp .env.example .env`, sửa bằng trình soạn thảo và đặt `SECRET_KEY` thành chuỗi ngẫu nhiên riêng ít nhất 32 ký tự.

Các giá trị database/MinIO mẫu dùng được cho local. Compose tự đổi địa chỉ database, Redis và MinIO sang hostname nội bộ container, đồng thời tạo bucket lưu trữ. Không dùng địa chỉ `localhost` để các container gọi nhau.

| Biến | Khi nào cần |
|---|---|
| `SECRET_KEY` | Khóa JWT riêng cho backend và worker |
| `GEMINI_API_KEY` | Copy key demo bên dưới vào cấu hình: Chatbox Agent và diễn giải XAI/báo cáo bằng LLM |
| `HF_API_TOKEN` | Copy token demo bên dưới vào cấu hình: query embedding/reranking cho RAG |
| `GEMINI_MODEL`, `HF_EMBEDDING_API_URL`, `HF_RERANKER_API_URL` | Chọn model/endpoint mà tài khoản của bạn có quyền dùng |
| `NEUROBOARD_SEED_DEMO` | Mặc định `true`: tạo tài khoản và bài đăng minh họa NeuroBoard; đặt `false` để không seed bài mẫu |

Sau khi copy template, điền hai key demo trong phần dưới vào file cấu hình. Người clone có thể dùng key của chủ dự án được chia sẻ để chạy demo nếu key còn hoạt động và còn hạn mức. Local dùng MinIO nên không cần tài khoản AWS hoặc Cloudflare R2.

<a id="llm-setup"></a>

#### Chatbox, RAG và mô hình LLM

**Copy cả hai dòng sau vào `.env` ở thư mục gốc khi chạy local, hoặc vào `.env.tunnel` khi dùng `docker-compose.tunnel.yml`.** Đây là key demo do chủ dự án cung cấp. Thay hai dòng đang để trống trong file cấu hình, không tạo biến trùng lặp:

```env
GEMINI_API_KEY=AIzaSyC0gyMW4k1jL9_BfmViDUnSnBmkCpa6K2M
HF_API_TOKEN=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
```

Giữ nguyên `GEMINI_MODEL`, `HF_EMBEDDING_API_URL` và `HF_RERANKER_API_URL` trong template. Backend/worker nạp key từ file cấu hình khi container được tạo; frontend Vercel chỉ gọi API backend.

- **Gemini** chạy qua API Internet, dùng `GEMINI_API_KEY` và `GEMINI_MODEL` trong `.env`. Mô hình MRI/tiên lượng PyTorch chạy trên máy local; Gemini phụ trách Chatbox, diễn giải và sinh nội dung báo cáo.
- **Hugging Face** cung cấp query embedding/reranking cho RAG qua `HF_API_TOKEN`, `HF_EMBEDDING_API_URL` và `HF_RERANKER_API_URL`. Kho tri thức XAI đã được đóng gói trong `backend/rag/xai/`.
- Key demo dùng chung hạn mức của chủ dự án. Nếu báo key không hợp lệ, hết quota hoặc model không được cấp quyền, thay bằng key riêng: [tạo Gemini key](https://ai.google.dev/gemini-api/docs/api-key), [tạo HF token](https://huggingface.co/settings/tokens) có quyền **Make calls to Inference Providers** theo [tài liệu HF](https://huggingface.co/docs/inference-providers/index). Kiểm tra model/endpoint tương ứng với tài khoản đang dùng.
- Các biến LLM được nạp vào **backend và worker**. Trên Vercel chỉ cấu hình `NEXT_PUBLIC_API_URL` để frontend gọi backend; không đặt key LLM trong biến `NEXT_PUBLIC_*`.

Sau khi đổi key/model trong `.env`:

```powershell
docker compose -f docker-compose.local.yml up -d --force-recreate backend worker
docker compose -f docker-compose.local.yml logs --tail 100 backend worker
```

Đăng nhập, mở robot và hỏi “Thống kê bệnh nhân của tôi” để kiểm tra Agent. Lỗi 401/403 thường cần kiểm tra key/quyền truy cập; lỗi 429 cần kiểm tra hạn mức và thử lại sau. Chạy model ảnh và mở các kết quả demo đã lưu vẫn được khi dịch vụ LLM bên ngoài tạm hết quota.

### 4. Build image và tạo container

```powershell
docker compose -f docker-compose.local.yml config --quiet
docker compose -f docker-compose.local.yml build backend frontend minio
docker compose -f docker-compose.local.yml run --rm --no-deps backend python scripts/check_local_setup.py
docker compose -f docker-compose.local.yml up -d
docker compose -f docker-compose.local.yml ps -a
curl.exe http://localhost:8001/health
```

Lần đầu build cần Internet và có thể mất nhiều phút. Image backend chứa Python/PyTorch và được worker dùng chung. MinIO và `mc` được build từ source chính thức ghim phiên bản vì registry MinIO cũ không còn tải được; Go compiler chạy trong Docker, không cần cài Go trên host.

Compose tạo network, volume và các container `db`, `redis`, `minio`, `backend`, `worker`, `frontend`. `minio-init` chạy một lần để tạo bucket, nên trạng thái **Exited (0)** của container này là bình thường. Backend đợi database/Redis/MinIO sẵn sàng; worker và frontend đợi backend. Không cần tạo container thủ công trong Docker Desktop.

Giữ container `minio-init` sau khi nó chạy xong. Nếu đã xóa và nút Start của Docker Desktop báo `could not find minio-init`, chạy lại `docker compose -f docker-compose.local.yml up -d` để tạo lại container còn thiếu.

Luôn dùng `-f docker-compose.local.yml` cho bộ local mới. File này dùng project `neuroproject-local` và volume riêng, không phụ thuộc volume của máy tác giả hay tự nạp `docker-compose.override.yml` của máy khác. Host cần trống cổng 3000, 8001, 9000 và 9001. API local dùng 8001 nên có thể chạy cùng backend tunnel ở 8000. Đổi `LOCAL_BACKEND_PORT` trong `.env` nếu cần, rồi tạo lại backend/frontend bằng `up -d`.

### 5. Mở web và kiểm tra workflow

| Dịch vụ | Địa chỉ | Đăng nhập local mặc định |
|---|---|---|
| Web | [http://localhost:3000/login](http://localhost:3000/login) | `admin` / `123456` (researcher) |
| Swagger API | [http://localhost:8001/docs](http://localhost:8001/docs) | Dùng `POST /auth/login` khi endpoint cần JWT |
| MinIO console | [http://localhost:9001](http://localhost:9001) | `admin` / `password123`, hoặc giá trị trong `.env` |

Backend trả `{"status":"ok"}` ở `/health`. Đăng nhập `admin` để thấy sẵn `UCSF-003` và `UCSF-001` trong danh sách bệnh nhân; mở hồ sơ để xem ảnh, kết quả và XAI đã có. Có thể tạo thêm bệnh nhân và tải MRI mới để thử pipeline. Worker phải chạy để job AI được xử lý.

Kiểm tra thủ công sau khi đăng nhập:

1. Mở **NeuroBoard** trong menu hoặc [localhost:3000/neuroboard](http://localhost:3000/neuroboard). Với seed mặc định, phải thấy bài minh họa; thử đăng một bài, bình luận và tải lại trang.
2. Bấm biểu tượng robot để mở **NeuroDiagnosis Agent**. Kiểm tra mở/đóng và lịch sử chat, gửi câu hỏi đơn giản và chờ trả lời sau khi copy key demo vào cấu hình.
3. Tạo bệnh nhân thử, upload MRI, kiểm tra ảnh preview, chạy phân tích và mở kết quả. Với ảnh phát hiện u, kiểm tra thêm XAI/phân đoạn; ảnh dự đoán không có u có thể bỏ qua các bước này.
4. Nếu có file phù hợp, upload WSI/RNA và nhập dữ liệu lâm sàng cho cùng bệnh nhân rồi chạy phân tích đa mô thức.
5. Dùng `stop`, sau đó `up -d` và đăng nhập lại: bệnh nhân, kết quả, bài đăng và hội thoại đã lưu phải còn. Hai ca demo được đọc từ gói trong repo; dữ liệu người dùng tạo thêm được giữ trong volume local.

Tài khoản bác sĩ mặc định `doctor_lan` / `123456` và `doctor_minh` / `123456` cũng được tạo khi khởi động; có thể dùng để thử chức năng theo vai trò. Đổi mật khẩu nếu chia sẻ backend cho người khác.

Health check chỉ xác nhận API trả lời. Để kiểm tra cả khả năng nạp các model:

```powershell
docker compose -f docker-compose.local.yml exec worker python scripts/check_local_setup.py --load-models
```

Lệnh này dùng thêm RAM vì nạp một bản model trong tiến trình riêng; chạy trước khi gửi job AI. Lần đầu backbone DenseNet121 có thể tải weights từ Internet, được giữ trong volume `torch_cache`. CPU chạy chậm hơn GPU, đặc biệt với WSI/tiên lượng đa mô thức.

### Dữ liệu có sẵn sau khi clone

Lần khởi động đầu, backend tự đọc [gói demo](backend/demo_data/README.md), kiểm tra checksum, chép file vào MinIO local và tạo hai hồ sơ thuộc tài khoản `admin`:

| Bệnh nhân | Dữ liệu có sẵn |
|---|---|
| `UCSF-003` | 3 ảnh MRI, 3 bộ WSI × 100 tile, RNA CSV, dữ liệu lâm sàng, kết quả và ảnh XAI đã lưu |
| `UCSF-001` | 3 ảnh MRI, kết quả và ảnh XAI đã lưu; nguồn hiện không có WSI/RNA/lâm sàng |

WSI/RNA của ca `UCSF-003` là mẫu TCGA-12-1093 đã được gán vào hồ sơ để thử chức năng upload đa mô thức. Đây là dữ liệu demo, không phải bộ dữ liệu sinh học ghép cặp của UCSF. Lịch sử kết quả phản ánh dữ liệu tại thời điểm chạy từng lần.

Không cần import SQL, tải lại file hay cung cấp R2 key để xem hai ca này. Khởi động lại không tạo trùng và không ghi đè hồ sơ đã có cùng mã. Để bỏ qua việc tạo ca mẫu, đặt `LOCAL_DEMO_SEED=false` trong `.env` trước lần chạy đầu. Cờ này không xóa dữ liệu đã tạo. Khi chạy lại AI, cần worker đang hoạt động; lần đầu encoder có thể tải thêm pretrained weights qua Internet.

### 6. Xem log, dừng và chạy lại

```powershell
docker compose -f docker-compose.local.yml logs -f backend worker frontend
docker compose -f docker-compose.local.yml stop
docker compose -f docker-compose.local.yml up -d
```

`Ctrl+C` thoát xem log. `stop` giữ dữ liệu; `down` gỡ container/network nhưng giữ named volume. **`down -v` xóa volume database, MinIO và cache**, chỉ dùng khi chủ động muốn xóa dữ liệu. Sau khi sửa `.env`, dùng `up -d --force-recreate backend worker`; `restart` đơn thuần không nạp lại biến môi trường container.

Muốn dùng NVIDIA GPU, xem phần [GPU trong SETUP_GUIDE.md](SETUP_GUIDE.md#6-chay-voi-nvidia-gpu-tuy-chon). Có thể chạy frontend bằng Node.js riêng theo hướng dẫn trong cùng file.

## Website Deployment

Frontend của dự án đã deploy trên **Vercel**: [https://neurodiagnosisai.vercel.app/login](https://neurodiagnosisai.vercel.app/login).

**Tình trạng backend:** kinh phí AWS đã hết nên backend hiện không còn được duy trì trên AWS. Để web Vercel gọi được API và chạy AI, người vận hành phải chạy backend/worker bằng Docker trên máy cá nhân, mở Cloudflare Tunnel, rồi đặt URL tunnel vào biến môi trường của project Vercel và redeploy frontend. Khi máy backend hoặc tunnel tắt, giao diện Vercel có thể vẫn mở nhưng đăng nhập, dữ liệu và inference sẽ không hoạt động.

```text
Trình duyệt -> frontend Vercel -> HTTPS Cloudflare Tunnel
                              -> FastAPI trên máy cá nhân :8001
                              -> PostgreSQL + Redis + Celery worker (CPU/GPU)
                              -> MinIO local (ảnh và kết quả)
```

### Cách A: Dùng ngay bộ local có sẵn dữ liệu, không cần R2

Đây là cách phù hợp cho người vừa clone repo. Hoàn thành phần [chạy local](#getting-started) trước; giữ stack `neuroproject-local` chạy, bao gồm backend, worker, database, Redis và MinIO. Web Vercel sẽ truy cập **chính dữ liệu local này**, gồm hai bệnh nhân demo và các dữ liệu bạn tạo thêm.

1. Trong `.env`, đặt origin frontend (không có `/login`):

   ```env
   FRONTEND_URL=https://neurodiagnosisai.vercel.app
   CORS_ORIGINS=http://localhost:3000,https://neurodiagnosisai.vercel.app
   ```

   Nếu dùng project Vercel riêng, thay domain bằng domain của project đó. Copy key LLM ở [phần cấu hình Chatbox](#llm-setup) vào `.env`.

2. Nạp lại cấu hình và kiểm tra API:

   ```powershell
   docker compose -f docker-compose.local.yml up -d --force-recreate backend worker
   curl.exe http://localhost:8001/health
   ```

3. Cài [cloudflared](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/), mở terminal mới và chạy:

   ```powershell
   winget install --id Cloudflare.cloudflared --exact
   cloudflared --version
   cloudflared tunnel --protocol http2 --url http://localhost:8001
   ```

   Giữ terminal tunnel chạy. Nếu `winget` vừa cài xong mà chưa tìm thấy lệnh, mở một PowerShell mới rồi chạy `cloudflared`. Nếu đã đổi `LOCAL_BACKEND_PORT`, thay `8001` bằng cổng đó.

4. Copy URL HTTPS được in ra, chẳng hạn `https://abc-def-xyz.trycloudflare.com`, kiểm tra `curl.exe https://abc-def-xyz.trycloudflare.com/health` trả `{"status":"ok"}`; sau đó thực hiện phần **Gán URL vào Vercel và redeploy** bên dưới.

5. Đăng nhập web Vercel bằng tài khoản của database local: mặc định `admin` / `123456`. Thử mở ảnh bệnh nhân, upload và chạy pipeline. File/ảnh được frontend truy cập qua backend `/media/...`, nên không cần mở riêng MinIO ra Internet.

Quick Tunnel phù hợp để thử login, hồ sơ và pipeline. **Chatbox dùng SSE; Cloudflare Quick Tunnel không hỗ trợ SSE.** Để dùng đầy đủ Chatbox trên Vercel, tạo [tunnel có tên](TUNNEL_DEPLOY_GUIDE.md#tunnel-co-ten-cho-chatbox) với domain của bạn và trỏ `service` về `http://localhost:8001`; dùng hostname HTTPS cố định đó trong `NEXT_PUBLIC_API_URL`. Chatbox trên web localhost gọi API trực tiếp. [Giới hạn Quick Tunnel](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/#limitations).

### Cách B: Bộ `neuroproject-tunnel` dùng Cloudflare R2

Cách này dành cho người muốn lưu ảnh trên R2 như bộ deploy của tác giả. Nó dùng API cổng **8000**, database riêng và cần credentials R2 của người vận hành; không tự có hai ca demo của bộ local. Có thể chạy cùng bộ local cổng 8001. Nếu chỉ cần demo sau khi clone, dùng **Cách A**.

#### 1. Chuẩn bị backend Docker và R2

Clone và tải model như phần local. Cài [cloudflared](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/) trên máy chạy backend; Windows có thể dùng:

```powershell
winget install --id Cloudflare.cloudflared --exact
cloudflared --version
Copy-Item .env.tunnel.example .env.tunnel
notepad .env.tunnel
```

Mở terminal mới nếu `cloudflared` chưa được nhận diện sau khi cài. Trong [Cloudflare R2](https://developers.cloudflare.com/r2/get-started/), tạo bucket **`medical-data`**, lấy endpoint S3 và API credentials có quyền đọc/ghi bucket. Điền `.env.tunnel`:

```env
POSTGRES_PASSWORD=<mat-khau-database-rieng-dung-chu-va-so>
SECRET_KEY=<khoa-JWT-ngau-nhien-rieng-it-nhat-32-ky-tu>
MINIO_URL=<account-id>.r2.cloudflarestorage.com
MINIO_ACCESS_KEY=<R2-access-key-id>
MINIO_SECRET_KEY=<R2-secret-access-key>
MINIO_SECURE=true
MINIO_REGION=auto
MINIO_BUCKET=medical-data
R2_BUCKET=medical-data
RNA_BUCKET=medical-data
MINIO_PUBLIC_URL=
FRONTEND_URL=https://neurodiagnosisai.vercel.app
CORS_ORIGINS=http://localhost:3000,https://neurodiagnosisai.vercel.app
```

Thay các placeholder bằng giá trị thật; copy hai dòng `GEMINI_API_KEY` và `HF_API_TOKEN` ở [phần cấu hình Chatbox](#llm-setup) vào `.env.tunnel`. `MINIO_URL` chỉ chứa hostname, không có `https://` hoặc tên bucket. Bộ `neuroproject-tunnel` dùng R2, không chạy MinIO local; ảnh được phục vụ qua API `/media/...` để frontend truy cập bằng cùng URL backend.

#### 2. Khởi động backend và worker

Đảm bảo cổng 8000 chưa bị một ứng dụng khác chiếm, rồi chạy:

```powershell
docker compose -f docker-compose.tunnel.yml --env-file .env.tunnel config --quiet
docker compose -f docker-compose.tunnel.yml --env-file .env.tunnel build backend
docker compose -f docker-compose.tunnel.yml --env-file .env.tunnel up -d
docker compose -f docker-compose.tunnel.yml --env-file .env.tunnel ps
curl.exe http://localhost:8000/health
```

Mặc định CPU; để tăng tốc, thêm `-f docker-compose.gpu.yml` sau file tunnel vào mọi lệnh build/up/kiểm tra của stack. Database local và database tunnel ở volume riêng; chuyển chế độ không tự chuyển hồ sơ/dữ liệu giữa hai nơi.

#### 3. Mở tunnel và lấy URL HTTPS

Mở terminal khác, giữ terminal này chạy:

```powershell
cloudflared tunnel --protocol http2 --url http://localhost:8000
```

Copy URL được in ra, ví dụ `https://abc-def-xyz.trycloudflare.com`, rồi kiểm tra:

```powershell
curl.exe https://abc-def-xyz.trycloudflare.com/health
```

Thay URL ví dụ bằng URL thực tế của bạn. Kết quả cần là `{"status":"ok"}`. Quick Tunnel không cần tài khoản Cloudflare; R2 cần tài khoản và credentials riêng.

Chatbox dùng streaming SSE. Quick Tunnel có giới hạn SSE; để dùng đầy đủ chat streaming và có URL ổn định, xem phần **Tunnel có tên cho Chatbox** trong [TUNNEL_DEPLOY_GUIDE.md](TUNNEL_DEPLOY_GUIDE.md). Trên localhost, Chatbox gọi API trực tiếp và không chịu giới hạn của tunnel.

### Gán URL vào Vercel và redeploy (áp dụng cả Cách A và B)

Người có quyền quản lý project Vercel thực hiện:

1. Mở **Vercel Dashboard → project → Settings → Environment Variables** (hoặc mục Environment Variables bên trong môi trường Production).
2. Thêm/sửa biến **`NEXT_PUBLIC_API_URL`**, giá trị là URL gốc HTTPS của tunnel, ví dụ `https://abc-def-xyz.trycloudflare.com`. Không thêm `/login`, `/docs` hoặc `/health`.
3. Chọn **Production**; chọn thêm **Preview** nếu muốn bản preview gọi cùng backend. Lưu cấu hình.
4. Mở **Deployments → deployment production mới nhất → Redeploy** và đợi trạng thái **Ready**. Biến `NEXT_PUBLIC_*` được đưa vào frontend lúc build; sửa biến rồi chỉ reload trình duyệt chưa đủ. [Tài liệu biến môi trường Vercel](https://vercel.com/docs/environment-variables).
5. Mở lại [web demo](https://neurodiagnosisai.vercel.app/login), đăng nhập bằng tài khoản của database backend đang chạy, thử tải MRI và chạy job. Trong DevTools → Network, request API phải đi tới URL tunnel vừa cấu hình.

Nếu tự tạo project Vercel: chọn **Add New → Project**, kết nối GitHub và cấp quyền repo `VuongQuocAn/NeuroProject`, chọn **Import**, đặt **Framework Preset: Next.js**, **Root Directory: frontend**, **Build Command: npm run build**, giữ Output Directory mặc định và thêm biến trên trước khi Deploy. Push lên production branch sau khi kết nối Git sẽ tự tạo deployment mới theo [Git integration](https://vercel.com/docs/git). Dùng URL Vercel của project riêng trong `FRONTEND_URL` và `CORS_ORIGINS`, không thêm đường dẫn `/login` vào origin.

### Mỗi lần mở lại backend/tunnel

Khởi động Docker stack đã chọn (local 8001 hoặc tunnel/R2 8000), mở `cloudflared` trỏ đúng cổng, copy URL mới → sửa `NEXT_PUBLIC_API_URL` → redeploy Vercel → kiểm tra `/health` và đăng nhập. URL Quick Tunnel đổi sau mỗi lần tạo lại; Docker, backend, worker, Internet và tiến trình tunnel cần tiếp tục chạy suốt buổi demo. Tunnel có tên giữ hostname cố định nên không cần sửa URL mỗi lần mở lại.

Bạn có thể gửi link website Vercel cho người xem mà không cấp quyền quản lý Vercel. Clone repo và chạy local cũng không cần quyền trên project Vercel của tác giả. Để sửa biến môi trường của website tác giả, cần chủ project hoặc thành viên có quyền thực hiện; mỗi project Vercel dùng chung một URL backend production tại một thời điểm.

Chi tiết R2, CORS, đổi giữa local/tunnel và lỗi thường gặp: [TUNNEL_DEPLOY_GUIDE.md](TUNNEL_DEPLOY_GUIDE.md). Cloudflare Quick Tunnel dùng cho demo, không có bảo đảm uptime và không hỗ trợ SSE; nếu phiên bản frontend dùng chat streaming thì cần tunnel có tên/domain ổn định hỗ trợ luồng đó. [Giới hạn Quick Tunnel](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/#limitations).

<a id="user-guide"></a>

## Hướng dẫn sử dụng toàn bộ web

**Tài khoản đăng nhập demo:** tên đăng nhập `admin`, mật khẩu `123456`.

Hướng dẫn này áp dụng cho web chạy local tại [localhost:3000](http://localhost:3000/login) và frontend [Vercel](https://neurodiagnosisai.vercel.app/login) khi backend/tunnel đang hoạt động. Đăng nhập trước khi thao tác. Local có sẵn hai bệnh nhân `UCSF-003`, `UCSF-001`; có thể tạo thêm bệnh nhân để thử upload mà vẫn giữ các ca mẫu.

Các ảnh dưới đây là ảnh màn hình bạn cung cấp, được giữ nguyên tỷ lệ và căn giữa. Bấm vào ảnh trong GitHub để mở bản gốc lớn hơn. ID ảnh và một số giá trị có thể khác trong bản clone vì database local cấp ID mới.

<a id="guide-patients"></a>

### 1. Trang bệnh nhân: tìm hồ sơ và tạo bệnh nhân mới

Vào **Bệnh nhân** để xem danh sách. Bảng hiển thị mã bệnh nhân, họ tên, tuổi/giới tính, lần khám gần nhất, chẩn đoán chính, điểm rủi ro AI và nút **Xem chi tiết**. Ô tìm kiếm giúp lọc theo tên hoặc mã. **Xuất dữ liệu đánh giá (CSV)** xuất dữ liệu nghiên cứu có trong tài khoản. Các thẻ thống kê và một số mục “Tải lên gần đây” trong giao diện có thể chứa số liệu minh họa; dùng hồ sơ/API để đối chiếu dữ liệu thực tế.

<p align="center">
  <a href="assets/readme/user-guide/01-patients.png"><img src="assets/readme/user-guide/01-patients.png" width="1200" alt="Danh sách bệnh nhân và nút Thêm bệnh nhân mới" /></a>
</p>

Để tạo hồ sơ:

1. Nhấn **Thêm bệnh nhân mới**.
2. Nhập **Họ và tên** — trường bắt buộc.
3. Nhập **Mã bệnh nhân (External ID)** riêng, ví dụ `DEMO-001`. Mã này dùng để chọn đúng hồ sơ khi upload MRI/WSI/RNA và lâm sàng; tránh trùng mã đã có.
4. Điền **Tuổi**, chọn **Giới tính**, rồi nhấn **Tạo bệnh nhân**. Nếu bỏ trống tuổi, hồ sơ thể hiện chưa có thông tin tuổi.
5. Mở **Xem chi tiết** hoặc tiếp tục tải dữ liệu vào hồ sơ vừa tạo. Hồ sơ thuộc tài khoản tạo nó.

<p align="center">
  <a href="assets/readme/user-guide/02-create-patient.png"><img src="assets/readme/user-guide/02-create-patient.png" width="1200" alt="Biểu mẫu tạo bệnh nhân gồm họ tên, mã bệnh nhân, tuổi và giới tính" /></a>
</p>

<a id="guide-upload"></a>

### 2. Trang Upload: MRI, WSI, RNA và lâm sàng

Trước mỗi lần tải lên, chọn/nhập đúng mã bệnh nhân đã tồn tại. Các tab dùng chung bệnh nhân đang chọn. Dữ liệu được ghi vào PostgreSQL/MinIO ở local hoặc PostgreSQL/R2 ở bản tunnel; đổi tab không tạo một bệnh nhân mới.

#### 2.1. MRI: ảnh đơn hoặc chuỗi DICOM

Mở tab **MRI (Ảnh)**, nhấn **Chọn file / Chuỗi ảnh MRI**, chọn một ảnh PNG/JPG/DICOM được hỗ trợ, nhiều file ảnh/DICOM, hoặc ZIP chứa chuỗi ảnh. Nhấn **Tải lên MRI** hay **Cập nhật** theo trạng thái hiện tại. Ảnh đơn tạo một lần chụp; nhiều file/ZIP đi qua luồng series, quét các lát, tổng hợp quyết định và chọn lát đại diện. ZIP phải chứa file ảnh/DICOM đọc được; tránh file nén lồng nhau.

Để thử nhanh, giải nén [MRI_test.zip](sample_data/MRI_test.zip), chọn **một ảnh** rồi chạy phân tích. Không gộp ảnh của các ca khác nhau thành một chuỗi để diễn giải như dữ liệu của cùng một bệnh nhân. Bộ thử có ảnh raster, không phải bộ DICOM gốc.

<p align="center">
  <a href="assets/readme/user-guide/03-upload-mri.png"><img src="assets/readme/user-guide/03-upload-mri.png" width="1200" alt="Tab upload ảnh MRI đơn, nhiều file hoặc ZIP chuỗi ảnh" /></a>
</p>

#### 2.2. WSI: tile mô bệnh học

Mở tab **WSI (Mô bệnh học)**, nhấn **Chọn file ZIP / WSI Tiles**, chọn [WSI.zip](sample_data/TCGA-12-1093/WSI.zip) hoặc các tile ảnh được hỗ trợ, rồi nhấn **Lọc và Tải lên WSI**. Hệ thống bỏ các file không phải ảnh, lọc tile không phù hợp và chọn tối đa 200 tile theo bộ lọc CNN. ZIP mẫu có 100 tile đã xử lý nên số tile lưu có thể nhỏ hơn giới hạn 200.

Luồng này nhận **tile ảnh**, không tự chuyển một file SVS/whole-slide gốc thành các tile. Dữ liệu WSI được gắn vào bệnh nhân đã chọn và dùng khi chạy tiên lượng; nó không thay ảnh MRI đang được chẩn đoán.

<p align="center">
  <a href="assets/readme/user-guide/04-upload-wsi.png"><img src="assets/readme/user-guide/04-upload-wsi.png" width="1200" alt="Tab upload ZIP chứa các tile WSI mô bệnh học" /></a>
</p>

#### 2.3. RNA: ma trận biểu hiện gene

Mở tab **RNA**, nhấn **Chọn file RNA**, chọn [RNA_sequence.csv](sample_data/TCGA-12-1093/RNA_sequence.csv), rồi **Tải lên RNA**. Chấp nhận CSV/TSV có một dòng mẫu, các cột gene và giá trị biểu hiện số. Đây là bảng biểu hiện gene dùng cho mô hình, không phải dữ liệu đọc thô FASTQ/BAM.

File một mẫu có thể tải vào bất kỳ hồ sơ thuộc tài khoản đang đăng nhập: hệ thống gán mã bệnh nhân trong **bản lưu** cho hồ sơ được chọn và giữ nguyên các giá trị biểu hiện. Không cần sửa mã `TCGA-12-1093` trong file mẫu trước khi thử. Với file nhiều mẫu, hệ thống cần tìm đúng dòng của bệnh nhân; file không xác định được dòng sẽ bị từ chối thay vì gộp dữ liệu của nhiều người.

<p align="center">
  <a href="assets/readme/user-guide/05-upload-rna.png"><img src="assets/readme/user-guide/05-upload-rna.png" width="1200" alt="Tab upload file CSV hoặc TSV biểu hiện RNA-seq cho bệnh nhân được chọn" /></a>
</p>

WSI/RNA mẫu có nguồn từ `TCGA-12-1093`. Gán chúng vào một hồ sơ/MRI khác dùng để thử chức năng đa mô thức; việc gán này không tạo ra bộ dữ liệu sinh học ghép cặp của cùng một bệnh nhân.

#### 2.4. Lâm sàng: ý nghĩa và cách nhập từng chỉ số

Mở tab **Lâm sàng**, kiểm tra mã bệnh nhân, điền thông tin từ hồ sơ/xét nghiệm và nhấn **Lưu thông tin lâm sàng**. Giao diện hiện yêu cầu nhập KI-67 khi lưu; các ô còn lại có thể để **Chưa xác định** khi chưa có kết quả.

| Trường | Ý nghĩa | Cách nhập |
|---|---|---|
| **Mã bệnh nhân / Patient ID** | Mã liên kết dữ liệu với đúng hồ sơ, không phải một chỉ số sinh học | Dùng mã hồ sơ đang thao tác |
| **KI-67 (%)** | Tỷ lệ tế bào trong mẫu mô có dấu ấn tăng sinh Ki-67; phản ánh mức hoạt động phân chia tế bào | Nhập phần trăm `0–100`, ví dụ `15` nghĩa là 15%; không nhập `0.15` để biểu diễn 15% |
| **Bậc u / WHO Grade** | Bậc mô bệnh học/phân loại của khối u; khác với giai đoạn lan rộng của bệnh | Chọn II, III hoặc IV theo kết quả chuyên môn; không suy ra grade từ màu heatmap |
| **Đột biến IDH** | Trạng thái đột biến gene IDH, thường xét IDH1/IDH2; là thông tin phân tử liên quan phân nhóm glioma | Chọn có/không đột biến, hoặc chưa xác định |
| **MGMT Methylation** | Trạng thái methyl hóa vùng promoter của gene sửa chữa DNA MGMT; là dấu ấn phân tử được xét nghiệm trên mẫu phù hợp | Chọn methylated/unmethylated, hoặc chưa xác định; không thay bằng phần trăm KI-67 |

Định nghĩa KI-67 và grade tham khảo [NCI — Ki-67](https://www.cancer.gov/publications/dictionaries/cancer-terms/def/ki-67-proliferation-index), [NCI — Tumor grade](https://www.cancer.gov/publications/dictionaries/cancer-terms/def/tumor-grade). Bối cảnh IDH/MGMT tham khảo [NCI — CNS tumors](https://www.cancer.gov/types/brain/hp/adult-brain-treatment-pdq), [NCI/SEER — MGMT](https://staging.seer.cancer.gov/eod_public/input/3.4/brain/mgmt/).

Tuổi và giới tính lấy từ hồ sơ bệnh nhân. API còn có các trường tiền sử điều trị, trạng thái ban đầu và marker sinh hóa, nhưng không phải tất cả đều có ô nhập trên tab này. Bản encoder hiện đưa KI-67, tuổi, giới tính, grade và một số đặc trưng MRI/điều trị vào vector lâm sàng; IDH/MGMT được lưu trong hồ sơ, chưa được ánh xạ thành hai đầu vào riêng của encoder hiện tại.

<p align="center">
  <a href="assets/readme/user-guide/06-upload-clinical.png"><img src="assets/readme/user-guide/06-upload-clinical.png" width="1200" alt="Form KI-67, WHO Grade, đột biến IDH và MGMT methylation" /></a>
</p>

#### 2.5. Chạy tiên lượng với dữ liệu thiếu khuyết

**Chỉ cần MRI cũng có thể chạy chẩn đoán và nhánh tiên lượng khi MRI phát hiện u.** Không bắt buộc đủ cả MRI, WSI, RNA và lâm sàng. Nếu có các dữ liệu bổ sung, upload chúng vào cùng bệnh nhân rồi chạy phân tích tổng hợp. Với series, hệ thống dùng quy trình tổng hợp và lát đại diện của lần chụp đó.

Mô hình dùng vector và mask cho các nhánh dữ liệu: WSI/RNA thiếu sẽ có đầu vào đệm và trọng số chú ý bị mask; các nhánh có dữ liệu tham gia fusion. Nhánh “Clinical” hiện cũng chứa đặc trưng rút ra từ MRI nên đôi khi vẫn có trọng số dù chưa nhập hồ sơ lâm sàng thủ công. Khi thêm/thay RNA, WSI hoặc lâm sàng, hệ thống nhận diện bộ đầu vào thay đổi để tạo tác vụ tiên lượng mới, thay vì trả nhầm kết quả của bộ dữ liệu cũ.

Nhấn **Chạy phân tích tổng hợp** khi dữ liệu sẵn sàng và đợi tác vụ hoàn tất. **Lần chạy đầu có thể chậm hơn** vì worker nạp các model, tải pretrained weights cho backbone/bộ lọc khi chưa có cache và chuẩn bị dữ liệu. CPU, chuỗi MRI nhiều lát và WSI nhiều tile cũng mất lâu hơn. Worker chạy nền nên có thể tiếp tục xem hồ sơ trong lúc xử lý; tránh bấm chạy liên tục tạo nhiều công việc.

Nếu MRI không phát hiện u, ứng dụng dừng các bước phân loại u/tiên lượng tương ứng và hiển thị nhánh **Không phát hiện khối u**, không tạo một risk score hợp lệ cho lần đó.

<a id="guide-patient-detail"></a>

### 3. Trang chi tiết bệnh nhân: chạy thêm và xem các lần trước

Trang này có hai luồng chính:

1. **Chẩn đoán thêm:** nhấn **Tải dữ liệu mới** hoặc nút cập nhật MRI/WSI/RNA/lâm sàng, chọn đúng bệnh nhân, upload dữ liệu và chạy pipeline. Với ảnh sẵn có chưa được phân tích, dùng nút phân tích của dòng ảnh đó.
2. **Xem kết quả trước:** mỗi dòng MRI là một lần chụp, có thời gian, trạng thái AI (`Ready`, `Pending`, `Done`, `Failed`), loại u, confidence, risk score/risk group và thao tác mở kết quả. Nhấn thumbnail để phóng to; mở chi tiết của đúng dòng để xem XAI và tiên lượng của lần đó. Nút tải báo cáo xuất PDF của ảnh được chọn. Nút thùng rác xóa sau hộp thoại xác nhận.

Các thẻ bên cạnh cho biết bệnh nhân có MRI, WSI, RNA hay lâm sàng chưa; WSI đếm tile, không phải số lần chụp MRI. **Sửa thông tin** cập nhật hồ sơ hành chính. Kết quả qua các lần chụp được giữ để xem diễn tiến và tạo báo cáo lịch sử.

<p align="center">
  <a href="assets/readme/user-guide/07-patient-detail.png"><img src="assets/readme/user-guide/07-patient-detail.png" width="1200" alt="Chi tiết bệnh nhân với từng lần chụp MRI, kết quả và các thao tác" /></a>
</p>

<a id="guide-results"></a>

### 4. Trang kết quả chẩn đoán: đọc MRI, review và tiên lượng

#### 4.1. Bằng chứng MRI và quyền xác nhận của bác sĩ

Trang kết quả hiển thị MRI gốc, **Detection/BBox** khoanh vùng nghi ngờ, ROI, **Segmentation/Mask**, **Tumor contour** và nhãn phân loại. Nhấn ảnh MRI/ảnh kết quả để mở bản lớn hơn. `BBox` là tọa độ vùng phát hiện; **Detection confidence** nói về quyết định phát hiện, còn **Classification confidence** là độ tin cậy của nhãn u. Confidence không phải độ chính xác đo trên một tập test, cũng không phải xác suất kết luận y khoa chắc chắn.

**Khi classification confidence ≤ 95% và chưa có review, hệ thống hiển thị yêu cầu màu đỏ để bác sĩ xem xét/xác nhận ngay.** Bác sĩ vẫn có quyền xác nhận hoặc chỉnh lại nhãn **ở bất kỳ mức confidence nào**, kể cả trên 95% và ca đã xác nhận. Nhấn **Xác nhận / Review phân loại / Xác nhận lại**, chọn nhãn chuyên gia, bổ sung nhận xét và lưu. Hệ thống giữ **AI label** ban đầu, **Expert label**, **Final label** và trạng thái review để đối chiếu; Agent không tự thay quyết định bác sĩ.

<p align="center">
  <a href="assets/readme/user-guide/08-results-classification.png"><img src="assets/readme/user-guide/08-results-classification.png" width="1200" alt="Thông tin phân loại MRI, confidence, class probabilities và chức năng review của bác sĩ" /></a>
</p>

Phần **MRI Core XAI** gồm ODAM giải thích phát hiện, Seg-Eigen-CAM giải thích phân đoạn và Finer-CAM giải thích phân loại. Vùng nóng là nơi mô hình chú ý cho tác vụ đang giải thích; không nên đọc nó như một đường biên khối u được chuyên gia xác nhận.

#### 4.2. Risk score và risk group

**Risk score** là đầu ra nguy cơ tương đối của mô hình fusion/CoxPH: trong cùng cấu hình mô hình, điểm cao hơn tương ứng nguy cơ mô hình ước lượng cao hơn. Nó có thể âm và không phải phần trăm. **Risk group** là nhãn được ứng dụng suy ra từ điểm bằng các ngưỡng hiện tại:

| Khoảng risk score | Risk group |
|---|---|
| `score ≤ -0.5` | `Low` — thấp |
| `-0.5 < score ≤ 0.5` | `Medium` — trung bình |
| `0.5 < score ≤ 1.5` | `High` — cao |
| `score > 1.5` | `Very High` — rất cao |

Ví dụ báo cáo mẫu có risk score khoảng `1.3270`, thuộc `High`. Đây là ngưỡng hiển thị của bản prototype; cần xem cùng đầu vào, XAI và đánh giá chuyên môn, không suy ra thời gian sống trực tiếp từ một điểm số.

#### 4.3. Heatmap CAM tiên lượng

**Grad-CAM, Grad-CAM++ và Layer-CAM** trong phần Multimodal Risk XAI giải thích nhánh hình ảnh đóng góp vào **risk score**, khác với Finer-CAM giải thích **nhãn phân loại u**. Các phương pháp có cách tổng hợp tín hiệu khác nhau nên vùng chú ý không nhất thiết trùng hoàn toàn. Nhấn từng ảnh để phóng to, đối chiếu với MRI/ROI và mask.

<p align="center">
  <a href="assets/readme/user-guide/10-prognosis-cam.png"><img src="assets/readme/user-guide/10-prognosis-cam.png" width="1200" alt="Risk score, risk group và ba heatmap Grad-CAM, Grad-CAM++ và Layer-CAM tiên lượng" /></a>
</p>

#### 4.4. Survival Probability vs Time

Trục ngang là thời gian theo **tháng** (`0, 6, 12, 18, 24, 30, 36`); trục dọc là giá trị xác suất sống còn mà ứng dụng vẽ tại các mốc. Đường giảm nhanh hơn thể hiện tiên lượng mà mô hình đánh giá bất lợi hơn. Đây là đường cong riêng cho kết quả đang xem, khác với biểu đồ risk score giữa nhiều lần chẩn đoán ở trang Lịch sử.

Bản triển khai hiện xây dựng đường cong từ baseline cố định và risk score theo `S(t) = S0(t)^exp(score)`, có chặn score để ổn định số học. Vì vậy hình bậc thang trong web/PDF là minh họa đầu ra tiên lượng của prototype, **không phải ước lượng Kaplan–Meier tính từ dữ liệu theo dõi của bệnh nhân này**, và chưa phải xác suất đã được hiệu chuẩn lâm sàng.

<p align="center">
  <a href="assets/readme/user-guide/11-survival-probability.png"><img src="assets/readme/user-guide/11-survival-probability.png" width="1200" alt="Biểu đồ Survival Probability vs Time tại các mốc tháng" /></a>
</p>

#### 4.5. Fusion Attention Weights và RNA-seq Feature Importance

**Fusion Attention Weights** hiển thị trọng số chú ý của MRI, WSI, RNA và Clinical trong phép tổng hợp của lần chạy này. Các nhánh thiếu được mask; phần trăm thể hiện mức chú ý của mô hình, không phải tỷ lệ chính xác hay bằng chứng nhân quả. Trong ví dụ, RNA chiếm khoảng 60.9%, MRI 25.7%, WSI 5.3% và Clinical 8.2%; các giá trị thay đổi theo dữ liệu và mô hình.

**Phân tích dấu ấn phân tử / RNA-seq Feature Importance** hiển thị tối đa **10 gene có giá trị biểu hiện đầu vào và đóng góp tuyệt đối lớn nhất** vào dự đoán risk score theo Input × Gradient. Nó không đơn thuần là danh sách gene có biểu hiện cao nhất:

| Thành phần | Cách đọc |
|---|---|
| **Gene / Ensembl ID** | Tên gene đã ánh xạ; nếu không ánh xạ được có thể giữ mã Ensembl |
| **Importance** | Đóng góp có dấu vào đầu ra nguy cơ; độ dài thanh thể hiện độ lớn tuyệt đối |
| **Đỏ / High Risk** | Đóng góp dương, đẩy risk score theo hướng tăng trong mô hình |
| **Xanh / Protective** | Đóng góp âm, đẩy risk score theo hướng giảm trong mô hình |
| **Expression** | Giá trị biểu hiện mà mô hình dùng sau tiền xử lý; không mặc định là số lượng đọc thô |

Chỉ có phần gene khi có RNA và attribution hợp lệ; có thể dưới 10 gene khi không đủ đóng góp đáng kể. “Protective/High Risk” mô tả dấu đóng góp trong lần chạy, không khẳng định gene đó tự gây hoặc bảo vệ khỏi bệnh. Ví dụ PDF có LAMTOR4 đóng góp dương, PPFIA1 đóng góp âm; không dùng danh sách này để tự quyết định điều trị.

<p align="center">
  <a href="assets/readme/user-guide/09-fusion-rna-importance.png"><img src="assets/readme/user-guide/09-fusion-rna-importance.png" width="1200" alt="Trọng số fusion bốn nhánh và biểu đồ top 10 gene đóng góp vào dự đoán rủi ro" /></a>
</p>

#### 4.6. Báo cáo MRI PDF

Nút tải báo cáo ở dòng ảnh/trang kết quả xuất bằng chứng và diễn giải của lần chụp được chọn. [Kết quả chẩn đoán MRI mẫu — PDF](assets/readme/reports/MRI_diagnosis_report.pdf) được đính kèm trong repo: gồm thông tin ca và confidence, MRI/BBox/ROI/mask/contour, MRI Core XAI, risk score/risk group, CAM tiên lượng, trọng số fusion, survival curve và bảng gene. Báo cáo là bản chụp kết quả tại lúc xuất; chạy lại hoặc review sau đó có thể tạo nội dung mới khác bản PDF này.

<a id="guide-agent"></a>

### 5. Chatbox Agentic: hỏi theo ngữ cảnh và thực hiện workflow

Nhấn biểu tượng robot ở góc màn hình để mở **NeuroDiagnosis Agent**. Agent nhận ngữ cảnh trang đang mở, bệnh nhân/ảnh đang chọn và vùng ảnh nếu có; dùng planner → kiểm tra tool → truy vấn dữ liệu → tổng hợp câu trả lời. Có thể hỏi kiến thức chung, nhưng câu hỏi về bệnh nhân cần đúng hồ sơ hoặc mã/ID cụ thể.

Các chức năng hiện có trong code:

| Nhóm | Chức năng và ví dụ |
|---|---|
| **Hồ sơ bệnh nhân** | Đọc thông tin hành chính, lâm sàng, tình trạng dữ liệu; “Tóm tắt hồ sơ bệnh nhân này” |
| **Kết quả một ảnh** | Giải thích nhãn AI, confidence, nhãn bác sĩ/final label, risk score, XAI; “Giải thích kết quả MRI đang mở” |
| **Lịch sử và timeline** | Đọc toàn bộ các lần chẩn đoán có trong hệ thống, so sánh nhãn/confidence/risk, chỉ ra thay đổi và dữ liệu thiếu; “Phân tích lịch sử chẩn đoán của UCSF-003” |
| **Giải thích XAI và báo cáo** | Trình bày bằng chứng ảnh/heatmap và tóm tắt diễn giải/báo cáo có trong ngữ cảnh; không suy đoán một ROI không có dữ liệu |
| **Thống kê hệ thống trong phạm vi tài khoản** | Đếm bệnh nhân, ảnh, chẩn đoán, bệnh nhân có RNA, số ca từng mang nhãn u; phân biệt số người với số lần chẩn đoán |
| **Thống kê review** | Đếm ca confidence ≤95%, ca chờ bác sĩ review và review đã hoàn thành; “Có bao nhiêu ca cần xem xét lại?” |
| **Thông báo** | Liệt kê cảnh báo review và dữ liệu tiên lượng bất thường, liên kết tới workflow liên quan |
| **Chẩn đoán MRI nhanh** | Đính kèm MRI trong chat; chọn bệnh nhân nếu chưa có ngữ cảnh; upload, chạy tác vụ MRI, theo dõi hoàn thành và xem tóm tắt/kết quả chi tiết |
| **Kiến thức chung** | Hỏi thuật ngữ MRI/u não hoặc cách dùng web; phân biệt kiến thức chung với dữ liệu hồ sơ thực tế |
| **Quản lý hội thoại** | Tạo chat mới (`+`), mở lịch sử, khôi phục phiên, xóa sau xác nhận; backend có luồng tóm tắt/cắt bớt lịch sử để quản lý ngữ cảnh |
| **Giao diện và dẫn hướng** | Mở/đóng, mở rộng cửa sổ, xem ảnh lớn trong câu trả lời, chuyển tới trang kết quả của workflow MRI |

Agent truy vấn dữ liệu theo quyền tài khoản, nhớ ngữ cảnh hội thoại và phân biệt **AI label / Expert label / Final label / Review status**. Nếu chưa biết bệnh nhân hay image ID, cần chọn hồ sơ để tiếp tục. Agent giải thích và hỗ trợ điều hướng; việc sửa nhãn cuối phải qua thao tác review/xác nhận của bác sĩ, không tự sửa nhãn từ một câu trò chuyện.

Các luồng tạo diễn giải XAI/báo cáo còn dùng kho RAG của dự án và LLM. Gemini key cho phép gọi Agent/LLM; Hugging Face token phục vụ embedding/reranking của RAG. Copy hai key demo ở [phần cấu hình Chatbox/LLM](#llm-setup) vào file cấu hình; dịch vụ vẫn cần Internet và hạn mức còn hiệu lực.

<p align="center">
  <a href="assets/readme/user-guide/12-agent-chat.png"><img src="assets/readme/user-guide/12-agent-chat.png" width="640" alt="Chatbox Agent phân tích lịch sử bệnh nhân, timeline và kết quả MRI" /></a>
</p>

<a id="guide-history"></a>

### 6. Lịch sử chẩn đoán và cập nhật báo cáo

#### 6.1. Bảng tổng hợp lịch sử

Mở **Lịch sử chẩn đoán**. Dùng tìm kiếm theo tên/mã, lọc risk group/review, chọn thứ tự và nhấn **Làm mới** để cập nhật bảng.

| Cột | Nội dung |
|---|---|
| **Mã / Tên bệnh nhân** | Xác định hồ sơ |
| **Chẩn đoán cuối** | Thời điểm của lần chẩn đoán gần nhất |
| **Nhãn phân loại** | Nhãn kết quả gần nhất, có xét trạng thái xác nhận/chỉnh sửa |
| **Confidence** | Độ tin cậy phân loại AI của lần đó |
| **Risk score / Risk group** | Điểm và nhóm nguy cơ tương ứng, không phải tổng cộng của các lần |
| **Review** | Chờ review, không cần review, đã xác nhận hoặc đã chỉnh nhãn |
| **Số lần** | Số lần chẩn đoán được ghi nhận trong timeline, không phải số tile WSI |
| **Thao tác** | Sinh/cập nhật báo cáo, mở báo cáo và tải PDF tùy trạng thái |

Nhấn **Sinh báo cáo** để tổng hợp lịch sử. Khi chạy một chẩn đoán mới hoặc thay đổi kết quả/review làm dữ liệu báo cáo khác trước, hệ thống phát hiện báo cáo cũ **Cần cập nhật**. Cần nhấn **Sinh/Cập nhật báo cáo** để LLM tạo lại nội dung từ dữ liệu mới; không hiểu rằng bản narrative/PDF cũ tự được viết lại ngay khi job AI hoàn tất. **Làm mới** tải trạng thái, còn **Sinh báo cáo** tạo lại nội dung.

<p align="center">
  <a href="assets/readme/user-guide/13-history-list.png"><img src="assets/readme/user-guide/13-history-list.png" width="1200" alt="Bảng lịch sử với lần chẩn đoán cuối, confidence, risk, review và trạng thái báo cáo cần cập nhật" /></a>
</p>

#### 6.2. Báo cáo chi tiết và bảng từng lần chẩn đoán

Mở báo cáo của bệnh nhân để xem số lần, kết quả mới nhất, risk score/risk group, tóm tắt, xu hướng phân loại/nguy cơ và kết luận tổng hợp. Bảng **Timeline** liệt kê từng lần theo thứ tự thời gian: số lần, ngày, thumbnail MRI, loại dữ liệu, nhãn phân loại, confidence, risk và trạng thái. Nhấn ảnh để xem lớn, đối chiếu bằng chứng giữa các mốc.

<p align="center">
  <a href="assets/readme/user-guide/14-history-timeline.png"><img src="assets/readme/user-guide/14-history-timeline.png" width="1200" alt="Báo cáo lịch sử bệnh nhân và timeline từng lần MRI" /></a>
</p>

#### 6.3. Sơ đồ diễn tiến tiên lượng

Biểu đồ **Diễn tiến tiên lượng theo thời gian** nối các risk score đã có qua từng lần chẩn đoán; trục ngang thể hiện lần chẩn đoán, trục dọc là risk score. Điểm đánh dấu kèm nhóm nguy cơ giúp so sánh tăng/giảm. Kết quả không có risk hợp lệ không được diễn giải như điểm 0 bình thường. Cần xét việc MRI/WSI/RNA/lâm sàng hoặc mô hình thay đổi giữa các lần trước khi kết luận đây là diễn tiến sinh học.

<p align="center">
  <a href="assets/readme/user-guide/15-history-risk-chart.png"><img src="assets/readme/user-guide/15-history-risk-chart.png" width="1200" alt="Biểu đồ risk score của bệnh nhân qua các lần chẩn đoán" /></a>
</p>

[Báo cáo lịch sử UCSF-003 mẫu — PDF](assets/readme/reports/patient_history_report_UCSF-003.pdf) gồm tóm tắt, diễn tiến phân loại, diễn tiến risk, kết luận và timeline. Dùng chức năng tải PDF trong web để xuất bản mới sau khi cập nhật báo cáo.

<a id="guide-neuroboard"></a>

### 7. NeuroBoard / Tumor Board: trao đổi và hội chẩn ca khó

Mở **NeuroBoard** từ menu. Có **Feed**, **Review Queue**, **My Cases**, **Saved** và bộ lọc bài thường/Clinical Case. Tìm kiếm nội dung, tải lại feed hoặc xem danh sách bác sĩ ở cột bên cạnh. Các tài khoản/bài demo được tạo để thử chức năng; quyền đọc hồ sơ và quyền xem một case được chia sẻ là các phạm vi riêng.

<p align="center">
  <a href="assets/readme/user-guide/16-neuroboard-feed.png"><img src="assets/readme/user-guide/16-neuroboard-feed.png" width="1200" alt="NeuroBoard với Feed, Review Queue, My Cases, Saved và trình soạn bài" /></a>
</p>

#### 7.1. Bài đăng thường: thích, bình luận và lưu

Chọn **Bài đăng thường**, nhập nội dung, thêm ảnh nếu cần và nhấn **Post**. Người dùng có thể nhấn **Hữu ích/Thích**, mở phần **bình luận** để trao đổi và nhấn **Lưu** để tìm lại trong **Saved**. Bài dài có **See more**; ảnh có thể mở lớn. Các thao tác và nội dung phải tuân theo quyền của tài khoản đang đăng nhập.

<p align="center">
  <a href="assets/readme/user-guide/17-neuroboard-normal-post.png"><img src="assets/readme/user-guide/17-neuroboard-normal-post.png" width="800" alt="Bài đăng thường với nút Hữu ích, bình luận và Lưu" /></a>
</p>

#### 7.2. Clinical Case: chia sẻ một ca cần hội chẩn

Chọn **Clinical Case** trong trình soạn bài, chọn bệnh nhân/lần chụp đã có kết quả, nhập vấn đề cần hội chẩn rồi đăng. Case hiển thị nhãn AI, confidence, risk score/risk group và bằng chứng MRI/BBox/mask/contour/XAI của lần chụp được chọn. Dùng **Mở chi tiết case** để vào không gian hội chẩn; không nhầm một ảnh gallery minh họa với kết quả của một lần chụp khác.

<p align="center">
  <a href="assets/readme/user-guide/18-neuroboard-clinical-case.png"><img src="assets/readme/user-guide/18-neuroboard-clinical-case.png" width="800" alt="Clinical Case với nhãn AI, confidence, risk và các ảnh bằng chứng" /></a>
</p>

#### 7.3. Bounding box và ROI comment gửi vào cuộc hội chẩn

Trong chi tiết case:

1. Chọn ảnh/bằng chứng muốn bàn luận, ví dụ MRI hoặc một ảnh XAI.
2. Kéo chuột khoanh **bounding box / ROI** trên vùng quan tâm. ROI comment là vùng bác sĩ đánh dấu để trao đổi, không tự thay BBox phát hiện của AI.
3. Nhập nhận xét về vùng đó, nhấn **Gửi ROI comment**.
4. Nhận xét kèm tọa độ/vùng ảnh xuất hiện trong **Trao đổi ROI trong case** để các bác sĩ có quyền xem case cùng đối chiếu. Có thể trả lời hoặc thu hồi nhận xét theo quyền trong giao diện.

<p align="center">
  <a href="assets/readme/user-guide/19-neuroboard-roi-comment.png"><img src="assets/readme/user-guide/19-neuroboard-roi-comment.png" width="1200" alt="Chi tiết ca với ROI bounding box trên MRI và trao đổi vùng quan tâm giữa bác sĩ" /></a>
</p>

Đây là chia sẻ/thảo luận trong NeuroBoard của ứng dụng. Chỉ ghi nhận nhãn cuối qua luồng review được phép; một ROI comment không tự trở thành kết luận chẩn đoán hay thay đổi kết quả của mô hình.

<a id="guide-no-tumor"></a>

### 8. Nhánh không phát hiện khối u

Khi detector không phát hiện khối u, trang kết quả hiển thị cảnh báo **Không phát hiện khối u. Cần tham khảo ý kiến chuyên gia** cùng ảnh detection. Bác sĩ cần xem xét lại MRI và bối cảnh lâm sàng; đây là kết quả âm tính của mô hình, không phải bảo đảm người bệnh không có tổn thương. Các trường phân loại u và risk/tiên lượng của lần này không được dùng như một kết quả u hợp lệ.

<p align="center">
  <a href="assets/readme/user-guide/20-no-tumor.png"><img src="assets/readme/user-guide/20-no-tumor.png" width="1200" alt="Kết quả không phát hiện khối u và khuyến nghị bác sĩ kiểm tra lại" /></a>
</p>

<a id="guide-test-files"></a>

### 9. Quy trình thử web bằng các file có sẵn trong repo

1. Chạy local, đăng nhập `admin / 123456`; mở `UCSF-003` hoặc `UCSF-001` để kiểm tra dữ liệu/kết quả có sẵn.
2. Tạo bệnh nhân mới với mã riêng, ví dụ `DEMO-001`.
3. Giải nén [MRI_test.zip](sample_data/MRI_test.zip), upload một ảnh vào `DEMO-001`, chạy chẩn đoán/tiên lượng để thử trường hợp chỉ có MRI. Lần đầu có thể mất lâu hơn.
4. Trong cùng hồ sơ, upload [WSI.zip](sample_data/TCGA-12-1093/WSI.zip), [RNA_sequence.csv](sample_data/TCGA-12-1093/RNA_sequence.csv), nhập KI-67/grade nếu muốn thử, rồi chạy lại tổng hợp. Đây là ca demo ghép dữ liệu để kiểm tra kỹ thuật.
5. Mở kết quả của đúng ảnh, phóng to MRI/XAI, thử review nhãn, xem risk, survival, fusion attention và các gene khi có RNA.
6. Mở Lịch sử, sinh/cập nhật báo cáo và tải PDF. Hỏi Agent về hồ sơ/lịch sử; thử tạo một Clinical Case và gửi ROI comment qua tài khoản bác sĩ.
7. Dùng `docker compose -f docker-compose.local.yml stop` rồi `up -d`; dữ liệu đã lưu trong volume vẫn còn. Các ảnh/PDF minh họa trong README là bản ghi tại thời điểm chụp/xuất, không hứa kết quả của mọi lần chạy mới giống hệt số trong ảnh.

Xem [danh mục file thử](sample_data/README.md), [gói bệnh nhân tự tạo](backend/demo_data/README.md), [báo cáo MRI mẫu](assets/readme/reports/MRI_diagnosis_report.pdf) và [báo cáo lịch sử mẫu](assets/readme/reports/patient_history_report_UCSF-003.pdf).
